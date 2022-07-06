//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "../src/Compiler.hpp"
#include "../src/cascading/CascadingCommandStreamGenerator.hpp"
#include "../src/cascading/CombinerDFS.hpp"
#include "../src/cascading/StripeHelper.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <fstream>

using namespace ethosn::support_library;
using namespace ethosn::support_library::impl;
using namespace ethosn::command_stream;
using namespace ethosn::command_stream::cascading;
using namespace cascading_compiler;
using PleKernelId = ethosn::command_stream::cascading::PleKernelId;

//////////////////////////////////////////////////////////////////////////////////////////////
// Cascading Command Stream Generation Testing Classes
//////////////////////////////////////////////////////////////////////////////////////////////

class StandalonePleOpGraph
{
public:
    StandalonePleOpGraph()
    {
        const std::set<uint32_t> operationIds = { 0 };
        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;
        const std::vector<Buffer*>& buffers = m_OpGraph.GetBuffers();
        const std::vector<Op*>& ops         = m_OpGraph.GetOps();
        // Plan inputDramPlan
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 160, 160, 3 }, TensorShape{ 0, 0, 0, 0 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_OperationId = 0;
        m_OpGraph.GetBuffers().back()->m_BufferType  = BufferType::Input;
        m_OpGraph.GetBuffers().back()->m_DebugTag    = "InputDramBuffer";

        m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        m_OpGraph.GetOps()[0]->m_DebugTag     = "InputDmaOp";
        m_OpGraph.GetOps()[0]->m_OperationIds = { 0 };

        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Plan standalone plePlan
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 160, 160, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                     TraversalOrder::Xyz, 4, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateInputSramBuffer";
        m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000000F;
        Buffer* sramBuffer                        = buffers.back();
        sramBuffer->m_SlotSizeInBytes             = 1 * 8 * 8 * 16;
        m_OpGraph.SetProducer(buffers.back(), ops.back());
        auto pleOp =
            std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::LEAKY_RELU,
                                    BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                    TensorShape{ 1, 8, 8, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp.get()->m_Offset     = 0x000000FF;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp =
            AddPleToOpGraph(m_OpGraph, Lifetime::Cascade, TensorShape{ 1, 8, 8, 32 }, numMemoryStripes,
                            std::move(pleOp), TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
        m_OpGraph.GetBuffers().back()->m_Offset = 0x00000F00;
        m_OpGraph.AddConsumer(sramBuffer, ops.back(), 0);

        m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        m_OpGraph.GetOps()[0]->m_DebugTag = "OutputDmaOp";

        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Plan outputDramPlan
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_OperationId        = 1;
        m_OpGraph.GetBuffers().back()->m_ProducerOutputIndx = 0;
        m_OpGraph.GetBuffers().back()->m_BufferType         = BufferType::Output;
        m_OpGraph.GetBuffers().back()->m_DebugTag           = "OutputDramBuffer";

        m_OpGraph.SetProducer(buffers.back(), ops.back());

        bool dumpOpGraphToFile = true;
        if (dumpOpGraphToFile)
        {
            std::ofstream stream("CascadingCommandStreamGenerator PleOnlySchedulerAgent Input.dot");
            SaveOpGraphToDot(m_OpGraph, stream, DetailLevel::High);
        }

        ETHOSN_UNUSED(outBufferAndPleOp);
    }

    OpGraph GetMergedOpGraph()
    {
        return m_OpGraph;
    }

private:
    OwnedOpGraph m_OpGraph;
};

class MceOpGraph
{
public:
    MceOpGraph()
    {
        const std::set<uint32_t> operationIds = { 0 };
        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;
        const std::vector<Buffer*>& buffers = m_OpGraph.GetBuffers();
        const std::vector<Op*>& ops         = m_OpGraph.GetOps();

        // Plan inputDramPlan
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 160, 160, 3 }, TensorShape{ 0, 0, 0, 0 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_OperationId = 0;
        m_OpGraph.GetBuffers().back()->m_BufferType  = BufferType::Input;
        m_OpGraph.GetBuffers().back()->m_DebugTag    = "InputDramBuffer";
        m_OpGraph.GetBuffers().back()->m_Offset      = 0x00000F0A;

        m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        m_OpGraph.GetOps()[0]->m_DebugTag     = "InputDmaOp";
        m_OpGraph.GetOps()[0]->m_OperationIds = { 0 };
        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Plan inputSramPlan
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 160, 160, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                     TraversalOrder::Zxy, 4, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag        = "InputSramBuffer";
        m_OpGraph.GetBuffers().back()->m_Offset          = 0x00000F0F;
        m_OpGraph.GetBuffers().back()->m_NumStripes      = 4;
        m_OpGraph.GetBuffers().back()->m_SlotSizeInBytes = 8 * 8 * 16;
        m_OpGraph.SetProducer(buffers.back(), ops.back());

        Buffer* ptrInputBuffer = m_OpGraph.GetBuffers().back();
        inputStripeSize        = utils::TotalSizeBytesNHWCB(ptrInputBuffer->m_StripeShape);
        inputZeroPoint         = ptrInputBuffer->m_QuantizationInfo.GetZeroPoint();

        // Plan weightDramPlan
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT,
                                                     TensorShape{ 1, 3, 1, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        m_OpGraph.GetBuffers().back()->m_DebugTag       = "WeightDramBuffer";
        encodedWeights                                  = std::make_shared<EncodedWeights>();
        encodedWeights->m_Data                          = { 1, 2, 3, 4 };
        encodedWeights->m_MaxSize                       = 10;
        encodedWeights->m_Metadata                      = { { 0, 2 }, { 2, 2 } };
        m_OpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights;

        m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::WEIGHT));
        m_OpGraph.GetOps().back()->m_DebugTag = "WeightDmaOp";
        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Plan weightSramPlan
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT,
                                                     TensorShape{ 1, 3, 1, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                     TraversalOrder::Xyz, 4, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag        = "WeightSramBuffer";
        m_OpGraph.GetBuffers().back()->m_Offset          = 0x00000FF0;
        m_OpGraph.GetBuffers().back()->m_NumStripes      = 3;
        m_OpGraph.GetBuffers().back()->m_SizeInBytes     = encodedWeights->m_MaxSize;
        m_OpGraph.GetBuffers().back()->m_SlotSizeInBytes = encodedWeights->m_MaxSize;
        m_OpGraph.SetProducer(buffers.back(), ops.back());

        Buffer* ptrWeightBuffer = m_OpGraph.GetBuffers().back();
        weightSize              = ptrWeightBuffer->m_SizeInBytes / ptrWeightBuffer->m_NumStripes;
        kernelHeight            = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[0]);
        kernelWidth             = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[1]);

        // Plan mcePlePlan
        m_OpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        m_OpGraph.GetOps().back()->m_DebugTag = "MceOp";

        m_OpGraph.AddConsumer(ptrInputBuffer, ops.back(), 0);    // connect input sram buffer
        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 1);    // connect weights sram buffer

        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputPleInputSramBuffer";
        m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F00F;
        Buffer* pleInBuffer                       = buffers.back();

        m_OpGraph.SetProducer(buffers.back(), ops.back());

        ifmDeltaDefaultHeight = 0;
        ifmDeltaDefaultWidth  = 1;
        ifmDeltaEdgeHeight    = static_cast<int8_t>(ptrInputBuffer->m_TensorShape[1] - pleInBuffer->m_TensorShape[1]);
        ifmDeltaEdgeWidth     = static_cast<int8_t>(ptrInputBuffer->m_TensorShape[2] - pleInBuffer->m_TensorShape[2]);

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        pleOp = std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                        BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                        TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp.get()->m_Offset     = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp =
            AddPleToOpGraph(m_OpGraph, Lifetime::Cascade, TensorShape{ 1, 4, 4, 32 }, numMemoryStripes,
                            std::move(pleOp), TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
        m_OpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        m_OpGraph.AddConsumer(pleInBuffer, ops.back(), 0);

        m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        m_OpGraph.GetOps()[0]->m_DebugTag = "OutputDmaOp";
        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Plan outputDramPlan
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_OperationId        = 2;
        m_OpGraph.GetBuffers().back()->m_ProducerOutputIndx = 0;
        m_OpGraph.GetBuffers().back()->m_BufferType         = BufferType::Output;
        m_OpGraph.GetBuffers().back()->m_DebugTag           = "OutputDramBuffer";
        m_OpGraph.SetProducer(buffers.back(), ops.back());

        bool dumpOpGraphToFile = false;
        if (dumpOpGraphToFile)
        {
            std::ofstream stream("CascadingCommandStreamGenerator_MceSchedulerAgent_Output.dot");
            SaveOpGraphToDot(m_OpGraph, stream, DetailLevel::High);
        }

        ETHOSN_UNUSED(outBufferAndPleOp);
    }

    OpGraph GetMergedOpGraph()
    {
        return m_OpGraph;
    }

    uint32_t getInputStripeSize()
    {
        return inputStripeSize;
    }

    uint32_t getWeightSize()
    {
        return weightSize;
    }

    int32_t getInputZeroPoint()
    {
        return inputZeroPoint;
    }

    uint8_t getKernelHeight()
    {
        return kernelHeight;
    }

    uint8_t getKernelWidth()
    {
        return kernelWidth;
    }

    int8_t getIfmDeltaDefaultHeight()
    {
        return ifmDeltaDefaultHeight;
    }

    int8_t getIfmDeltaDefaultWidth()
    {
        return ifmDeltaDefaultWidth;
    }

    int8_t getIfmDeltaEdgeHeight()
    {
        return ifmDeltaEdgeHeight;
    }

    int8_t getIfmDeltaEdgeWidth()
    {
        return ifmDeltaEdgeWidth;
    }

private:
    GraphOfParts graph;

    Plan inputDramPlan;
    Plan inputSramPlan;
    Plan weightDramPlan;
    Plan weightSramPlan;
    Plan mcePlePlan;
    Plan outputDramPlan;

    std::shared_ptr<EncodedWeights> encodedWeights;

    std::unique_ptr<PleOp> pleOp;

    uint32_t inputStripeSize;
    uint32_t weightSize;
    int32_t inputZeroPoint;

    uint8_t kernelHeight;
    uint8_t kernelWidth;
    int8_t ifmDeltaDefaultHeight;
    int8_t ifmDeltaDefaultWidth;
    int8_t ifmDeltaEdgeHeight;
    int8_t ifmDeltaEdgeWidth;

    Combination comb;

    OwnedOpGraph m_OpGraph;
};

// This class creates a network consisting of an Intermediate Dram Buffer with multiple consumers
class MceOpGraphIntermediateDramBuffers
{
public:
    MceOpGraphIntermediateDramBuffers()
    {
        const std::set<uint32_t> operationIds = { 0 };
        const std::vector<Buffer*>& buffers   = m_OpGraph.GetBuffers();
        const std::vector<Op*>& ops           = m_OpGraph.GetOps();

        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 160, 160, 3 }, TensorShape{ 0, 0, 0, 0 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_OperationId = 0;
        m_OpGraph.GetBuffers().back()->m_BufferType  = BufferType::Input;
        m_OpGraph.GetBuffers().back()->m_DebugTag    = "InputDramBuffer";

        m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        m_OpGraph.GetOps()[0]->m_DebugTag     = "InputDmaOp";
        m_OpGraph.GetOps()[0]->m_OperationIds = { 0 };
        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 160, 160, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramBuffer";
        m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000000F;
        m_OpGraph.SetProducer(buffers.back(), ops.back());
        Buffer* inputSramBuffer = buffers.back();

        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT,
                                                     TensorShape{ 1, 1, 3, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        m_OpGraph.GetBuffers().back()->m_DebugTag       = "WeightsDramBuffer";
        encodedWeights                                  = std::make_shared<EncodedWeights>();
        encodedWeights->m_Data                          = { 1, 2, 3, 4 };
        encodedWeights->m_MaxSize                       = 10;
        encodedWeights->m_Metadata                      = { { 0, 2 }, { 2, 2 } };
        m_OpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights;

        m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::WEIGHT));
        m_OpGraph.GetOps()[1]->m_DebugTag = "WeightsDmaOp";
        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                     TraversalOrder::Xyz, 4, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag = "WeightsSramBuffer";
        m_OpGraph.GetBuffers().back()->m_Offset   = 0x000000F0;
        m_OpGraph.SetProducer(buffers.back(), ops.back());

        m_OpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

        m_OpGraph.AddConsumer(inputSramBuffer, ops.back(), 0);
        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 1);

        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag = "PleSramBuffer";
        m_OpGraph.GetBuffers().back()->m_Offset   = 0x000000FF;
        m_OpGraph.SetProducer(buffers.back(), ops.back());

        m_OpGraph.GetOps()[2]->m_DebugTag = "Mce";

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        m_OpGraph.AddOp(
            std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                    BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                    TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true));
        m_OpGraph.GetOps()[3]->m_DebugTag = "Ple";
        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Get the PleOp from the OpGraph, check that it is indeed a PleOp and set the Offset
        Op* maybePleOp = ops.back();
        CHECK(IsPleOp(maybePleOp));
        PleOp* actualPleOp    = static_cast<PleOp*>(maybePleOp);
        actualPleOp->m_Offset = 0x00000F00;

        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 80, 80, 24 }, TensorShape{ 1, 4, 4, 32 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramBuffer";
        m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000F0F;
        m_OpGraph.SetProducer(buffers.back(), ops.back());

        m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        ops.back()->m_DebugTag = "InputDma";
        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
        m_OpGraph.GetBuffers().back()->m_DebugTag   = "IntermediateDramBuffer";
        m_OpGraph.SetProducer(buffers.back(), ops.back());

        Buffer* intermediateDramBuffer = buffers.back();

        m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        ops.back()->m_DebugTag = "OutputDmaBranchA";
        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Plan B
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 80, 80, 24 }, TensorShape{ 1, 8, 8, 32 },
                                                     TraversalOrder::Xyz, 4, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag = "SramBufferBranchA";
        m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000FF0;
        m_OpGraph.SetProducer(buffers.back(), ops.back());

        m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        m_OpGraph.GetOps()[0]->m_DebugTag = "DmaOpBranchA";
        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Plan C
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_OperationId        = 2;
        m_OpGraph.GetBuffers().back()->m_ProducerOutputIndx = 0;
        m_OpGraph.GetBuffers().back()->m_BufferType         = BufferType::Output;
        m_OpGraph.GetBuffers().back()->m_DebugTag           = "OutputDramBufferBranchA";
        m_OpGraph.SetProducer(buffers.back(), ops.back());

        m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        ops.back()->m_DebugTag = "OutputDmaBranchB";
        m_OpGraph.AddConsumer(intermediateDramBuffer, ops.back(), 0);

        // Plan D
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 80, 80, 24 }, TensorShape{ 1, 8, 8, 32 },
                                                     TraversalOrder::Xyz, 4, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag = "SramBufferBranchB";
        m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000FFF;
        m_OpGraph.SetProducer(buffers.back(), ops.back());

        m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        m_OpGraph.GetOps()[0]->m_DebugTag = "DmaOpBranchB";
        m_OpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Plan E
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_OperationId        = 2;
        m_OpGraph.GetBuffers().back()->m_ProducerOutputIndx = 0;
        m_OpGraph.GetBuffers().back()->m_BufferType         = BufferType::Output;
        m_OpGraph.GetBuffers().back()->m_DebugTag           = "OutputDramBufferBranchB";
        m_OpGraph.SetProducer(buffers.back(), ops.back());

        bool dumpOutputGraphToFile = false;
        if (dumpOutputGraphToFile)
        {
            std::ofstream stream("IntermediateDramBufferLifetime Test Output.dot");
            SaveOpGraphToDot(m_OpGraph, stream, DetailLevel::High);
        }
    }

    OpGraph GetMergedOpGraph()
    {
        return m_OpGraph;
    }

private:
    GraphOfParts graph;

    Plan planA;
    Plan planB;
    Plan planC;
    Plan planD;
    Plan planE;

    std::shared_ptr<EncodedWeights> encodedWeights;

    Combination comb;
    OwnedOpGraph m_OpGraph;
};

class ConcatOpGraph
{
public:
    ConcatOpGraph()
    {
        // Plan concatPlan
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 16, 16, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                     TraversalOrder::Xyz, 4, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag    = "Input1DramBuffer";
        m_OpGraph.GetBuffers().back()->m_Offset      = 0x00000FFF;
        m_OpGraph.GetBuffers().back()->m_OperationId = 0;
        m_OpGraph.GetBuffers().back()->m_BufferType  = BufferType::Input;
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 16, 8, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                     TraversalOrder::Xyz, 4, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag    = "Input2DramBuffer";
        m_OpGraph.GetBuffers().back()->m_Offset      = 0x0000F000;
        m_OpGraph.GetBuffers().back()->m_OperationId = 0;
        m_OpGraph.GetBuffers().back()->m_BufferType  = BufferType::Input;
        m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                     TensorShape{ 1, 16, 24, 3 }, TensorShape{ 1, 16, 24, 3 },
                                                     TraversalOrder::Xyz, 0, QuantizationInfo()));
        m_OpGraph.GetBuffers().back()->m_DebugTag           = "OutputDramBuffer";
        m_OpGraph.GetBuffers().back()->m_Offset             = 0x0000F00F;
        m_OpGraph.GetBuffers().back()->m_OperationId        = 2;
        m_OpGraph.GetBuffers().back()->m_ProducerOutputIndx = 0;
        m_OpGraph.GetBuffers().back()->m_BufferType         = BufferType::Output;
        m_OpGraph.AddOp(std::make_unique<ConcatOp>(CascadingBufferFormat::NHWCB));
        m_OpGraph.GetOps()[0]->m_DebugTag = "ConcatOp";
        m_OpGraph.AddConsumer(m_OpGraph.GetBuffers()[0], m_OpGraph.GetOps()[0], 0);
        m_OpGraph.AddConsumer(m_OpGraph.GetBuffers()[1], m_OpGraph.GetOps()[0], 1);
        m_OpGraph.SetProducer(m_OpGraph.GetBuffers()[2], m_OpGraph.GetOps()[0]);

        bool dumpOutputGraphToFile = false;
        if (dumpOutputGraphToFile)
        {
            std::ofstream stream("Concat_Graph_Merged.dot");
            SaveOpGraphToDot(m_OpGraph, stream, DetailLevel::High);
        }
    }

    OpGraph GetMergedOpGraph()
    {
        return m_OpGraph;
    }

private:
    GraphOfParts graph;
    Plan input1DramPlan;
    Plan input2DramPlan;
    Plan concatPlan;
    Plan outputDramPlan;
    Combination comb;
    OwnedOpGraph m_OpGraph;
};

class TwoMceDramIntermediateOpGraph
{
public:
    TwoMceDramIntermediateOpGraph()
    {

        // Plan inputDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 160, 160, 3 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_OperationId = 0;
        mergedOpGraph.GetBuffers().back()->m_BufferType  = BufferType::Input;
        mergedOpGraph.GetBuffers().back()->m_DebugTag    = "InputDramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset      = 0x00000F0A;

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps()[0]->m_OperationIds = { 1 };
        mergedOpGraph.GetOps().back()->m_DebugTag = "InputDmaOp";

        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(), 0);

        // Plan inputSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 160, 160, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag   = "InputSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
        mergedOpGraph.GetBuffers().back()->m_NumStripes = 4;

        Buffer* inputSramBuffer = mergedOpGraph.GetBuffers().back();

        mergedOpGraph.SetProducer(inputSramBuffer, mergedOpGraph.GetOps().back());

        Buffer* ptrInputBuffer = mergedOpGraph.GetBuffers().back();
        inputStripeSize        = utils::TotalSizeBytesNHWCB(ptrInputBuffer->m_StripeShape);
        inputZeroPoint         = ptrInputBuffer->m_QuantizationInfo.GetZeroPoint();

        // Plan weightDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT,
                                                         TensorShape{ 1, 1, 3, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        mergedOpGraph.GetBuffers().back()->m_DebugTag       = "WeightDramBuffer";
        encodedWeights                                      = std::make_shared<EncodedWeights>();
        encodedWeights->m_Data                              = { 1, 2, 3, 4 };
        encodedWeights->m_MaxSize                           = 10;
        encodedWeights->m_Metadata                          = { { 0, 2 }, { 2, 2 } };
        mergedOpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights;

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::WEIGHT));
        mergedOpGraph.GetOps().back()->m_DebugTag = "WeightDmaOp";

        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(), 0);

        // Plan weightSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT,
                                                         TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag    = "WeightSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        mergedOpGraph.GetBuffers().back()->m_NumStripes  = 3;
        mergedOpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights->m_MaxSize;

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        Buffer* ptrWeightBuffer = mergedOpGraph.GetBuffers().back();
        weightSize              = ptrWeightBuffer->m_SizeInBytes / ptrWeightBuffer->m_NumStripes;
        kernelHeight            = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[0]);
        kernelWidth             = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[1]);

        mergedOpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        mergedOpGraph.GetOps().back()->m_DebugTag = "MceOp";

        mergedOpGraph.AddConsumer(inputSramBuffer, mergedOpGraph.GetOps().back(), 0);
        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(), 1);

        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag = "OutputPleInputSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset   = 0x0000F00F;
        Buffer* pleInBuffer                           = mergedOpGraph.GetBuffers().back();

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        ifmDeltaDefaultHeight = static_cast<int8_t>(inputSramBuffer->m_TensorShape[1] - pleInBuffer->m_TensorShape[1]);
        ifmDeltaDefaultWidth  = static_cast<int8_t>(inputSramBuffer->m_TensorShape[2] - pleInBuffer->m_TensorShape[2]);
        ifmDeltaEdgeHeight    = static_cast<int8_t>(inputSramBuffer->m_TensorShape[1] - pleInBuffer->m_TensorShape[1]);
        ifmDeltaEdgeWidth     = static_cast<int8_t>(inputSramBuffer->m_TensorShape[2] - pleInBuffer->m_TensorShape[2]);

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        pleOp = std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                        BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                        TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp.get()->m_Offset = 0x0000F0F0;
        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp =
            AddPleToOpGraph(mergedOpGraph, Lifetime::Cascade, TensorShape{ 1, 4, 4, 32 }, numMemoryStripes,
                            std::move(pleOp), TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), { 0 });
        mergedOpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mergedOpGraph.AddConsumer(pleInBuffer, outBufferAndPleOp.second, 0);

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps().back()->m_DebugTag = "intermediateDmaOp";

        mergedOpGraph.AddConsumer(outBufferAndPleOp.first, mergedOpGraph.GetOps().back(), 0);

        // Plan intermediateDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
        mergedOpGraph.GetBuffers().back()->m_DebugTag   = "intermediateDramBuffer";

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps().back()->m_DebugTag = "intermediateSramDmaOp";
        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(), 0);

        // Plan intermediateSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 80, 80, 24 }, TensorShape{ 1, 8, 8, 16 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
        mergedOpGraph.GetBuffers().back()->m_DebugTag   = "intermediateSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
        mergedOpGraph.GetBuffers().back()->m_NumStripes = 4;
        Buffer* intermediateSramBuffer                  = mergedOpGraph.GetBuffers().back();

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        // Plan weight2DramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT,
                                                         TensorShape{ 1, 1, 3, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        mergedOpGraph.GetBuffers().back()->m_DebugTag       = "Weight2DramBuffer";
        encodedWeights2                                     = std::make_shared<EncodedWeights>();
        encodedWeights2->m_Data                             = { 1, 2, 3, 4 };
        encodedWeights2->m_MaxSize                          = 10;
        encodedWeights2->m_Metadata                         = { { 0, 2 }, { 2, 2 } };
        mergedOpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights2;

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::WEIGHT));
        mergedOpGraph.GetOps().back()->m_DebugTag = "Weight2DmaOp";

        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(), 0);

        // Plan weightSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT,
                                                         TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag    = "Weight2SramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        mergedOpGraph.GetBuffers().back()->m_NumStripes  = 3;
        mergedOpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights2->m_MaxSize;

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        Buffer* ptrWeightBuffer2 = mergedOpGraph.GetBuffers().back();
        weightSize2              = ptrWeightBuffer2->m_SizeInBytes / ptrWeightBuffer2->m_NumStripes;
        kernelHeight2            = static_cast<uint8_t>(ptrWeightBuffer2->m_TensorShape[0]);
        kernelWidth2             = static_cast<uint8_t>(ptrWeightBuffer2->m_TensorShape[1]);

        // Plan mcePlePlan
        mergedOpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        mergedOpGraph.GetOps().back()->m_DebugTag = "MceOp2";

        mergedOpGraph.AddConsumer(intermediateSramBuffer, mergedOpGraph.GetOps().back(), 0);
        mergedOpGraph.AddConsumer(ptrWeightBuffer2, mergedOpGraph.GetOps().back(), 1);

        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag = "outputPleInputSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset   = 0x0000F00F;
        Buffer* pleInBuffer2                          = mergedOpGraph.GetBuffers().back();

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        ifmDeltaDefaultHeight =
            static_cast<int8_t>(intermediateSramBuffer->m_TensorShape[1] - pleInBuffer2->m_TensorShape[1]);
        ifmDeltaDefaultWidth =
            static_cast<int8_t>(intermediateSramBuffer->m_TensorShape[2] - pleInBuffer2->m_TensorShape[2]);
        ifmDeltaEdgeHeight =
            static_cast<int8_t>(intermediateSramBuffer->m_TensorShape[1] - pleInBuffer2->m_TensorShape[1]);
        ifmDeltaEdgeWidth =
            static_cast<int8_t>(intermediateSramBuffer->m_TensorShape[2] - pleInBuffer2->m_TensorShape[2]);

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        auto pleOp2 =
            std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                    BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                    TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp2.get()->m_Offset    = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp2 =
            AddPleToOpGraph(mergedOpGraph, Lifetime::Cascade, TensorShape{ 1, 4, 4, 32 }, numMemoryStripes,
                            std::move(pleOp2), TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), { 1 });
        mergedOpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mergedOpGraph.AddConsumer(pleInBuffer2, outBufferAndPleOp2.second, 0);

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps().back()->m_DebugTag = "outputDmaOp";

        mergedOpGraph.AddConsumer(outBufferAndPleOp2.first, mergedOpGraph.GetOps().back(), 0);

        // Plan outputDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_OperationId        = 2;
        mergedOpGraph.GetBuffers().back()->m_ProducerOutputIndx = 0;
        mergedOpGraph.GetBuffers().back()->m_BufferType         = BufferType::Output;
        mergedOpGraph.GetBuffers().back()->m_DebugTag           = "outputDramBuffer";

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        bool dumpOutputGraphToFile = false;
        if (dumpOutputGraphToFile)
        {
            std::ofstream stream("CascadingCommandStreamGenerator_TwoMceSchedulerAgent_Output.dot");
            SaveOpGraphToDot(mergedOpGraph, stream, DetailLevel::High);
        }

        ETHOSN_UNUSED(outBufferAndPleOp);
        ETHOSN_UNUSED(outBufferAndPleOp2);
    }

    OpGraph GetMergedOpGraph()
    {
        return mergedOpGraph;
    }

    uint32_t getInputStripeSize()
    {
        return inputStripeSize;
    }

    uint32_t getWeightSize()
    {
        return weightSize;
    }

    int32_t getInputZeroPoint()
    {
        return inputZeroPoint;
    }

    uint8_t getKernelHeight()
    {
        return kernelHeight;
    }

    uint8_t getKernelWidth()
    {
        return kernelWidth;
    }

    int8_t getIfmDeltaDefaultHeight()
    {
        return ifmDeltaDefaultHeight;
    }

    int8_t getIfmDeltaDefaultWidth()
    {
        return ifmDeltaDefaultWidth;
    }

    int8_t getIfmDeltaEdgeHeight()
    {
        return ifmDeltaEdgeHeight;
    }

    int8_t getIfmDeltaEdgeWidth()
    {
        return ifmDeltaEdgeWidth;
    }

private:
    GraphOfParts graph;

    Plan inputDramPlan;
    Plan inputSramPlan;
    Plan weightDramPlan;
    Plan weightSramPlan;
    Plan mcePlePlan;
    Plan intermediateDramPlan;
    Plan intermediateSramPlan;

    Plan weight2DramPlan;
    Plan weight2SramPlan;
    Plan mcePle2Plan;
    Plan outputDramPlan;

    std::shared_ptr<EncodedWeights> encodedWeights;
    std::shared_ptr<EncodedWeights> encodedWeights2;

    std::unique_ptr<PleOp> pleOp;
    std::unique_ptr<PleOp> pleOp2;

    uint32_t inputStripeSize;
    uint32_t weightSize;
    uint32_t weightSize2;
    int32_t inputZeroPoint;

    uint8_t kernelHeight;
    uint8_t kernelWidth;
    uint8_t kernelHeight2;
    uint8_t kernelWidth2;
    int8_t ifmDeltaDefaultHeight;
    int8_t ifmDeltaDefaultWidth;
    int8_t ifmDeltaEdgeHeight;
    int8_t ifmDeltaEdgeWidth;

    Combination comb;
    OwnedOpGraph mergedOpGraph;
};

class TwoMceSramIntermediateOpGraph
{
public:
    TwoMceSramIntermediateOpGraph()
    {
        const std::set<uint32_t> operationIds = { 0 };
        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;

        //// Plan inputDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 160, 160, 3 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_OperationId = 0;
        mergedOpGraph.GetBuffers().back()->m_BufferType  = BufferType::Input;
        mergedOpGraph.GetBuffers().back()->m_DebugTag    = "InputDramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset      = 0x00000F0A;

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps().back()->m_DebugTag = "InputDmaOp";
        mergedOpGraph.GetOps()[0]->m_OperationIds = { 0 };

        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(), 0);

        // Plan inputSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 160, 160, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag   = "InputSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
        mergedOpGraph.GetBuffers().back()->m_NumStripes = 4;
        auto inputSramBuffer                            = mergedOpGraph.GetBuffers().back();

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        Buffer* ptrInputBuffer = mergedOpGraph.GetBuffers().back();
        inputStripeSize        = utils::TotalSizeBytesNHWCB(ptrInputBuffer->m_StripeShape);
        inputZeroPoint         = ptrInputBuffer->m_QuantizationInfo.GetZeroPoint();

        // Plan weightDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT,
                                                         TensorShape{ 1, 1, 3, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        mergedOpGraph.GetBuffers().back()->m_DebugTag       = "WeightDramBuffer";
        encodedWeights                                      = std::make_shared<EncodedWeights>();
        encodedWeights->m_Data                              = { 1, 2, 3, 4 };
        encodedWeights->m_MaxSize                           = 10;
        encodedWeights->m_Metadata                          = { { 0, 2 }, { 2, 2 } };
        mergedOpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights;

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::WEIGHT));
        mergedOpGraph.GetOps().back()->m_DebugTag = "WeightDmaOp";

        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(), 0);

        // Plan weightSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT,
                                                         TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag    = "WeightSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        mergedOpGraph.GetBuffers().back()->m_NumStripes  = 3;
        mergedOpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights->m_MaxSize;

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        Buffer* ptrWeightBuffer = mergedOpGraph.GetBuffers().back();
        weightSize              = ptrWeightBuffer->m_SizeInBytes / ptrWeightBuffer->m_NumStripes;
        kernelHeight            = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[0]);
        kernelWidth             = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[1]);

        // Plan mcePlePlan
        mergedOpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        mergedOpGraph.GetOps().back()->m_DebugTag = "MceOp";

        mergedOpGraph.AddConsumer(inputSramBuffer, mergedOpGraph.GetOps().back(), 0);
        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(), 1);

        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag = "OutputPleInputSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset   = 0x0000F00F;
        Buffer* pleInBuffer                           = mergedOpGraph.GetBuffers().back();

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        ifmDeltaDefaultHeight = static_cast<int8_t>(inputSramBuffer->m_TensorShape[1] -
                                                    mergedOpGraph.GetBuffers().back()->m_TensorShape[1]);
        ifmDeltaDefaultWidth  = static_cast<int8_t>(inputSramBuffer->m_TensorShape[2] -
                                                   mergedOpGraph.GetBuffers().back()->m_TensorShape[2]);
        ifmDeltaEdgeHeight    = static_cast<int8_t>(inputSramBuffer->m_TensorShape[1] -
                                                 mergedOpGraph.GetBuffers().back()->m_TensorShape[1]);
        ifmDeltaEdgeWidth     = static_cast<int8_t>(inputSramBuffer->m_TensorShape[2] -
                                                mergedOpGraph.GetBuffers().back()->m_TensorShape[2]);

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        pleOp = std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                        BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                        TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp.get()->m_Offset     = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp =
            AddPleToOpGraph(mergedOpGraph, Lifetime::Cascade, TensorShape{ 1, 4, 4, 32 }, numMemoryStripes,
                            std::move(pleOp), TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
        mergedOpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mergedOpGraph.AddConsumer(pleInBuffer, outBufferAndPleOp.second, 0);

        // Plan weight2DramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT,
                                                         TensorShape{ 1, 1, 3, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        mergedOpGraph.GetBuffers().back()->m_DebugTag       = "Weight2DramBuffer";
        encodedWeights2                                     = std::make_shared<EncodedWeights>();
        encodedWeights2->m_Data                             = { 1, 2, 3, 4 };
        encodedWeights2->m_MaxSize                          = 10;
        encodedWeights2->m_Metadata                         = { { 0, 2 }, { 2, 2 } };
        mergedOpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights2;

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::WEIGHT));
        mergedOpGraph.GetOps().back()->m_DebugTag = "Weight2DmaOp";

        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(), 0);

        // Plan weightSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT,
                                                         TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag    = "Weight2SramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        mergedOpGraph.GetBuffers().back()->m_NumStripes  = 3;
        mergedOpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights2->m_MaxSize;

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        Buffer* ptrWeightBuffer2 = mergedOpGraph.GetBuffers().back();
        weightSize2              = ptrWeightBuffer2->m_SizeInBytes / ptrWeightBuffer2->m_NumStripes;
        kernelHeight2            = static_cast<uint8_t>(ptrWeightBuffer2->m_TensorShape[0]);
        kernelWidth2             = static_cast<uint8_t>(ptrWeightBuffer2->m_TensorShape[1]);

        // Plan mcePlePlan
        mergedOpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        mergedOpGraph.GetOps().back()->m_DebugTag = "MceOp2";

        mergedOpGraph.AddConsumer(outBufferAndPleOp.first, mergedOpGraph.GetOps().back(), 0);
        mergedOpGraph.AddConsumer(ptrWeightBuffer2, mergedOpGraph.GetOps().back(), 1);

        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag = "outputPleInputSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset   = 0x0000F00F;
        Buffer* pleInBuffer2                          = mergedOpGraph.GetBuffers().back();

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        pleOp2 =
            std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                    BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                    TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp2.get()->m_Offset    = 0x0000F0F0;
        pleOp2->m_LoadKernel      = false;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp2 =
            AddPleToOpGraph(mergedOpGraph, Lifetime::Cascade, TensorShape{ 1, 4, 4, 32 }, numMemoryStripes,
                            std::move(pleOp2), TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
        mergedOpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mergedOpGraph.AddConsumer(pleInBuffer2, outBufferAndPleOp2.second, 0);

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps().back()->m_DebugTag = "outputDmaOp";

        mergedOpGraph.AddConsumer(outBufferAndPleOp2.first, mergedOpGraph.GetOps().back(), 0);

        // Plan outputDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_OperationId        = 2;
        mergedOpGraph.GetBuffers().back()->m_ProducerOutputIndx = 0;
        mergedOpGraph.GetBuffers().back()->m_BufferType         = BufferType::Output;
        mergedOpGraph.GetBuffers().back()->m_DebugTag           = "outputDramBuffer";

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        bool dumpOutputGraphToFile = false;
        if (dumpOutputGraphToFile)
        {
            std::ofstream stream("CascadingCommandStreamGenerator_TwoMceSchedulerAgent_Output.dot");
            SaveOpGraphToDot(mergedOpGraph, stream, DetailLevel::High);
        }

        ETHOSN_UNUSED(outBufferAndPleOp);
        ETHOSN_UNUSED(outBufferAndPleOp2);
    }

    OpGraph GetMergedOpGraph()
    {
        return mergedOpGraph;
    }

    uint32_t getInputStripeSize()
    {
        return inputStripeSize;
    }

    uint32_t getWeightSize()
    {
        return weightSize;
    }

    int32_t getInputZeroPoint()
    {
        return inputZeroPoint;
    }

    uint8_t getKernelHeight()
    {
        return kernelHeight;
    }

    uint8_t getKernelWidth()
    {
        return kernelWidth;
    }

    int8_t getIfmDeltaDefaultHeight()
    {
        return ifmDeltaDefaultHeight;
    }

    int8_t getIfmDeltaDefaultWidth()
    {
        return ifmDeltaDefaultWidth;
    }

    int8_t getIfmDeltaEdgeHeight()
    {
        return ifmDeltaEdgeHeight;
    }

    int8_t getIfmDeltaEdgeWidth()
    {
        return ifmDeltaEdgeWidth;
    }

private:
    GraphOfParts graph;

    Plan inputDramPlan;
    Plan inputSramPlan;
    Plan weightDramPlan;
    Plan weightSramPlan;
    Plan mcePlePlan;

    Plan weight2DramPlan;
    Plan weight2SramPlan;
    Plan mcePle2Plan;
    Plan outputDramPlan;

    std::shared_ptr<EncodedWeights> encodedWeights;
    std::shared_ptr<EncodedWeights> encodedWeights2;

    std::unique_ptr<PleOp> pleOp;
    std::unique_ptr<PleOp> pleOp2;

    uint32_t inputStripeSize;
    uint32_t weightSize;
    uint32_t weightSize2;
    int32_t inputZeroPoint;

    uint8_t kernelHeight;
    uint8_t kernelWidth;
    uint8_t kernelHeight2;
    uint8_t kernelWidth2;
    int8_t ifmDeltaDefaultHeight;
    int8_t ifmDeltaDefaultWidth;
    int8_t ifmDeltaEdgeHeight;
    int8_t ifmDeltaEdgeWidth;

    Combination comb;
    OwnedOpGraph mergedOpGraph;
};

class StridedConvOpGraph
{
public:
    StridedConvOpGraph(uint32_t padLeft, uint32_t padTop, TensorShape weightTensorShape, TensorShape outputTensorShape)
    {
        const std::set<uint32_t> operationIds = { 0 };
        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;

        // Plan inputDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 5, 5, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_OperationId = 0;
        mergedOpGraph.GetBuffers().back()->m_BufferType  = BufferType::Input;
        mergedOpGraph.GetBuffers().back()->m_DebugTag    = "InputDramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset      = 0x00000F0A;

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps().back()->m_DebugTag = "InputDmaOp";
        mergedOpGraph.GetOps()[0]->m_OperationIds = { 0 };

        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(), 0);

        // Plan inputSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 5, 5, 1 }, TensorShape{ 1, 5, 5, 1 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag   = "InputSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
        mergedOpGraph.GetBuffers().back()->m_NumStripes = 1;

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        Buffer* ptrInputBuffer = mergedOpGraph.GetBuffers().back();
        inputStripeSize        = utils::TotalSizeBytesNHWCB(ptrInputBuffer->m_StripeShape);
        inputZeroPoint         = ptrInputBuffer->m_QuantizationInfo.GetZeroPoint();

        // Plan weightDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT,
                                                         weightTensorShape, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        mergedOpGraph.GetBuffers().back()->m_DebugTag       = "WeightDramBuffer";
        encodedWeights                                      = std::make_shared<EncodedWeights>();
        encodedWeights->m_Data                              = { 1, 2, 3, 4 };
        encodedWeights->m_MaxSize                           = 10;
        encodedWeights->m_Metadata                          = { { 0, 2 }, { 2, 2 } };
        mergedOpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights;

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::WEIGHT));
        mergedOpGraph.GetOps().back()->m_DebugTag = "WeightDmaOp";

        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(), 0);

        // Plan weightSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT,
                                                         weightTensorShape, weightTensorShape, TraversalOrder::Xyz, 4,
                                                         QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag    = "WeightSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        mergedOpGraph.GetBuffers().back()->m_NumStripes  = 1;
        mergedOpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights->m_MaxSize;

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        Buffer* ptrWeightBuffer = mergedOpGraph.GetBuffers().back();
        weightSize              = ptrWeightBuffer->m_SizeInBytes / ptrWeightBuffer->m_NumStripes;
        kernelHeight            = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[1]);
        kernelWidth             = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[2]);

        // Plan mcePlePlan
        mergedOpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 5, 5, 1 }, TensorShape{ 1, 5, 5, 1 }, outputTensorShape,
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        mergedOpGraph.GetOps().back()->m_DebugTag = "MceOp Stride 1x1";

        mergedOpGraph.AddConsumer(ptrInputBuffer, mergedOpGraph.GetOps().back(), 0);    // connect input sram buffer
        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(),
                                  1);    // connect weights sram buffer

        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 5, 5, 1 }, TensorShape{ 1, 5, 5, 1 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag   = "OutputPleInputSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset     = 0x0000F00F;
        mergedOpGraph.GetBuffers().back()->m_NumStripes = 1;
        Buffer* pleInBuffer                             = mergedOpGraph.GetBuffers().back();

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        ifmDeltaHeight =
            static_cast<int8_t>(ptrInputBuffer->m_TensorShape[1] - mergedOpGraph.GetBuffers().back()->m_TensorShape[1]);
        ifmDeltaWidth =
            static_cast<int8_t>(ptrInputBuffer->m_TensorShape[2] - mergedOpGraph.GetBuffers().back()->m_TensorShape[2]);

        // Adding an Interleave PLE kernel to the plan
        pleOp =
            std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::INTERLEAVE_2X2_2_2,
                                    BlockConfig{ 16u, 16u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 5, 5, 1 } },
                                    TensorShape{ 1, 5, 5, 1 }, ethosn::command_stream::DataType::U8, true);
        pleOp.get()->m_Offset     = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp =
            AddPleToOpGraph(mergedOpGraph, Lifetime::Cascade, TensorShape{ 1, 5, 5, 1 }, numMemoryStripes,
                            std::move(pleOp), TensorShape{ 1, 5, 5, 1 }, QuantizationInfo(), operationIds);
        mergedOpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mergedOpGraph.AddConsumer(pleInBuffer, outBufferAndPleOp.second, 0);

        // Plan weight2DramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT,
                                                         weightTensorShape, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        mergedOpGraph.GetBuffers().back()->m_DebugTag       = "Weight2DramBuffer";
        encodedWeights2                                     = std::make_shared<EncodedWeights>();
        encodedWeights2->m_Data                             = { 1, 2, 3, 4 };
        encodedWeights2->m_MaxSize                          = 10;
        encodedWeights2->m_Metadata                         = { { 0, 2 }, { 2, 2 } };
        mergedOpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights2;

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::WEIGHT));
        mergedOpGraph.GetOps()[0]->m_DebugTag = "Weight2DmaOp";

        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(), 0);

        // Plan weightSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT,
                                                         weightTensorShape, weightTensorShape, TraversalOrder::Xyz, 4,
                                                         QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag    = "Weight2SramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        mergedOpGraph.GetBuffers().back()->m_NumStripes  = 1;
        mergedOpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights2->m_MaxSize;

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        Buffer* ptrWeightBuffer2 = mergedOpGraph.GetBuffers().back();
        weightSize2              = ptrWeightBuffer2->m_SizeInBytes / ptrWeightBuffer2->m_NumStripes;
        kernelHeight2            = static_cast<uint8_t>(ptrWeightBuffer2->m_TensorShape[1]);
        kernelWidth2             = static_cast<uint8_t>(ptrWeightBuffer2->m_TensorShape[2]);

        // Plan mcePlePlan
        mergedOpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 5, 5, 1 }, outputTensorShape, outputTensorShape,
            TraversalOrder::Xyz, Stride(2, 2), padLeft, padTop, 0, 255));
        (static_cast<MceOp*>(mergedOpGraph.GetOps().back()))->m_uninterleavedInputShape = TensorShape{ 1, 5, 5, 1 };
        mergedOpGraph.GetOps().back()->m_DebugTag                                       = "MceOp Stride 2x2";

        mergedOpGraph.AddConsumer(outBufferAndPleOp.first, mergedOpGraph.GetOps().back(),
                                  0);    // connect input sram buffer
        mergedOpGraph.AddConsumer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back(),
                                  1);    // connect weights sram buffer

        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                         outputTensorShape, outputTensorShape, TraversalOrder::Xyz, 4,
                                                         QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag   = "outputPleInputSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset     = 0x0000F00F;
        mergedOpGraph.GetBuffers().back()->m_NumStripes = 1;
        Buffer* pleInBuffer2                            = mergedOpGraph.GetBuffers().back();

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        pleOp2 = std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                         BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ outputTensorShape },
                                         outputTensorShape, ethosn::command_stream::DataType::U8, true);
        pleOp2.get()->m_Offset    = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp2 =
            AddPleToOpGraph(mergedOpGraph, Lifetime::Cascade, outputTensorShape, numMemoryStripes, std::move(pleOp2),
                            outputTensorShape, QuantizationInfo(), operationIds);
        mergedOpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mergedOpGraph.AddConsumer(pleInBuffer2, mergedOpGraph.GetOps().back(), 0);

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps().back()->m_DebugTag = "outputDmaOp";
        mergedOpGraph.AddConsumer(outBufferAndPleOp2.first, mergedOpGraph.GetOps().back(), 0);

        // Plan outputDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                         outputTensorShape, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_OperationId        = 2;
        mergedOpGraph.GetBuffers().back()->m_ProducerOutputIndx = 0;
        mergedOpGraph.GetBuffers().back()->m_BufferType         = BufferType::Output;
        mergedOpGraph.GetBuffers().back()->m_DebugTag           = "outputDramBuffer";

        mergedOpGraph.SetProducer(mergedOpGraph.GetBuffers().back(), mergedOpGraph.GetOps().back());

        bool dumpOutputGraphToFile = false;
        if (dumpOutputGraphToFile)
        {
            std::ofstream stream("CommandStreamGenerator_StridedConvOpGraph_Output.dot");
            SaveOpGraphToDot(mergedOpGraph, stream, DetailLevel::High);
        }

        ETHOSN_UNUSED(outBufferAndPleOp);
        ETHOSN_UNUSED(outBufferAndPleOp2);
    }

    OpGraph GetMergedOpGraph()
    {
        return mergedOpGraph;
    }

    uint32_t getInputStripeSize()
    {
        return inputStripeSize;
    }

    uint32_t getWeightSize()
    {
        return weightSize;
    }

    int32_t getInputZeroPoint()
    {
        return inputZeroPoint;
    }

    uint8_t getKernelHeight()
    {
        return kernelHeight;
    }

    uint8_t getKernelWidth()
    {
        return kernelWidth;
    }

    int8_t getIfmDeltaHeight()
    {
        return ifmDeltaHeight;
    }

    int8_t getIfmDeltaWidth()
    {
        return ifmDeltaWidth;
    }

private:
    GraphOfParts graph;

    Plan inputDramPlan;
    Plan inputSramPlan;
    Plan weightDramPlan;
    Plan weightSramPlan;
    Plan mcePlePlan;

    Plan weight2DramPlan;
    Plan weight2SramPlan;
    Plan mcePle2Plan;
    Plan outputDramPlan;

    std::shared_ptr<EncodedWeights> encodedWeights;
    std::shared_ptr<EncodedWeights> encodedWeights2;

    std::unique_ptr<PleOp> pleOp;
    std::unique_ptr<PleOp> pleOp2;

    uint32_t inputStripeSize;
    uint32_t weightSize;
    uint32_t weightSize2;
    int32_t inputZeroPoint;

    uint8_t kernelHeight;
    uint8_t kernelWidth;
    uint8_t kernelHeight2;
    uint8_t kernelWidth2;
    int8_t ifmDeltaHeight;
    int8_t ifmDeltaWidth;

    Combination comb;
    OwnedOpGraph mergedOpGraph;
};

class TwoInputsForPleOpGraph
{
public:
    TwoInputsForPleOpGraph()
    {
        const std::set<uint32_t> operationIds = { 0 };
        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;

        const std::vector<Buffer*>& buffers = mergedOpGraph.GetBuffers();
        const std::vector<Op*>& ops         = mergedOpGraph.GetOps();

        // Plan inputDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 160, 160, 3 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_OperationId = 0;
        mergedOpGraph.GetBuffers().back()->m_BufferType  = BufferType::Input;
        mergedOpGraph.GetBuffers().back()->m_DebugTag    = "InputDramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset      = 0x00000F0A;

        // Glue glueInputDram_InputSram
        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps()[0]->m_DebugTag     = "InputDmaOp";
        mergedOpGraph.GetOps()[0]->m_OperationIds = { 0 };

        mergedOpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Plan inputSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 160, 160, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag   = "InputSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
        mergedOpGraph.GetBuffers().back()->m_NumStripes = 4;

        mergedOpGraph.SetProducer(buffers.back(), ops.back());

        Buffer* ptrInputBuffer = buffers.back();
        inputStripeSize        = utils::TotalSizeBytesNHWCB(ptrInputBuffer->m_StripeShape);
        inputZeroPoint         = ptrInputBuffer->m_QuantizationInfo.GetZeroPoint();

        // Plan weightDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT,
                                                         TensorShape{ 1, 1, 3, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        mergedOpGraph.GetBuffers().back()->m_DebugTag       = "WeightDramBuffer";
        encodedWeights                                      = std::make_shared<EncodedWeights>();
        encodedWeights->m_Data                              = { 1, 2, 3, 4 };
        encodedWeights->m_MaxSize                           = 10;
        encodedWeights->m_Metadata                          = { { 0, 2 }, { 2, 2 } };
        mergedOpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights;

        // Glue glueWeightDram_WeightSram
        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::WEIGHT));
        mergedOpGraph.GetOps()[0]->m_DebugTag = "WeightDmaOp";

        mergedOpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Plan weightSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT,
                                                         TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag    = "WeightSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        mergedOpGraph.GetBuffers().back()->m_NumStripes  = 3;
        mergedOpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights->m_MaxSize;

        mergedOpGraph.SetProducer(buffers.back(), ops.back());

        Buffer* ptrWeightBuffer = buffers.back();
        weightSize              = ptrWeightBuffer->m_SizeInBytes / ptrWeightBuffer->m_NumStripes;
        kernelHeight            = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[1]);
        kernelWidth             = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[2]);

        // Plan mcePlePlan
        mergedOpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        mergedOpGraph.GetOps()[0]->m_DebugTag = "MceOp";

        mergedOpGraph.AddConsumer(ptrInputBuffer, ops.back(), 0);
        mergedOpGraph.AddConsumer(ptrWeightBuffer, ops.back(), 1);

        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag = "OutputPleInputSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset   = 0x0000F00F;

        mergedOpGraph.SetProducer(buffers.back(), ops.back());

        Buffer* pleInputBuffer = buffers.back();

        ifmDeltaHeight = static_cast<int8_t>(ptrInputBuffer->m_TensorShape[1] - buffers.back()->m_TensorShape[1]);
        ifmDeltaWidth  = static_cast<int8_t>(ptrInputBuffer->m_TensorShape[2] - buffers.back()->m_TensorShape[2]);

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        pleOp = std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                        BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                        TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp.get()->m_Offset     = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp =
            AddPleToOpGraph(mergedOpGraph, Lifetime::Cascade, TensorShape{ 1, 4, 4, 32 }, numMemoryStripes,
                            std::move(pleOp), TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
        buffers.back()->m_Offset = 0X0000F0FF;

        mergedOpGraph.AddConsumer(pleInputBuffer, outBufferAndPleOp.second, 0);

        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps()[0]->m_DebugTag = "intermediateDmaOp";

        mergedOpGraph.AddConsumer(outBufferAndPleOp.first, ops.back(), 0);

        // Plan intermediateDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
        mergedOpGraph.GetBuffers().back()->m_DebugTag   = "intermediateDramBuffer";

        mergedOpGraph.SetProducer(buffers.back(), ops.back());

        // Glue glueintermediateDram_intermediateSram
        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps()[0]->m_DebugTag = "intermediateSramDmaOp";

        mergedOpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Plan intermediateSramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 80, 80, 24 }, TensorShape{ 1, 8, 8, 16 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
        mergedOpGraph.GetBuffers().back()->m_DebugTag   = "intermediateSramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
        mergedOpGraph.GetBuffers().back()->m_NumStripes = 4;

        mergedOpGraph.SetProducer(buffers.back(), ops.back());

        Buffer* pleInput0 = buffers.back();

        // Plan input2DramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 1, 3, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_OperationId = 0;
        mergedOpGraph.GetBuffers().back()->m_BufferType  = BufferType::Input;
        mergedOpGraph.GetBuffers().back()->m_DebugTag    = "Input2DramBuffer";

        // Glue glueInput2Dram_Input2Sram
        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps()[0]->m_DebugTag = "Input2DmaOp";

        mergedOpGraph.AddConsumer(buffers.back(), ops.back(), 0);

        // Plan input2SramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_DebugTag   = "Input2SramBuffer";
        mergedOpGraph.GetBuffers().back()->m_Offset     = 0x00000FF0;
        mergedOpGraph.GetBuffers().back()->m_NumStripes = 3;

        mergedOpGraph.SetProducer(buffers.back(), ops.back());

        Buffer* pleInput1 = buffers.back();

        pleOp2 =
            std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::ADDITION_RESCALE,
                                    BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                    TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp2.get()->m_Offset    = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp2 =
            AddPleToOpGraph(mergedOpGraph, Lifetime::Cascade, TensorShape{ 1, 4, 4, 32 }, numMemoryStripes,
                            std::move(pleOp2), TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
        mergedOpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;

        mergedOpGraph.AddConsumer(pleInput0, outBufferAndPleOp2.second, 0);
        mergedOpGraph.AddConsumer(pleInput1, outBufferAndPleOp2.second, 1);

        // Glue glueOutputSram_OutputDram
        mergedOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        mergedOpGraph.GetOps()[0]->m_DebugTag = "outputDmaOp";

        mergedOpGraph.AddConsumer(outBufferAndPleOp2.first, ops.back(), 0);

        // Plan outputDramPlan
        mergedOpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
        mergedOpGraph.GetBuffers().back()->m_OperationId        = 2;
        mergedOpGraph.GetBuffers().back()->m_ProducerOutputIndx = 0;
        mergedOpGraph.GetBuffers().back()->m_BufferType         = BufferType::Output;
        mergedOpGraph.GetBuffers().back()->m_DebugTag           = "outputDramBuffer";

        mergedOpGraph.SetProducer(buffers.back(), ops.back());

        bool dumpOutputGraphToFile = false;
        if (dumpOutputGraphToFile)
        {
            std::ofstream stream("CascadingCommandStreamGenerator_TwoInputForPle_Output.dot");
            SaveOpGraphToDot(mergedOpGraph, stream, DetailLevel::High);
        }
    }

    OpGraph GetMergedOpGraph()
    {
        return mergedOpGraph;
    }

    uint32_t getInputStripeSize()
    {
        return inputStripeSize;
    }

    uint32_t getWeightSize()
    {
        return weightSize;
    }

    int32_t getInputZeroPoint()
    {
        return inputZeroPoint;
    }

    uint8_t getKernelHeight()
    {
        return kernelHeight;
    }

    uint8_t getKernelWidth()
    {
        return kernelWidth;
    }

    int8_t getIfmDeltaHeight()
    {
        return ifmDeltaHeight;
    }

    int8_t getIfmDeltaWidth()
    {
        return ifmDeltaWidth;
    }

private:
    GraphOfParts graph;

    Plan inputDramPlan;
    Plan inputSramPlan;
    Plan weightDramPlan;
    Plan weightSramPlan;
    Plan mcePlePlan;
    Plan intermediateDramPlan;
    Plan intermediateSramPlan;

    Plan input2DramPlan;
    Plan input2SramPlan;
    Plan twoInputsPlePlan;
    Plan outputDramPlan;

    std::shared_ptr<EncodedWeights> encodedWeights;

    std::unique_ptr<PleOp> pleOp;
    std::unique_ptr<PleOp> pleOp2;

    uint32_t inputStripeSize;
    uint32_t weightSize;
    int32_t inputZeroPoint;

    uint8_t kernelHeight;
    uint8_t kernelWidth;
    int8_t ifmDeltaHeight;
    int8_t ifmDeltaWidth;

    Combination comb;
    OwnedOpGraph mergedOpGraph;
};

//////////////////////////////////////////////////////////////////////////////////////////////
// Command Stream Agents Order Tests
//////////////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("StandalonePleOpGraph Command Stream Agents Order Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph opGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph        = opGraph.GetMergedOpGraph();

    const std::set<uint32_t> operationIds = { 0 };
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    CHECK(commandStream.size() == 4);
    CHECK(commandStream[0].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[1].data.type == AgentType::PLE_LOADER);
    CHECK(commandStream[2].data.type == AgentType::PLE_SCHEDULER);
    CHECK(commandStream[3].data.type == AgentType::OFM_STREAMER);
}

TEST_CASE("MceOpGraph Command Stream Agents Order Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph opGraph    = MceOpGraph();
    OpGraph mergedOpGraph = opGraph.GetMergedOpGraph();

    const std::set<uint32_t> operationIds = { 0 };
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    CHECK(commandStream.size() == 6);
    CHECK(commandStream[0].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[1].data.type == AgentType::WGT_STREAMER);
    CHECK(commandStream[2].data.type == AgentType::PLE_LOADER);
    CHECK(commandStream[3].data.type == AgentType::MCE_SCHEDULER);
    CHECK(commandStream[4].data.type == AgentType::PLE_SCHEDULER);
    CHECK(commandStream[5].data.type == AgentType::OFM_STREAMER);
}

TEST_CASE("MceOpGraphIntermediateDramBuffers Command Stream Agents Order Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraphIntermediateDramBuffers opGraph = MceOpGraphIntermediateDramBuffers();
    OpGraph mergedOpGraph                     = opGraph.GetMergedOpGraph();

    const std::set<uint32_t> operationIds = { 0 };
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    CHECK(commandStream.size() == 10);
    CHECK(commandStream[0].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[1].data.type == AgentType::WGT_STREAMER);
    CHECK(commandStream[2].data.type == AgentType::PLE_LOADER);
    CHECK(commandStream[3].data.type == AgentType::MCE_SCHEDULER);
    CHECK(commandStream[4].data.type == AgentType::PLE_SCHEDULER);
    CHECK(commandStream[5].data.type == AgentType::OFM_STREAMER);
    CHECK(commandStream[6].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[7].data.type == AgentType::OFM_STREAMER);
    CHECK(commandStream[8].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[9].data.type == AgentType::OFM_STREAMER);
}

TEST_CASE("TwoMceDramIntermediateOpGraph Command Stream Agents Order Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceDramIntermediateOpGraph opGraph = TwoMceDramIntermediateOpGraph();
    OpGraph mergedOpGraph                 = opGraph.GetMergedOpGraph();

    const std::set<uint32_t> operationIds = { 0 };
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    CHECK(commandStream.size() == 12);
    CHECK(commandStream[0].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[1].data.type == AgentType::WGT_STREAMER);
    CHECK(commandStream[2].data.type == AgentType::PLE_LOADER);
    CHECK(commandStream[3].data.type == AgentType::MCE_SCHEDULER);
    CHECK(commandStream[4].data.type == AgentType::PLE_SCHEDULER);
    CHECK(commandStream[5].data.type == AgentType::OFM_STREAMER);
    CHECK(commandStream[6].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[7].data.type == AgentType::WGT_STREAMER);
    CHECK(commandStream[8].data.type == AgentType::PLE_LOADER);
    CHECK(commandStream[9].data.type == AgentType::MCE_SCHEDULER);
    CHECK(commandStream[10].data.type == AgentType::PLE_SCHEDULER);
    CHECK(commandStream[11].data.type == AgentType::OFM_STREAMER);
}

TEST_CASE("MceOpGraphIntermediateSramBuffers Command Stream Agents Order Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceSramIntermediateOpGraph opGraph = TwoMceSramIntermediateOpGraph();
    OpGraph mergedOpGraph                 = opGraph.GetMergedOpGraph();

    const std::set<uint32_t> operationIds = { 0 };
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    CHECK(commandStream.size() == 9);
    CHECK(commandStream[0].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[1].data.type == AgentType::WGT_STREAMER);
    CHECK(commandStream[2].data.type == AgentType::PLE_LOADER);
    CHECK(commandStream[3].data.type == AgentType::MCE_SCHEDULER);
    CHECK(commandStream[4].data.type == AgentType::PLE_SCHEDULER);
    CHECK(commandStream[5].data.type == AgentType::WGT_STREAMER);
    CHECK(commandStream[6].data.type == AgentType::MCE_SCHEDULER);
    CHECK(commandStream[7].data.type == AgentType::PLE_SCHEDULER);
    CHECK(commandStream[8].data.type == AgentType::OFM_STREAMER);
}

TEST_CASE("TwoInputsForPleOpGraph Command Stream Agents Order Test", "[CascadingCommandStreamGenerator]")
{
    TwoInputsForPleOpGraph opGraph = TwoInputsForPleOpGraph();
    OpGraph mergedOpGraph          = opGraph.GetMergedOpGraph();

    const std::set<uint32_t> operationIds = { 0 };
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    CHECK(commandStream.size() == 11);
    CHECK(commandStream[0].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[1].data.type == AgentType::WGT_STREAMER);
    CHECK(commandStream[2].data.type == AgentType::PLE_LOADER);
    CHECK(commandStream[3].data.type == AgentType::MCE_SCHEDULER);
    CHECK(commandStream[4].data.type == AgentType::PLE_SCHEDULER);
    CHECK(commandStream[5].data.type == AgentType::OFM_STREAMER);
    CHECK(commandStream[6].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[7].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[8].data.type == AgentType::PLE_LOADER);
    CHECK(commandStream[9].data.type == AgentType::PLE_SCHEDULER);
    CHECK(commandStream[10].data.type == AgentType::OFM_STREAMER);
}

TEST_CASE("StridedConvOpGraph Command Stream Agents Order Test", "[CascadingCommandStreamGenerator]")
{
    StridedConvOpGraph opGraph = StridedConvOpGraph(1, 1, { 3, 3, 1, 1 }, { 1, 3, 3, 1 });
    OpGraph mergedOpGraph      = opGraph.GetMergedOpGraph();

    const std::set<uint32_t> operationIds = { 0 };
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    CHECK(commandStream.size() == 10);
    CHECK(commandStream[0].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[1].data.type == AgentType::WGT_STREAMER);
    CHECK(commandStream[2].data.type == AgentType::PLE_LOADER);
    CHECK(commandStream[3].data.type == AgentType::MCE_SCHEDULER);
    CHECK(commandStream[4].data.type == AgentType::PLE_SCHEDULER);
    CHECK(commandStream[5].data.type == AgentType::WGT_STREAMER);
    CHECK(commandStream[6].data.type == AgentType::PLE_LOADER);
    CHECK(commandStream[7].data.type == AgentType::MCE_SCHEDULER);
    CHECK(commandStream[8].data.type == AgentType::PLE_SCHEDULER);
    CHECK(commandStream[9].data.type == AgentType::OFM_STREAMER);
}

TEST_CASE("ConcatOpGraph Command Stream Agents Order Test", "[CascadingCommandStreamGenerator]")
{
    ConcatOpGraph opGraph = ConcatOpGraph();
    OpGraph mergedOpGraph = opGraph.GetMergedOpGraph();

    const std::set<uint32_t> operationIds = { 0 };
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    CHECK(commandStream.size() == 4);
    CHECK(commandStream[0].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[1].data.type == AgentType::OFM_STREAMER);
    CHECK(commandStream[2].data.type == AgentType::IFM_STREAMER);
    CHECK(commandStream[3].data.type == AgentType::OFM_STREAMER);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Agent Data Tests
//////////////////////////////////////////////////////////////////////////////////////////////

// IfmStreamer Agent Data Test
TEST_CASE("IfmStreamer Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent = commandStream[0];
    const IfmS& ifmSData   = ifmSAgent.data.ifm;

    CHECK(ifmSData.fmData.dramOffset == 0);
    CHECK(ifmSData.fmData.bufferId == 1);
    CHECK(ifmSData.fmData.dataType == FmsDataType::NHWCB);

    CHECK(ifmSData.fmData.fcafInfo.signedActivation == false);
    CHECK(ifmSData.fmData.fcafInfo.zeroPoint == 0);

    CHECK(ifmSData.fmData.tile.baseAddr == 3855);
    CHECK(ifmSData.fmData.tile.numSlots == 4);
    CHECK(ifmSData.fmData.tile.slotSize == 128);

    CHECK(ifmSData.fmData.dfltStripeSize.height == 8);
    CHECK(ifmSData.fmData.dfltStripeSize.width == 8);
    CHECK(ifmSData.fmData.dfltStripeSize.channels == 16);

    CHECK(ifmSData.fmData.edgeStripeSize.height == 8);
    CHECK(ifmSData.fmData.edgeStripeSize.width == 8);
    CHECK(ifmSData.fmData.edgeStripeSize.channels == 3);

    CHECK(ifmSData.fmData.supertensorSizeInCells.width == 20);
    CHECK(ifmSData.fmData.supertensorSizeInCells.channels == 1);

    CHECK(ifmSData.fmData.numStripes.height == 20);
    CHECK(ifmSData.fmData.numStripes.width == 20);
    CHECK(ifmSData.fmData.numStripes.channels == 1);

    CHECK(ifmSData.fmData.stripeIdStrides.height == 20);
    CHECK(ifmSData.fmData.stripeIdStrides.width == 1);
    CHECK(ifmSData.fmData.stripeIdStrides.channels == 1);
}

// WeightStreamer Agent Data Test
TEST_CASE("WeightStreamer Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& wgtSAgent = commandStream[1];
    const WgtS& wgtSData   = wgtSAgent.data.wgt;

    CHECK(wgtSData.bufferId == 2);
    CHECK(wgtSData.metadataBufferId == 3);

    CHECK(wgtSData.tile.baseAddr == 0x00000FF0);
    CHECK(wgtSData.tile.numSlots == 3);
    CHECK(wgtSData.tile.slotSize == 2);

    CHECK(wgtSData.numStripes.ifmChannels == 1);
    CHECK(wgtSData.numStripes.ofmChannels == 1);

    CHECK(wgtSData.stripeIdStrides.ifmChannels == 1);
    CHECK(wgtSData.stripeIdStrides.ofmChannels == 1);
}

// MceScheduler Agent Data Test
TEST_CASE("MceScheduler Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent = commandStream[3];
    const MceS& mceSData   = mceSAgent.data.mce;

    CHECK(mceSData.ifmTile.baseAddr == 0x00000F0F);
    CHECK(mceSData.ifmTile.numSlots == 4);
    CHECK(mceSData.ifmTile.slotSize == mceOpGraph.getInputStripeSize() / hwCaps.GetNumberOfSrams());

    CHECK(mceSData.wgtTile.baseAddr == 0x00000FF0);
    CHECK(mceSData.wgtTile.numSlots == 3);
    CHECK(mceSData.wgtTile.slotSize == 2);

    CHECK(mceSData.blockSize.width == 16);
    CHECK(mceSData.blockSize.height == 16);

    CHECK(mceSData.dfltStripeSize.ofmHeight == 8);
    CHECK(mceSData.dfltStripeSize.ofmWidth == 8);
    CHECK(mceSData.dfltStripeSize.ofmChannels == 8);
    CHECK(mceSData.dfltStripeSize.ifmChannels == 16);

    CHECK(mceSData.edgeStripeSize.ofmHeight == 1);
    CHECK(mceSData.edgeStripeSize.ofmWidth == 8);
    CHECK(mceSData.edgeStripeSize.ofmChannels == 8);
    CHECK(mceSData.edgeStripeSize.ifmChannels == 3);

    CHECK(mceSData.numStripes.ofmHeight == 3);
    CHECK(mceSData.numStripes.ofmWidth == 2);
    CHECK(mceSData.numStripes.ofmChannels == 2);
    CHECK(mceSData.numStripes.ifmChannels == 1);

    CHECK(mceSData.stripeIdStrides.ofmHeight == 2);
    CHECK(mceSData.stripeIdStrides.ofmWidth == 1);
    CHECK(mceSData.stripeIdStrides.ofmChannels == 6);
    CHECK(mceSData.stripeIdStrides.ifmChannels == 1);

    CHECK(mceSData.convStrideXy.x == 1);
    CHECK(mceSData.convStrideXy.y == 1);

    CHECK(mceSData.ifmZeroPoint == mceOpGraph.getInputZeroPoint());
    CHECK(mceSData.mceOpMode == cascading::MceOperation::CONVOLUTION);
    CHECK(mceSData.algorithm == cascading::MceAlgorithm::DIRECT);

    CHECK(mceSData.filterShape[0].height == mceOpGraph.getKernelHeight());
    CHECK(mceSData.filterShape[0].width == mceOpGraph.getKernelWidth());

    CHECK(mceSData.padding[0].left == 0);
    CHECK(mceSData.padding[0].top == 0);

    CHECK(mceSData.ifmDeltaDefault[0].height == mceOpGraph.getIfmDeltaDefaultHeight());
    CHECK(mceSData.ifmDeltaDefault[0].width == mceOpGraph.getIfmDeltaDefaultWidth());
    CHECK(mceSData.ifmDeltaEdge[0].height == mceOpGraph.getIfmDeltaEdgeHeight());
    CHECK(mceSData.ifmDeltaEdge[0].width == mceOpGraph.getIfmDeltaEdgeWidth());

    CHECK(mceSData.reluActiv.max == 255);
    CHECK(mceSData.reluActiv.min == 0);

    CHECK(mceSData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);
}

TEST_CASE("MceScheduler Agent Data Test - 1x1 Convolution - 2x2 Stride", "[CascadingCommandStreamGenerator]")
{
    StridedConvOpGraph stridedConvGraph = StridedConvOpGraph(0, 0, { 1, 1, 1, 1 }, { 1, 2, 2, 1 });
    OpGraph mergedOpGraph               = stridedConvGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent = commandStream[7];
    const MceS& mceSData   = mceSAgent.data.mce;

    // Submap 0
    CHECK(mceSData.filterShape[0].height == 1);
    CHECK(mceSData.filterShape[0].width == 1);
    CHECK(mceSData.padding[0].left == 0);
    CHECK(mceSData.padding[0].top == 0);
    CHECK(mceSData.ifmDeltaDefault[0].height == 1);
    CHECK(mceSData.ifmDeltaDefault[0].width == 1);
    CHECK(mceSData.ifmDeltaEdge[0].height == 1);
    CHECK(mceSData.ifmDeltaEdge[0].width == 1);

    // Submap 1
    CHECK(mceSData.filterShape[1].height == 1);
    CHECK(mceSData.filterShape[1].width == 0);

    // Submap 2
    CHECK(mceSData.filterShape[2].height == 0);
    CHECK(mceSData.filterShape[2].width == 1);

    // Submap 3
    CHECK(mceSData.filterShape[3].height == 0);
    CHECK(mceSData.filterShape[3].width == 0);

    CHECK(mceSData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);
}

TEST_CASE("MceScheduler Agent Data Test - 2x2 Convolution - 2x2 Stride - Valid Padding",
          "[CascadingCommandStreamGenerator]")
{
    StridedConvOpGraph stridedConvGraph = StridedConvOpGraph(0, 0, { 2, 2, 1, 1 }, { 1, 2, 2, 1 });
    OpGraph mergedOpGraph               = stridedConvGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent = commandStream[7];
    const MceS& mceSData   = mceSAgent.data.mce;

    // Submap 0
    CHECK(mceSData.filterShape[0].height == 1);
    CHECK(mceSData.filterShape[0].width == 1);
    CHECK(mceSData.padding[0].left == 0);
    CHECK(mceSData.padding[0].top == 0);
    CHECK(mceSData.ifmDeltaDefault[0].height == 1);
    CHECK(mceSData.ifmDeltaDefault[0].width == 1);
    CHECK(mceSData.ifmDeltaEdge[0].height == 1);
    CHECK(mceSData.ifmDeltaEdge[0].width == 1);

    // Submap 1
    CHECK(mceSData.filterShape[1].height == 1);
    CHECK(mceSData.filterShape[1].width == 1);
    CHECK(mceSData.padding[1].left == 0);
    CHECK(mceSData.padding[1].top == 0);
    CHECK(mceSData.ifmDeltaDefault[1].height == 1);
    CHECK(mceSData.ifmDeltaDefault[1].width == 0);
    CHECK(mceSData.ifmDeltaEdge[1].height == 1);
    CHECK(mceSData.ifmDeltaEdge[1].width == 0);

    // Submap 2
    CHECK(mceSData.filterShape[2].height == 1);
    CHECK(mceSData.filterShape[2].width == 1);
    CHECK(mceSData.padding[2].left == 0);
    CHECK(mceSData.padding[2].top == 0);
    CHECK(mceSData.ifmDeltaDefault[2].height == 0);
    CHECK(mceSData.ifmDeltaDefault[2].width == 1);
    CHECK(mceSData.ifmDeltaEdge[2].height == 0);
    CHECK(mceSData.ifmDeltaEdge[2].width == 1);

    // Submap 3
    CHECK(mceSData.filterShape[3].height == 1);
    CHECK(mceSData.filterShape[3].width == 1);
    CHECK(mceSData.padding[3].left == 0);
    CHECK(mceSData.padding[3].top == 0);
    CHECK(mceSData.ifmDeltaDefault[3].height == 0);
    CHECK(mceSData.ifmDeltaDefault[3].width == 0);
    CHECK(mceSData.ifmDeltaEdge[3].height == 0);
    CHECK(mceSData.ifmDeltaEdge[3].width == 0);

    CHECK(mceSData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);
}

TEST_CASE("MceScheduler Agent Data Test - 3x3 Convolution - 2x2 Stride - Valid Padding",
          "[CascadingCommandStreamGenerator]")
{
    StridedConvOpGraph stridedConvGraph = StridedConvOpGraph(0, 0, { 3, 3, 1, 1 }, { 1, 2, 2, 1 });
    OpGraph mergedOpGraph               = stridedConvGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent = commandStream[7];
    const MceS& mceSData   = mceSAgent.data.mce;

    // Submap 0
    CHECK(mceSData.filterShape[0].height == 2);
    CHECK(mceSData.filterShape[0].width == 2);
    CHECK(mceSData.padding[0].left == 0);
    CHECK(mceSData.padding[0].top == 0);
    CHECK(mceSData.ifmDeltaDefault[0].height == 1);
    CHECK(mceSData.ifmDeltaDefault[0].width == 1);
    CHECK(mceSData.ifmDeltaEdge[0].height == 1);
    CHECK(mceSData.ifmDeltaEdge[0].width == 1);

    // Submap 1
    CHECK(mceSData.filterShape[1].height == 2);
    CHECK(mceSData.filterShape[1].width == 1);
    CHECK(mceSData.padding[1].left == 0);
    CHECK(mceSData.padding[1].top == 0);
    CHECK(mceSData.ifmDeltaDefault[1].height == 1);
    CHECK(mceSData.ifmDeltaDefault[1].width == 0);
    CHECK(mceSData.ifmDeltaEdge[1].height == 1);
    CHECK(mceSData.ifmDeltaEdge[1].width == 0);

    // Submap 2
    CHECK(mceSData.filterShape[2].height == 1);
    CHECK(mceSData.filterShape[2].width == 2);
    CHECK(mceSData.padding[2].left == 0);
    CHECK(mceSData.padding[2].top == 0);
    CHECK(mceSData.ifmDeltaDefault[2].height == 0);
    CHECK(mceSData.ifmDeltaDefault[2].width == 1);
    CHECK(mceSData.ifmDeltaEdge[2].height == 0);
    CHECK(mceSData.ifmDeltaEdge[2].width == 1);

    // Submap 3
    CHECK(mceSData.filterShape[3].height == 1);
    CHECK(mceSData.filterShape[3].width == 1);
    CHECK(mceSData.padding[3].left == 0);
    CHECK(mceSData.padding[3].top == 0);
    CHECK(mceSData.ifmDeltaDefault[3].height == 0);
    CHECK(mceSData.ifmDeltaDefault[3].width == 0);
    CHECK(mceSData.ifmDeltaEdge[3].height == 0);
    CHECK(mceSData.ifmDeltaEdge[3].width == 0);

    CHECK(mceSData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);
}

TEST_CASE("MceScheduler Agent Data Test - 3x3 Convolution - 2x2 Stride - Same Padding",
          "[CascadingCommandStreamGenerator]")
{
    StridedConvOpGraph stridedConvGraph = StridedConvOpGraph(1, 1, { 3, 3, 1, 1 }, { 1, 3, 3, 1 });
    OpGraph mergedOpGraph               = stridedConvGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent = commandStream[7];
    const MceS& mceSData   = mceSAgent.data.mce;

    // Submap 0
    CHECK(mceSData.filterShape[0].height == 1);
    CHECK(mceSData.filterShape[0].width == 1);
    CHECK(mceSData.padding[0].left == 0);
    CHECK(mceSData.padding[0].top == 0);
    CHECK(mceSData.ifmDeltaDefault[0].height == 0);
    CHECK(mceSData.ifmDeltaDefault[0].width == 0);
    CHECK(mceSData.ifmDeltaEdge[0].height == 0);
    CHECK(mceSData.ifmDeltaEdge[0].width == 0);

    // Submap 1
    CHECK(mceSData.filterShape[1].height == 1);
    CHECK(mceSData.filterShape[1].width == 2);
    CHECK(mceSData.padding[1].left == 1);
    CHECK(mceSData.padding[1].top == 0);
    CHECK(mceSData.ifmDeltaDefault[1].height == 0);
    CHECK(mceSData.ifmDeltaDefault[1].width == -1);
    CHECK(mceSData.ifmDeltaEdge[1].height == 0);
    CHECK(mceSData.ifmDeltaEdge[1].width == -1);

    // Submap 2
    CHECK(mceSData.filterShape[2].height == 2);
    CHECK(mceSData.filterShape[2].width == 1);
    CHECK(mceSData.padding[2].left == 0);
    CHECK(mceSData.padding[2].top == 1);
    CHECK(mceSData.ifmDeltaDefault[2].height == -1);
    CHECK(mceSData.ifmDeltaDefault[2].width == 0);
    CHECK(mceSData.ifmDeltaEdge[2].height == -1);
    CHECK(mceSData.ifmDeltaEdge[2].width == 0);

    // Submap 3
    CHECK(mceSData.filterShape[3].height == 2);
    CHECK(mceSData.filterShape[3].width == 2);
    CHECK(mceSData.padding[3].left == 1);
    CHECK(mceSData.padding[3].top == 1);
    CHECK(mceSData.ifmDeltaDefault[3].height == -1);
    CHECK(mceSData.ifmDeltaDefault[3].width == -1);
    CHECK(mceSData.ifmDeltaEdge[3].height == -1);
    CHECK(mceSData.ifmDeltaEdge[3].width == -1);

    CHECK(mceSData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);
}

// PleLoader Agent Data Test
TEST_CASE("PleLoader Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleLAgent = commandStream[2];
    const PleL& pleLData   = pleLAgent.data.pleL;

    CHECK(pleLData.sramAddr == 0x0000F0F0);
    CHECK(pleLData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);
}

// PleScheduler Agent Data Test
TEST_CASE("PleScheduler Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleSchedulerAgent = commandStream[4];

    // The network consists of all agent types. Here we test that the PleScheduler
    // agent is set correctly.
    CHECK(pleSchedulerAgent.data.pleS.ofmTile.baseAddr == 0x000F0FF);
    CHECK(pleSchedulerAgent.data.pleS.ofmTile.numSlots == 1);
    CHECK(pleSchedulerAgent.data.pleS.ofmTile.slotSize == 256);
    CHECK(pleSchedulerAgent.data.pleS.ofmZeroPoint == 0);

    CHECK(pleSchedulerAgent.data.pleS.dfltStripeSize.height == 4);
    CHECK(pleSchedulerAgent.data.pleS.dfltStripeSize.width == 4);
    CHECK(pleSchedulerAgent.data.pleS.dfltStripeSize.channels == 32);

    CHECK(pleSchedulerAgent.data.pleS.numStripes.height == 20);
    CHECK(pleSchedulerAgent.data.pleS.numStripes.width == 20);
    CHECK(pleSchedulerAgent.data.pleS.numStripes.channels == 1);

    CHECK(pleSchedulerAgent.data.pleS.edgeStripeSize.height == 4);
    CHECK(pleSchedulerAgent.data.pleS.edgeStripeSize.width == 4);
    CHECK(pleSchedulerAgent.data.pleS.edgeStripeSize.channels == 24);

    CHECK(pleSchedulerAgent.data.pleS.stripeIdStrides.height == 20);
    CHECK(pleSchedulerAgent.data.pleS.stripeIdStrides.width == 1);
    CHECK(pleSchedulerAgent.data.pleS.stripeIdStrides.channels == 400);

    CHECK(pleSchedulerAgent.data.pleS.inputMode == PleInputMode::MCE_ALL_OGS);

    CHECK(pleSchedulerAgent.data.pleS.pleKernelSramAddr == 0x0000F0F0);
    CHECK(pleSchedulerAgent.data.pleS.pleKernelId == PleKernelId::PASSTHROUGH_8X8_1);
}

// PleScheduler Standalone Agent Data Test
TEST_CASE("PleScheduler Standalone Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const std::set<uint32_t> operationIds = { 0 };
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleSAgent = commandStream[2];

    // The network consists of a standalone ple op and DMA ops. Here we test that
    // the PleScheduler agent is set correctly.
    CHECK(pleSAgent.data.pleS.ofmTile.baseAddr == 0x0000F00);
    CHECK(pleSAgent.data.pleS.ofmTile.numSlots == 1);
    CHECK(pleSAgent.data.pleS.ofmTile.slotSize == 256);
    CHECK(pleSAgent.data.pleS.ofmZeroPoint == 0);

    CHECK(pleSAgent.data.pleS.dfltStripeSize.height == 8);
    CHECK(pleSAgent.data.pleS.dfltStripeSize.width == 8);
    CHECK(pleSAgent.data.pleS.dfltStripeSize.channels == 32);

    CHECK(pleSAgent.data.pleS.numStripes.height == 10);
    CHECK(pleSAgent.data.pleS.numStripes.width == 10);
    CHECK(pleSAgent.data.pleS.numStripes.channels == 1);

    CHECK(pleSAgent.data.pleS.edgeStripeSize.height == 8);
    CHECK(pleSAgent.data.pleS.edgeStripeSize.width == 8);
    CHECK(pleSAgent.data.pleS.edgeStripeSize.channels == 24);

    CHECK(pleSAgent.data.pleS.stripeIdStrides.height == 10);
    CHECK(pleSAgent.data.pleS.stripeIdStrides.width == 1);
    CHECK(pleSAgent.data.pleS.stripeIdStrides.channels == 100);

    CHECK(pleSAgent.data.pleS.inputMode == PleInputMode::SRAM);

    CHECK(pleSAgent.data.pleS.pleKernelSramAddr == 0x000000FF);
    CHECK(pleSAgent.data.pleS.pleKernelId == PleKernelId::LEAKY_RELU_8X8_1);

    CHECK(pleSAgent.data.pleS.ifmTile0.baseAddr == 0x0000000F);
    CHECK(pleSAgent.data.pleS.ifmTile0.numSlots == 0);
    CHECK(pleSAgent.data.pleS.ifmTile0.slotSize == 128);

    CHECK(pleSAgent.data.pleS.ifmInfo0.zeroPoint == 0);
    CHECK(pleSAgent.data.pleS.ifmInfo0.multiplier == 32768);
    CHECK(pleSAgent.data.pleS.ifmInfo0.shift == 15);
}

// OfmStreamer Agent Data Test
TEST_CASE("OfmStreamer Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ofmSAgent = commandStream[5];
    const OfmS& ofmSData   = ofmSAgent.data.ofm;

    CHECK(ofmSData.fmData.dramOffset == 0);
    CHECK(ofmSData.fmData.bufferId == 4);
    CHECK(ofmSData.fmData.dataType == FmsDataType::NHWCB);

    CHECK(ofmSData.fmData.fcafInfo.signedActivation == false);
    CHECK(ofmSData.fmData.fcafInfo.zeroPoint == 0);

    CHECK(ofmSData.fmData.tile.baseAddr == 61695);
    CHECK(ofmSData.fmData.tile.numSlots == 1);
    CHECK(ofmSData.fmData.tile.slotSize == 256);

    CHECK(ofmSData.fmData.dfltStripeSize.height == 4);
    CHECK(ofmSData.fmData.dfltStripeSize.width == 4);
    CHECK(ofmSData.fmData.dfltStripeSize.channels == 32);

    CHECK(ofmSData.fmData.edgeStripeSize.height == 8);
    CHECK(ofmSData.fmData.edgeStripeSize.width == 8);
    CHECK(ofmSData.fmData.edgeStripeSize.channels == 24);

    CHECK(ofmSData.fmData.supertensorSizeInCells.width == 10);
    CHECK(ofmSData.fmData.supertensorSizeInCells.channels == 2);

    CHECK(ofmSData.fmData.numStripes.height == 20);
    CHECK(ofmSData.fmData.numStripes.width == 20);
    CHECK(ofmSData.fmData.numStripes.channels == 1);

    CHECK(ofmSData.fmData.stripeIdStrides.height == 20);
    CHECK(ofmSData.fmData.stripeIdStrides.width == 1);
    CHECK(ofmSData.fmData.stripeIdStrides.channels == 400);
}

// Concat Op Agent Data Test
TEST_CASE("Concat Op Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    ConcatOpGraph inputOutputMergeGraph = ConcatOpGraph();
    OpGraph mergedOpGraph               = inputOutputMergeGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent1 = commandStream[0];
    const Agent& ofmSAgent1 = commandStream[1];
    const Agent& ifmSAgent2 = commandStream[2];
    const Agent& ofmSAgent2 = commandStream[3];

    const IfmS& ifmSData1 = ifmSAgent1.data.ifm;
    const OfmS& ofmSData1 = ofmSAgent1.data.ofm;
    const IfmS& ifmSData2 = ifmSAgent2.data.ifm;
    const OfmS& ofmSData2 = ofmSAgent2.data.ofm;

    // IfmSData1
    CHECK(ifmSData1.fmData.bufferId == 2);
    CHECK(ifmSData1.fmData.dramOffset == 0);
    CHECK(ifmSData1.fmData.dataType == FmsDataType::NHWCB);

    CHECK(ifmSData1.fmData.fcafInfo.zeroPoint == 0);
    CHECK(ifmSData1.fmData.fcafInfo.signedActivation == false);

    CHECK(ifmSData1.fmData.tile.baseAddr == 0);
    CHECK(ifmSData1.fmData.tile.numSlots == 2);
    CHECK(ifmSData1.fmData.tile.slotSize == 128);

    CHECK(ifmSData1.fmData.dfltStripeSize.height == 8);
    CHECK(ifmSData1.fmData.dfltStripeSize.width == 8);
    CHECK(ifmSData1.fmData.dfltStripeSize.channels == 3);

    CHECK(ifmSData1.fmData.edgeStripeSize.height == 8);
    CHECK(ifmSData1.fmData.edgeStripeSize.width == 8);
    CHECK(ifmSData1.fmData.edgeStripeSize.channels == 3);

    CHECK(ifmSData1.fmData.supertensorSizeInCells.width == 2);
    CHECK(ifmSData1.fmData.supertensorSizeInCells.channels == 1);

    CHECK(ifmSData1.fmData.numStripes.height == 1);
    CHECK(ifmSData1.fmData.numStripes.width == 1);
    CHECK(ifmSData1.fmData.numStripes.channels == 1);

    CHECK(ifmSData1.fmData.stripeIdStrides.height == 1);
    CHECK(ifmSData1.fmData.stripeIdStrides.width == 1);
    CHECK(ifmSData1.fmData.stripeIdStrides.channels == 1);

    // ofmSData1
    CHECK(ofmSData1.fmData.bufferId == 1);
    CHECK(ofmSData1.fmData.dramOffset == 0);
    CHECK(ofmSData1.fmData.dataType == FmsDataType::NHWCB);

    CHECK(ofmSData1.fmData.fcafInfo.zeroPoint == 0);
    CHECK(ofmSData1.fmData.fcafInfo.signedActivation == false);

    CHECK(ofmSData1.fmData.tile.baseAddr == 0);
    CHECK(ofmSData1.fmData.tile.numSlots == 2);
    CHECK(ofmSData1.fmData.tile.slotSize == 128);

    CHECK(ofmSData1.fmData.dfltStripeSize.height == 8);
    CHECK(ofmSData1.fmData.dfltStripeSize.width == 8);
    CHECK(ofmSData1.fmData.dfltStripeSize.channels == 3);

    CHECK(ofmSData1.fmData.edgeStripeSize.height == 8);
    CHECK(ofmSData1.fmData.edgeStripeSize.width == 8);
    CHECK(ofmSData1.fmData.edgeStripeSize.channels == 3);

    CHECK(ofmSData1.fmData.supertensorSizeInCells.width == 3);
    CHECK(ofmSData1.fmData.supertensorSizeInCells.channels == 1);

    CHECK(ofmSData1.fmData.numStripes.height == 1);
    CHECK(ofmSData1.fmData.numStripes.width == 1);
    CHECK(ofmSData1.fmData.numStripes.channels == 1);

    CHECK(ofmSData1.fmData.stripeIdStrides.height == 1);
    CHECK(ofmSData1.fmData.stripeIdStrides.width == 1);
    CHECK(ofmSData1.fmData.stripeIdStrides.channels == 1);

    // ifmsData2
    CHECK(ifmSData2.fmData.bufferId == 3);
    CHECK(ifmSData2.fmData.dramOffset == 0);
    CHECK(ifmSData2.fmData.dataType == FmsDataType::NHWCB);

    CHECK(ifmSData2.fmData.fcafInfo.zeroPoint == 0);
    CHECK(ifmSData2.fmData.fcafInfo.signedActivation == false);

    CHECK(ifmSData2.fmData.tile.baseAddr == 256);
    CHECK(ifmSData2.fmData.tile.numSlots == 2);
    CHECK(ifmSData2.fmData.tile.slotSize == 128);

    CHECK(ifmSData2.fmData.dfltStripeSize.height == 8);
    CHECK(ifmSData2.fmData.dfltStripeSize.width == 8);
    CHECK(ifmSData2.fmData.dfltStripeSize.channels == 3);

    CHECK(ifmSData2.fmData.edgeStripeSize.height == 8);
    CHECK(ifmSData2.fmData.edgeStripeSize.width == 8);
    CHECK(ifmSData2.fmData.edgeStripeSize.channels == 3);

    CHECK(ifmSData2.fmData.supertensorSizeInCells.width == 1);
    CHECK(ifmSData2.fmData.supertensorSizeInCells.channels == 1);

    CHECK(ifmSData2.fmData.numStripes.height == 1);
    CHECK(ifmSData2.fmData.numStripes.width == 1);
    CHECK(ifmSData2.fmData.numStripes.channels == 1);

    CHECK(ifmSData2.fmData.stripeIdStrides.height == 1);
    CHECK(ifmSData2.fmData.stripeIdStrides.width == 1);
    CHECK(ifmSData2.fmData.stripeIdStrides.channels == 1);

    // ofmsData2
    CHECK(ofmSData2.fmData.bufferId == 1);
    CHECK(ofmSData2.fmData.dramOffset == 0x00000800);
    CHECK(ofmSData2.fmData.dataType == FmsDataType::NHWCB);

    CHECK(ofmSData2.fmData.fcafInfo.zeroPoint == 0);
    CHECK(ofmSData2.fmData.fcafInfo.signedActivation == false);

    CHECK(ofmSData2.fmData.tile.baseAddr == 256);
    CHECK(ofmSData2.fmData.tile.numSlots == 2);
    CHECK(ofmSData2.fmData.tile.slotSize == 128);

    CHECK(ofmSData2.fmData.dfltStripeSize.height == 8);
    CHECK(ofmSData2.fmData.dfltStripeSize.width == 8);
    CHECK(ofmSData2.fmData.dfltStripeSize.channels == 3);

    CHECK(ofmSData2.fmData.edgeStripeSize.height == 8);
    CHECK(ofmSData2.fmData.edgeStripeSize.width == 8);
    CHECK(ofmSData2.fmData.edgeStripeSize.channels == 3);

    CHECK(ofmSData2.fmData.supertensorSizeInCells.width == 3);
    CHECK(ofmSData2.fmData.supertensorSizeInCells.channels == 1);

    CHECK(ofmSData2.fmData.numStripes.height == 1);
    CHECK(ofmSData2.fmData.numStripes.width == 1);
    CHECK(ofmSData2.fmData.numStripes.channels == 1);

    CHECK(ofmSData2.fmData.stripeIdStrides.height == 1);
    CHECK(ofmSData2.fmData.stripeIdStrides.width == 1);
    CHECK(ofmSData2.fmData.stripeIdStrides.channels == 1);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Read After Write Dependency Tests
//////////////////////////////////////////////////////////////////////////////////////////////

// IfmStreamer Agent - Read After Write Dependency Test
TEST_CASE("IfmStreamer-OfmStreamer ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    ConcatOpGraph inputOutputMergeGraph = ConcatOpGraph();
    OpGraph mergedOpGraph               = inputOutputMergeGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ofmSAgent1 = commandStream[1];
    const Agent& ofmSAgent2 = commandStream[3];

    const Dependency& readDependency1 = ofmSAgent1.info.readDependencies.at(0);
    const Dependency& readDependency2 = ofmSAgent2.info.readDependencies.at(0);

    // ifmS1 -> ofmS1
    CHECK(readDependency1.relativeAgentId == 1);
    CHECK(readDependency1.outerRatio.other == 1);
    CHECK(readDependency1.outerRatio.self == 1);
    CHECK(readDependency1.innerRatio.other == 1);
    CHECK(readDependency1.innerRatio.self == 1);
    CHECK(readDependency1.boundary == 0);
    // ifmS2 -> ofmS2
    CHECK(readDependency2.relativeAgentId == 1);
    CHECK(readDependency2.outerRatio.other == 1);
    CHECK(readDependency2.outerRatio.self == 1);
    CHECK(readDependency2.innerRatio.other == 1);
    CHECK(readDependency2.innerRatio.self == 1);
    CHECK(readDependency2.boundary == 0);
}

// MceScheduler Agent - Read After Write Dependency Test
TEST_CASE("MceScheduler-IfmStreamer ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent           = commandStream[0];
    const Agent& mceSAgent           = commandStream[3];
    const Dependency& readDependency = mceSAgent.info.readDependencies.at(0);

    uint32_t numberOfMceStripesPerRow =
        mceSAgent.data.mce.numStripes.ofmWidth * mceSAgent.data.mce.numStripes.ifmChannels;
    uint32_t numberOfIfmStripesPerRow =
        ifmSAgent.data.ifm.fmData.numStripes.width * ifmSAgent.data.ifm.fmData.numStripes.channels;

    CHECK(readDependency.relativeAgentId == 3);
    CHECK(readDependency.outerRatio.other == numberOfIfmStripesPerRow);
    CHECK(readDependency.outerRatio.self == numberOfMceStripesPerRow);
    CHECK(readDependency.innerRatio.other == 1);
    CHECK(readDependency.innerRatio.self == 1);
    CHECK(readDependency.boundary == 1);
}

// MceScheduler Agent - Read After Write Dependency Test
TEST_CASE("MceScheduler-WeightStreamer ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent           = commandStream[3];
    const Dependency& readDependency = mceSAgent.info.readDependencies.at(1);

    CHECK(readDependency.relativeAgentId == 2);
    CHECK(readDependency.outerRatio.other == 3);
    CHECK(readDependency.outerRatio.self == 12);
    CHECK(readDependency.innerRatio.other == 1);
    CHECK(readDependency.innerRatio.self == 6);
    CHECK(readDependency.boundary == 0);
}

// PleScheduler Agent - Read After Write Dependency Test
TEST_CASE("PleScheduler-IfmStreamer ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent           = commandStream[0];
    const Agent& pleSAgent           = commandStream[2];
    const Dependency& readDependency = pleSAgent.info.readDependencies.at(1);

    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;
    uint32_t numberOfIfmStripes = ifmSAgent.data.ifm.fmData.numStripes.height *
                                  ifmSAgent.data.ifm.fmData.numStripes.width *
                                  ifmSAgent.data.ifm.fmData.numStripes.channels;

    CHECK(readDependency.relativeAgentId == 2);
    CHECK(readDependency.outerRatio.other == numberOfIfmStripes / 100);    // 100 is the common factor between
    CHECK(readDependency.outerRatio.self ==
          numberOfPleStripes / 100);    // 400 (IfmStripes), 100 (PleStripes), 0 (boundary)
    CHECK(readDependency.innerRatio.other == 4);
    CHECK(readDependency.innerRatio.self == 1);
    CHECK(readDependency.boundary == 0);
}

// PleScheduler Agent - Read After Write Dependency Test
TEST_CASE("PleScheduler-MceScheduler ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent           = commandStream[3];
    const Agent& pleSAgent           = commandStream[4];
    const Dependency& readDependency = pleSAgent.info.readDependencies.at(1);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ofmChannels;
    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    CHECK(readDependency.relativeAgentId == 1);
    CHECK(readDependency.outerRatio.other == numberOfMceStripes);
    CHECK(readDependency.outerRatio.self == numberOfPleStripes);
    CHECK(readDependency.innerRatio.other == 2);
    CHECK(readDependency.innerRatio.self == 1);
    CHECK(readDependency.boundary == 1);
}

// MceScheduler Agent - Read After Write Dependency Test
TEST_CASE("MceScheduler-PleScheduler ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceSramIntermediateOpGraph mceOpGraph = TwoMceSramIntermediateOpGraph();
    OpGraph mergedOpGraph                    = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent           = commandStream[6];
    const Dependency& readDependency = mceSAgent.info.readDependencies.at(0);

    CHECK(readDependency.relativeAgentId == 2);
    CHECK(readDependency.outerRatio.other == 100);
    CHECK(readDependency.outerRatio.self == 3);
    CHECK(readDependency.innerRatio.other == 33);
    CHECK(readDependency.innerRatio.self == 1);
    CHECK(readDependency.boundary == 1);
}

// PleScheduler Agent - Read After Write Dependency Test
TEST_CASE("PleScheduler-PleLoader ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleSAgent           = commandStream[4];
    const Dependency& readDependency = pleSAgent.info.readDependencies.at(0);

    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    CHECK(readDependency.relativeAgentId == 2);
    CHECK(readDependency.outerRatio.other == 1);
    CHECK(readDependency.outerRatio.self == numberOfPleStripes);
    CHECK(readDependency.innerRatio.other == 1);
    CHECK(readDependency.innerRatio.self == numberOfPleStripes);
    CHECK(readDependency.boundary == 0);
}

// OfmStreamer Agent - Read After Write Dependency Test
TEST_CASE("OfmStreamer-IfmStreamer ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceDramIntermediateOpGraph twoMceOpMergeGraph = TwoMceDramIntermediateOpGraph();
    OpGraph mergedOpGraph                            = twoMceOpMergeGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ofmSAgent           = commandStream[5];
    const Dependency& readDependency = ofmSAgent.info.readDependencies.at(0);

    CHECK(readDependency.relativeAgentId == 1);
    CHECK(readDependency.outerRatio.other == 1);
    CHECK(readDependency.outerRatio.self == 1);
    CHECK(readDependency.innerRatio.other == 1);
    CHECK(readDependency.innerRatio.self == 1);
    CHECK(readDependency.boundary == 0);
}

// OfmStreamer Agent - Read After Write Dependency Test
TEST_CASE("OfmStreamer-PleScheduler ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ofmSAgent           = commandStream[3];
    const Dependency& readDependency = ofmSAgent.info.readDependencies.at(0);

    CHECK(readDependency.relativeAgentId == 1);
    CHECK(readDependency.outerRatio.other == 1);
    CHECK(readDependency.outerRatio.self == 1);
    CHECK(readDependency.innerRatio.other == 1);
    CHECK(readDependency.innerRatio.self == 1);
    CHECK(readDependency.boundary == 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Sram Overlap Dependency Tests
//////////////////////////////////////////////////////////////////////////////////////////////

// WeightStreamer Agent - Sram Overlap Dependency Test
TEST_CASE("WeightStreamer-OfmStreamer SramOverlapDependency Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceDramIntermediateOpGraph twoMceOpMergeGraph = TwoMceDramIntermediateOpGraph();
    OpGraph mergedOpGraph                            = twoMceOpMergeGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& wgtSAgent           = commandStream[7];
    const Dependency& readDependency = wgtSAgent.info.readDependencies.at(0);

    CHECK(readDependency.relativeAgentId == 2);
    CHECK(readDependency.outerRatio.other == 400);
    CHECK(readDependency.outerRatio.self == 1);
    CHECK(readDependency.innerRatio.other == 400);
    CHECK(readDependency.innerRatio.self == 1);
    CHECK(readDependency.boundary == 0);

    ETHOSN_UNUSED(commandStream);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Write After Read Dependency Tests
//////////////////////////////////////////////////////////////////////////////////////////////

// IfmStreamer Agent - Write After Read Dependency Test
TEST_CASE("IfmStreamer-MceScheduler WriteAfterReadDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent            = commandStream[0];
    const Dependency& writeDependency = ifmSAgent.info.writeDependencies.at(0);

    CHECK(writeDependency.relativeAgentId == 3);
    CHECK(writeDependency.outerRatio.other == 2);
    CHECK(writeDependency.outerRatio.self == 20);
    CHECK(writeDependency.innerRatio.other == 1);
    CHECK(writeDependency.innerRatio.self == 1);
    CHECK(writeDependency.boundary == 1);
}

// IfmStreamer Agent - Write After Read Dependency Test
TEST_CASE("IfmStreamer-PleScheduler WriteAfterReadDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent            = commandStream[0];
    const Agent& pleSAgent            = commandStream[2];
    const Dependency& writeDependency = ifmSAgent.info.writeDependencies.at(0);

    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;
    uint32_t numberOfIfmStripes = ifmSAgent.data.ifm.fmData.numStripes.height *
                                  ifmSAgent.data.ifm.fmData.numStripes.width *
                                  ifmSAgent.data.ifm.fmData.numStripes.channels;

    CHECK(writeDependency.relativeAgentId == 2);
    CHECK(writeDependency.outerRatio.other == numberOfPleStripes / 100);    // 100 is the common factor between
    CHECK(writeDependency.outerRatio.self ==
          numberOfIfmStripes / 100);    // 400 (PleStripes), 100 (IfmStripes), 0 (boundary)
    CHECK(writeDependency.innerRatio.other == 1);
    CHECK(writeDependency.innerRatio.self == 4);
    CHECK(writeDependency.boundary == 0);
}

// IfmStreamer Agent - Write After Read Dependency Test
TEST_CASE("IfmStreamer-OfmStreamer WriteAfterReadDependency Test", "[CascadingCommandStreamGenerator]")
{
    ConcatOpGraph inputOutputMergeGraph = ConcatOpGraph();
    OpGraph mergedOpGraph               = inputOutputMergeGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent1 = commandStream[0];
    const Agent& ifmSAgent2 = commandStream[2];

    const Dependency& writeDependency1 = ifmSAgent1.info.writeDependencies.at(0);
    const Dependency& writeDependency2 = ifmSAgent2.info.writeDependencies.at(0);

    // ifmS1 -> ofmS1
    CHECK(writeDependency1.relativeAgentId == 1);
    CHECK(writeDependency1.outerRatio.other == 1);
    CHECK(writeDependency1.outerRatio.self == 1);
    CHECK(writeDependency1.innerRatio.other == 1);
    CHECK(writeDependency1.innerRatio.self == 1);
    CHECK(writeDependency1.boundary == 0);
    // ifmS2 -> ofmS2
    CHECK(writeDependency2.relativeAgentId == 1);
    CHECK(writeDependency2.outerRatio.other == 1);
    CHECK(writeDependency2.outerRatio.self == 1);
    CHECK(writeDependency2.innerRatio.other == 1);
    CHECK(writeDependency2.innerRatio.self == 1);
    CHECK(writeDependency2.boundary == 0);
}

// WeightStreamer Agent - Write After Read Dependency Test
TEST_CASE("WeightStreamer-MceScheduler WriteAfterReadDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& wgtSAgent            = commandStream[1];
    const Dependency& writeDependency = wgtSAgent.info.writeDependencies.at(0);

    CHECK(writeDependency.relativeAgentId == 2);
    CHECK(writeDependency.outerRatio.other == 12);
    CHECK(writeDependency.outerRatio.self == 3);
    CHECK(writeDependency.innerRatio.other == 6);
    CHECK(writeDependency.innerRatio.self == 1);
    CHECK(writeDependency.boundary == 0);
}

// PleScheduler Agent - Write After Read Dependency Test
TEST_CASE("PleScheduler-MceScheduler WriteAfterReadDependency Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceSramIntermediateOpGraph mceOpGraph = TwoMceSramIntermediateOpGraph();
    OpGraph mergedOpGraph                    = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent            = commandStream[6];
    const Agent& pleSAgent            = commandStream[4];
    const Dependency& writeDependency = pleSAgent.info.writeDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ofmChannels;
    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    CHECK(writeDependency.relativeAgentId == 2);
    CHECK(writeDependency.outerRatio.other == numberOfMceStripes / 4);    // 4 is the common factor between
    CHECK(writeDependency.outerRatio.self ==
          numberOfPleStripes / 4);    // 400 (PleStripes), 12 (MceStripes), 4 (boundary)
    CHECK(writeDependency.innerRatio.other == 1);
    CHECK(writeDependency.innerRatio.self == 33);
    CHECK(writeDependency.boundary == 1);
}

// PleScheduler Agent - Write After Read Dependency Test
TEST_CASE("PleScheduler-OfmStreamer WriteAfterReadDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleSAgent            = commandStream[2];
    const Dependency& writeDependency = pleSAgent.info.writeDependencies.at(0);

    CHECK(writeDependency.relativeAgentId == 1);
    CHECK(writeDependency.outerRatio.other == 1);
    CHECK(writeDependency.outerRatio.self == 1);
    CHECK(writeDependency.innerRatio.other == 1);
    CHECK(writeDependency.innerRatio.self == 1);
    CHECK(writeDependency.boundary == 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Schedule Time Dependency Tests
//////////////////////////////////////////////////////////////////////////////////////////////

// IfmStreamer Agent - Schedule Time Dependency Test
TEST_CASE("IfmStreamer-MceScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent               = commandStream[0];
    const Dependency& scheduleDependency = ifmSAgent.info.scheduleDependencies.at(0);

    CHECK(scheduleDependency.relativeAgentId == 3);
    CHECK(scheduleDependency.outerRatio.other == 2);
    CHECK(scheduleDependency.outerRatio.self == 20);
    CHECK(scheduleDependency.innerRatio.other == 1);
    CHECK(scheduleDependency.innerRatio.self == 1);
    CHECK(scheduleDependency.boundary == 1);
}

// IfmStreamer Agent - Schedule Time Dependency Test
TEST_CASE("IfmStreamer-PleScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent               = commandStream[0];
    const Agent& pleSAgent               = commandStream[2];
    const Dependency& scheduleDependency = ifmSAgent.info.scheduleDependencies.at(0);

    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;
    uint32_t numberOfIfmStripes = ifmSAgent.data.ifm.fmData.numStripes.height *
                                  ifmSAgent.data.ifm.fmData.numStripes.width *
                                  ifmSAgent.data.ifm.fmData.numStripes.channels;

    CHECK(scheduleDependency.relativeAgentId == 2);
    CHECK(scheduleDependency.outerRatio.other == numberOfPleStripes / 100);    // 100 is the common factor between
    CHECK(scheduleDependency.outerRatio.self ==
          numberOfIfmStripes / 100);    // 400 (PleStripes), 100 (IfmStripes), 0 (boundary)
    CHECK(scheduleDependency.innerRatio.other == 1);
    CHECK(scheduleDependency.innerRatio.self == 4);
    CHECK(scheduleDependency.boundary == 0);
}

// IfmStreamer Agent - Schedule Time Dependency Test
TEST_CASE("IfmStreamer-OfmStreamer ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    ConcatOpGraph inputOutputMergeGraph = ConcatOpGraph();
    OpGraph mergedOpGraph               = inputOutputMergeGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent1 = commandStream[0];
    const Agent& ifmSAgent2 = commandStream[2];

    const Dependency& scheduleDependency1 = ifmSAgent1.info.scheduleDependencies.at(0);
    const Dependency& scheduleDependency2 = ifmSAgent2.info.scheduleDependencies.at(0);

    // ifmS1 -> ofmS1
    CHECK(scheduleDependency1.relativeAgentId == 1);
    CHECK(scheduleDependency1.outerRatio.other == 1);
    CHECK(scheduleDependency1.outerRatio.self == 1);
    CHECK(scheduleDependency1.innerRatio.other == 1);
    CHECK(scheduleDependency1.innerRatio.self == 1);
    CHECK(scheduleDependency1.boundary == 0);
    // ifmS2 -> ofmS2
    CHECK(scheduleDependency2.relativeAgentId == 1);
    CHECK(scheduleDependency2.outerRatio.other == 1);
    CHECK(scheduleDependency2.outerRatio.self == 1);
    CHECK(scheduleDependency2.innerRatio.other == 1);
    CHECK(scheduleDependency2.innerRatio.self == 1);
    CHECK(scheduleDependency2.boundary == 0);
}

// WeightStreamer Agent - Schedule Time Dependency Test
TEST_CASE("WeightStreamer-MceScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& wgtSAgent               = commandStream[1];
    const Dependency& scheduleDependency = wgtSAgent.info.scheduleDependencies.at(0);

    CHECK(scheduleDependency.relativeAgentId == 2);
    CHECK(scheduleDependency.outerRatio.other == 12);
    CHECK(scheduleDependency.outerRatio.self == 3);
    CHECK(scheduleDependency.innerRatio.other == 6);
    CHECK(scheduleDependency.innerRatio.self == 1);
    CHECK(scheduleDependency.boundary == 0);
}

// MceScheduler Agent - Schedule Time Dependency Test
TEST_CASE("MceScheduler-PleScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent               = commandStream[3];
    const Agent& pleSAgent               = commandStream[4];
    const Dependency& scheduleDependency = mceSAgent.info.scheduleDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ofmChannels;
    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    CHECK(scheduleDependency.relativeAgentId == 1);
    CHECK(scheduleDependency.outerRatio.other == numberOfPleStripes);
    CHECK(scheduleDependency.outerRatio.self == numberOfMceStripes);
    CHECK(scheduleDependency.innerRatio.other == 1);
    CHECK(scheduleDependency.innerRatio.self == 70);
    CHECK(scheduleDependency.boundary == 1);
}

// PleScheduler Agent - Schedule Time Dependency Test
TEST_CASE("PleScheduler-MceScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceSramIntermediateOpGraph mceOpGraph = TwoMceSramIntermediateOpGraph();
    OpGraph mergedOpGraph                    = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent               = commandStream[6];
    const Agent& pleSAgent               = commandStream[4];
    const Dependency& scheduleDependency = pleSAgent.info.scheduleDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ofmChannels;
    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    CHECK(scheduleDependency.relativeAgentId == 2);
    CHECK(scheduleDependency.outerRatio.other == numberOfMceStripes / 4);    // 4 is the common factor between
    CHECK(scheduleDependency.outerRatio.self ==
          numberOfPleStripes / 4);    // 400 (PleStripes), 12 (MceStripes), 4 (boundary)
    CHECK(scheduleDependency.innerRatio.other == 1);
    CHECK(scheduleDependency.innerRatio.self == 33);
    CHECK(scheduleDependency.boundary == 1);
}

// PleLoader Agent - Schedule Time Dependency Test
TEST_CASE("PleLoader-MceScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleLAgent               = commandStream[2];
    const Agent& mceSAgent               = commandStream[3];
    const Dependency& scheduleDependency = pleLAgent.info.scheduleDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ifmChannels;

    CHECK(scheduleDependency.relativeAgentId == 1);
    CHECK(scheduleDependency.outerRatio.other == numberOfMceStripes);
    CHECK(scheduleDependency.outerRatio.self == 1);
    CHECK(scheduleDependency.innerRatio.other == numberOfMceStripes);
    CHECK(scheduleDependency.innerRatio.self == 1);
    CHECK(scheduleDependency.boundary == 0);
}

// PleLoader Agent - Schedule Time Dependency Test
TEST_CASE("PleLoader-PleScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleLAgent               = commandStream[1];
    const Agent& pleSAgent               = commandStream[2];
    const Dependency& scheduleDependency = pleLAgent.info.scheduleDependencies.at(0);

    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    CHECK(scheduleDependency.relativeAgentId == 1);
    CHECK(scheduleDependency.outerRatio.other == numberOfPleStripes);
    CHECK(scheduleDependency.outerRatio.self == 1);
    CHECK(scheduleDependency.innerRatio.other == numberOfPleStripes);
    CHECK(scheduleDependency.innerRatio.self == 1);
    CHECK(scheduleDependency.boundary == 0);
}

// PleScheduler Agent - Schedule Time Dependency Test
TEST_CASE("PleScheduler-OfmStreamer ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleSAgent               = commandStream[2];
    const Dependency& scheduleDependency = pleSAgent.info.scheduleDependencies.at(0);

    CHECK(scheduleDependency.relativeAgentId == 1);
    CHECK(scheduleDependency.outerRatio.other == 1);
    CHECK(scheduleDependency.outerRatio.self == 1);
    CHECK(scheduleDependency.innerRatio.other == 1);
    CHECK(scheduleDependency.innerRatio.self == 1);
    CHECK(scheduleDependency.boundary == 0);
}

// OfmStreamer Agent - Schedule Time Dependency Test
TEST_CASE("OfmStreamer-IfmStreamer ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceDramIntermediateOpGraph twoMceOpMergeGraph = TwoMceDramIntermediateOpGraph();
    OpGraph mergedOpGraph                            = twoMceOpMergeGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ofmSAgent               = commandStream[5];
    const Agent& ifmSAgent               = commandStream[6];
    const Dependency& scheduleDependency = ofmSAgent.info.scheduleDependencies.at(0);

    uint32_t numOfOfmStripes = ofmSAgent.data.ofm.fmData.numStripes.height *
                               ofmSAgent.data.ofm.fmData.numStripes.width *
                               ofmSAgent.data.ofm.fmData.numStripes.channels;
    uint32_t numOfIfmStripes = ifmSAgent.data.ifm.fmData.numStripes.height *
                               ifmSAgent.data.ifm.fmData.numStripes.width *
                               ifmSAgent.data.ifm.fmData.numStripes.channels;

    CHECK(scheduleDependency.relativeAgentId == 1);
    CHECK(scheduleDependency.outerRatio.other == numOfIfmStripes);
    CHECK(scheduleDependency.outerRatio.self == numOfOfmStripes);
    CHECK(scheduleDependency.innerRatio.other == 1);
    CHECK(scheduleDependency.innerRatio.self == 400);
    CHECK(scheduleDependency.boundary == 0);
}

// Producer-Consumer Agent - Intermediate Dram Buffer Lifetime Test
// Manually creates a network consisting of a Glue with an Intermediate Dram Buffer, to test the lifetime logic of the CascadingCommandStreamGenerator.
// The topology is chosen to test cases including:
//      * Intermediate Dram Buffers with branches, whose end of Lifetime depends on their last consumer Op.
TEST_CASE("Producer-Consumer IntermediateDramBufferLifetime Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraphIntermediateDramBuffers mceOpGraphIntermediateBuffers = MceOpGraphIntermediateDramBuffers();
    OpGraph mergedOpGraph                                           = mceOpGraphIntermediateBuffers.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    // Create CascadingCommandStreamGenerator object and generate command stream
    DebuggingContext debuggingContext{ CompilationOptions::DebugInfo() };
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt,
                                                           debuggingContext);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    // Use dedicated functions to retrieve private OpGraph, IntermdiateDramBufToBufIdMapping and BufferManager
    for (Buffer* buffer : commandStreamGenerator.GetMergedOpGraph().GetBuffers())
    {
        if (buffer->m_Location == Location::Dram && buffer->m_BufferType.value() == BufferType::Intermediate)
        {
            // Retrieve Buffer Id for a Dram Buffer using m_DramBufToBufIdMapping.
            // Buffer Id is internal to m_BufferManager
            auto buffId = commandStreamGenerator.GetDramBufToBufIdMapping().at(buffer);

            const BufferManager& bufferManager = commandStreamGenerator.GetBufferManager();

            // Use Buffer Id to retrieve the appropriate Buffer's CompilerBufferInfo and use that to check the Lifetimes.
            CHECK(bufferManager.GetBuffers().at(buffId).m_LifetimeStart == 5);
            CHECK(bufferManager.GetBuffers().at(buffId).m_LifetimeEnd == 9);
        }
    }
}
