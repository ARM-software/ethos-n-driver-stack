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
// Cascading Compiler Testing Classes
//////////////////////////////////////////////////////////////////////////////////////////////

class StandalonePleOpGraph
{
public:
    StandalonePleOpGraph()
    {
        auto& parts       = graph.m_Parts;
        auto& connections = graph.m_Connections;

        auto inputDramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
        auto inputSramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
        auto plePart        = std::make_unique<MockPart>(graph.GeneratePartId());
        auto outputDramPart = std::make_unique<MockPart>(graph.GeneratePartId());

        PartId inputDramPartId  = inputDramPart->GetPartId();
        PartId inputSramPartId  = inputSramPart->GetPartId();
        PartId plePartId        = plePart->GetPartId();
        PartId outputDramPartId = outputDramPart->GetPartId();

        parts.push_back(std::move(inputDramPart));
        parts.push_back(std::move(inputSramPart));
        parts.push_back(std::move(plePart));
        parts.push_back(std::move(outputDramPart));

        PartOutputSlot inputDramPartOutputSlot0 = { inputDramPartId, 0 };

        PartInputSlot inputSramPartInputSlot0   = { inputSramPartId, 0 };
        PartOutputSlot inputSramPartOutputSlot0 = { inputSramPartId, 0 };

        PartInputSlot plePartInputSlot0   = { plePartId, 0 };
        PartOutputSlot plePartOutputSlot0 = { plePartId, 0 };

        PartInputSlot outputDramPartInputSlot0 = { outputDramPartId, 0 };

        connections[inputSramPartInputSlot0]  = inputDramPartOutputSlot0;
        connections[plePartInputSlot0]        = inputSramPartOutputSlot0;
        connections[outputDramPartInputSlot0] = plePartOutputSlot0;

        const std::set<uint32_t> operationIds = { 0 };
        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;

        // Plan inputDramPlan
        inputDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDramBuffer";
        inputDramPlan.m_OutputMappings = { { inputDramPlan.m_OpGraph.GetBuffers()[0], inputDramPartOutputSlot0 } };

        // Glue glueInputDram_InputSram
        glueInputDram_InputSram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueInputDram_InputSram.m_Graph.GetOps()[0]->m_DebugTag = "InputDmaOp";
        glueInputDram_InputSram.m_InputSlot                     = { glueInputDram_InputSram.m_Graph.GetOps()[0], 0 };
        glueInputDram_InputSram.m_Output.push_back(glueInputDram_InputSram.m_Graph.GetOps()[0]);

        // Plan inputSramPlan
        inputSramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                     TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramBuffer";
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000000F;
        inputSramPlan.m_InputMappings  = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartInputSlot0 } };
        inputSramPlan.m_OutputMappings = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartOutputSlot0 } };

        // Plan standalone plePlan
        plePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                             TensorShape{ 1, 160, 160, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                             TraversalOrder::Xyz, 4, QuantizationInfo()));
        plePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateInputSramBuffer";
        plePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x000000F0;
        pleOp = std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::LEAKY_RELU,
                                        BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                        TensorShape{ 1, 8, 8, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp.get()->m_Offset     = 0x000000FF;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp    = AddPleToOpGraph(plePlan.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                                 TensorShape{ 1, 8, 8, 32 }, numMemoryStripes, std::move(pleOp),
                                                 TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
        plePlan.m_OpGraph.GetBuffers().back()->m_Offset = 0x00000F00;
        plePlan.m_OpGraph.AddConsumer(plePlan.m_OpGraph.GetBuffers()[0], plePlan.m_OpGraph.GetOps()[0], 0);

        plePlan.m_InputMappings  = { { plePlan.m_OpGraph.GetBuffers()[0], plePartInputSlot0 } };
        plePlan.m_OutputMappings = { { plePlan.m_OpGraph.GetBuffers()[1], plePartOutputSlot0 } };

        // Glue glueOutputSram_OutputDram
        glueOutputSram_OutputDram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueOutputSram_OutputDram.m_Graph.GetOps()[0]->m_DebugTag = "OutputDmaOp";
        glueOutputSram_OutputDram.m_InputSlot = { glueOutputSram_OutputDram.m_Graph.GetOps()[0], 0 };
        glueOutputSram_OutputDram.m_Output.push_back(glueOutputSram_OutputDram.m_Graph.GetOps()[0]);

        // Plan outputDramPlan
        outputDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 80, 80, 24 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        outputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
        outputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDramBuffer";
        outputDramPlan.m_InputMappings = { { outputDramPlan.m_OpGraph.GetBuffers()[0], outputDramPartInputSlot0 } };

        Elem elemInputDram  = { std::make_shared<Plan>(std::move(inputDramPlan)),
                               { { inputSramPartInputSlot0, { &glueInputDram_InputSram, true } } } };
        Elem elemInputSram  = { std::make_shared<Plan>(std::move(inputSramPlan)), {} };
        Elem elemPle        = { std::make_shared<Plan>(std::move(plePlan)),
                         { { outputDramPartInputSlot0, { &glueOutputSram_OutputDram, true } } } };
        Elem elemOutputDram = { std::make_shared<Plan>(std::move(outputDramPlan)), {} };

        comb.m_Elems.insert(std::make_pair(0, elemInputDram));
        comb.m_PartIdsInOrder.push_back(0);
        comb.m_Elems.insert(std::make_pair(1, elemInputSram));
        comb.m_PartIdsInOrder.push_back(1);
        comb.m_Elems.insert(std::make_pair(2, elemPle));
        comb.m_PartIdsInOrder.push_back(2);
        comb.m_Elems.insert(std::make_pair(3, elemOutputDram));
        comb.m_PartIdsInOrder.push_back(3);

        bool dumpInputGraphToFile = false;
        if (dumpInputGraphToFile)
        {
            std::ofstream stream("CascadingCommandStreamGenerator PleOnlySchedulerAgent Input.dot");
            SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
        }

        mergedOpGraph = GetOpGraphForCombination(comb, graph);

        bool dumpOutputGraphToFile = false;
        if (dumpOutputGraphToFile)
        {
            std::ofstream stream("CascadingCommandStreamGenerator PleOnlySchedulerAgent Output.dot");
            SaveOpGraphToDot(mergedOpGraph, stream, DetailLevel::High);
        }

        ETHOSN_UNUSED(outBufferAndPleOp);
    }

    OpGraph GetMergedOpGraph()
    {
        return mergedOpGraph;
    }

private:
    GraphOfParts graph;

    Plan inputDramPlan;
    Glue glueInputDram_InputSram;
    Plan inputSramPlan;
    Plan plePlan;
    Glue glueOutputSram_OutputDram;
    Plan outputDramPlan;

    std::unique_ptr<PleOp> pleOp;

    Combination comb;
    OpGraph mergedOpGraph;
};

class MceOpGraph
{
public:
    MceOpGraph()
    {
        auto& parts       = graph.m_Parts;
        auto& connections = graph.m_Connections;

        auto inputDramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
        auto inputSramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
        auto weightDramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto weightSramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto mcePlePart     = std::make_unique<MockPart>(graph.GeneratePartId());
        auto outputDramPart = std::make_unique<MockPart>(graph.GeneratePartId());

        PartId inputDramPartId  = inputDramPart->GetPartId();
        PartId inputSramPartId  = inputSramPart->GetPartId();
        PartId weightDramPartId = weightDramPart->GetPartId();
        PartId weightSramPartId = weightSramPart->GetPartId();
        PartId mcePlePartId     = mcePlePart->GetPartId();
        PartId outputDramPartId = outputDramPart->GetPartId();

        parts.push_back(std::move(inputDramPart));
        parts.push_back(std::move(inputSramPart));
        parts.push_back(std::move(weightDramPart));
        parts.push_back(std::move(weightSramPart));
        parts.push_back(std::move(mcePlePart));
        parts.push_back(std::move(outputDramPart));

        PartOutputSlot inputDramPartOutputSlot0  = { inputDramPartId, 0 };
        PartOutputSlot weightDramPartOutputSlot0 = { weightDramPartId, 0 };

        PartInputSlot inputSramPartInputSlot0   = { inputSramPartId, 0 };
        PartOutputSlot inputSramPartOutputSlot0 = { inputSramPartId, 0 };

        PartInputSlot weightSramPartInputSlot0   = { weightSramPartId, 0 };
        PartOutputSlot weightSramPartOutputSlot0 = { weightSramPartId, 0 };

        PartInputSlot mcePlePartInputSlot0   = { mcePlePartId, 0 };
        PartInputSlot mcePlePartInputSlot1   = { mcePlePartId, 1 };
        PartOutputSlot mcePlePartOutputSlot0 = { mcePlePartId, 0 };

        PartInputSlot outputDramPartInputSlot0 = { outputDramPartId, 0 };

        connections[inputSramPartInputSlot0]  = inputDramPartOutputSlot0;
        connections[weightSramPartInputSlot0] = weightDramPartOutputSlot0;
        connections[mcePlePartInputSlot0]     = inputSramPartOutputSlot0;
        connections[mcePlePartInputSlot1]     = weightSramPartOutputSlot0;
        connections[outputDramPartInputSlot0] = mcePlePartOutputSlot0;

        const std::set<uint32_t> operationIds = { 0 };
        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;

        // Plan inputDramPlan
        inputDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDramBuffer";
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000F0A;
        inputDramPlan.m_OutputMappings = { { inputDramPlan.m_OpGraph.GetBuffers()[0], inputDramPartOutputSlot0 } };

        // Glue glueInputDram_InputSram
        glueInputDram_InputSram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueInputDram_InputSram.m_Graph.GetOps()[0]->m_DebugTag = "InputDmaOp";
        glueInputDram_InputSram.m_InputSlot                     = { glueInputDram_InputSram.m_Graph.GetOps()[0], 0 };
        glueInputDram_InputSram.m_Output.push_back(glueInputDram_InputSram.m_Graph.GetOps()[0]);

        // Plan inputSramPlan
        inputSramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                     TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputSramBuffer";
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes = 4;
        inputSramPlan.m_InputMappings  = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartInputSlot0 } };
        inputSramPlan.m_OutputMappings = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartOutputSlot0 } };

        Buffer* ptrInputBuffer = inputSramPlan.m_OpGraph.GetBuffers().back();
        inputStripeSize        = utils::TotalSizeBytesNHWCB(ptrInputBuffer->m_StripeShape);
        inputZeroPoint         = ptrInputBuffer->m_QuantizationInfo.GetZeroPoint();

        // Plan weightDramPlan
        weightDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT, TensorShape{ 1, 3, 1, 1 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        weightDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        weightDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag       = "WeightDramBuffer";
        encodedWeights                                                 = std::make_shared<EncodedWeights>();
        encodedWeights->m_Data                                         = { 1, 2, 3, 4 };
        encodedWeights->m_MaxSize                                      = 10;
        encodedWeights->m_Metadata                                     = { { 0, 2 }, { 2, 2 } };
        weightDramPlan.m_OpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights;
        weightDramPlan.m_OutputMappings = { { weightDramPlan.m_OpGraph.GetBuffers()[0], weightDramPartOutputSlot0 } };

        // Glue glueWeightDram_WeightSram
        glueWeightDram_WeightSram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueWeightDram_WeightSram.m_Graph.GetOps()[0]->m_DebugTag = "WeightDmaOp";
        glueWeightDram_WeightSram.m_InputSlot = { glueWeightDram_WeightSram.m_Graph.GetOps()[0], 0 };
        glueWeightDram_WeightSram.m_Output.push_back(glueWeightDram_WeightSram.m_Graph.GetOps()[0]);

        // Plan weightSramPlan
        weightSramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT, TensorShape{ 1, 3, 1, 1 },
                                     TensorShape{ 1, 1, 16, 1 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag    = "WeightSramBuffer";
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes  = 3;
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights->m_MaxSize;
        weightSramPlan.m_InputMappings  = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartInputSlot0 } };
        weightSramPlan.m_OutputMappings = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartOutputSlot0 } };

        Buffer* ptrWeightBuffer = weightSramPlan.m_OpGraph.GetBuffers().back();
        weightSize              = ptrWeightBuffer->m_SizeInBytes / ptrWeightBuffer->m_NumStripes;
        kernelHeight            = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[0]);
        kernelWidth             = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[1]);

        // Plan mcePlePlan
        mcePlePlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                     TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateInputSramBuffer";
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000FFF;
        mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 3, 1, 1 }, TensorShape{ 1, 16, 1, 1 },
                                                                TraversalOrder::Xyz, 4, QuantizationInfo()));
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateWeightSramBuffer";
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F000;
        mcePlePlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 17, 16, 16 },
                                     TensorShape{ 1, 17, 16, 16 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputPleInputSramBuffer";
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F00F;

        mcePlePlan.m_OpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        mcePlePlan.m_OpGraph.GetOps()[0]->m_DebugTag = "MceOp";

        mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePlan.m_OpGraph.GetOps()[0], 0);
        mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePlan.m_OpGraph.GetOps()[0], 1);
        mcePlePlan.m_OpGraph.SetProducer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[0]);

        ifmDeltaHeight = static_cast<int8_t>(inputSramPlan.m_OpGraph.GetBuffers()[0]->m_TensorShape[1] -
                                             mcePlePlan.m_OpGraph.GetBuffers()[2]->m_TensorShape[1]);
        ifmDeltaWidth  = static_cast<int8_t>(inputSramPlan.m_OpGraph.GetBuffers()[0]->m_TensorShape[2] -
                                            mcePlePlan.m_OpGraph.GetBuffers()[2]->m_TensorShape[2]);

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        pleOp = std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                        BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                        TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp.get()->m_Offset     = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp    = AddPleToOpGraph(mcePlePlan.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                                 TensorShape{ 1, 4, 4, 32 }, numMemoryStripes, std::move(pleOp),
                                                 TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[1], 0);

        mcePlePlan.m_InputMappings  = { { mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePartInputSlot0 },
                                       { mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePartInputSlot1 } };
        mcePlePlan.m_OutputMappings = { { mcePlePlan.m_OpGraph.GetBuffers()[3], mcePlePartOutputSlot0 } };

        // Glue glueOutputSram_OutputDram
        glueOutputSram_OutputDram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueOutputSram_OutputDram.m_Graph.GetOps()[0]->m_DebugTag = "OutputDmaOp";
        glueOutputSram_OutputDram.m_InputSlot = { glueOutputSram_OutputDram.m_Graph.GetOps()[0], 0 };
        glueOutputSram_OutputDram.m_Output.push_back(glueOutputSram_OutputDram.m_Graph.GetOps()[0]);

        // Plan outputDramPlan
        outputDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 80, 80, 24 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        outputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
        outputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDramBuffer";
        outputDramPlan.m_InputMappings = { { outputDramPlan.m_OpGraph.GetBuffers()[0], outputDramPartInputSlot0 } };

        Elem elemInputDram  = { std::make_shared<Plan>(std::move(inputDramPlan)),
                               { { inputSramPartInputSlot0, { &glueInputDram_InputSram, true } } } };
        Elem elemInputSram  = { std::make_shared<Plan>(std::move(inputSramPlan)), {} };
        Elem elemWeightDram = { std::make_shared<Plan>(std::move(weightDramPlan)),
                                { { weightSramPartInputSlot0, { &glueWeightDram_WeightSram, true } } } };
        Elem elemWeightSram = { std::make_shared<Plan>(std::move(weightSramPlan)), {} };
        Elem elemMcePle     = { std::make_shared<Plan>(std::move(mcePlePlan)),
                            { { outputDramPartInputSlot0, { &glueOutputSram_OutputDram, true } } } };
        Elem elemOutputDram = { std::make_shared<Plan>(std::move(outputDramPlan)), {} };

        comb.m_Elems.insert(std::make_pair(0, elemInputDram));
        comb.m_PartIdsInOrder.push_back(0);
        comb.m_Elems.insert(std::make_pair(1, elemInputSram));
        comb.m_PartIdsInOrder.push_back(1);
        comb.m_Elems.insert(std::make_pair(2, elemWeightDram));
        comb.m_PartIdsInOrder.push_back(2);
        comb.m_Elems.insert(std::make_pair(3, elemWeightSram));
        comb.m_PartIdsInOrder.push_back(3);
        comb.m_Elems.insert(std::make_pair(4, elemMcePle));
        comb.m_PartIdsInOrder.push_back(4);
        comb.m_Elems.insert(std::make_pair(5, elemOutputDram));
        comb.m_PartIdsInOrder.push_back(5);

        bool dumpInputGraphToFile = false;
        if (dumpInputGraphToFile)
        {
            std::ofstream stream("CascadingCommandStreamGenerator_MceSchedulerAgent_Input.dot");
            SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
        }

        mergedOpGraph = GetOpGraphForCombination(comb, graph);

        bool dumpOutputGraphToFile = false;
        if (dumpOutputGraphToFile)
        {
            std::ofstream stream("CascadingCommandStreamGenerator_MceSchedulerAgent_Output.dot");
            SaveOpGraphToDot(mergedOpGraph, stream, DetailLevel::High);
        }

        ETHOSN_UNUSED(outBufferAndPleOp);
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
    Glue glueInputDram_InputSram;
    Plan inputSramPlan;
    Plan weightDramPlan;
    Glue glueWeightDram_WeightSram;
    Plan weightSramPlan;
    Plan mcePlePlan;
    Glue glueOutputSram_OutputDram;
    Plan outputDramPlan;

    std::shared_ptr<EncodedWeights> encodedWeights;

    std::unique_ptr<PleOp> pleOp;

    uint32_t inputStripeSize;
    uint32_t weightSize;
    int32_t inputZeroPoint;

    uint8_t kernelHeight;
    uint8_t kernelWidth;
    int8_t ifmDeltaHeight;
    int8_t ifmDeltaWidth;

    Combination comb;
    OpGraph mergedOpGraph;
};

// This class creates a network consisting of an Intermediate Dram Buffer with multiple consumers
class MceOpGraphIntermediateDramBuffers
{
public:
    MceOpGraphIntermediateDramBuffers()
    {
        // Create graph:
        //                /-> D (SramBuffer) - E (DramBuffer)
        //  A (Mce + Ple) ->  B (SramBuffer) - C (DramBuffer)
        //
        auto& parts       = graph.m_Parts;
        auto& connections = graph.m_Connections;

        auto pA        = std::make_unique<MockPart>(graph.GeneratePartId());
        auto pB        = std::make_unique<MockPart>(graph.GeneratePartId());
        auto pC        = std::make_unique<MockPart>(graph.GeneratePartId());
        auto pD        = std::make_unique<MockPart>(graph.GeneratePartId());
        auto pE        = std::make_unique<MockPart>(graph.GeneratePartId());
        PartId partAId = pA->GetPartId();
        PartId partBId = pB->GetPartId();
        PartId partCId = pC->GetPartId();
        PartId partDId = pD->GetPartId();
        PartId partEId = pE->GetPartId();
        parts.push_back(std::move(pA));
        parts.push_back(std::move(pB));
        parts.push_back(std::move(pC));
        parts.push_back(std::move(pD));
        parts.push_back(std::move(pE));

        PartOutputSlot partAOutputSlot0 = { partAId, 0 };
        PartOutputSlot partAOutputSlot1 = { partAId, 1 };

        PartInputSlot partBInputSlot0   = { partBId, 0 };
        PartOutputSlot partBOutputSlot0 = { partBId, 0 };

        PartInputSlot partCInputSlot0 = { partCId, 0 };

        PartInputSlot partDInputSlot0   = { partDId, 0 };
        PartOutputSlot partDOutputSlot0 = { partDId, 0 };

        PartInputSlot partEInputSlot0 = { partEId, 0 };

        connections[partBInputSlot0] = partAOutputSlot0;
        connections[partCInputSlot0] = partBOutputSlot0;
        connections[partDInputSlot0] = partAOutputSlot1;
        connections[partEInputSlot0] = partDOutputSlot0;

        // Plan A
        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 160, 160, 3 }, TensorShape{ 0, 0, 0, 0 },
                                                           TraversalOrder::Xyz, 0, QuantizationInfo()));
        planA.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
        planA.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDramBuffer";

        planA.m_OpGraph.AddOp(std::make_unique<DmaOp>());
        planA.m_OpGraph.GetOps()[0]->m_DebugTag = "InputDmaOp";

        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 160, 160, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                           TraversalOrder::Xyz, 0, QuantizationInfo()));
        planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramBuffer";
        planA.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000000F;

        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT,
                                                           TensorShape{ 1, 1, 3, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                           TraversalOrder::Xyz, 0, QuantizationInfo()));
        planA.m_OpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        planA.m_OpGraph.GetBuffers().back()->m_DebugTag       = "WeightsDramBuffer";
        encodedWeights                                        = std::make_shared<EncodedWeights>();
        encodedWeights->m_Data                                = { 1, 2, 3, 4 };
        encodedWeights->m_MaxSize                             = 10;
        encodedWeights->m_Metadata                            = { { 0, 2 }, { 2, 2 } };
        planA.m_OpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights;

        planA.m_OpGraph.AddOp(std::make_unique<DmaOp>());
        planA.m_OpGraph.GetOps()[1]->m_DebugTag = "WeightsDmaOp";

        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                           TraversalOrder::Xyz, 4, QuantizationInfo()));
        planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "WeightsSramBuffer";
        planA.m_OpGraph.GetBuffers().back()->m_Offset   = 0x000000F0;

        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                           TraversalOrder::Xyz, 0, QuantizationInfo()));
        planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "PleSramBuffer";
        planA.m_OpGraph.GetBuffers().back()->m_Offset   = 0x000000FF;

        planA.m_OpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        planA.m_OpGraph.GetOps()[2]->m_DebugTag = "Mce";

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        planA.m_OpGraph.AddOp(
            std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                    BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                    TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true));
        planA.m_OpGraph.GetOps()[3]->m_DebugTag = "Ple";

        // Get the PleOp from the OpGraph, check that it is indeed a PleOp and set the Offset
        Op* maybePleOp = planA.m_OpGraph.GetOp(3);
        REQUIRE(IsPleOp(maybePleOp));
        PleOp* actualPleOp    = static_cast<PleOp*>(maybePleOp);
        actualPleOp->m_Offset = 0x00000F00;

        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 80, 80, 24 }, TensorShape{ 1, 4, 4, 32 },
                                                           TraversalOrder::Xyz, 0, QuantizationInfo()));
        planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramBuffer";
        planA.m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000F0F;

        planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[0], planA.m_OpGraph.GetOps()[0], 0);
        planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[1], planA.m_OpGraph.GetOps()[0]);
        planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[1], planA.m_OpGraph.GetOps()[2], 0);
        planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[2], planA.m_OpGraph.GetOps()[1], 0);
        planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[3], planA.m_OpGraph.GetOps()[1]);
        planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[3], planA.m_OpGraph.GetOps()[2], 1);
        planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[4], planA.m_OpGraph.GetOps()[2]);
        planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[4], planA.m_OpGraph.GetOps()[3], 0);
        planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[5], planA.m_OpGraph.GetOps()[3]);
        planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[5], partAOutputSlot0 },
                                   { planA.m_OpGraph.GetBuffers()[5], partAOutputSlot1 } };

        // GlueA_B
        glueA_B.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueA_B.m_Graph.GetOps()[0]->m_DebugTag = "InputDma";

        glueA_B.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueA_B.m_Graph.GetOps()[1]->m_DebugTag = "OutputDmaBranchA";

        glueA_B.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueA_B.m_Graph.GetOps()[2]->m_DebugTag = "OutputDmaBranchB";

        glueA_B.m_InputSlot = { glueA_B.m_Graph.GetOps()[0], 0 };
        glueA_B.m_Output.push_back(glueA_B.m_Graph.GetOps()[1]);
        glueA_B.m_Output.push_back(glueA_B.m_Graph.GetOps()[2]);
        glueA_B.m_OutDmaOffset = 1;

        glueA_B.m_Graph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                           TraversalOrder::Xyz, 0, QuantizationInfo()));
        glueA_B.m_Graph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
        glueA_B.m_Graph.GetBuffers().back()->m_DebugTag   = "IntermediateDramBuffer";

        glueA_B.m_Graph.AddConsumer(glueA_B.m_Graph.GetBuffers()[0], glueA_B.m_Graph.GetOps()[1], 0);
        glueA_B.m_Graph.AddConsumer(glueA_B.m_Graph.GetBuffers()[0], glueA_B.m_Graph.GetOps()[2], 0);
        glueA_B.m_Graph.SetProducer(glueA_B.m_Graph.GetBuffers()[0], glueA_B.m_Graph.GetOps()[0]);

        // Plan B
        planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 80, 80, 24 }, TensorShape{ 1, 8, 8, 32 },
                                                           TraversalOrder::Xyz, 4, QuantizationInfo()));
        planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "SramBufferBranchA";
        planB.m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000FF0;

        planB.m_InputMappings  = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot0 } };
        planB.m_OutputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot0 } };

        // GlueB_C
        glueB_C.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueB_C.m_Graph.GetOps()[0]->m_DebugTag = "DmaOpBranchA";

        glueB_C.m_InputSlot = { glueB_C.m_Graph.GetOps()[0], 0 };
        glueB_C.m_Output.push_back(glueB_C.m_Graph.GetOps()[0]);

        // Plan C
        planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                           TraversalOrder::Xyz, 0, QuantizationInfo()));
        planC.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
        planC.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDramBufferBranchA";

        planC.m_InputMappings = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot0 } };

        // Plan D
        planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 80, 80, 24 }, TensorShape{ 1, 8, 8, 32 },
                                                           TraversalOrder::Xyz, 4, QuantizationInfo()));
        planD.m_OpGraph.GetBuffers().back()->m_DebugTag = "SramBufferBranchB";
        planD.m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000FFF;

        planD.m_InputMappings  = { { planD.m_OpGraph.GetBuffers()[0], partDInputSlot0 } };
        planD.m_OutputMappings = { { planD.m_OpGraph.GetBuffers()[0], partDOutputSlot0 } };

        // GlueD_E
        glueD_E.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueD_E.m_Graph.GetOps()[0]->m_DebugTag = "DmaOpBranchB";

        glueD_E.m_InputSlot = { glueD_E.m_Graph.GetOps()[0], 0 };
        glueD_E.m_Output.push_back(glueD_E.m_Graph.GetOps()[0]);

        // Plan E
        planE.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                           TraversalOrder::Xyz, 0, QuantizationInfo()));
        planE.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
        planE.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDramBufferBranchB";

        planE.m_InputMappings = { { planE.m_OpGraph.GetBuffers()[0], partEInputSlot0 } };

        // Add to Combination all the Plans and Glues
        Elem elemA = { std::make_shared<Plan>(std::move(planA)),
                       { { partBInputSlot0, { &glueA_B, true } }, { partDInputSlot0, { &glueA_B, true } } } };
        Elem elemB = { std::make_shared<Plan>(std::move(planB)), { { partCInputSlot0, { &glueB_C, true } } } };
        Elem elemC = { std::make_shared<Plan>(std::move(planC)), {} };
        Elem elemD = { std::make_shared<Plan>(std::move(planD)), { { partEInputSlot0, { &glueD_E, true } } } };
        Elem elemE = { std::make_shared<Plan>(std::move(planE)), {} };

        comb.m_Elems.insert(std::make_pair(0, elemA));
        comb.m_PartIdsInOrder.push_back(0);
        comb.m_Elems.insert(std::make_pair(1, elemB));
        comb.m_PartIdsInOrder.push_back(1);
        comb.m_Elems.insert(std::make_pair(2, elemC));
        comb.m_PartIdsInOrder.push_back(2);
        comb.m_Elems.insert(std::make_pair(3, elemD));
        comb.m_PartIdsInOrder.push_back(3);
        comb.m_Elems.insert(std::make_pair(4, elemE));
        comb.m_PartIdsInOrder.push_back(4);

        bool dumpInputGraphToFile = false;
        if (dumpInputGraphToFile)
        {
            std::ofstream stream("IntermediateDramBufferLifetime Test Input.dot");
            SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
        }

        mergedOpGraph = GetOpGraphForCombination(comb, graph);

        bool dumpOutputGraphToFile = false;
        if (dumpOutputGraphToFile)
        {
            std::ofstream stream("IntermediateDramBufferLifetime Test Output.dot");
            SaveOpGraphToDot(mergedOpGraph, stream, DetailLevel::High);
        }
    }

    OpGraph GetMergedOpGraph()
    {
        return mergedOpGraph;
    }

private:
    GraphOfParts graph;

    Plan planA;
    Glue glueA_B;
    Plan planB;
    Glue glueB_C;
    Plan planC;
    Plan planD;
    Glue glueD_E;
    Plan planE;

    std::shared_ptr<EncodedWeights> encodedWeights;

    Combination comb;
    OpGraph mergedOpGraph;
};

class ConcatOpGraph
{
public:
    ConcatOpGraph()
    {
        auto& parts = graph.m_Parts;

        auto concatPart = std::make_unique<MockPart>(graph.GeneratePartId());

        PartId concatPartId = concatPart->GetPartId();

        parts.push_back(std::move(concatPart));

        PartInputSlot concatPartInputSlot0   = { concatPartId, 0 };
        PartInputSlot concatPartInputSlot1   = { concatPartId, 1 };
        PartOutputSlot concatPartOutputSlot0 = { concatPartId, 0 };

        const std::set<uint32_t> operationIds = { 0 };

        // Plan concatPlan
        concatPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 16, 16, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                                TraversalOrder::Xyz, 4, QuantizationInfo()));
        concatPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "Input1DramBuffer";
        concatPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000FFF;
        concatPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
        concatPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 16, 8, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                                TraversalOrder::Xyz, 4, QuantizationInfo()));
        concatPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "Input2DramBuffer";
        concatPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x0000F000;
        concatPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
        concatPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 24, 3 },
                                     TensorShape{ 1, 16, 24, 3 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        concatPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDramBuffer";
        concatPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x0000F00F;
        concatPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
        concatPlan.m_OpGraph.AddOp(std::make_unique<ConcatOp>());
        concatPlan.m_OpGraph.GetOps()[0]->m_DebugTag = "ConcatOp";
        concatPlan.m_OpGraph.AddConsumer(concatPlan.m_OpGraph.GetBuffers()[0], concatPlan.m_OpGraph.GetOps()[0], 0);
        concatPlan.m_OpGraph.AddConsumer(concatPlan.m_OpGraph.GetBuffers()[1], concatPlan.m_OpGraph.GetOps()[0], 1);
        concatPlan.m_OpGraph.SetProducer(concatPlan.m_OpGraph.GetBuffers()[2], concatPlan.m_OpGraph.GetOps()[0]);
        concatPlan.m_InputMappings  = { { concatPlan.m_OpGraph.GetBuffers()[0], concatPartInputSlot0 },
                                       { concatPlan.m_OpGraph.GetBuffers()[1], concatPartInputSlot1 } };
        concatPlan.m_OutputMappings = { { concatPlan.m_OpGraph.GetBuffers()[2], concatPartOutputSlot0 } };

        Elem elemInput1Dram = { std::make_shared<Plan>(std::move(input1DramPlan)), {} };
        Elem elemInput2Dram = { std::make_shared<Plan>(std::move(input2DramPlan)), {} };
        Elem elemConcat     = { std::make_shared<Plan>(std::move(concatPlan)), {} };
        Elem elemOutputDram = { std::make_shared<Plan>(std::move(outputDramPlan)), {} };

        comb.m_Elems.insert(std::make_pair(0, elemConcat));
        comb.m_PartIdsInOrder.push_back(0);

        bool dumpInputGraphToFile = false;
        if (dumpInputGraphToFile)
        {
            std::ofstream stream("Concat_Graph.dot");
            SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
        }

        mergedOpGraph = GetOpGraphForCombination(comb, graph);

        bool dumpOutputGraphToFile = false;
        if (dumpOutputGraphToFile)
        {
            std::ofstream stream("Concat_Graph_Merged.dot");
            SaveOpGraphToDot(mergedOpGraph, stream, DetailLevel::High);
        }
    }

    OpGraph GetMergedOpGraph()
    {
        return mergedOpGraph;
    }

private:
    GraphOfParts graph;
    Plan input1DramPlan;
    Plan input2DramPlan;
    Plan concatPlan;
    Plan outputDramPlan;
    Combination comb;
    OpGraph mergedOpGraph;
};

class TwoMceDramIntermediateOpGraph
{
public:
    TwoMceDramIntermediateOpGraph()
    {
        auto& parts       = graph.m_Parts;
        auto& connections = graph.m_Connections;

        auto inputDramPart        = std::make_unique<MockPart>(graph.GeneratePartId());
        auto inputSramPart        = std::make_unique<MockPart>(graph.GeneratePartId());
        auto weightDramPart       = std::make_unique<MockPart>(graph.GeneratePartId());
        auto weightSramPart       = std::make_unique<MockPart>(graph.GeneratePartId());
        auto mcePlePart           = std::make_unique<MockPart>(graph.GeneratePartId());
        auto intermediateDramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto intermediateSramPart = std::make_unique<MockPart>(graph.GeneratePartId());

        auto weight2DramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto weight2SramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto mcePle2Part     = std::make_unique<MockPart>(graph.GeneratePartId());
        auto outputDramPart  = std::make_unique<MockPart>(graph.GeneratePartId());

        PartId inputDramPartId        = inputDramPart->GetPartId();
        PartId inputSramPartId        = inputSramPart->GetPartId();
        PartId weightDramPartId       = weightDramPart->GetPartId();
        PartId weightSramPartId       = weightSramPart->GetPartId();
        PartId mcePlePartId           = mcePlePart->GetPartId();
        PartId intermediateDramPartId = intermediateDramPart->GetPartId();
        PartId intermediateSramPartId = intermediateSramPart->GetPartId();

        PartId weight2DramPartId = weight2DramPart->GetPartId();
        PartId weight2SramPartId = weight2SramPart->GetPartId();
        PartId mcePle2PartId     = mcePle2Part->GetPartId();
        PartId outputDramPartId  = outputDramPart->GetPartId();

        parts.push_back(std::move(inputDramPart));
        parts.push_back(std::move(inputSramPart));
        parts.push_back(std::move(weightDramPart));
        parts.push_back(std::move(weightSramPart));
        parts.push_back(std::move(mcePlePart));
        parts.push_back(std::move(intermediateDramPart));
        parts.push_back(std::move(intermediateSramPart));

        parts.push_back(std::move(weight2DramPart));
        parts.push_back(std::move(weight2SramPart));
        parts.push_back(std::move(mcePle2Part));
        parts.push_back(std::move(outputDramPart));

        PartOutputSlot inputDramPartOutputSlot0  = { inputDramPartId, 0 };
        PartOutputSlot weightDramPartOutputSlot0 = { weightDramPartId, 0 };

        PartInputSlot inputSramPartInputSlot0   = { inputSramPartId, 0 };
        PartOutputSlot inputSramPartOutputSlot0 = { inputSramPartId, 0 };

        PartInputSlot weightSramPartInputSlot0   = { weightSramPartId, 0 };
        PartOutputSlot weightSramPartOutputSlot0 = { weightSramPartId, 0 };

        PartInputSlot mcePlePartInputSlot0   = { mcePlePartId, 0 };
        PartInputSlot mcePlePartInputSlot1   = { mcePlePartId, 1 };
        PartOutputSlot mcePlePartOutputSlot0 = { mcePlePartId, 0 };

        PartInputSlot intermediateDramPartInputSlot0   = { intermediateDramPartId, 0 };
        PartOutputSlot intermediateDramPartOutputSlot0 = { intermediateDramPartId, 0 };

        PartInputSlot intermediateSramPartInputSlot0   = { intermediateSramPartId, 0 };
        PartOutputSlot intermediateSramPartOutputSlot0 = { intermediateSramPartId, 0 };

        PartOutputSlot weight2DramPartOutputSlot0 = { weight2DramPartId, 0 };

        PartInputSlot weight2SramPartInputSlot0   = { weight2SramPartId, 0 };
        PartOutputSlot weight2SramPartOutputSlot0 = { weight2SramPartId, 0 };

        PartInputSlot mcePle2PartInputSlot0   = { mcePle2PartId, 0 };
        PartInputSlot mcePle2PartInputSlot1   = { mcePle2PartId, 1 };
        PartOutputSlot mcePle2PartOutputSlot0 = { mcePle2PartId, 0 };

        PartInputSlot outputDramPartInputSlot0 = { outputDramPartId, 0 };

        connections[inputSramPartInputSlot0]        = inputDramPartOutputSlot0;
        connections[weightSramPartInputSlot0]       = weightDramPartOutputSlot0;
        connections[mcePlePartInputSlot0]           = inputSramPartOutputSlot0;
        connections[mcePlePartInputSlot1]           = weightSramPartOutputSlot0;
        connections[intermediateDramPartInputSlot0] = mcePlePartOutputSlot0;
        connections[intermediateSramPartInputSlot0] = intermediateDramPartOutputSlot0;

        connections[weight2SramPartInputSlot0] = weight2DramPartOutputSlot0;
        connections[mcePle2PartInputSlot0]     = intermediateSramPartOutputSlot0;
        connections[mcePle2PartInputSlot1]     = weight2SramPartOutputSlot0;
        connections[outputDramPartInputSlot0]  = mcePle2PartOutputSlot0;

        const std::set<uint32_t> operationIds = { 0 };
        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;

        // Plan inputDramPlan
        inputDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDramBuffer";
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000F0A;
        inputDramPlan.m_OutputMappings = { { inputDramPlan.m_OpGraph.GetBuffers()[0], inputDramPartOutputSlot0 } };

        // Glue glueInputDram_InputSram
        glueInputDram_InputSram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueInputDram_InputSram.m_Graph.GetOps()[0]->m_DebugTag = "InputDmaOp";
        glueInputDram_InputSram.m_InputSlot                     = { glueInputDram_InputSram.m_Graph.GetOps()[0], 0 };
        glueInputDram_InputSram.m_Output.push_back(glueInputDram_InputSram.m_Graph.GetOps()[0]);

        // Plan inputSramPlan
        inputSramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                     TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputSramBuffer";
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes = 4;
        inputSramPlan.m_InputMappings  = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartInputSlot0 } };
        inputSramPlan.m_OutputMappings = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartOutputSlot0 } };

        Buffer* ptrInputBuffer = inputSramPlan.m_OpGraph.GetBuffers().back();
        inputStripeSize        = utils::TotalSizeBytesNHWCB(ptrInputBuffer->m_StripeShape);
        inputZeroPoint         = ptrInputBuffer->m_QuantizationInfo.GetZeroPoint();

        // Plan weightDramPlan
        weightDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT, TensorShape{ 1, 1, 3, 1 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        weightDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        weightDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag       = "WeightDramBuffer";
        encodedWeights                                                 = std::make_shared<EncodedWeights>();
        encodedWeights->m_Data                                         = { 1, 2, 3, 4 };
        encodedWeights->m_MaxSize                                      = 10;
        encodedWeights->m_Metadata                                     = { { 0, 2 }, { 2, 2 } };
        weightDramPlan.m_OpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights;
        weightDramPlan.m_OutputMappings = { { weightDramPlan.m_OpGraph.GetBuffers()[0], weightDramPartOutputSlot0 } };

        // Glue glueWeightDram_WeightSram
        glueWeightDram_WeightSram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueWeightDram_WeightSram.m_Graph.GetOps()[0]->m_DebugTag = "WeightDmaOp";
        glueWeightDram_WeightSram.m_InputSlot = { glueWeightDram_WeightSram.m_Graph.GetOps()[0], 0 };
        glueWeightDram_WeightSram.m_Output.push_back(glueWeightDram_WeightSram.m_Graph.GetOps()[0]);

        // Plan weightSramPlan
        weightSramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT, TensorShape{ 1, 1, 3, 1 },
                                     TensorShape{ 1, 1, 16, 1 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag    = "WeightSramBuffer";
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes  = 3;
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights->m_MaxSize;
        weightSramPlan.m_InputMappings  = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartInputSlot0 } };
        weightSramPlan.m_OutputMappings = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartOutputSlot0 } };

        Buffer* ptrWeightBuffer = weightSramPlan.m_OpGraph.GetBuffers().back();
        weightSize              = ptrWeightBuffer->m_SizeInBytes / ptrWeightBuffer->m_NumStripes;
        kernelHeight            = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[0]);
        kernelWidth             = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[1]);

        // Plan mcePlePlan
        mcePlePlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                     TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateInputSramBuffer";
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000FFF;
        mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                                TraversalOrder::Xyz, 4, QuantizationInfo()));
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateWeightSramBuffer";
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F000;
        mcePlePlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 17, 16, 16 },
                                     TensorShape{ 1, 17, 16, 16 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputPleInputSramBuffer";
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F00F;

        mcePlePlan.m_OpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        mcePlePlan.m_OpGraph.GetOps()[0]->m_DebugTag = "MceOp";

        mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePlan.m_OpGraph.GetOps()[0], 0);
        mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePlan.m_OpGraph.GetOps()[0], 1);
        mcePlePlan.m_OpGraph.SetProducer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[0]);

        ifmDeltaHeight = static_cast<int8_t>(inputSramPlan.m_OpGraph.GetBuffers()[0]->m_TensorShape[1] -
                                             mcePlePlan.m_OpGraph.GetBuffers()[2]->m_TensorShape[1]);
        ifmDeltaWidth  = static_cast<int8_t>(inputSramPlan.m_OpGraph.GetBuffers()[0]->m_TensorShape[2] -
                                            mcePlePlan.m_OpGraph.GetBuffers()[2]->m_TensorShape[2]);

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        pleOp = std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                        BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                        TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp.get()->m_Offset     = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp    = AddPleToOpGraph(mcePlePlan.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                                 TensorShape{ 1, 4, 4, 32 }, numMemoryStripes, std::move(pleOp),
                                                 TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[1], 0);

        mcePlePlan.m_InputMappings  = { { mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePartInputSlot0 },
                                       { mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePartInputSlot1 } };
        mcePlePlan.m_OutputMappings = { { mcePlePlan.m_OpGraph.GetBuffers()[3], mcePlePartOutputSlot0 } };

        // Glue glueintermediateSram_intermediateDram
        glueintermediateSram_intermediateDram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueintermediateSram_intermediateDram.m_Graph.GetOps()[0]->m_DebugTag = "intermediateDmaOp";
        glueintermediateSram_intermediateDram.m_InputSlot = { glueintermediateSram_intermediateDram.m_Graph.GetOps()[0],
                                                              0 };
        glueintermediateSram_intermediateDram.m_Output.push_back(
            glueintermediateSram_intermediateDram.m_Graph.GetOps()[0]);

        // Plan intermediateDramPlan
        intermediateDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 80, 80, 24 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        intermediateDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
        intermediateDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "intermediateDramBuffer";
        intermediateDramPlan.m_InputMappings  = { { intermediateDramPlan.m_OpGraph.GetBuffers()[0],
                                                   intermediateDramPartInputSlot0 } };
        intermediateDramPlan.m_OutputMappings = { { intermediateDramPlan.m_OpGraph.GetBuffers()[0],
                                                    intermediateDramPartOutputSlot0 } };

        // Glue glueintermediateDram_intermediateSram
        glueintermediateDram_intermediateSram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueintermediateDram_intermediateSram.m_Graph.GetOps()[0]->m_DebugTag = "intermediateSramDmaOp";
        glueintermediateDram_intermediateSram.m_InputSlot = { glueintermediateDram_intermediateSram.m_Graph.GetOps()[0],
                                                              0 };
        glueintermediateDram_intermediateSram.m_Output.push_back(
            glueintermediateDram_intermediateSram.m_Graph.GetOps()[0]);

        // Plan intermediateSramPlan
        intermediateSramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 80, 80, 24 },
                                     TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        intermediateSramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
        intermediateSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "intermediateSramBuffer";
        intermediateSramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
        intermediateSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes = 4;
        intermediateSramPlan.m_InputMappings  = { { intermediateSramPlan.m_OpGraph.GetBuffers()[0],
                                                   intermediateSramPartInputSlot0 } };
        intermediateSramPlan.m_OutputMappings = { { intermediateSramPlan.m_OpGraph.GetBuffers()[0],
                                                    intermediateSramPartOutputSlot0 } };

        // Plan weight2DramPlan
        weight2DramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT, TensorShape{ 1, 1, 3, 1 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        weight2DramPlan.m_OpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        weight2DramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag       = "Weight2DramBuffer";
        encodedWeights2                                                 = std::make_shared<EncodedWeights>();
        encodedWeights2->m_Data                                         = { 1, 2, 3, 4 };
        encodedWeights2->m_MaxSize                                      = 10;
        encodedWeights2->m_Metadata                                     = { { 0, 2 }, { 2, 2 } };
        weight2DramPlan.m_OpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights2;
        weight2DramPlan.m_OutputMappings                                = { { weight2DramPlan.m_OpGraph.GetBuffers()[0],
                                               weight2DramPartOutputSlot0 } };

        // Glue glueWeightDram_WeightSram
        glueWeight2Dram_Weight2Sram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueWeight2Dram_Weight2Sram.m_Graph.GetOps()[0]->m_DebugTag = "Weight2DmaOp";
        glueWeight2Dram_Weight2Sram.m_InputSlot = { glueWeight2Dram_Weight2Sram.m_Graph.GetOps()[0], 0 };
        glueWeight2Dram_Weight2Sram.m_Output.push_back(glueWeight2Dram_Weight2Sram.m_Graph.GetOps()[0]);

        // Plan weightSramPlan
        weight2SramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT, TensorShape{ 1, 1, 3, 1 },
                                     TensorShape{ 1, 1, 16, 1 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        weight2SramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag    = "Weight2SramBuffer";
        weight2SramPlan.m_OpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        weight2SramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes  = 3;
        weight2SramPlan.m_OpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights2->m_MaxSize;
        weight2SramPlan.m_InputMappings  = { { weight2SramPlan.m_OpGraph.GetBuffers()[0], weight2SramPartInputSlot0 } };
        weight2SramPlan.m_OutputMappings = { { weight2SramPlan.m_OpGraph.GetBuffers()[0],
                                               weight2SramPartOutputSlot0 } };

        Buffer* ptrWeightBuffer2 = weight2SramPlan.m_OpGraph.GetBuffers().back();
        weightSize2              = ptrWeightBuffer2->m_SizeInBytes / ptrWeightBuffer2->m_NumStripes;
        kernelHeight2            = static_cast<uint8_t>(ptrWeightBuffer2->m_TensorShape[0]);
        kernelWidth2             = static_cast<uint8_t>(ptrWeightBuffer2->m_TensorShape[1]);

        // Plan mcePlePlan
        mcePle2Plan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 80, 80, 24 },
                                     TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateInput2SramBuffer";
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000FFF;
        mcePle2Plan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                 TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                                 TraversalOrder::Xyz, 4, QuantizationInfo()));
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateWeight2SramBuffer";
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F000;
        mcePle2Plan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 17, 16, 16 },
                                     TensorShape{ 1, 17, 16, 16 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_DebugTag = "outputPleInputSramBuffer";
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F00F;

        mcePle2Plan.m_OpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        mcePle2Plan.m_OpGraph.GetOps()[0]->m_DebugTag = "MceOp2";

        mcePle2Plan.m_OpGraph.AddConsumer(mcePle2Plan.m_OpGraph.GetBuffers()[0], mcePle2Plan.m_OpGraph.GetOps()[0], 0);
        mcePle2Plan.m_OpGraph.AddConsumer(mcePle2Plan.m_OpGraph.GetBuffers()[1], mcePle2Plan.m_OpGraph.GetOps()[0], 1);
        mcePle2Plan.m_OpGraph.SetProducer(mcePle2Plan.m_OpGraph.GetBuffers()[2], mcePle2Plan.m_OpGraph.GetOps()[0]);

        ifmDeltaHeight = static_cast<int8_t>(intermediateSramPlan.m_OpGraph.GetBuffers()[0]->m_TensorShape[1] -
                                             mcePle2Plan.m_OpGraph.GetBuffers()[2]->m_TensorShape[1]);
        ifmDeltaWidth  = static_cast<int8_t>(intermediateSramPlan.m_OpGraph.GetBuffers()[0]->m_TensorShape[2] -
                                            mcePle2Plan.m_OpGraph.GetBuffers()[2]->m_TensorShape[2]);

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        pleOp2 =
            std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                    BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                    TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp2.get()->m_Offset    = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp2   = AddPleToOpGraph(mcePle2Plan.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                                  TensorShape{ 1, 4, 4, 32 }, numMemoryStripes, std::move(pleOp2),
                                                  TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mcePle2Plan.m_OpGraph.AddConsumer(mcePle2Plan.m_OpGraph.GetBuffers()[2], mcePle2Plan.m_OpGraph.GetOps()[1], 0);

        mcePle2Plan.m_InputMappings  = { { mcePle2Plan.m_OpGraph.GetBuffers()[0], mcePle2PartInputSlot0 },
                                        { mcePle2Plan.m_OpGraph.GetBuffers()[1], mcePle2PartInputSlot1 } };
        mcePle2Plan.m_OutputMappings = { { mcePle2Plan.m_OpGraph.GetBuffers()[3], mcePle2PartOutputSlot0 } };

        // Glue glueOutputSram_OutputDram
        glueOutputSram_OutputDram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueOutputSram_OutputDram.m_Graph.GetOps()[0]->m_DebugTag = "outputDmaOp";
        glueOutputSram_OutputDram.m_InputSlot = { glueOutputSram_OutputDram.m_Graph.GetOps()[0], 0 };
        glueOutputSram_OutputDram.m_Output.push_back(glueOutputSram_OutputDram.m_Graph.GetOps()[0]);

        // Plan outputDramPlan
        outputDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 80, 80, 24 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        outputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
        outputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "outputDramBuffer";
        outputDramPlan.m_InputMappings = { { outputDramPlan.m_OpGraph.GetBuffers()[0], outputDramPartInputSlot0 } };

        Elem elemInputDram        = { std::make_shared<Plan>(std::move(inputDramPlan)),
                               { { inputSramPartInputSlot0, { &glueInputDram_InputSram, true } } } };
        Elem elemInputSram        = { std::make_shared<Plan>(std::move(inputSramPlan)), {} };
        Elem elemWeightDram       = { std::make_shared<Plan>(std::move(weightDramPlan)),
                                { { weightSramPartInputSlot0, { &glueWeightDram_WeightSram, true } } } };
        Elem elemWeightSram       = { std::make_shared<Plan>(std::move(weightSramPlan)), {} };
        Elem elemMcePle           = { std::make_shared<Plan>(std::move(mcePlePlan)),
                            { { intermediateDramPartInputSlot0, { &glueintermediateSram_intermediateDram, true } } } };
        Elem elemintermediateDram = { std::make_shared<Plan>(std::move(intermediateDramPlan)),
                                      { { intermediateSramPartInputSlot0,
                                          { &glueintermediateDram_intermediateSram, true } } } };
        Elem elemintermediateSram = { std::make_shared<Plan>(std::move(intermediateSramPlan)), {} };

        Elem elemWeight2Dram = { std::make_shared<Plan>(std::move(weight2DramPlan)),
                                 { { weight2SramPartInputSlot0, { &glueWeight2Dram_Weight2Sram, true } } } };
        Elem elemWeight2Sram = { std::make_shared<Plan>(std::move(weight2SramPlan)), {} };
        Elem elemMcePle2     = { std::make_shared<Plan>(std::move(mcePle2Plan)),
                             { { outputDramPartInputSlot0, { &glueOutputSram_OutputDram, true } } } };
        Elem elemoutputDram  = { std::make_shared<Plan>(std::move(outputDramPlan)), {} };

        comb.m_Elems.insert(std::make_pair(0, elemInputDram));
        comb.m_PartIdsInOrder.push_back(0);
        comb.m_Elems.insert(std::make_pair(1, elemInputSram));
        comb.m_PartIdsInOrder.push_back(1);
        comb.m_Elems.insert(std::make_pair(2, elemWeightDram));
        comb.m_PartIdsInOrder.push_back(2);
        comb.m_Elems.insert(std::make_pair(3, elemWeightSram));
        comb.m_PartIdsInOrder.push_back(3);
        comb.m_Elems.insert(std::make_pair(4, elemMcePle));
        comb.m_PartIdsInOrder.push_back(4);
        comb.m_Elems.insert(std::make_pair(5, elemintermediateDram));
        comb.m_PartIdsInOrder.push_back(5);
        comb.m_Elems.insert(std::make_pair(6, elemintermediateSram));
        comb.m_PartIdsInOrder.push_back(6);

        comb.m_Elems.insert(std::make_pair(7, elemWeight2Dram));
        comb.m_PartIdsInOrder.push_back(7);
        comb.m_Elems.insert(std::make_pair(8, elemWeight2Sram));
        comb.m_PartIdsInOrder.push_back(8);
        comb.m_Elems.insert(std::make_pair(9, elemMcePle2));
        comb.m_PartIdsInOrder.push_back(9);
        comb.m_Elems.insert(std::make_pair(10, elemoutputDram));
        comb.m_PartIdsInOrder.push_back(10);

        bool dumpInputGraphToFile = false;
        if (dumpInputGraphToFile)
        {
            std::ofstream stream("CascadingCommandStreamGenerator_TwoMceSchedulerAgent_Input.dot");
            SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
        }

        mergedOpGraph = GetOpGraphForCombination(comb, graph);

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
    Glue glueInputDram_InputSram;
    Plan inputSramPlan;
    Plan weightDramPlan;
    Glue glueWeightDram_WeightSram;
    Plan weightSramPlan;
    Plan mcePlePlan;
    Glue glueintermediateSram_intermediateDram;
    Plan intermediateDramPlan;
    Glue glueintermediateDram_intermediateSram;
    Plan intermediateSramPlan;

    Plan weight2DramPlan;
    Glue glueWeight2Dram_Weight2Sram;
    Plan weight2SramPlan;
    Plan mcePle2Plan;
    Glue glueOutputSram_OutputDram;
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
    OpGraph mergedOpGraph;
};

class TwoMceSramIntermediateOpGraph
{
public:
    TwoMceSramIntermediateOpGraph()
    {
        auto& parts       = graph.m_Parts;
        auto& connections = graph.m_Connections;

        auto inputDramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
        auto inputSramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
        auto weightDramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto weightSramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto mcePlePart     = std::make_unique<MockPart>(graph.GeneratePartId());

        auto weight2DramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto weight2SramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto mcePle2Part     = std::make_unique<MockPart>(graph.GeneratePartId());
        auto outputDramPart  = std::make_unique<MockPart>(graph.GeneratePartId());

        PartId inputDramPartId  = inputDramPart->GetPartId();
        PartId inputSramPartId  = inputSramPart->GetPartId();
        PartId weightDramPartId = weightDramPart->GetPartId();
        PartId weightSramPartId = weightSramPart->GetPartId();
        PartId mcePlePartId     = mcePlePart->GetPartId();

        PartId weight2DramPartId = weight2DramPart->GetPartId();
        PartId weight2SramPartId = weight2SramPart->GetPartId();
        PartId mcePle2PartId     = mcePle2Part->GetPartId();
        PartId outputDramPartId  = outputDramPart->GetPartId();

        parts.push_back(std::move(inputDramPart));
        parts.push_back(std::move(inputSramPart));
        parts.push_back(std::move(weightDramPart));
        parts.push_back(std::move(weightSramPart));
        parts.push_back(std::move(mcePlePart));

        parts.push_back(std::move(weight2DramPart));
        parts.push_back(std::move(weight2SramPart));
        parts.push_back(std::move(mcePle2Part));
        parts.push_back(std::move(outputDramPart));

        PartOutputSlot inputDramPartOutputSlot0  = { inputDramPartId, 0 };
        PartOutputSlot weightDramPartOutputSlot0 = { weightDramPartId, 0 };

        PartInputSlot inputSramPartInputSlot0   = { inputSramPartId, 0 };
        PartOutputSlot inputSramPartOutputSlot0 = { inputSramPartId, 0 };

        PartInputSlot weightSramPartInputSlot0   = { weightSramPartId, 0 };
        PartOutputSlot weightSramPartOutputSlot0 = { weightSramPartId, 0 };

        PartInputSlot mcePlePartInputSlot0   = { mcePlePartId, 0 };
        PartInputSlot mcePlePartInputSlot1   = { mcePlePartId, 1 };
        PartOutputSlot mcePlePartOutputSlot0 = { mcePlePartId, 0 };

        PartOutputSlot weight2DramPartOutputSlot0 = { weight2DramPartId, 0 };

        PartInputSlot weight2SramPartInputSlot0   = { weight2SramPartId, 0 };
        PartOutputSlot weight2SramPartOutputSlot0 = { weight2SramPartId, 0 };

        PartInputSlot mcePle2PartInputSlot0   = { mcePle2PartId, 0 };
        PartInputSlot mcePle2PartInputSlot1   = { mcePle2PartId, 1 };
        PartOutputSlot mcePle2PartOutputSlot0 = { mcePle2PartId, 0 };

        PartInputSlot outputDramPartInputSlot0 = { outputDramPartId, 0 };

        connections[inputSramPartInputSlot0]  = inputDramPartOutputSlot0;
        connections[weightSramPartInputSlot0] = weightDramPartOutputSlot0;
        connections[mcePlePartInputSlot0]     = inputSramPartOutputSlot0;
        connections[mcePlePartInputSlot1]     = weightSramPartOutputSlot0;

        connections[weight2SramPartInputSlot0] = weight2DramPartOutputSlot0;
        connections[mcePle2PartInputSlot0]     = mcePlePartOutputSlot0;
        connections[mcePle2PartInputSlot1]     = weight2SramPartOutputSlot0;
        connections[outputDramPartInputSlot0]  = mcePle2PartOutputSlot0;

        const std::set<uint32_t> operationIds = { 0 };
        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;

        // Plan inputDramPlan
        inputDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDramBuffer";
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000F0A;
        inputDramPlan.m_OutputMappings = { { inputDramPlan.m_OpGraph.GetBuffers()[0], inputDramPartOutputSlot0 } };

        // Glue glueInputDram_InputSram
        glueInputDram_InputSram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueInputDram_InputSram.m_Graph.GetOps()[0]->m_DebugTag = "InputDmaOp";
        glueInputDram_InputSram.m_InputSlot                     = { glueInputDram_InputSram.m_Graph.GetOps()[0], 0 };
        glueInputDram_InputSram.m_Output.push_back(glueInputDram_InputSram.m_Graph.GetOps()[0]);

        // Plan inputSramPlan
        inputSramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                     TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputSramBuffer";
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes = 4;
        inputSramPlan.m_InputMappings  = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartInputSlot0 } };
        inputSramPlan.m_OutputMappings = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartOutputSlot0 } };

        Buffer* ptrInputBuffer = inputSramPlan.m_OpGraph.GetBuffers().back();
        inputStripeSize        = utils::TotalSizeBytesNHWCB(ptrInputBuffer->m_StripeShape);
        inputZeroPoint         = ptrInputBuffer->m_QuantizationInfo.GetZeroPoint();

        // Plan weightDramPlan
        weightDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT, TensorShape{ 1, 1, 3, 1 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        weightDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        weightDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag       = "WeightDramBuffer";
        encodedWeights                                                 = std::make_shared<EncodedWeights>();
        encodedWeights->m_Data                                         = { 1, 2, 3, 4 };
        encodedWeights->m_MaxSize                                      = 10;
        encodedWeights->m_Metadata                                     = { { 0, 2 }, { 2, 2 } };
        weightDramPlan.m_OpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights;
        weightDramPlan.m_OutputMappings = { { weightDramPlan.m_OpGraph.GetBuffers()[0], weightDramPartOutputSlot0 } };

        // Glue glueWeightDram_WeightSram
        glueWeightDram_WeightSram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueWeightDram_WeightSram.m_Graph.GetOps()[0]->m_DebugTag = "WeightDmaOp";
        glueWeightDram_WeightSram.m_InputSlot = { glueWeightDram_WeightSram.m_Graph.GetOps()[0], 0 };
        glueWeightDram_WeightSram.m_Output.push_back(glueWeightDram_WeightSram.m_Graph.GetOps()[0]);

        // Plan weightSramPlan
        weightSramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT, TensorShape{ 1, 1, 3, 1 },
                                     TensorShape{ 1, 1, 16, 1 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag    = "WeightSramBuffer";
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes  = 3;
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights->m_MaxSize;
        weightSramPlan.m_InputMappings  = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartInputSlot0 } };
        weightSramPlan.m_OutputMappings = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartOutputSlot0 } };

        Buffer* ptrWeightBuffer = weightSramPlan.m_OpGraph.GetBuffers().back();
        weightSize              = ptrWeightBuffer->m_SizeInBytes / ptrWeightBuffer->m_NumStripes;
        kernelHeight            = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[0]);
        kernelWidth             = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[1]);

        // Plan mcePlePlan
        mcePlePlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                     TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateInputSramBuffer";
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000FFF;
        mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                                TraversalOrder::Xyz, 4, QuantizationInfo()));
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateWeightSramBuffer";
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F000;
        mcePlePlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 17, 16, 16 },
                                     TensorShape{ 1, 17, 16, 16 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputPleInputSramBuffer";
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F00F;

        mcePlePlan.m_OpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        mcePlePlan.m_OpGraph.GetOps()[0]->m_DebugTag = "MceOp";

        mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePlan.m_OpGraph.GetOps()[0], 0);
        mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePlan.m_OpGraph.GetOps()[0], 1);
        mcePlePlan.m_OpGraph.SetProducer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[0]);

        ifmDeltaHeight = static_cast<int8_t>(inputSramPlan.m_OpGraph.GetBuffers()[0]->m_TensorShape[1] -
                                             mcePlePlan.m_OpGraph.GetBuffers()[2]->m_TensorShape[1]);
        ifmDeltaWidth  = static_cast<int8_t>(inputSramPlan.m_OpGraph.GetBuffers()[0]->m_TensorShape[2] -
                                            mcePlePlan.m_OpGraph.GetBuffers()[2]->m_TensorShape[2]);

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        pleOp = std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                        BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                        TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp.get()->m_Offset     = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp    = AddPleToOpGraph(mcePlePlan.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                                 TensorShape{ 1, 4, 4, 32 }, numMemoryStripes, std::move(pleOp),
                                                 TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[1], 0);

        mcePlePlan.m_InputMappings  = { { mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePartInputSlot0 },
                                       { mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePartInputSlot1 } };
        mcePlePlan.m_OutputMappings = { { mcePlePlan.m_OpGraph.GetBuffers()[3], mcePlePartOutputSlot0 } };

        // Plan weight2DramPlan
        weight2DramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT, TensorShape{ 1, 1, 3, 1 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        weight2DramPlan.m_OpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        weight2DramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag       = "Weight2DramBuffer";
        encodedWeights2                                                 = std::make_shared<EncodedWeights>();
        encodedWeights2->m_Data                                         = { 1, 2, 3, 4 };
        encodedWeights2->m_MaxSize                                      = 10;
        encodedWeights2->m_Metadata                                     = { { 0, 2 }, { 2, 2 } };
        weight2DramPlan.m_OpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights2;
        weight2DramPlan.m_OutputMappings                                = { { weight2DramPlan.m_OpGraph.GetBuffers()[0],
                                               weight2DramPartOutputSlot0 } };

        // Glue glueWeightDram_WeightSram
        glueWeight2Dram_Weight2Sram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueWeight2Dram_Weight2Sram.m_Graph.GetOps()[0]->m_DebugTag = "Weight2DmaOp";
        glueWeight2Dram_Weight2Sram.m_InputSlot = { glueWeight2Dram_Weight2Sram.m_Graph.GetOps()[0], 0 };
        glueWeight2Dram_Weight2Sram.m_Output.push_back(glueWeight2Dram_Weight2Sram.m_Graph.GetOps()[0]);

        // Plan weightSramPlan
        weight2SramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT, TensorShape{ 1, 1, 3, 1 },
                                     TensorShape{ 1, 1, 16, 1 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        weight2SramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag    = "Weight2SramBuffer";
        weight2SramPlan.m_OpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        weight2SramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes  = 3;
        weight2SramPlan.m_OpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights2->m_MaxSize;
        weight2SramPlan.m_InputMappings  = { { weight2SramPlan.m_OpGraph.GetBuffers()[0], weight2SramPartInputSlot0 } };
        weight2SramPlan.m_OutputMappings = { { weight2SramPlan.m_OpGraph.GetBuffers()[0],
                                               weight2SramPartOutputSlot0 } };

        Buffer* ptrWeightBuffer2 = weight2SramPlan.m_OpGraph.GetBuffers().back();
        weightSize2              = ptrWeightBuffer2->m_SizeInBytes / ptrWeightBuffer2->m_NumStripes;
        kernelHeight2            = static_cast<uint8_t>(ptrWeightBuffer2->m_TensorShape[0]);
        kernelWidth2             = static_cast<uint8_t>(ptrWeightBuffer2->m_TensorShape[1]);

        // Plan mcePlePlan
        mcePle2Plan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 80, 80, 24 },
                                     TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateInput2SramBuffer";
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000FFF;
        mcePle2Plan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                 TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                                 TraversalOrder::Xyz, 4, QuantizationInfo()));
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateWeight2SramBuffer";
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F000;
        mcePle2Plan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 17, 16, 16 },
                                     TensorShape{ 1, 17, 16, 16 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_DebugTag = "outputPleInputSramBuffer";
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F00F;

        mcePle2Plan.m_OpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        mcePle2Plan.m_OpGraph.GetOps()[0]->m_DebugTag = "MceOp2";

        mcePle2Plan.m_OpGraph.AddConsumer(mcePle2Plan.m_OpGraph.GetBuffers()[0], mcePle2Plan.m_OpGraph.GetOps()[0], 0);
        mcePle2Plan.m_OpGraph.AddConsumer(mcePle2Plan.m_OpGraph.GetBuffers()[1], mcePle2Plan.m_OpGraph.GetOps()[0], 1);
        mcePle2Plan.m_OpGraph.SetProducer(mcePle2Plan.m_OpGraph.GetBuffers()[2], mcePle2Plan.m_OpGraph.GetOps()[0]);

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        pleOp2 =
            std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                    BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                    TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
        pleOp2.get()->m_Offset    = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp2   = AddPleToOpGraph(mcePle2Plan.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                                  TensorShape{ 1, 4, 4, 32 }, numMemoryStripes, std::move(pleOp2),
                                                  TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mcePle2Plan.m_OpGraph.AddConsumer(mcePle2Plan.m_OpGraph.GetBuffers()[2], mcePle2Plan.m_OpGraph.GetOps()[1], 0);

        mcePle2Plan.m_InputMappings  = { { mcePle2Plan.m_OpGraph.GetBuffers()[0], mcePle2PartInputSlot0 },
                                        { mcePle2Plan.m_OpGraph.GetBuffers()[1], mcePle2PartInputSlot1 } };
        mcePle2Plan.m_OutputMappings = { { mcePle2Plan.m_OpGraph.GetBuffers()[3], mcePle2PartOutputSlot0 } };

        // Glue glueOutputSram_OutputDram
        glueOutputSram_OutputDram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueOutputSram_OutputDram.m_Graph.GetOps()[0]->m_DebugTag = "outputDmaOp";
        glueOutputSram_OutputDram.m_InputSlot = { glueOutputSram_OutputDram.m_Graph.GetOps()[0], 0 };
        glueOutputSram_OutputDram.m_Output.push_back(glueOutputSram_OutputDram.m_Graph.GetOps()[0]);

        // Plan outputDramPlan
        outputDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 80, 80, 24 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        outputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
        outputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "outputDramBuffer";
        outputDramPlan.m_InputMappings = { { outputDramPlan.m_OpGraph.GetBuffers()[0], outputDramPartInputSlot0 } };

        Elem elemInputDram  = { std::make_shared<Plan>(std::move(inputDramPlan)),
                               { { inputSramPartInputSlot0, { &glueInputDram_InputSram, true } } } };
        Elem elemInputSram  = { std::make_shared<Plan>(std::move(inputSramPlan)), {} };
        Elem elemWeightDram = { std::make_shared<Plan>(std::move(weightDramPlan)),
                                { { weightSramPartInputSlot0, { &glueWeightDram_WeightSram, true } } } };
        Elem elemWeightSram = { std::make_shared<Plan>(std::move(weightSramPlan)), {} };
        Elem elemMcePle     = { std::make_shared<Plan>(std::move(mcePlePlan)), {} };

        Elem elemWeight2Dram = { std::make_shared<Plan>(std::move(weight2DramPlan)),
                                 { { weight2SramPartInputSlot0, { &glueWeight2Dram_Weight2Sram, true } } } };
        Elem elemWeight2Sram = { std::make_shared<Plan>(std::move(weight2SramPlan)), {} };
        Elem elemMcePle2     = { std::make_shared<Plan>(std::move(mcePle2Plan)),
                             { { outputDramPartInputSlot0, { &glueOutputSram_OutputDram, true } } } };
        Elem elemoutputDram  = { std::make_shared<Plan>(std::move(outputDramPlan)), {} };

        comb.m_Elems.insert(std::make_pair(0, elemInputDram));
        comb.m_PartIdsInOrder.push_back(0);
        comb.m_Elems.insert(std::make_pair(1, elemInputSram));
        comb.m_PartIdsInOrder.push_back(1);
        comb.m_Elems.insert(std::make_pair(2, elemWeightDram));
        comb.m_PartIdsInOrder.push_back(2);
        comb.m_Elems.insert(std::make_pair(3, elemWeightSram));
        comb.m_PartIdsInOrder.push_back(3);
        comb.m_Elems.insert(std::make_pair(4, elemMcePle));
        comb.m_PartIdsInOrder.push_back(4);

        comb.m_Elems.insert(std::make_pair(5, elemWeight2Dram));
        comb.m_PartIdsInOrder.push_back(5);
        comb.m_Elems.insert(std::make_pair(6, elemWeight2Sram));
        comb.m_PartIdsInOrder.push_back(6);
        comb.m_Elems.insert(std::make_pair(7, elemMcePle2));
        comb.m_PartIdsInOrder.push_back(7);
        comb.m_Elems.insert(std::make_pair(8, elemoutputDram));
        comb.m_PartIdsInOrder.push_back(8);

        bool dumpInputGraphToFile = false;
        if (dumpInputGraphToFile)
        {
            std::ofstream stream("CascadingCommandStreamGenerator_TwoMceSchedulerAgent_Input.dot");
            SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
        }

        mergedOpGraph = GetOpGraphForCombination(comb, graph);

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
    Glue glueInputDram_InputSram;
    Plan inputSramPlan;
    Plan weightDramPlan;
    Glue glueWeightDram_WeightSram;
    Plan weightSramPlan;
    Plan mcePlePlan;

    Plan weight2DramPlan;
    Glue glueWeight2Dram_Weight2Sram;
    Plan weight2SramPlan;
    Plan mcePle2Plan;
    Glue glueOutputSram_OutputDram;
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
    OpGraph mergedOpGraph;
};

class StridedConvOpGraph
{
public:
    StridedConvOpGraph(uint32_t padLeft, uint32_t padTop, TensorShape outputTensorShape)
    {
        auto& parts       = graph.m_Parts;
        auto& connections = graph.m_Connections;

        auto inputDramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
        auto inputSramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
        auto weightDramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto weightSramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto mcePlePart     = std::make_unique<MockPart>(graph.GeneratePartId());

        auto weight2DramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto weight2SramPart = std::make_unique<MockPart>(graph.GeneratePartId());
        auto mcePle2Part     = std::make_unique<MockPart>(graph.GeneratePartId());
        auto outputDramPart  = std::make_unique<MockPart>(graph.GeneratePartId());

        PartId inputDramPartId  = inputDramPart->GetPartId();
        PartId inputSramPartId  = inputSramPart->GetPartId();
        PartId weightDramPartId = weightDramPart->GetPartId();
        PartId weightSramPartId = weightSramPart->GetPartId();
        PartId mcePlePartId     = mcePlePart->GetPartId();

        PartId weight2DramPartId = weight2DramPart->GetPartId();
        PartId weight2SramPartId = weight2SramPart->GetPartId();
        PartId mcePle2PartId     = mcePle2Part->GetPartId();
        PartId outputDramPartId  = outputDramPart->GetPartId();

        parts.push_back(std::move(inputDramPart));
        parts.push_back(std::move(inputSramPart));
        parts.push_back(std::move(weightDramPart));
        parts.push_back(std::move(weightSramPart));
        parts.push_back(std::move(mcePlePart));

        parts.push_back(std::move(weight2DramPart));
        parts.push_back(std::move(weight2SramPart));
        parts.push_back(std::move(mcePle2Part));
        parts.push_back(std::move(outputDramPart));

        PartOutputSlot inputDramPartOutputSlot0  = { inputDramPartId, 0 };
        PartOutputSlot weightDramPartOutputSlot0 = { weightDramPartId, 0 };

        PartInputSlot inputSramPartInputSlot0   = { inputSramPartId, 0 };
        PartOutputSlot inputSramPartOutputSlot0 = { inputSramPartId, 0 };

        PartInputSlot weightSramPartInputSlot0   = { weightSramPartId, 0 };
        PartOutputSlot weightSramPartOutputSlot0 = { weightSramPartId, 0 };

        PartInputSlot mcePlePartInputSlot0   = { mcePlePartId, 0 };
        PartInputSlot mcePlePartInputSlot1   = { mcePlePartId, 1 };
        PartOutputSlot mcePlePartOutputSlot0 = { mcePlePartId, 0 };

        PartOutputSlot weight2DramPartOutputSlot0 = { weight2DramPartId, 0 };

        PartInputSlot weight2SramPartInputSlot0   = { weight2SramPartId, 0 };
        PartOutputSlot weight2SramPartOutputSlot0 = { weight2SramPartId, 0 };

        PartInputSlot mcePle2PartInputSlot0   = { mcePle2PartId, 0 };
        PartInputSlot mcePle2PartInputSlot1   = { mcePle2PartId, 1 };
        PartOutputSlot mcePle2PartOutputSlot0 = { mcePle2PartId, 0 };

        PartInputSlot outputDramPartInputSlot0 = { outputDramPartId, 0 };

        connections[inputSramPartInputSlot0]  = inputDramPartOutputSlot0;
        connections[weightSramPartInputSlot0] = weightDramPartOutputSlot0;
        connections[mcePlePartInputSlot0]     = inputSramPartOutputSlot0;
        connections[mcePlePartInputSlot1]     = weightSramPartOutputSlot0;

        connections[weight2SramPartInputSlot0] = weight2DramPartOutputSlot0;
        connections[mcePle2PartInputSlot0]     = mcePlePartOutputSlot0;
        connections[mcePle2PartInputSlot1]     = weight2SramPartOutputSlot0;
        connections[outputDramPartInputSlot0]  = mcePle2PartOutputSlot0;

        const std::set<uint32_t> operationIds = { 0 };
        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;

        // Plan inputDramPlan
        inputDramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                                   TensorShape{ 1, 5, 5, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                                   TraversalOrder::Xyz, 0, QuantizationInfo()));
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDramBuffer";
        inputDramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000F0A;
        inputDramPlan.m_OutputMappings = { { inputDramPlan.m_OpGraph.GetBuffers()[0], inputDramPartOutputSlot0 } };

        // Glue glueInputDram_InputSram
        glueInputDram_InputSram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueInputDram_InputSram.m_Graph.GetOps()[0]->m_DebugTag = "InputDmaOp";
        glueInputDram_InputSram.m_InputSlot                     = { glueInputDram_InputSram.m_Graph.GetOps()[0], 0 };
        glueInputDram_InputSram.m_Output.push_back(glueInputDram_InputSram.m_Graph.GetOps()[0]);

        // Plan inputSramPlan
        inputSramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                   TensorShape{ 1, 5, 5, 1 }, TensorShape{ 1, 5, 5, 1 },
                                                                   TraversalOrder::Xyz, 4, QuantizationInfo()));
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputSramBuffer";
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
        inputSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes = 1;
        inputSramPlan.m_InputMappings  = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartInputSlot0 } };
        inputSramPlan.m_OutputMappings = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartOutputSlot0 } };

        Buffer* ptrInputBuffer = inputSramPlan.m_OpGraph.GetBuffers().back();
        inputStripeSize        = utils::TotalSizeBytesNHWCB(ptrInputBuffer->m_StripeShape);
        inputZeroPoint         = ptrInputBuffer->m_QuantizationInfo.GetZeroPoint();

        // Plan weightDramPlan
        weightDramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT, TensorShape{ 3, 3, 2, 1 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        weightDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        weightDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag       = "WeightDramBuffer";
        encodedWeights                                                 = std::make_shared<EncodedWeights>();
        encodedWeights->m_Data                                         = { 1, 2, 3, 4 };
        encodedWeights->m_MaxSize                                      = 10;
        encodedWeights->m_Metadata                                     = { { 0, 2 }, { 2, 2 } };
        weightDramPlan.m_OpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights;
        weightDramPlan.m_OutputMappings = { { weightDramPlan.m_OpGraph.GetBuffers()[0], weightDramPartOutputSlot0 } };

        // Glue glueWeightDram_WeightSram
        glueWeightDram_WeightSram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueWeightDram_WeightSram.m_Graph.GetOps()[0]->m_DebugTag = "WeightDmaOp";
        glueWeightDram_WeightSram.m_InputSlot = { glueWeightDram_WeightSram.m_Graph.GetOps()[0], 0 };
        glueWeightDram_WeightSram.m_Output.push_back(glueWeightDram_WeightSram.m_Graph.GetOps()[0]);

        // Plan weightSramPlan
        weightSramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT, TensorShape{ 3, 3, 1, 1 },
                                     TensorShape{ 3, 3, 1, 1 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag    = "WeightSramBuffer";
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes  = 1;
        weightSramPlan.m_OpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights->m_MaxSize;
        weightSramPlan.m_InputMappings  = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartInputSlot0 } };
        weightSramPlan.m_OutputMappings = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartOutputSlot0 } };

        Buffer* ptrWeightBuffer = weightSramPlan.m_OpGraph.GetBuffers().back();
        weightSize              = ptrWeightBuffer->m_SizeInBytes / ptrWeightBuffer->m_NumStripes;
        kernelHeight            = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[1]);
        kernelWidth             = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[2]);

        // Plan mcePlePlan
        mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 5, 5, 1 }, TensorShape{ 1, 5, 5, 1 },
                                                                TraversalOrder::Xyz, 1, QuantizationInfo()));
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateInputSramBuffer";
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000FFF;
        mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 3, 3, 1 }, TensorShape{ 1, 3, 3, 1 },
                                                                TraversalOrder::Xyz, 1, QuantizationInfo()));
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateWeightSramBuffer";
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F000;
        mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 5, 5, 1 }, TensorShape{ 1, 5, 5, 1 },
                                                                TraversalOrder::Xyz, 0, QuantizationInfo()));
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputPleInputSramBuffer";
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x0000F00F;
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_NumStripes = 1;

        mcePlePlan.m_OpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 5, 5, 1 }, TensorShape{ 1, 5, 5, 1 }, outputTensorShape,
            TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
        mcePlePlan.m_OpGraph.GetOps()[0]->m_DebugTag = "MceOp Stride 1x1";

        mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePlan.m_OpGraph.GetOps()[0], 0);
        mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePlan.m_OpGraph.GetOps()[0], 1);
        mcePlePlan.m_OpGraph.SetProducer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[0]);

        ifmDeltaHeight = static_cast<int8_t>(inputSramPlan.m_OpGraph.GetBuffers()[0]->m_TensorShape[1] -
                                             mcePlePlan.m_OpGraph.GetBuffers()[2]->m_TensorShape[1]);
        ifmDeltaWidth  = static_cast<int8_t>(inputSramPlan.m_OpGraph.GetBuffers()[0]->m_TensorShape[2] -
                                            mcePlePlan.m_OpGraph.GetBuffers()[2]->m_TensorShape[2]);

        // Adding an Interleave PLE kernel to the plan
        pleOp =
            std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::INTERLEAVE_2X2_2_2,
                                    BlockConfig{ 16u, 16u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 5, 5, 1 } },
                                    TensorShape{ 1, 5, 5, 1 }, ethosn::command_stream::DataType::U8, true);
        pleOp.get()->m_Offset     = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp    = AddPleToOpGraph(mcePlePlan.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                                 TensorShape{ 1, 5, 5, 1 }, numMemoryStripes, std::move(pleOp),
                                                 TensorShape{ 1, 5, 5, 1 }, QuantizationInfo(), operationIds);
        mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[1], 0);

        mcePlePlan.m_InputMappings  = { { mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePartInputSlot0 },
                                       { mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePartInputSlot1 } };
        mcePlePlan.m_OutputMappings = { { mcePlePlan.m_OpGraph.GetBuffers()[3], mcePlePartOutputSlot0 } };

        // Plan weight2DramPlan
        weight2DramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::WEIGHT, TensorShape{ 3, 3, 1, 1 },
                                     TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
        weight2DramPlan.m_OpGraph.GetBuffers().back()->m_BufferType     = BufferType::ConstantDma;
        weight2DramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag       = "Weight2DramBuffer";
        encodedWeights2                                                 = std::make_shared<EncodedWeights>();
        encodedWeights2->m_Data                                         = { 1, 2, 3, 4 };
        encodedWeights2->m_MaxSize                                      = 10;
        encodedWeights2->m_Metadata                                     = { { 0, 2 }, { 2, 2 } };
        weight2DramPlan.m_OpGraph.GetBuffers().back()->m_EncodedWeights = encodedWeights2;
        weight2DramPlan.m_OutputMappings                                = { { weight2DramPlan.m_OpGraph.GetBuffers()[0],
                                               weight2DramPartOutputSlot0 } };

        // Glue glueWeightDram_WeightSram
        glueWeight2Dram_Weight2Sram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueWeight2Dram_Weight2Sram.m_Graph.GetOps()[0]->m_DebugTag = "Weight2DmaOp";
        glueWeight2Dram_Weight2Sram.m_InputSlot = { glueWeight2Dram_Weight2Sram.m_Graph.GetOps()[0], 0 };
        glueWeight2Dram_Weight2Sram.m_Output.push_back(glueWeight2Dram_Weight2Sram.m_Graph.GetOps()[0]);

        // Plan weightSramPlan
        weight2SramPlan.m_OpGraph.AddBuffer(
            std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT, TensorShape{ 3, 3, 1, 1 },
                                     TensorShape{ 3, 3, 1, 1 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
        weight2SramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag    = "Weight2SramBuffer";
        weight2SramPlan.m_OpGraph.GetBuffers().back()->m_Offset      = 0x00000FF0;
        weight2SramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes  = 1;
        weight2SramPlan.m_OpGraph.GetBuffers().back()->m_SizeInBytes = encodedWeights2->m_MaxSize;
        weight2SramPlan.m_InputMappings  = { { weight2SramPlan.m_OpGraph.GetBuffers()[0], weight2SramPartInputSlot0 } };
        weight2SramPlan.m_OutputMappings = { { weight2SramPlan.m_OpGraph.GetBuffers()[0],
                                               weight2SramPartOutputSlot0 } };

        Buffer* ptrWeightBuffer2 = weight2SramPlan.m_OpGraph.GetBuffers().back();
        weightSize2              = ptrWeightBuffer2->m_SizeInBytes / ptrWeightBuffer2->m_NumStripes;
        kernelHeight2            = static_cast<uint8_t>(ptrWeightBuffer2->m_TensorShape[1]);
        kernelWidth2             = static_cast<uint8_t>(ptrWeightBuffer2->m_TensorShape[2]);

        // Plan mcePlePlan
        mcePle2Plan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                 TensorShape{ 1, 5, 5, 1 }, TensorShape{ 1, 5, 5, 1 },
                                                                 TraversalOrder::Xyz, 1, QuantizationInfo()));
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateInput2SramBuffer";
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x00000FFF;
        mcePle2Plan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                 TensorShape{ 1, 3, 3, 1 }, TensorShape{ 1, 3, 3, 1 },
                                                                 TraversalOrder::Xyz, 1, QuantizationInfo()));
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateWeight2SramBuffer";
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_Offset   = 0x0000F000;
        mcePle2Plan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                                 outputTensorShape, outputTensorShape,
                                                                 TraversalOrder::Xyz, 4, QuantizationInfo()));
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "outputPleInputSramBuffer";
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x0000F00F;
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_NumStripes = 1;

        mcePle2Plan.m_OpGraph.AddOp(std::make_unique<MceOp>(
            Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
            BlockConfig{ 16u, 16u }, TensorShape{ 1, 5, 5, 1 }, outputTensorShape, outputTensorShape,
            TraversalOrder::Xyz, Stride(2, 2), padLeft, padTop, 0, 255));
        (static_cast<MceOp*>(mcePle2Plan.m_OpGraph.GetOps()[0]))->m_uninterleavedInputShape = TensorShape{ 1, 5, 5, 1 };

        mcePle2Plan.m_OpGraph.GetOps()[0]->m_DebugTag = "MceOp Stride 2x2";

        mcePle2Plan.m_OpGraph.AddConsumer(mcePle2Plan.m_OpGraph.GetBuffers()[0], mcePle2Plan.m_OpGraph.GetOps()[0], 0);
        mcePle2Plan.m_OpGraph.AddConsumer(mcePle2Plan.m_OpGraph.GetBuffers()[1], mcePle2Plan.m_OpGraph.GetOps()[0], 1);
        mcePle2Plan.m_OpGraph.SetProducer(mcePle2Plan.m_OpGraph.GetBuffers()[2], mcePle2Plan.m_OpGraph.GetOps()[0]);

        // Adding a passthrough PLE kernel to the plan
        // The PleKernelId is expected to be PASSTHROUGH_8x8_1
        pleOp2 = std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                         BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ outputTensorShape },
                                         outputTensorShape, ethosn::command_stream::DataType::U8, true);
        pleOp2.get()->m_Offset    = 0x0000F0F0;
        numMemoryStripes.m_Output = 1;
        auto outBufferAndPleOp2 =
            AddPleToOpGraph(mcePle2Plan.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz, outputTensorShape,
                            numMemoryStripes, std::move(pleOp2), outputTensorShape, QuantizationInfo(), operationIds);
        mcePle2Plan.m_OpGraph.GetBuffers().back()->m_Offset = 0X0000F0FF;
        mcePle2Plan.m_OpGraph.AddConsumer(mcePle2Plan.m_OpGraph.GetBuffers()[2], mcePle2Plan.m_OpGraph.GetOps()[1], 0);

        mcePle2Plan.m_InputMappings  = { { mcePle2Plan.m_OpGraph.GetBuffers()[0], mcePle2PartInputSlot0 },
                                        { mcePle2Plan.m_OpGraph.GetBuffers()[1], mcePle2PartInputSlot1 } };
        mcePle2Plan.m_OutputMappings = { { mcePle2Plan.m_OpGraph.GetBuffers()[3], mcePle2PartOutputSlot0 } };

        // Glue glueOutputSram_OutputDram
        glueOutputSram_OutputDram.m_Graph.AddOp(std::make_unique<DmaOp>());
        glueOutputSram_OutputDram.m_Graph.GetOps()[0]->m_DebugTag = "outputDmaOp";
        glueOutputSram_OutputDram.m_InputSlot = { glueOutputSram_OutputDram.m_Graph.GetOps()[0], 0 };
        glueOutputSram_OutputDram.m_Output.push_back(glueOutputSram_OutputDram.m_Graph.GetOps()[0]);

        // Plan outputDramPlan
        outputDramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                                    outputTensorShape, TensorShape{ 0, 0, 0, 0 },
                                                                    TraversalOrder::Xyz, 0, QuantizationInfo()));
        outputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
        outputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "outputDramBuffer";
        outputDramPlan.m_InputMappings = { { outputDramPlan.m_OpGraph.GetBuffers()[0], outputDramPartInputSlot0 } };

        Elem elemInputDram  = { std::make_shared<Plan>(std::move(inputDramPlan)),
                               { { inputSramPartInputSlot0, { &glueInputDram_InputSram, true } } } };
        Elem elemInputSram  = { std::make_shared<Plan>(std::move(inputSramPlan)), {} };
        Elem elemWeightDram = { std::make_shared<Plan>(std::move(weightDramPlan)),
                                { { weightSramPartInputSlot0, { &glueWeightDram_WeightSram, true } } } };
        Elem elemWeightSram = { std::make_shared<Plan>(std::move(weightSramPlan)), {} };
        Elem elemMcePle     = { std::make_shared<Plan>(std::move(mcePlePlan)), {} };

        Elem elemWeight2Dram = { std::make_shared<Plan>(std::move(weight2DramPlan)),
                                 { { weight2SramPartInputSlot0, { &glueWeight2Dram_Weight2Sram, true } } } };
        Elem elemWeight2Sram = { std::make_shared<Plan>(std::move(weight2SramPlan)), {} };
        Elem elemMcePle2     = { std::make_shared<Plan>(std::move(mcePle2Plan)),
                             { { outputDramPartInputSlot0, { &glueOutputSram_OutputDram, true } } } };
        Elem elemoutputDram  = { std::make_shared<Plan>(std::move(outputDramPlan)), {} };

        comb.m_Elems.insert(std::make_pair(0, elemInputDram));
        comb.m_PartIdsInOrder.push_back(0);
        comb.m_Elems.insert(std::make_pair(1, elemInputSram));
        comb.m_PartIdsInOrder.push_back(1);
        comb.m_Elems.insert(std::make_pair(2, elemWeightDram));
        comb.m_PartIdsInOrder.push_back(2);
        comb.m_Elems.insert(std::make_pair(3, elemWeightSram));
        comb.m_PartIdsInOrder.push_back(3);
        comb.m_Elems.insert(std::make_pair(4, elemMcePle));
        comb.m_PartIdsInOrder.push_back(4);

        comb.m_Elems.insert(std::make_pair(5, elemWeight2Dram));
        comb.m_PartIdsInOrder.push_back(5);
        comb.m_Elems.insert(std::make_pair(6, elemWeight2Sram));
        comb.m_PartIdsInOrder.push_back(6);
        comb.m_Elems.insert(std::make_pair(7, elemMcePle2));
        comb.m_PartIdsInOrder.push_back(7);
        comb.m_Elems.insert(std::make_pair(8, elemoutputDram));
        comb.m_PartIdsInOrder.push_back(8);

        bool dumpInputGraphToFile = false;
        if (dumpInputGraphToFile)
        {
            std::ofstream stream("CommandStreamGenerator_StridedConvOpGraph_Input.dot");
            SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
        }

        mergedOpGraph = GetOpGraphForCombination(comb, graph);

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
    Glue glueInputDram_InputSram;
    Plan inputSramPlan;
    Plan weightDramPlan;
    Glue glueWeightDram_WeightSram;
    Plan weightSramPlan;
    Plan mcePlePlan;

    Plan weight2DramPlan;
    Glue glueWeight2Dram_Weight2Sram;
    Plan weight2SramPlan;
    Plan mcePle2Plan;
    Glue glueOutputSram_OutputDram;
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
    OpGraph mergedOpGraph;
};

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

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent = commandStream[0];
    const IfmS& ifmSData   = ifmSAgent.data.ifm;

    REQUIRE(ifmSData.fmData.dramOffset == 0);
    REQUIRE(ifmSData.fmData.bufferId == 1);
    REQUIRE(ifmSData.fmData.dataType == FmsDataType::NHWCB);

    REQUIRE(ifmSData.fmData.fcafInfo.signedActivation == 0);
    REQUIRE(ifmSData.fmData.fcafInfo.zeroPoint == false);

    REQUIRE(ifmSData.fmData.tile.baseAddr == 3855);
    REQUIRE(ifmSData.fmData.tile.numSlots == 4);
    REQUIRE(ifmSData.fmData.tile.slotSize == 128);

    REQUIRE(ifmSData.fmData.dfltStripeSize.height == 8);
    REQUIRE(ifmSData.fmData.dfltStripeSize.width == 8);
    REQUIRE(ifmSData.fmData.dfltStripeSize.channels == 16);

    REQUIRE(ifmSData.fmData.edgeStripeSize.height == 8);
    REQUIRE(ifmSData.fmData.edgeStripeSize.width == 8);
    REQUIRE(ifmSData.fmData.edgeStripeSize.channels == 3);

    REQUIRE(ifmSData.fmData.supertensorSizeInCells.width == 20);
    REQUIRE(ifmSData.fmData.supertensorSizeInCells.channels == 1);

    REQUIRE(ifmSData.fmData.numStripes.height == 20);
    REQUIRE(ifmSData.fmData.numStripes.width == 20);
    REQUIRE(ifmSData.fmData.numStripes.channels == 1);

    REQUIRE(ifmSData.fmData.stripeIdStrides.height == 20);
    REQUIRE(ifmSData.fmData.stripeIdStrides.width == 1);
    REQUIRE(ifmSData.fmData.stripeIdStrides.channels == 1);
}

// WeightStreamer Agent Data Test
TEST_CASE("WeightStreamer Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& wgtSAgent = commandStream[1];
    const WgtS& wgtSData   = wgtSAgent.data.wgt;

    REQUIRE(wgtSData.bufferId == 2);
    REQUIRE(wgtSData.metadataBufferId == 3);

    REQUIRE(wgtSData.tile.baseAddr == 0x00000FF0);
    REQUIRE(wgtSData.tile.numSlots == 3);
    REQUIRE(wgtSData.tile.slotSize == 1);

    REQUIRE(wgtSData.numStripes.ifmChannels == 1);
    REQUIRE(wgtSData.numStripes.ofmChannels == 1);

    REQUIRE(wgtSData.stripeIdStrides.ifmChannels == 1);
    REQUIRE(wgtSData.stripeIdStrides.ofmChannels == 1);
}

// MceScheduler Agent Data Test
TEST_CASE("MceScheduler Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent = commandStream[3];
    const MceS& mceSData   = mceSAgent.data.mce;

    REQUIRE(mceSData.ifmTile.baseAddr == 0x00000F0F);
    REQUIRE(mceSData.ifmTile.numSlots == 4);
    REQUIRE(mceSData.ifmTile.slotSize == mceOpGraph.getInputStripeSize() / hwCaps.GetNumberOfSrams());

    REQUIRE(mceSData.wgtTile.baseAddr == 0x00000FF0);
    REQUIRE(mceSData.wgtTile.numSlots == 3);
    REQUIRE(mceSData.wgtTile.slotSize == 1);

    REQUIRE(mceSData.blockSize.width == 16);
    REQUIRE(mceSData.blockSize.height == 16);

    REQUIRE(mceSData.dfltStripeSize.ofmHeight == 8);
    REQUIRE(mceSData.dfltStripeSize.ofmWidth == 8);
    REQUIRE(mceSData.dfltStripeSize.ofmChannels == 8);
    REQUIRE(mceSData.dfltStripeSize.ifmChannels == 16);

    REQUIRE(mceSData.edgeStripeSize.ofmHeight == 1);
    REQUIRE(mceSData.edgeStripeSize.ofmWidth == 8);
    REQUIRE(mceSData.edgeStripeSize.ofmChannels == 8);
    REQUIRE(mceSData.edgeStripeSize.ifmChannels == 3);

    REQUIRE(mceSData.numStripes.ofmHeight == 3);
    REQUIRE(mceSData.numStripes.ofmWidth == 2);
    REQUIRE(mceSData.numStripes.ofmChannels == 2);
    REQUIRE(mceSData.numStripes.ifmChannels == 1);

    REQUIRE(mceSData.stripeIdStrides.ofmHeight == 2);
    REQUIRE(mceSData.stripeIdStrides.ofmWidth == 1);
    REQUIRE(mceSData.stripeIdStrides.ofmChannels == 6);
    REQUIRE(mceSData.stripeIdStrides.ifmChannels == 1);

    REQUIRE(mceSData.convStrideXy.x == 1);
    REQUIRE(mceSData.convStrideXy.y == 1);

    REQUIRE(mceSData.ifmZeroPoint == mceOpGraph.getInputZeroPoint());
    REQUIRE(mceSData.mceOpMode == cascading::MceOperation::CONVOLUTION);
    REQUIRE(mceSData.algorithm == cascading::MceAlgorithm::DIRECT);

    REQUIRE(mceSData.filterShape[0].height == mceOpGraph.getKernelHeight());
    REQUIRE(mceSData.filterShape[0].width == mceOpGraph.getKernelWidth());

    REQUIRE(mceSData.padding[0].left == 0);
    REQUIRE(mceSData.padding[0].top == 0);

    REQUIRE(mceSData.ifmDeltaDefault[0].height == mceOpGraph.getIfmDeltaHeight());
    REQUIRE(mceSData.ifmDeltaDefault[0].width == mceOpGraph.getIfmDeltaWidth());
    REQUIRE(mceSData.ifmDeltaEdge[0].height == mceOpGraph.getIfmDeltaHeight());
    REQUIRE(mceSData.ifmDeltaEdge[0].width == mceOpGraph.getIfmDeltaWidth());

    REQUIRE(mceSData.reluActiv.max == 255);
    REQUIRE(mceSData.reluActiv.min == 0);

    REQUIRE(mceSData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);
}

TEST_CASE("MceScheduler Agent Data Test - 3x3 Convolution - 2x2 Stride - Valid Padding",
          "[CascadingCommandStreamGenerator]")
{
    StridedConvOpGraph stridedConvGraph = StridedConvOpGraph(0, 0, { 1, 2, 2, 1 });
    OpGraph mergedOpGraph               = stridedConvGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent = commandStream[7];
    const MceS& mceSData   = mceSAgent.data.mce;

    // Submap 0
    REQUIRE(mceSData.filterShape[0].height == 2);
    REQUIRE(mceSData.filterShape[0].width == 2);
    REQUIRE(mceSData.padding[0].left == 0);
    REQUIRE(mceSData.padding[0].top == 0);
    REQUIRE(mceSData.ifmDeltaDefault[0].height == 1);
    REQUIRE(mceSData.ifmDeltaDefault[0].width == 1);
    REQUIRE(mceSData.ifmDeltaEdge[0].height == 1);
    REQUIRE(mceSData.ifmDeltaEdge[0].width == 1);

    // Submap 1
    REQUIRE(mceSData.filterShape[1].height == 2);
    REQUIRE(mceSData.filterShape[1].width == 1);
    REQUIRE(mceSData.padding[1].left == 0);
    REQUIRE(mceSData.padding[1].top == 0);
    REQUIRE(mceSData.ifmDeltaDefault[1].height == 1);
    REQUIRE(mceSData.ifmDeltaDefault[1].width == 0);
    REQUIRE(mceSData.ifmDeltaEdge[1].height == 1);
    REQUIRE(mceSData.ifmDeltaEdge[1].width == 0);

    // Submap 2
    REQUIRE(mceSData.filterShape[2].height == 1);
    REQUIRE(mceSData.filterShape[2].width == 2);
    REQUIRE(mceSData.padding[2].left == 0);
    REQUIRE(mceSData.padding[2].top == 0);
    REQUIRE(mceSData.ifmDeltaDefault[2].height == 0);
    REQUIRE(mceSData.ifmDeltaDefault[2].width == 1);
    REQUIRE(mceSData.ifmDeltaEdge[2].height == 0);
    REQUIRE(mceSData.ifmDeltaEdge[2].width == 1);

    // Submap 3
    REQUIRE(mceSData.filterShape[3].height == 1);
    REQUIRE(mceSData.filterShape[3].width == 1);
    REQUIRE(mceSData.padding[3].left == 0);
    REQUIRE(mceSData.padding[3].top == 0);
    REQUIRE(mceSData.ifmDeltaDefault[3].height == 0);
    REQUIRE(mceSData.ifmDeltaDefault[3].width == 0);
    REQUIRE(mceSData.ifmDeltaEdge[3].height == 0);
    REQUIRE(mceSData.ifmDeltaEdge[3].width == 0);

    REQUIRE(mceSData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);
}

TEST_CASE("MceScheduler Agent Data Test - 3x3 Convolution - 2x2 Stride - Same Padding",
          "[CascadingCommandStreamGenerator]")
{
    StridedConvOpGraph stridedConvGraph = StridedConvOpGraph(1, 1, { 1, 3, 3, 1 });
    OpGraph mergedOpGraph               = stridedConvGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent = commandStream[7];
    const MceS& mceSData   = mceSAgent.data.mce;

    // Submap 0
    REQUIRE(mceSData.filterShape[0].height == 1);
    REQUIRE(mceSData.filterShape[0].width == 1);
    REQUIRE(mceSData.padding[0].left == 0);
    REQUIRE(mceSData.padding[0].top == 0);
    REQUIRE(mceSData.ifmDeltaDefault[0].height == 0);
    REQUIRE(mceSData.ifmDeltaDefault[0].width == 0);
    REQUIRE(mceSData.ifmDeltaEdge[0].height == 0);
    REQUIRE(mceSData.ifmDeltaEdge[0].width == 0);

    // Submap 1
    REQUIRE(mceSData.filterShape[1].height == 1);
    REQUIRE(mceSData.filterShape[1].width == 2);
    REQUIRE(mceSData.padding[1].left == 1);
    REQUIRE(mceSData.padding[1].top == 0);
    REQUIRE(mceSData.ifmDeltaDefault[1].height == 0);
    REQUIRE(mceSData.ifmDeltaDefault[1].width == -1);
    REQUIRE(mceSData.ifmDeltaEdge[1].height == 0);
    REQUIRE(mceSData.ifmDeltaEdge[1].width == -1);

    // Submap 2
    REQUIRE(mceSData.filterShape[2].height == 2);
    REQUIRE(mceSData.filterShape[2].width == 1);
    REQUIRE(mceSData.padding[2].left == 0);
    REQUIRE(mceSData.padding[2].top == 1);
    REQUIRE(mceSData.ifmDeltaDefault[2].height == -1);
    REQUIRE(mceSData.ifmDeltaDefault[2].width == 0);
    REQUIRE(mceSData.ifmDeltaEdge[2].height == -1);
    REQUIRE(mceSData.ifmDeltaEdge[2].width == 0);

    // Submap 3
    REQUIRE(mceSData.filterShape[3].height == 2);
    REQUIRE(mceSData.filterShape[3].width == 2);
    REQUIRE(mceSData.padding[3].left == 1);
    REQUIRE(mceSData.padding[3].top == 1);
    REQUIRE(mceSData.ifmDeltaDefault[3].height == -1);
    REQUIRE(mceSData.ifmDeltaDefault[3].width == -1);
    REQUIRE(mceSData.ifmDeltaEdge[3].height == -1);
    REQUIRE(mceSData.ifmDeltaEdge[3].width == -1);

    REQUIRE(mceSData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);
}

// PleLoader Agent Data Test
TEST_CASE("PleLoader Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleLAgent = commandStream[2];
    const PleL& pleLData   = pleLAgent.data.pleL;

    REQUIRE(pleLData.sramAddr == 0x0000F0F0);
    REQUIRE(pleLData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);
}

// PleScheduler Agent Data Test
TEST_CASE("PleScheduler Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleSchedulerAgent = commandStream[4];

    // The network consists of all agent types. Here we test that the PleScheduler
    // agent is set correctly.
    REQUIRE(pleSchedulerAgent.data.pleS.ofmTile.baseAddr == 0x000F0FF);
    REQUIRE(pleSchedulerAgent.data.pleS.ofmTile.numSlots == 1);
    REQUIRE(pleSchedulerAgent.data.pleS.ofmTile.slotSize == 256);
    REQUIRE(pleSchedulerAgent.data.pleS.ofmZeroPoint == 0);

    REQUIRE(pleSchedulerAgent.data.pleS.dfltStripeSize.height == 4);
    REQUIRE(pleSchedulerAgent.data.pleS.dfltStripeSize.width == 4);
    REQUIRE(pleSchedulerAgent.data.pleS.dfltStripeSize.channels == 32);

    REQUIRE(pleSchedulerAgent.data.pleS.numStripes.height == 20);
    REQUIRE(pleSchedulerAgent.data.pleS.numStripes.width == 20);
    REQUIRE(pleSchedulerAgent.data.pleS.numStripes.channels == 1);

    REQUIRE(pleSchedulerAgent.data.pleS.edgeStripeSize.height == 4);
    REQUIRE(pleSchedulerAgent.data.pleS.edgeStripeSize.width == 4);
    REQUIRE(pleSchedulerAgent.data.pleS.edgeStripeSize.channels == 24);

    REQUIRE(pleSchedulerAgent.data.pleS.stripeIdStrides.height == 20);
    REQUIRE(pleSchedulerAgent.data.pleS.stripeIdStrides.width == 1);
    REQUIRE(pleSchedulerAgent.data.pleS.stripeIdStrides.channels == 400);

    REQUIRE(pleSchedulerAgent.data.pleS.inputMode == PleInputMode::MCE_ALL_OGS);

    REQUIRE(pleSchedulerAgent.data.pleS.pleKernelSramAddr == 0x0000F0F0);
    REQUIRE(pleSchedulerAgent.data.pleS.pleKernelId == PleKernelId::PASSTHROUGH_8X8_1);
}

// PleScheduler Standalone Agent Data Test
TEST_CASE("PleScheduler Standalone Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const std::set<uint32_t> operationIds = { 0 };
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleSAgent = commandStream[2];

    // The network consists of a standalone ple op and DMA ops. Here we test that
    // the PleScheduler agent is set correctly.
    REQUIRE(pleSAgent.data.pleS.ofmTile.baseAddr == 0x0000F00);
    REQUIRE(pleSAgent.data.pleS.ofmTile.numSlots == 1);
    REQUIRE(pleSAgent.data.pleS.ofmTile.slotSize == 256);
    REQUIRE(pleSAgent.data.pleS.ofmZeroPoint == 0);

    REQUIRE(pleSAgent.data.pleS.dfltStripeSize.height == 8);
    REQUIRE(pleSAgent.data.pleS.dfltStripeSize.width == 8);
    REQUIRE(pleSAgent.data.pleS.dfltStripeSize.channels == 32);

    REQUIRE(pleSAgent.data.pleS.numStripes.height == 10);
    REQUIRE(pleSAgent.data.pleS.numStripes.width == 10);
    REQUIRE(pleSAgent.data.pleS.numStripes.channels == 1);

    REQUIRE(pleSAgent.data.pleS.edgeStripeSize.height == 8);
    REQUIRE(pleSAgent.data.pleS.edgeStripeSize.width == 8);
    REQUIRE(pleSAgent.data.pleS.edgeStripeSize.channels == 24);

    REQUIRE(pleSAgent.data.pleS.stripeIdStrides.height == 10);
    REQUIRE(pleSAgent.data.pleS.stripeIdStrides.width == 1);
    REQUIRE(pleSAgent.data.pleS.stripeIdStrides.channels == 100);

    REQUIRE(pleSAgent.data.pleS.inputMode == PleInputMode::SRAM);

    REQUIRE(pleSAgent.data.pleS.pleKernelSramAddr == 0x000000FF);
    REQUIRE(pleSAgent.data.pleS.pleKernelId == PleKernelId::LEAKY_RELU_8X8_1);

    REQUIRE(pleSAgent.data.pleS.ifmTile0.baseAddr == 0x0000000F);
    REQUIRE(pleSAgent.data.pleS.ifmTile0.numSlots == 0);
    REQUIRE(pleSAgent.data.pleS.ifmTile0.slotSize == 128);

    REQUIRE(pleSAgent.data.pleS.ifmInfo0.zeroPoint == 0);
    REQUIRE(pleSAgent.data.pleS.ifmInfo0.multiplier == 32768);
    REQUIRE(pleSAgent.data.pleS.ifmInfo0.shift == 15);
}

// OfmStreamer Agent Data Test
TEST_CASE("OfmStreamer Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ofmSAgent = commandStream[5];
    const OfmS& ofmSData   = ofmSAgent.data.ofm;

    REQUIRE(ofmSData.fmData.dramOffset == 0);
    REQUIRE(ofmSData.fmData.bufferId == 4);
    REQUIRE(ofmSData.fmData.dataType == FmsDataType::NHWCB);

    REQUIRE(ofmSData.fmData.fcafInfo.signedActivation == 0);
    REQUIRE(ofmSData.fmData.fcafInfo.zeroPoint == false);

    REQUIRE(ofmSData.fmData.tile.baseAddr == 61695);
    REQUIRE(ofmSData.fmData.tile.numSlots == 1);
    REQUIRE(ofmSData.fmData.tile.slotSize == 256);

    REQUIRE(ofmSData.fmData.dfltStripeSize.height == 4);
    REQUIRE(ofmSData.fmData.dfltStripeSize.width == 4);
    REQUIRE(ofmSData.fmData.dfltStripeSize.channels == 32);

    REQUIRE(ofmSData.fmData.edgeStripeSize.height == 4);
    REQUIRE(ofmSData.fmData.edgeStripeSize.width == 4);
    REQUIRE(ofmSData.fmData.edgeStripeSize.channels == 24);

    REQUIRE(ofmSData.fmData.supertensorSizeInCells.width == 10);
    REQUIRE(ofmSData.fmData.supertensorSizeInCells.channels == 2);

    REQUIRE(ofmSData.fmData.numStripes.height == 20);
    REQUIRE(ofmSData.fmData.numStripes.width == 20);
    REQUIRE(ofmSData.fmData.numStripes.channels == 1);

    REQUIRE(ofmSData.fmData.stripeIdStrides.height == 20);
    REQUIRE(ofmSData.fmData.stripeIdStrides.width == 1);
    REQUIRE(ofmSData.fmData.stripeIdStrides.channels == 1);
}

// Concat Op Agent Data Test
TEST_CASE("Concat Op Agent Data Test", "[CascadingCommandStreamGenerator]")
{
    ConcatOpGraph inputOutputMergeGraph = ConcatOpGraph();
    OpGraph mergedOpGraph               = inputOutputMergeGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
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
    REQUIRE(ifmSData1.fmData.bufferId == 2);
    REQUIRE(ifmSData1.fmData.dramOffset == 0);
    REQUIRE(ifmSData1.fmData.dataType == FmsDataType::NHWCB);

    REQUIRE(ifmSData1.fmData.fcafInfo.zeroPoint == 0);
    REQUIRE(ifmSData1.fmData.fcafInfo.signedActivation == false);

    REQUIRE(ifmSData1.fmData.tile.baseAddr == 0);
    REQUIRE(ifmSData1.fmData.tile.numSlots == 2);
    REQUIRE(ifmSData1.fmData.tile.slotSize == 128);

    REQUIRE(ifmSData1.fmData.dfltStripeSize.height == 8);
    REQUIRE(ifmSData1.fmData.dfltStripeSize.width == 8);
    REQUIRE(ifmSData1.fmData.dfltStripeSize.channels == 3);

    REQUIRE(ifmSData1.fmData.edgeStripeSize.height == 8);
    REQUIRE(ifmSData1.fmData.edgeStripeSize.width == 8);
    REQUIRE(ifmSData1.fmData.edgeStripeSize.channels == 3);

    REQUIRE(ifmSData1.fmData.supertensorSizeInCells.width == 2);
    REQUIRE(ifmSData1.fmData.supertensorSizeInCells.channels == 1);

    REQUIRE(ifmSData1.fmData.numStripes.height == 1);
    REQUIRE(ifmSData1.fmData.numStripes.width == 1);
    REQUIRE(ifmSData1.fmData.numStripes.channels == 1);

    REQUIRE(ifmSData1.fmData.stripeIdStrides.height == 1);
    REQUIRE(ifmSData1.fmData.stripeIdStrides.width == 1);
    REQUIRE(ifmSData1.fmData.stripeIdStrides.channels == 1);

    // ofmSData1
    REQUIRE(ofmSData1.fmData.bufferId == 1);
    REQUIRE(ofmSData1.fmData.dramOffset == 0);
    REQUIRE(ofmSData1.fmData.dataType == FmsDataType::NHWCB);

    REQUIRE(ofmSData1.fmData.fcafInfo.zeroPoint == 0);
    REQUIRE(ofmSData1.fmData.fcafInfo.signedActivation == false);

    REQUIRE(ofmSData1.fmData.tile.baseAddr == 0);
    REQUIRE(ofmSData1.fmData.tile.numSlots == 2);
    REQUIRE(ofmSData1.fmData.tile.slotSize == 128);

    REQUIRE(ofmSData1.fmData.dfltStripeSize.height == 8);
    REQUIRE(ofmSData1.fmData.dfltStripeSize.width == 8);
    REQUIRE(ofmSData1.fmData.dfltStripeSize.channels == 3);

    REQUIRE(ofmSData1.fmData.edgeStripeSize.height == 8);
    REQUIRE(ofmSData1.fmData.edgeStripeSize.width == 8);
    REQUIRE(ofmSData1.fmData.edgeStripeSize.channels == 3);

    REQUIRE(ofmSData1.fmData.supertensorSizeInCells.width == 3);
    REQUIRE(ofmSData1.fmData.supertensorSizeInCells.channels == 1);

    REQUIRE(ofmSData1.fmData.numStripes.height == 1);
    REQUIRE(ofmSData1.fmData.numStripes.width == 1);
    REQUIRE(ofmSData1.fmData.numStripes.channels == 1);

    REQUIRE(ofmSData1.fmData.stripeIdStrides.height == 1);
    REQUIRE(ofmSData1.fmData.stripeIdStrides.width == 1);
    REQUIRE(ofmSData1.fmData.stripeIdStrides.channels == 1);

    // ifmsData2
    REQUIRE(ifmSData2.fmData.bufferId == 3);
    REQUIRE(ifmSData2.fmData.dramOffset == 0);
    REQUIRE(ifmSData2.fmData.dataType == FmsDataType::NHWCB);

    REQUIRE(ifmSData2.fmData.fcafInfo.zeroPoint == 0);
    REQUIRE(ifmSData2.fmData.fcafInfo.signedActivation == false);

    REQUIRE(ifmSData2.fmData.tile.baseAddr == 256);
    REQUIRE(ifmSData2.fmData.tile.numSlots == 2);
    REQUIRE(ifmSData2.fmData.tile.slotSize == 128);

    REQUIRE(ifmSData2.fmData.dfltStripeSize.height == 8);
    REQUIRE(ifmSData2.fmData.dfltStripeSize.width == 8);
    REQUIRE(ifmSData2.fmData.dfltStripeSize.channels == 3);

    REQUIRE(ifmSData2.fmData.edgeStripeSize.height == 8);
    REQUIRE(ifmSData2.fmData.edgeStripeSize.width == 8);
    REQUIRE(ifmSData2.fmData.edgeStripeSize.channels == 3);

    REQUIRE(ifmSData2.fmData.supertensorSizeInCells.width == 1);
    REQUIRE(ifmSData2.fmData.supertensorSizeInCells.channels == 1);

    REQUIRE(ifmSData2.fmData.numStripes.height == 1);
    REQUIRE(ifmSData2.fmData.numStripes.width == 1);
    REQUIRE(ifmSData2.fmData.numStripes.channels == 1);

    REQUIRE(ifmSData2.fmData.stripeIdStrides.height == 1);
    REQUIRE(ifmSData2.fmData.stripeIdStrides.width == 1);
    REQUIRE(ifmSData2.fmData.stripeIdStrides.channels == 1);

    // ofmsData2
    REQUIRE(ofmSData2.fmData.bufferId == 1);
    REQUIRE(ofmSData2.fmData.dramOffset == 0x00000800);
    REQUIRE(ofmSData2.fmData.dataType == FmsDataType::NHWCB);

    REQUIRE(ofmSData2.fmData.fcafInfo.zeroPoint == 0);
    REQUIRE(ofmSData2.fmData.fcafInfo.signedActivation == false);

    REQUIRE(ofmSData2.fmData.tile.baseAddr == 256);
    REQUIRE(ofmSData2.fmData.tile.numSlots == 2);
    REQUIRE(ofmSData2.fmData.tile.slotSize == 128);

    REQUIRE(ofmSData2.fmData.dfltStripeSize.height == 8);
    REQUIRE(ofmSData2.fmData.dfltStripeSize.width == 8);
    REQUIRE(ofmSData2.fmData.dfltStripeSize.channels == 3);

    REQUIRE(ofmSData2.fmData.edgeStripeSize.height == 8);
    REQUIRE(ofmSData2.fmData.edgeStripeSize.width == 8);
    REQUIRE(ofmSData2.fmData.edgeStripeSize.channels == 3);

    REQUIRE(ofmSData2.fmData.supertensorSizeInCells.width == 3);
    REQUIRE(ofmSData2.fmData.supertensorSizeInCells.channels == 1);

    REQUIRE(ofmSData2.fmData.numStripes.height == 1);
    REQUIRE(ofmSData2.fmData.numStripes.width == 1);
    REQUIRE(ofmSData2.fmData.numStripes.channels == 1);

    REQUIRE(ofmSData2.fmData.stripeIdStrides.height == 1);
    REQUIRE(ofmSData2.fmData.stripeIdStrides.width == 1);
    REQUIRE(ofmSData2.fmData.stripeIdStrides.channels == 1);
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

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ofmSAgent1 = commandStream[1];
    const Agent& ofmSAgent2 = commandStream[3];

    const Dependency& readDependency1 = ofmSAgent1.info.readDependencies.at(0);
    const Dependency& readDependency2 = ofmSAgent2.info.readDependencies.at(0);

    // ifmS1 -> ofmS1
    REQUIRE(readDependency1.relativeAgentId == 1);
    REQUIRE(readDependency1.outerRatio.other == 1);
    REQUIRE(readDependency1.outerRatio.self == 1);
    REQUIRE(readDependency1.innerRatio.other == 1);
    REQUIRE(readDependency1.innerRatio.self == 1);
    REQUIRE(readDependency1.boundary == 0);
    // ifmS2 -> ofmS2
    REQUIRE(readDependency2.relativeAgentId == 1);
    REQUIRE(readDependency2.outerRatio.other == 1);
    REQUIRE(readDependency2.outerRatio.self == 1);
    REQUIRE(readDependency2.innerRatio.other == 1);
    REQUIRE(readDependency2.innerRatio.self == 1);
    REQUIRE(readDependency2.boundary == 0);
}

// MceScheduler Agent - Read After Write Dependency Test
TEST_CASE("MceScheduler-IfmStreamer ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator][.]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent           = commandStream[0];
    const Agent& mceSAgent           = commandStream[3];
    const Dependency& readDependency = mceSAgent.info.readDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ifmChannels;
    uint32_t numberOfIfmStripes = ifmSAgent.data.ifm.fmData.numStripes.height *
                                  ifmSAgent.data.ifm.fmData.numStripes.width *
                                  ifmSAgent.data.ifm.fmData.numStripes.channels;

    REQUIRE(readDependency.relativeAgentId == 3);
    REQUIRE(readDependency.outerRatio.other == numberOfIfmStripes);
    REQUIRE(readDependency.outerRatio.self == numberOfMceStripes);
    REQUIRE(readDependency.innerRatio.other == 1);
    REQUIRE(readDependency.innerRatio.self == 1);
    REQUIRE(readDependency.boundary == 0);
}

// MceScheduler Agent - Read After Write Dependency Test
TEST_CASE("MceScheduler-WeightStreamer ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent           = commandStream[3];
    const Dependency& readDependency = mceSAgent.info.readDependencies.at(1);

    REQUIRE(readDependency.relativeAgentId == 2);
    REQUIRE(readDependency.outerRatio.other == 1);
    REQUIRE(readDependency.outerRatio.self == 6);
    REQUIRE(readDependency.innerRatio.other == 1);
    REQUIRE(readDependency.innerRatio.self == 6);
    REQUIRE(readDependency.boundary == 0);
}

// PleScheduler Agent - Read After Write Dependency Test
TEST_CASE("PleScheduler-IfmStreamer ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
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

    REQUIRE(readDependency.relativeAgentId == 2);
    REQUIRE(readDependency.outerRatio.other == numberOfIfmStripes);
    REQUIRE(readDependency.outerRatio.self == numberOfPleStripes);
    REQUIRE(readDependency.innerRatio.other == 1);
    REQUIRE(readDependency.innerRatio.self == 1);
    REQUIRE(readDependency.boundary == 1);
}

// PleScheduler Agent - Read After Write Dependency Test
TEST_CASE("PleScheduler-MceScheduler ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent           = commandStream[3];
    const Agent& pleSAgent           = commandStream[4];
    const Dependency& readDependency = pleSAgent.info.readDependencies.at(1);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ofmChannels;
    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    REQUIRE(readDependency.relativeAgentId == 1);
    REQUIRE(readDependency.outerRatio.other == numberOfMceStripes);
    REQUIRE(readDependency.outerRatio.self == numberOfPleStripes);
    REQUIRE(readDependency.innerRatio.other == 70);
    REQUIRE(readDependency.innerRatio.self == 1);
    REQUIRE(readDependency.boundary == 1);
}

// MceScheduler Agent - Read After Write Dependency Test
TEST_CASE("MceScheduler-PleScheduler ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceSramIntermediateOpGraph mceOpGraph = TwoMceSramIntermediateOpGraph();
    OpGraph mergedOpGraph                    = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent           = commandStream[7];
    const Agent& pleSAgent           = commandStream[4];
    const Dependency& readDependency = mceSAgent.info.readDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ofmChannels;
    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    REQUIRE(readDependency.relativeAgentId == 3);
    REQUIRE(readDependency.outerRatio.other == numberOfPleStripes);
    REQUIRE(readDependency.outerRatio.self == numberOfMceStripes);
    REQUIRE(readDependency.innerRatio.other == 1);
    REQUIRE(readDependency.innerRatio.self == 70);
    REQUIRE(readDependency.boundary == 1);
}

// PleScheduler Agent - Read After Write Dependency Test
TEST_CASE("PleScheduler-PleLoader ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleSAgent           = commandStream[4];
    const Dependency& readDependency = pleSAgent.info.readDependencies.at(0);

    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    REQUIRE(readDependency.relativeAgentId == 2);
    REQUIRE(readDependency.outerRatio.other == 1);
    REQUIRE(readDependency.outerRatio.self == numberOfPleStripes);
    REQUIRE(readDependency.innerRatio.other == 1);
    REQUIRE(readDependency.innerRatio.self == numberOfPleStripes);
    REQUIRE(readDependency.boundary == 0);
}

// OfmStreamer Agent - Read After Write Dependency Test
TEST_CASE("OfmStreamer-IfmStreamer ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceDramIntermediateOpGraph twoMceOpMergeGraph = TwoMceDramIntermediateOpGraph();
    OpGraph mergedOpGraph                            = twoMceOpMergeGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ofmSAgent           = commandStream[5];
    const Dependency& readDependency = ofmSAgent.info.readDependencies.at(0);

    REQUIRE(readDependency.relativeAgentId == 1);
    REQUIRE(readDependency.outerRatio.other == 1);
    REQUIRE(readDependency.outerRatio.self == 1);
    REQUIRE(readDependency.innerRatio.other == 1);
    REQUIRE(readDependency.innerRatio.self == 1);
    REQUIRE(readDependency.boundary == 0);

    ETHOSN_UNUSED(commandStream);
}

// OfmStreamer Agent - Read After Write Dependency Test
TEST_CASE("OfmStreamer-PleScheduler ReadAfterWriteDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ofmSAgent           = commandStream[3];
    const Dependency& readDependency = ofmSAgent.info.readDependencies.at(0);

    REQUIRE(readDependency.relativeAgentId == 1);
    REQUIRE(readDependency.outerRatio.other == 1);
    REQUIRE(readDependency.outerRatio.self == 1);
    REQUIRE(readDependency.innerRatio.other == 1);
    REQUIRE(readDependency.innerRatio.self == 1);
    REQUIRE(readDependency.boundary == 0);
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

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& wgtSAgent           = commandStream[1];
    const Dependency& readDependency = wgtSAgent.info.readDependencies.at(0);

    REQUIRE(readDependency.relativeAgentId == 1);
    REQUIRE(readDependency.outerRatio.other == 400);
    REQUIRE(readDependency.outerRatio.self == 1);
    REQUIRE(readDependency.innerRatio.other == 400);
    REQUIRE(readDependency.innerRatio.self == 1);
    REQUIRE(readDependency.boundary == 0);

    ETHOSN_UNUSED(commandStream);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Write After Read Dependency Tests
//////////////////////////////////////////////////////////////////////////////////////////////

// IfmStreamer Agent - Write After Read Dependency Test
TEST_CASE("IfmStreamer-MceScheduler WriteAfterReadDependency Test", "[CascadingCommandStreamGenerator][.]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent            = commandStream[0];
    const Agent& mceSAgent            = commandStream[3];
    const Dependency& writeDependency = ifmSAgent.info.writeDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ifmChannels;
    uint32_t numberOfIfmStripes = ifmSAgent.data.ifm.fmData.numStripes.height *
                                  ifmSAgent.data.ifm.fmData.numStripes.width *
                                  ifmSAgent.data.ifm.fmData.numStripes.channels;

    REQUIRE(writeDependency.relativeAgentId == 3);
    REQUIRE(writeDependency.outerRatio.other == numberOfMceStripes);
    REQUIRE(writeDependency.outerRatio.self == numberOfIfmStripes);
    REQUIRE(writeDependency.innerRatio.other == 1);
    REQUIRE(writeDependency.innerRatio.self == 1);
    REQUIRE(writeDependency.boundary == 0);
}

// IfmStreamer Agent - Write After Read Dependency Test
TEST_CASE("IfmStreamer-PleScheduler WriteAfterReadDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
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

    REQUIRE(writeDependency.relativeAgentId == 2);
    REQUIRE(writeDependency.outerRatio.other == numberOfPleStripes);
    REQUIRE(writeDependency.outerRatio.self == numberOfIfmStripes);
    REQUIRE(writeDependency.innerRatio.other == 1);
    REQUIRE(writeDependency.innerRatio.self == 1);
    REQUIRE(writeDependency.boundary == 1);
}

// IfmStreamer Agent - Write After Read Dependency Test
TEST_CASE("IfmStreamer-OfmStreamer WriteAfterReadDependency Test", "[CascadingCommandStreamGenerator]")
{
    ConcatOpGraph inputOutputMergeGraph = ConcatOpGraph();
    OpGraph mergedOpGraph               = inputOutputMergeGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent1 = commandStream[0];
    const Agent& ifmSAgent2 = commandStream[2];

    const Dependency& writeDependency1 = ifmSAgent1.info.writeDependencies.at(0);
    const Dependency& writeDependency2 = ifmSAgent2.info.writeDependencies.at(0);

    // ifmS1 -> ofmS1
    REQUIRE(writeDependency1.relativeAgentId == 1);
    REQUIRE(writeDependency1.outerRatio.other == 1);
    REQUIRE(writeDependency1.outerRatio.self == 1);
    REQUIRE(writeDependency1.innerRatio.other == 1);
    REQUIRE(writeDependency1.innerRatio.self == 1);
    REQUIRE(writeDependency1.boundary == 0);
    // ifmS2 -> ofmS2
    REQUIRE(writeDependency2.relativeAgentId == 1);
    REQUIRE(writeDependency2.outerRatio.other == 1);
    REQUIRE(writeDependency2.outerRatio.self == 1);
    REQUIRE(writeDependency2.innerRatio.other == 1);
    REQUIRE(writeDependency2.innerRatio.self == 1);
    REQUIRE(writeDependency2.boundary == 0);
}

// WeightStreamer Agent - Write After Read Dependency Test
TEST_CASE("WeightStreamer-MceScheduler WriteAfterReadDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& wgtSAgent            = commandStream[1];
    const Dependency& writeDependency = wgtSAgent.info.writeDependencies.at(0);

    REQUIRE(writeDependency.relativeAgentId == 2);
    REQUIRE(writeDependency.outerRatio.other == 6);
    REQUIRE(writeDependency.outerRatio.self == 1);
    REQUIRE(writeDependency.innerRatio.other == 6);
    REQUIRE(writeDependency.innerRatio.self == 1);
    REQUIRE(writeDependency.boundary == 0);
}

// MceScheduler Agent - Write After Read Dependency Test
TEST_CASE("PleScheduler-MceScheduler WriteAfterReadDependency Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceSramIntermediateOpGraph mceOpGraph = TwoMceSramIntermediateOpGraph();
    OpGraph mergedOpGraph                    = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent            = commandStream[7];
    const Agent& pleSAgent            = commandStream[4];
    const Dependency& writeDependency = pleSAgent.info.writeDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ofmChannels;
    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    REQUIRE(writeDependency.relativeAgentId == 3);
    REQUIRE(writeDependency.outerRatio.other == numberOfMceStripes);
    REQUIRE(writeDependency.outerRatio.self == numberOfPleStripes);
    REQUIRE(writeDependency.innerRatio.other == 70);
    REQUIRE(writeDependency.innerRatio.self == 1);
    REQUIRE(writeDependency.boundary == 1);
}

// PleScheduler Agent - Write After Read Dependency Test
TEST_CASE("PleScheduler-OfmStreamer WriteAfterReadDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleSAgent            = commandStream[2];
    const Dependency& writeDependency = pleSAgent.info.writeDependencies.at(0);

    REQUIRE(writeDependency.relativeAgentId == 1);
    REQUIRE(writeDependency.outerRatio.other == 1);
    REQUIRE(writeDependency.outerRatio.self == 1);
    REQUIRE(writeDependency.innerRatio.other == 1);
    REQUIRE(writeDependency.innerRatio.self == 1);
    REQUIRE(writeDependency.boundary == 0);
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

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent               = commandStream[3];
    const Agent& pleSAgent               = commandStream[4];
    const Dependency& scheduleDependency = mceSAgent.info.scheduleDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ofmChannels;
    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    REQUIRE(scheduleDependency.relativeAgentId == 1);
    REQUIRE(scheduleDependency.outerRatio.other == numberOfPleStripes);
    REQUIRE(scheduleDependency.outerRatio.self == numberOfMceStripes);
    REQUIRE(scheduleDependency.innerRatio.other == 1);
    REQUIRE(scheduleDependency.innerRatio.self == 70);
    REQUIRE(scheduleDependency.boundary == 1);
}

// IfmStreamer Agent - Schedule Time Dependency Test
TEST_CASE("IfmStreamer-PleScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
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

    REQUIRE(scheduleDependency.relativeAgentId == 2);
    REQUIRE(scheduleDependency.outerRatio.other == numberOfPleStripes);
    REQUIRE(scheduleDependency.outerRatio.self == numberOfIfmStripes);
    REQUIRE(scheduleDependency.innerRatio.other == 1);
    REQUIRE(scheduleDependency.innerRatio.self == 1);
    REQUIRE(scheduleDependency.boundary == 1);
}

// IfmStreamer Agent - Schedule Time Dependency Test
TEST_CASE("IfmStreamer-OfmStreamer ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    ConcatOpGraph inputOutputMergeGraph = ConcatOpGraph();
    OpGraph mergedOpGraph               = inputOutputMergeGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ifmSAgent1 = commandStream[0];
    const Agent& ifmSAgent2 = commandStream[2];

    const Dependency& scheduleDependency1 = ifmSAgent1.info.scheduleDependencies.at(0);
    const Dependency& scheduleDependency2 = ifmSAgent2.info.scheduleDependencies.at(0);

    // ifmS1 -> ofmS1
    REQUIRE(scheduleDependency1.relativeAgentId == 1);
    REQUIRE(scheduleDependency1.outerRatio.other == 1);
    REQUIRE(scheduleDependency1.outerRatio.self == 1);
    REQUIRE(scheduleDependency1.innerRatio.other == 1);
    REQUIRE(scheduleDependency1.innerRatio.self == 1);
    REQUIRE(scheduleDependency1.boundary == 0);
    // ifmS2 -> ofmS2
    REQUIRE(scheduleDependency2.relativeAgentId == 1);
    REQUIRE(scheduleDependency2.outerRatio.other == 1);
    REQUIRE(scheduleDependency2.outerRatio.self == 1);
    REQUIRE(scheduleDependency2.innerRatio.other == 1);
    REQUIRE(scheduleDependency2.innerRatio.self == 1);
    REQUIRE(scheduleDependency2.boundary == 0);
}

// WeightStreamer Agent - Schedule Time Dependency Test
TEST_CASE("WeightStreamer-MceScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent               = commandStream[3];
    const Agent& pleSAgent               = commandStream[4];
    const Dependency& scheduleDependency = mceSAgent.info.scheduleDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ofmChannels;
    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    REQUIRE(scheduleDependency.relativeAgentId == 1);
    REQUIRE(scheduleDependency.outerRatio.other == numberOfPleStripes);
    REQUIRE(scheduleDependency.outerRatio.self == numberOfMceStripes);
    REQUIRE(scheduleDependency.innerRatio.other == 1);
    REQUIRE(scheduleDependency.innerRatio.self == 70);
    REQUIRE(scheduleDependency.boundary == 1);
}

// MceScheduler Agent - Schedule Time Dependency Test
TEST_CASE("MceScheduler-PleScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent               = commandStream[3];
    const Agent& pleSAgent               = commandStream[4];
    const Dependency& scheduleDependency = mceSAgent.info.scheduleDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ofmChannels;
    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    REQUIRE(scheduleDependency.relativeAgentId == 1);
    REQUIRE(scheduleDependency.outerRatio.other == numberOfPleStripes);
    REQUIRE(scheduleDependency.outerRatio.self == numberOfMceStripes);
    REQUIRE(scheduleDependency.innerRatio.other == 1);
    REQUIRE(scheduleDependency.innerRatio.self == 70);
    REQUIRE(scheduleDependency.boundary == 1);
}

// PleScheduler Agent - Schedule Time Dependency Test
TEST_CASE("PleScheduler-MceScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceSramIntermediateOpGraph mceOpGraph = TwoMceSramIntermediateOpGraph();
    OpGraph mergedOpGraph                    = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& mceSAgent               = commandStream[7];
    const Agent& pleSAgent               = commandStream[4];
    const Dependency& scheduleDependency = pleSAgent.info.scheduleDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ofmChannels;
    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    REQUIRE(scheduleDependency.relativeAgentId == 3);
    REQUIRE(scheduleDependency.outerRatio.other == numberOfMceStripes);
    REQUIRE(scheduleDependency.outerRatio.self == numberOfPleStripes);
    REQUIRE(scheduleDependency.innerRatio.other == 70);
    REQUIRE(scheduleDependency.innerRatio.self == 1);
    REQUIRE(scheduleDependency.boundary == 1);
}

// PleLoader Agent - Schedule Time Dependency Test
TEST_CASE("PleLoader-MceScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleLAgent               = commandStream[2];
    const Agent& mceSAgent               = commandStream[3];
    const Dependency& scheduleDependency = pleLAgent.info.scheduleDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ifmChannels;

    REQUIRE(scheduleDependency.relativeAgentId == 1);
    REQUIRE(scheduleDependency.outerRatio.other == numberOfMceStripes);
    REQUIRE(scheduleDependency.outerRatio.self == 1);
    REQUIRE(scheduleDependency.innerRatio.other == numberOfMceStripes);
    REQUIRE(scheduleDependency.innerRatio.self == 1);
    REQUIRE(scheduleDependency.boundary == 0);
}

// PleLoader Agent - Schedule Time Dependency Test
TEST_CASE("PleLoader-PleScheduler ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleLAgent               = commandStream[1];
    const Agent& pleSAgent               = commandStream[2];
    const Dependency& scheduleDependency = pleLAgent.info.scheduleDependencies.at(0);

    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    REQUIRE(scheduleDependency.relativeAgentId == 1);
    REQUIRE(scheduleDependency.outerRatio.other == numberOfPleStripes);
    REQUIRE(scheduleDependency.outerRatio.self == 1);
    REQUIRE(scheduleDependency.innerRatio.other == numberOfPleStripes);
    REQUIRE(scheduleDependency.innerRatio.self == 1);
    REQUIRE(scheduleDependency.boundary == 0);
}

// PleScheduler Agent - Schedule Time Dependency Test
TEST_CASE("PleScheduler-OfmStreamer ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& pleSAgent               = commandStream[2];
    const Dependency& scheduleDependency = pleSAgent.info.scheduleDependencies.at(0);

    REQUIRE(scheduleDependency.relativeAgentId == 1);
    REQUIRE(scheduleDependency.outerRatio.other == 1);
    REQUIRE(scheduleDependency.outerRatio.self == 1);
    REQUIRE(scheduleDependency.innerRatio.other == 1);
    REQUIRE(scheduleDependency.innerRatio.self == 1);
    REQUIRE(scheduleDependency.boundary == 0);
}

// OfmStreamer Agent - Schedule Time Dependency Test
TEST_CASE("OfmStreamer-IfmStreamer ScheduleTimeDependency Test", "[CascadingCommandStreamGenerator]")
{
    TwoMceDramIntermediateOpGraph twoMceOpMergeGraph = TwoMceDramIntermediateOpGraph();
    OpGraph mergedOpGraph                            = twoMceOpMergeGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = commandStreamGenerator.Generate();

    const std::vector<Agent>& commandStream = commandStreamGenerator.GetCommandStreamOfAgents();

    const Agent& ofmSAgent               = commandStream[5];
    const Dependency& scheduleDependency = ofmSAgent.info.scheduleDependencies.at(0);

    REQUIRE(scheduleDependency.relativeAgentId == 1);
    REQUIRE(scheduleDependency.outerRatio.other == 200);
    REQUIRE(scheduleDependency.outerRatio.self == 400);
    REQUIRE(scheduleDependency.innerRatio.other == 1);
    REQUIRE(scheduleDependency.innerRatio.self == 400);
    REQUIRE(scheduleDependency.boundary == 0);
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
    CascadingCommandStreamGenerator commandStreamGenerator(mergedOpGraph, operationIds, hwCaps, compOpt);
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
            REQUIRE(bufferManager.GetBuffers().at(buffId).m_LifetimeStart == 5);
            REQUIRE(bufferManager.GetBuffers().at(buffId).m_LifetimeEnd == 9);
        }
    }
}
