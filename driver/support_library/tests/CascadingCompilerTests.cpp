//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/cascading/CascadingCompiler.hpp"
#include "../src/cascading/CombinerDFS.hpp"
#include "../src/cascading/PartUtils.hpp"
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
// Agent Data Tests
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
            std::ofstream stream("CascadingCompiler MceSchedulerAgent Input.dot");
            SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
        }

        mergedOpGraph = GetOpGraphForCombination(comb, graph);

        bool dumpOutputGraphToFile = false;
        if (dumpOutputGraphToFile)
        {
            std::ofstream stream("PleOnly_Output2.dot");
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
        inputStripeSize        = CalculateBufferSize(ptrInputBuffer->m_StripeShape, ptrInputBuffer->m_Format);
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
        kernelHeight            = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[1]);
        kernelWidth             = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[2]);

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
            std::ofstream stream("CascadingCompiler MceSchedulerAgent Input.dot");
            SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
        }

        mergedOpGraph = GetOpGraphForCombination(comb, graph);

        bool dumpOutputGraphToFile = false;
        if (dumpOutputGraphToFile)
        {
            std::ofstream stream("CascadingCompiler_MceSchedulerAgent_Output.dot");
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

// IfmStreamer Agent Data Test
TEST_CASE("IfmStreamer Agent Data Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("WeightStreamer Agent Data Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("MceScheduler Agent Data Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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

    REQUIRE(mceSData.filterShape.height == mceOpGraph.getKernelHeight());
    REQUIRE(mceSData.filterShape.width == mceOpGraph.getKernelWidth());

    REQUIRE(mceSData.padding.left == 0);
    REQUIRE(mceSData.padding.top == 0);

    REQUIRE(mceSData.ifmDeltaDefault.height == mceOpGraph.getIfmDeltaHeight());
    REQUIRE(mceSData.ifmDeltaDefault.width == mceOpGraph.getIfmDeltaWidth());
    REQUIRE(mceSData.ifmDeltaEdge.height == mceOpGraph.getIfmDeltaHeight());
    REQUIRE(mceSData.ifmDeltaEdge.width == mceOpGraph.getIfmDeltaWidth());

    REQUIRE(mceSData.reluActiv.max == 255);
    REQUIRE(mceSData.reluActiv.min == 0);

    REQUIRE(mceSData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);
}

// PleLoader Agent Data Test
TEST_CASE("PleLoader Agent Data Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

    const Agent& pleLAgent = commandStream[2];
    const PleL& pleLData   = pleLAgent.data.pleL;

    REQUIRE(pleLData.sramAddr == 0x0000F0F0);
    REQUIRE(pleLData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);
}

// PleScheduler Agent Data Test
TEST_CASE("PleScheduler Agent Data Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();
    Agent pleSchedulerAgent          = commandStream[4];

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
TEST_CASE("PleScheduler Standalone Agent Data Test", "[CascadingCompiler]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const std::set<uint32_t> operationIds = { 0 };
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("OfmStreamer Agent Data Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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

//////////////////////////////////////////////////////////////////////////////////////////////
// Read After Write Dependency Tests
//////////////////////////////////////////////////////////////////////////////////////////////

// IfmStreamer Agent - Read After Write Dependency Test
TEST_CASE("IfmStreamer-OfmStreamer ReadAfterWriteDependency Test", "[CascadingCompiler]")
{}

// MceScheduler Agent - Read After Write Dependency Test
TEST_CASE("MceScheduler-IfmStreamer ReadAfterWriteDependency Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
    REQUIRE(readDependency.boundary == 1);
}

// MceScheduler Agent - Read After Write Dependency Test
TEST_CASE("MceScheduler-WeightStreamer ReadAfterWriteDependency Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("PleScheduler-IfmStreamer ReadAfterWriteDependency Test", "[CascadingCompiler]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("PleScheduler-MceScheduler ReadAfterWriteDependency Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
    REQUIRE(readDependency.boundary == 0);
}

// PleScheduler Agent - Read After Write Dependency Test
TEST_CASE("PleScheduler-PleLoader ReadAfterWriteDependency Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("OfmStreamer-IfmStreamer ReadAfterWriteDependency Test", "[CascadingCompiler]")
{}

// OfmStreamer Agent - Read After Write Dependency Test
TEST_CASE("OfmStreamer-PleScheduler ReadAfterWriteDependency Test", "[CascadingCompiler]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("WeightStreamer-OfmStreamer SramOverlpaDependency Test", "[CascadingCompiler]")
{}

//////////////////////////////////////////////////////////////////////////////////////////////
// Write After Read Dependency Tests
//////////////////////////////////////////////////////////////////////////////////////////////

// IfmStreamer Agent - Write After Read Dependency Test
TEST_CASE("IfmStreamer-MceScheduler WriteAfterReadDependency Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
    REQUIRE(writeDependency.boundary == 1);
}

// IfmStreamer Agent - Write After Read Dependency Test
TEST_CASE("IfmStreamer-PleScheduler WriteAfterReadDependency Test", "[CascadingCompiler]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("IfmStreamer-OfmStreamer WriteAfterReadDependency Test", "[CascadingCompiler]")
{}

// WeightStreamer Agent - Write After Read Dependency Test
TEST_CASE("WeightStreamer-MceScheduler WriteAfterReadDependency Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("MceScheduler-PleScheduler WriteAfterReadDependency Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

    const Agent& mceSAgent            = commandStream[3];
    const Agent& pleSAgent            = commandStream[4];
    const Dependency& writeDependency = mceSAgent.info.writeDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth *
                                  mceSAgent.data.mce.numStripes.ofmChannels;
    uint32_t numberOfPleStripes = pleSAgent.data.pleS.numStripes.height * pleSAgent.data.pleS.numStripes.width *
                                  pleSAgent.data.pleS.numStripes.channels;

    REQUIRE(writeDependency.relativeAgentId == 1);
    REQUIRE(writeDependency.outerRatio.other == numberOfPleStripes);
    REQUIRE(writeDependency.outerRatio.self == numberOfMceStripes);
    REQUIRE(writeDependency.innerRatio.other == 1);
    REQUIRE(writeDependency.innerRatio.self == 70);
    REQUIRE(writeDependency.boundary == 1);
}

// PleScheduler Agent - Write After Read Dependency Test
TEST_CASE("PleScheduler-OfmStreamer WriteAfterReadDependency Test", "[CascadingCompiler]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("IfmStreamer-MceScheduler ScheduleTimeDependency Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("IfmStreamer-PleScheduler ScheduleTimeDependency Test", "[CascadingCompiler]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("IfmStreamer-OfmStreamer ScheduleTimeDependency Test", "[CascadingCompiler]")
{}

// WeightStreamer Agent - Schedule Time Dependency Test
TEST_CASE("WeightStreamer-MceScheduler ScheduleTimeDependency Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("MceScheduler-PleScheduler ScheduleTimeDependency Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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

// PleLoader Agent - Schedule Time Dependency Test
TEST_CASE("PleLoader-MceScheduler ScheduleTimeDependency Test", "[CascadingCompiler]")
{
    MceOpGraph mceOpGraph = MceOpGraph();
    OpGraph mergedOpGraph = mceOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("PleLoader-PleScheduler ScheduleTimeDependency Test", "[CascadingCompiler]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("PleScheduler-OfmStreamer ScheduleTimeDependency Test", "[CascadingCompiler]")
{
    StandalonePleOpGraph saPleOpGraph = StandalonePleOpGraph();
    OpGraph mergedOpGraph             = saPleOpGraph.GetMergedOpGraph();

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps     = GetEthosN78HwCapabilities();
    const std::set<uint32_t> operationIds = { 0 };

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

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
TEST_CASE("OfmStreamer-IfmStreamer ScheduleTimeDependency Test", "[CascadingCompiler]")
{}
