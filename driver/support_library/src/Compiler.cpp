//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Compiler.hpp"

#include "ConversionPass.hpp"
#include "GraphNodes.hpp"
#include "McePlePass.hpp"
#include "PlePass.hpp"
#include "Section.hpp"
#include "SramAllocator.hpp"

#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>

namespace ethosn
{
namespace support_library
{

using namespace utils;

uint32_t CalculateBufferSize(const TensorShape& shape, command_stream::DataFormat dataFormat)
{
    assert(dataFormat == command_stream::DataFormat::NHWC || dataFormat == command_stream::DataFormat::NHWCB ||
           dataFormat == command_stream::DataFormat::NHWCB_COMPRESSED);

    if (dataFormat == command_stream::DataFormat::NHWCB_COMPRESSED)
    {
        return TotalSizeBytesNHWCBCompressed(shape);
    }
    else if (dataFormat == command_stream::DataFormat::NHWCB)
    {
        return TotalSizeBytesNHWCB(shape);
    }
    else
    {
        return TotalSizeBytes(shape);
    }
}

std::vector<std::unique_ptr<IStrategy>> GenerateAllowedStrategies(const CompilationOptions& m_Options)
{
    std::vector<std::unique_ptr<IStrategy>> result;
    // We try the "best" strategies first until we find one which is appropriate
    // This may change in the future when we use a dynamic programming approach
    if (m_Options.m_Strategy3)
    {
        result.push_back(std::make_unique<Strategy3>());
    }
    if (m_Options.m_Strategy0)
    {
        result.push_back(std::make_unique<Strategy0>());
    }
    if (m_Options.m_Strategy1)
    {
        result.push_back(std::make_unique<Strategy1>());
    }
    if (m_Options.m_Strategy4)
    {
        result.push_back(std::make_unique<Strategy4>());
    }
    if (m_Options.m_Strategy6)
    {
        result.push_back(std::make_unique<Strategy6>());
    }
    if (m_Options.m_Strategy7)
    {
        result.push_back(std::make_unique<Strategy7>());
    }
    return result;
}

std::vector<command_stream::BlockConfig> GenerateAllowedBlockConfigs(const CompilationOptions& m_Options)
{
    using namespace command_stream;
    std::vector<BlockConfig> result;

    if (m_Options.m_BlockConfig16x16)
    {
        result.emplace_back(16u, 16u);
    }
    if (m_Options.m_BlockConfig32x8)
    {
        result.emplace_back(32u, 8u);
    }
    if (m_Options.m_BlockConfig8x32)
    {
        result.emplace_back(8u, 32u);
    }
    if (m_Options.m_BlockConfig8x8)
    {
        result.emplace_back(8u, 8u);
    }
    return result;
}

Compiler::Compiler(const Network& network,
                   const FirmwareAndHardwareCapabilities& fwAndHwCapabilities,
                   const CompilationOptions& compilationOptions,
                   const EstimationOptions& estimationOptions)
    : m_Network(network)
    , m_DumpRam(compilationOptions.m_DumpRam)
    , m_InitialSramDump(compilationOptions.m_InitialSramDump)
    , m_AllowedStrategies(GenerateAllowedStrategies(compilationOptions))
    , m_AllowedBlockConfigs(GenerateAllowedBlockConfigs(compilationOptions))
    , m_Capabilities(fwAndHwCapabilities)
    , m_DumpDebugFiles(compilationOptions.m_DumpDebugFiles)
    , m_DebugDir(compilationOptions.m_DebugDir)
    , m_DisableWinograd(compilationOptions.m_DisableWinograd)
    , m_EnableIntermediateCompression(compilationOptions.m_EnableIntermediateCompression)
    , m_EstimationOptions(estimationOptions)
{}

std::unique_ptr<CompiledNetwork> Compiler::Compile()
{
    try
    {
        Convert();
        Prepare();
        Generate();
    }
    catch (const NotSupportedException& e)
    {
        // Either we failed compilation or there was not enough SRAM to convert NHWCB to NHWC
        // NNXSW-2802: Temporary fix to print the error but need better approach  for error reporting from support library.
        std::cerr << "Error: " << e.what() << std::endl;
        return std::unique_ptr<CompiledNetworkImpl>(nullptr);
    }

    // The compiler will need to split the network into supported subgraphs and have the appropriate ids for each.
    // See the Support Library public interface design note for more details.
    // For now we're just passing the full network ids through.
    std::set<uint32_t> compiledOperationIds = m_Network.GetOperationIds();

    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = std::make_unique<CompiledNetworkImpl>(
        m_BufferManager.GetConstantDmaData(), m_BufferManager.GetConstantControlUnitData(),
        m_BufferManager.GetBuffers(), compiledOperationIds);

    return compiledNetwork;
}

NetworkPerformanceData Compiler::EstimatePerformance()
{
    try
    {
        Convert();
        Prepare();
    }
    catch (const NotSupportedException&)
    {
        // Conversion and preparation can throw by not creating a valid graph but we should still be able to estimate it.
    }
    Estimate();

    return m_PerformanceStream;
}

void Compiler::Convert()
{
    m_Graph = Graph(m_Network, m_Capabilities);

    DumpGraph("GraphInitial.dot");
}

void Compiler::Optimize()
{
    bool madeChange;
    do
    {
        madeChange = false;
        for (Node* node : m_Graph.GetNodesSorted())
        {
            ConcatNode* concatenationNode        = dynamic_cast<ConcatNode*>(node);
            ConstantNode* constantNode           = dynamic_cast<ConstantNode*>(node);
            FormatConversionNode* conversionNode = dynamic_cast<FormatConversionNode*>(node);
            OutputNode* outputNode               = dynamic_cast<OutputNode*>(node);
            RequantizeNode* requantizeNode       = dynamic_cast<RequantizeNode*>(node);
            ReinterpretNode* reinterpetNode      = dynamic_cast<ReinterpretNode*>(node);
            // Two adjacent format conversions which perform opposite conversions can be eliminated:
            //
            //   X (NHWCB) -->  FormatConversionNode to NHWC  -->  FormatConversionNode to NHWCB -->
            //
            //  Becomes
            //
            //  X (NHWCB) -->
            if (conversionNode && conversionNode->GetOutputs().size() == 1 &&
                conversionNode->GetOptimizationHint() != OptimizationHint::DoNotMerge)
            {
                FormatConversionNode* nextFormatConversionNode =
                    dynamic_cast<FormatConversionNode*>(conversionNode->GetOutput(0)->GetDestination());
                if (nextFormatConversionNode &&
                    nextFormatConversionNode->GetOptimizationHint() != OptimizationHint::DoNotMerge)
                {
                    if (conversionNode->GetInputFormat(0) == nextFormatConversionNode->GetFormat())
                    {
                        m_Graph.CollapseEdge(conversionNode->GetInput(0));
                        m_Graph.CollapseEdge(nextFormatConversionNode->GetInput(0));
                        madeChange = true;
                        break;
                    }
                }
            }
            // Two adjacent requantize nodes can be merged:
            //
            //   X -->  RequantizeNode to (0.1, 74) --> RequantizeNode to (1, -84)  -->
            //
            //  Becomes
            //
            //  X -->  RequantizeNode to (1, -84) -->
            else if (requantizeNode && requantizeNode->GetOutputs().size() == 1 &&
                     dynamic_cast<RequantizeNode*>(requantizeNode->GetOutput(0)->GetDestination()))
            {
                // Add the corresponding ids from the first requantize node (the removed one) to the second one (the one we are keeping)
                RequantizeNode* nextNode =
                    dynamic_cast<RequantizeNode*>(requantizeNode->GetOutput(0)->GetDestination());
                nextNode->AddCorrespondingOperationIDs(requantizeNode->GetCorrespondingOperationIds());

                m_Graph.CollapseNode(requantizeNode);
                madeChange = true;
                break;
            }
            // A reinterpret followed by a requantize can be reordered so the requantize is first.
            // This is required to be able to do the requantize as part of a preceding MceOperation
            //
            //  X -->  ReinterpretNode --> RequantizeNode to (-1, 84) -->
            //
            //  Becomes
            //
            //  X --> RequantizeNode to (-1, 84) --> ReinterpretNode -->
            else if (reinterpetNode && reinterpetNode->GetOutputs().size() == 1 &&
                     dynamic_cast<RequantizeNode*>(reinterpetNode->GetOutput(0)->GetDestination()))
            {
                Node* oldRequantNode = dynamic_cast<RequantizeNode*>(reinterpetNode->GetOutput(0)->GetDestination());
                Node* newRequant     = m_Graph.CreateAndAddNode<RequantizeNode>(
                    reinterpetNode->GetInputShape(0), oldRequantNode->GetQuantizationInfo(),
                    oldRequantNode->GetInputFormat(0), oldRequantNode->GetCorrespondingOperationIds());
                m_Graph.SplitEdge(reinterpetNode->GetInput(0), newRequant);
                m_Graph.CollapseNode(oldRequantNode);
                madeChange = true;
                break;
            }
            // A concat followed by a requantize can be reordered so that the requantize occurs on each input of the concat.
            // This is required to be able to do the requantize as part of a preceding MceOperation
            //
            //  X0 -->
            //  X1 -->  ConcatNode  --> RequantizeNode to (-1, 84) -->
            //  X2 -->
            //
            //  Becomes
            //
            //  X0 --> RequantizeNode to (-1, 84) -->
            //  X1 --> RequantizeNode to (-1, 84) --> ConcatNode -->
            //  X2 --> RequantizeNode to (-1, 84) -->
            else if (concatenationNode && concatenationNode->GetOutputs().size() == 1 &&
                     dynamic_cast<RequantizeNode*>(concatenationNode->GetOutput(0)->GetDestination()))
            {
                Node* oldRequantNode = dynamic_cast<RequantizeNode*>(concatenationNode->GetOutput(0)->GetDestination());
                for (uint32_t i = 0; i < concatenationNode->GetInputs().size(); ++i)
                {
                    Node* newRequant = m_Graph.CreateAndAddNode<RequantizeNode>(
                        concatenationNode->GetInputShape(i), oldRequantNode->GetQuantizationInfo(),
                        concatenationNode->GetInputFormat(i), oldRequantNode->GetCorrespondingOperationIds());
                    m_Graph.SplitEdge(concatenationNode->GetInput(i), newRequant);
                }
                m_Graph.CollapseNode(oldRequantNode);
                madeChange = true;
                break;
            }
            // This is for use case of concatenation to concatenation in the graph, for example
            // Before:
            // concatNode0      concatNode1
            //     \                /
            //         concatNode2
            // After:
            //         concatNode2
            //
            else if (concatenationNode && concatenationNode->GetInputs().size() > 1 &&
                     concatenationNode->GetOptimizationHint() != OptimizationHint::DoNotMerge)
            {
                for (uint32_t i = 0; i < concatenationNode->GetInputs().size(); ++i)
                {
                    ConcatNode* prevConcatenationNode =
                        dynamic_cast<ConcatNode*>(concatenationNode->GetInput(i)->GetSource());
                    if (prevConcatenationNode)
                    {
                        // preserve the corresponding ID from the concat node we are removing
                        concatenationNode->AddCorrespondingOperationIDs(
                            prevConcatenationNode->GetCorrespondingOperationIds());
                        m_Graph.CollapseNode(prevConcatenationNode);
                        madeChange = true;
                        break;
                    }
                }
            }
            // Remove unconnected nodes
            // Before:
            // Node0   Node1
            //         /
            //      Node2
            // After:
            //        Node1
            //         /
            //      Node2
            //
            else if (outputNode == nullptr && node->GetOutputs().size() == 0)
            {
                m_Graph.RemoveNode(node);
                madeChange = true;
                break;
            }
            // Merge Constant node with ReinterpretNode if any.
            // Before:
            //         ConstantNode
            //         /
            //      ReinterpretNode
            // After:
            //        ConstantNode
            //
            else if (constantNode && constantNode->GetOutputs().size() == 1 &&
                     constantNode->GetFormat() == CompilerDataFormat::NHWC &&
                     dynamic_cast<ReinterpretNode*>(constantNode->GetOutput(0)->GetDestination()))
            {
                // Statically reshape the constant node shape.
                ReinterpretNode* reinterpetNode =
                    dynamic_cast<ReinterpretNode*>(constantNode->GetOutput(0)->GetDestination());
                const TensorInfo constantInfo(reinterpetNode->GetShape(), constantNode->GetConstantDataType(),
                                              DataFormat::NHWC, constantNode->GetQuantizationInfo());
                Node* newConstantNode = m_Graph.CreateAndAddNode<ConstantNode>(
                    constantInfo, constantNode->GetConstantData(), node->GetCorrespondingOperationIds());
                // preserve the operation ids from the nodes that are being removed
                newConstantNode->AddCorrespondingOperationIDs(reinterpetNode->GetCorrespondingOperationIds());

                m_Graph.InsertNodeAfter(reinterpetNode, newConstantNode);
                m_Graph.CollapseNode(reinterpetNode);
                m_Graph.CollapseNode(constantNode);
                madeChange = true;
                break;
            }
            // Merge Constant node with FormatConversionNode if any.
            // Before:
            //         ConstantNode
            //         /
            //      FormatConversionNode
            // After:
            //        ConstantNode
            //
            else if (constantNode && constantNode->GetOutputs().size() == 1 &&
                     constantNode->GetFormat() == CompilerDataFormat::NHWC &&
                     dynamic_cast<FormatConversionNode*>(constantNode->GetOutput(0)->GetDestination()))
            {
                m_Graph.CollapseEdge(constantNode->GetOutput(0));
                madeChange = true;
                break;
            }
            // Replace Constant node and Addition node with a new MceOperationNode.
            // Before:
            // constantNode          inputNode
            //          \                /
            //      StandalonePleOperationNode
            // After:
            //                inputNode
            //                   /
            //   MceOperationNode (identity depthwise where the bias is the constant)
            //
            else if (constantNode && constantNode->GetOutputs().size() == 1 &&
                     constantNode->GetFormat() == CompilerDataFormat::NHWC &&
                     dynamic_cast<StandalonePleOperationNode*>(constantNode->GetOutput(0)->GetDestination()))
            {
                StandalonePleOperationNode* pleOperationNode =
                    dynamic_cast<StandalonePleOperationNode*>(constantNode->GetOutput(0)->GetDestination());

                if (pleOperationNode->GetKernelOperation() == command_stream::PleOperation::ADDITION ||
                    pleOperationNode->GetKernelOperation() == command_stream::PleOperation::ADDITION_RESCALE)
                {
                    // if input shape is { 1, 1, 1, C } add an identity depthwise instead where the bias values are the constant vals from the bias add
                    bool isConstantBroadcastAddChannels = constantNode->GetShape()[0] == 1 &&
                                                          constantNode->GetShape()[1] == 1 &&
                                                          constantNode->GetShape()[2] == 1;

                    if (isConstantBroadcastAddChannels)
                    {
                        const TensorInfo constantLayerInfo(constantNode->GetShape(),
                                                           constantNode->GetConstantDataType(), DataFormat::NHWC,
                                                           constantNode->GetQuantizationInfo());

                        std::vector<uint8_t> constantLayerData = constantNode->GetConstantData();
                        const Padding& padding                 = { 0, 0, 0, 0 };

                        // Assume there is only one constant input (and only 2 inputs total).
                        // In this case the input to the depthwise will be the non constant one.
                        uint8_t idxOfInput = 0;

                        // If the constant one is at idx 0, then it must be the other one.
                        if (dynamic_cast<ConstantNode*>(pleOperationNode->GetInput(0)->GetSource()))
                        {
                            idxOfInput = 1;
                        }

                        Node* inputNode = pleOperationNode->GetInput(idxOfInput)->GetSource();

                        const TensorShape inputShape = inputNode->GetShape();

                        if (inputShape[3] == constantNode->GetShape()[3])
                        {

                            const QuantizationInfo& outputQuantInfo =
                                pleOperationNode->GetOutput(0)->GetSource()->GetQuantizationInfo();

                            TensorShape outputShape = pleOperationNode->GetOutput(0)->GetSource()->GetShape();

                            const uint32_t numIfm = inputShape[3];
                            // Since the constant input is being requantized, the weight scale and values must be chosen
                            // A weight scale and data must satisify the following requirements:
                            //   - the resulting weight data for the identity convolution doesn't saturate
                            //       (i.e. must be between 1 and 255)
                            //   - inputQuantScale * weightQuantScale needs to be less than the outputQuantScale
                            //       (See CalculateQuantizedMultiplierSmallerThanOne in Utils.hpp)
                            const float weightScaleUpperBound =
                                std::min(outputQuantInfo.m_Scale / inputNode->GetQuantizationInfo().m_Scale, 1.f);
                            constexpr float weightScaleLowerBound = (1.f / 255.f);
                            if (weightScaleUpperBound < weightScaleLowerBound)
                            {
                                throw NotSupportedException("Couldn't choose appropriate weight scale for bias add");
                            }
                            const float weightScaleTarget = (weightScaleUpperBound + weightScaleLowerBound) / 2.f;
                            // The reciprical of the scale needs to be a whole number to minimize rounding error.
                            const float weightScaleRecipRounded = std::round(1.f / weightScaleTarget);
                            const float weightScale             = 1.f / weightScaleRecipRounded;
                            const float newConstantLayerScale = weightScale * inputNode->GetQuantizationInfo().m_Scale;

                            std::vector<uint8_t> weightsData(1 * 1 * 1 * numIfm,
                                                             static_cast<uint8_t>(weightScaleRecipRounded));

                            TensorInfo weightInfo{
                                { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale }
                            };

                            QuantizationInfo constantNodeQuantizationInfo = constantNode->GetQuantizationInfo();
                            if (constantNode->GetConstantDataType() == DataType::UINT8_QUANTIZED)
                            {
                                std::vector<int32_t> newConstantLayerData;
                                for (uint32_t k = 0; k < constantLayerData.size(); ++k)
                                {
                                    float fpValue = constantNodeQuantizationInfo.m_Scale *
                                                    static_cast<float>((constantLayerData.at(k) -
                                                                        constantNodeQuantizationInfo.m_ZeroPoint));
                                    newConstantLayerData.push_back(
                                        static_cast<int32_t>(std::round(fpValue / newConstantLayerScale)));
                                }
                                Node* mceNode = m_Graph.CreateAndAddNode<MceOperationNode>(
                                    inputShape, outputShape, outputQuantInfo, weightInfo, weightsData,
                                    constantLayerInfo, newConstantLayerData, Stride{ 1, 1 }, 1, padding.m_Top,
                                    padding.m_Left, ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION,
                                    CompilerDataFormat::NHWCB, node->GetCorrespondingOperationIds());

                                mceNode->AddCorrespondingOperationIDs(pleOperationNode->GetCorrespondingOperationIds());

                                m_Graph.InsertNodeAfter(inputNode, mceNode);
                                m_Graph.CollapseEdge(mceNode->GetOutput(0));
                                madeChange = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    } while (madeChange);
}

void Compiler::Prepare()
{
    // This is an iterative process, where we modify the graph as necessary to prepare it for Generation.
    uint32_t numIterations = 0;
    // Set an upper limit for the number of iterations in case we have a bug somewhere.
    // This should not be required because we only keep iterating if we make a change to the graph, and we should
    // only change something if we know it will help. However if we have a bug we may get stuck in a case where we
    // repeatedly modify the graph thinking it will help, but it does not.
    // Note that this limit is set based on the size of the *initial* graph (the graph may grow in size).
    const uint32_t maxIterations = static_cast<uint32_t>(m_Graph.GetNodes().size()) * 10;
    while (true)
    {
        DumpGraph(std::string("GraphPrepareIteration") + std::to_string(numIterations) + "_Pre.dot");

        Optimize();
        CreatePasses();

        DumpGraph(std::string("GraphPrepareIteration") + std::to_string(numIterations) + "_Post.dot");

        if (IsPrepared())
        {
            CreateSections();
            break;
        }

        ++numIterations;

        // Modify graph based on previous attempt. Make a copy as we may add/remove nodes as we fix.
        std::vector<Node*> nodes = m_Graph.GetNodesSorted();
        bool madeChange          = false;    // Record if we were able to make a change to the graph
        // First try making less severe changes and then only escalate to more severe changes if necessary.
        // This prevents making potentially suboptimal changes to the graph that aren't necessary.
        for (FixGraphSeverity severity = FixGraphSeverity::Lowest; severity <= FixGraphSeverity::Highest;
             severity                  = utils::NextEnumValue(severity))
        {
            for (auto& n : nodes)
            {
                madeChange |= n->FixGraph(m_Graph, severity);
                // Note we don't break immedately if a change was made because for large graphs it might be very
                // slow making only one change at a time.
            }
            if (madeChange)
            {
                break;
            }
        }

        if (!madeChange || numIterations > maxIterations)
        {
            std::string errorMsg =
                std::string("Unable to prepare graph after ") + std::to_string(numIterations) + " iterations.";

            errorMsg += "The operation(s) with the following ids have failed to compile:";

            std::vector<uint32_t> failedOpIds;

            // Find which nodes failed and correlate them with the original operations
            for (auto& n : nodes)
            {
                if (!(n->IsPrepared()))
                {
                    for (auto id : n->GetCorrespondingOperationIds())
                    {
                        failedOpIds.push_back(id);
                    }
                }
            }

            // Remove duplicates
            std::sort(failedOpIds.begin(), failedOpIds.end());

            std::vector<uint32_t>::iterator iter =
                std::unique(failedOpIds.begin(), failedOpIds.begin() + failedOpIds.size());

            failedOpIds.resize(std::distance(failedOpIds.begin(), iter));

            for (auto o : failedOpIds)
            {
                errorMsg += " " + std::to_string(o);
            }

            throw NotSupportedException(errorMsg.c_str());
        }

        // Clear passes for next attempt
        m_Passes.clear();
        for (auto& n : m_Graph.GetNodes())
        {
            n->Reset();
        }
    }
}

bool Compiler::IsPrepared()
{
    for (const auto& n : m_Graph.GetNodes())
    {
        if (!n->IsPrepared())
        {
            return false;
        }
    }

    return true;
}

void Compiler::CreatePasses()
{
    std::vector<IStrategy*> strategies = utils::GetRawPointers(m_AllowedStrategies);
    std::vector<Node*> sortedNodes     = m_Graph.GetNodesSorted();
    SramAllocator sramAllocator(m_Capabilities.GetTotalSramSize() / m_Capabilities.GetNumberOfSrams());
    for (Node* n : sortedNodes)
    {
        if (n->GetPass() == nullptr)
        {
            const size_t passId = m_Passes.size();
            std::unique_ptr<Pass> p;
            if (!p)
            {
                p = McePlePass::CreateGreedily(m_Capabilities, passId, strategies, m_AllowedBlockConfigs,
                                               m_EnableIntermediateCompression, !m_DisableWinograd, n, sramAllocator);
            }
            if (!p)
            {
                p = PlePass::CreateGreedily(m_Capabilities, passId, n, sramAllocator);
            }
            if (!p)
            {
                p = ConversionPass::CreateGreedily(m_Capabilities, passId, n, sramAllocator);
            }

            if (p)
            {
                m_Passes.push_back(std::move(p));
            }
            n->PrepareAfterPassAssignment(sramAllocator);
        }
    }
}

void Compiler::CreateSections()
{
    // NNXSW-1221: Implement a search algorithm to partition the network into sections
    // For now, each section will only have one pass (SISO or MISO)
    for (auto& p : m_Passes)
    {
        command_stream::SectionType sectionType = (p->GetNodes().front()->GetInputs().size() > 1)
                                                      ? command_stream::SectionType::MISO
                                                      : command_stream::SectionType::SISO;

        std::string sectionId = std::to_string(m_Sections.size());

        std::unique_ptr<ethosn::support_library::Section> section =
            std::make_unique<Section>(sectionId, sectionType, p.get());

        p->SetSection(section.get());

        m_Sections.push_back(std::move(section));
    }
}

void Compiler::Generate()
{
    std::vector<Node*> sorted = m_Graph.GetNodesSorted();

    // If an initial dump is requested, add the sram dump command at the head of the stream.
    if (m_InitialSramDump)
    {
        ethosn::command_stream::DumpSram cmdStrDumpSram;
        std::string dumpName = std::string("initial_ce");
        std::copy(dumpName.begin(), dumpName.end(), cmdStrDumpSram.m_Filename().begin());
        m_CommandStream.EmplaceBack(cmdStrDumpSram);
    }

    for (Node* n : sorted)
    {
        n->Generate(m_CommandStream, m_BufferManager, m_DumpRam);
    }

    DumpGraph("GraphFinal.dot");

    m_BufferManager.AddCommandStream(m_CommandStream);

    m_BufferManager.Allocate();
}

void Compiler::EstimateCascading(bool current)
{
    if (!current)
    {
        std::vector<PassPerformanceData> perfStream = m_PerformanceStream.m_Stream;
        constexpr double factor                     = 0.2f;

        uint32_t sramFootprint            = 0;
        uint32_t numCascadingNodes        = 0;
        PassPerformanceData* previousNode = nullptr;

        // There are two possible cascading strategies:
        // - Input feature map streaming, only for the first node of the section
        // - Weight streaming while all the input feature maps are stationary
        for (PassPerformanceData& node : perfStream)
        {
            PassStats& current = node.m_Stats;

            sramFootprint += static_cast<uint32_t>(
                (current.m_Input.m_MemoryStats.m_DramParallel + current.m_Input.m_MemoryStats.m_DramNonParallel) *
                factor);
            sramFootprint += static_cast<uint32_t>(current.m_Weights.m_MemoryStats.m_DramParallel +
                                                   current.m_Weights.m_MemoryStats.m_DramNonParallel);

            // This is a sequence of cascade-able nodes.
            if (numCascadingNodes > 0 && previousNode)
            {
                PassStats& previous = previousNode->m_Stats;

                // The current node is not already cascaded with the previous node and the cascaded section fits in
                // Sram.
                if (current.m_Input.m_MemoryStats.m_Sram == 0 && sramFootprint <= m_Capabilities.GetTotalSramSize())
                {
                    // Two or more nodes can be cascaded
                    if (numCascadingNodes == 1)
                    {
                        const uint32_t dramNonParallel = previous.m_Input.m_MemoryStats.m_DramNonParallel;
                        const uint32_t dramParallel    = previous.m_Input.m_MemoryStats.m_DramParallel;

                        // Update inputs statistics
                        previous.m_Input.m_MemoryStats.m_DramNonParallel =
                            static_cast<uint32_t>((dramNonParallel + dramParallel) * factor);
                        previous.m_Input.m_MemoryStats.m_DramParallel =
                            static_cast<uint32_t>((dramNonParallel + dramParallel) * (1 - factor));
                    }
                    else
                    {
                        // Update inputs statistics
                        previous.m_Input.m_MemoryStats.m_Sram = previous.m_Input.m_MemoryStats.m_DramNonParallel +
                                                                previous.m_Input.m_MemoryStats.m_DramParallel;
                        previous.m_Input.m_MemoryStats.m_DramNonParallel = 0;
                        previous.m_Input.m_MemoryStats.m_DramParallel    = 0;
                        // Update weights statistics
                        previous.m_Weights.m_MemoryStats.m_DramParallel =
                            previous.m_Weights.m_MemoryStats.m_DramNonParallel +
                            previous.m_Weights.m_MemoryStats.m_DramParallel;
                        previous.m_Weights.m_MemoryStats.m_DramNonParallel = 0;
                    }

                    // Update outputs statistics
                    previous.m_Output.m_MemoryStats.m_Sram = previous.m_Output.m_MemoryStats.m_DramNonParallel +
                                                             previous.m_Output.m_MemoryStats.m_DramParallel;
                    previous.m_Output.m_MemoryStats.m_DramNonParallel = 0;
                    previous.m_Output.m_MemoryStats.m_DramParallel    = 0;
                    ++numCascadingNodes;
                }
                else
                {
                    // The current node cannot be cascaded with the previous node, update the statistics for the
                    // previous node to account for this.
                    if (previous.m_Input.m_MemoryStats.m_Sram == 0)
                    {
                        // Update inputs statistics
                        previous.m_Input.m_MemoryStats.m_Sram = previous.m_Input.m_MemoryStats.m_DramNonParallel +
                                                                previous.m_Input.m_MemoryStats.m_DramParallel;
                        previous.m_Input.m_MemoryStats.m_DramNonParallel = 0;
                        previous.m_Input.m_MemoryStats.m_DramParallel    = 0;

                        // Update outputs statistics
                        const uint32_t dramNonParallel = previous.m_Output.m_MemoryStats.m_DramNonParallel;
                        const uint32_t dramParallel    = previous.m_Output.m_MemoryStats.m_DramParallel;

                        previous.m_Output.m_MemoryStats.m_DramNonParallel =
                            static_cast<uint32_t>((dramParallel + dramNonParallel) * factor);
                        previous.m_Output.m_MemoryStats.m_DramParallel =
                            static_cast<uint32_t>((dramParallel + dramNonParallel) * (1 - factor));

                        // Update weights statistics
                        previous.m_Weights.m_MemoryStats.m_DramParallel =
                            previous.m_Weights.m_MemoryStats.m_DramNonParallel +
                            previous.m_Weights.m_MemoryStats.m_DramParallel;
                        previous.m_Weights.m_MemoryStats.m_DramNonParallel = 0;
                    }
                    // Check if it can do at least weight streaming
                    else if (current.m_Input.m_MemoryStats.m_Sram != 0)
                    {
                        // Update weights statistics
                        current.m_Weights.m_MemoryStats.m_DramParallel =
                            current.m_Weights.m_MemoryStats.m_DramNonParallel +
                            current.m_Weights.m_MemoryStats.m_DramParallel;
                        current.m_Weights.m_MemoryStats.m_DramNonParallel = 0;
                    }

                    numCascadingNodes = 0;
                    sramFootprint     = 0;
                }
            }
            else
            {
                // This is the first node of a potential section.
                if (numCascadingNodes == 0 && previousNode)
                {
                    PassStats& previous = previousNode->m_Stats;

                    // Check if weight streaming
                    if (previous.m_Input.m_MemoryStats.m_Sram != 0 && current.m_Input.m_MemoryStats.m_Sram != 0)
                    {
                        // Update weights statistics
                        current.m_Weights.m_MemoryStats.m_DramParallel =
                            current.m_Weights.m_MemoryStats.m_DramNonParallel +
                            current.m_Weights.m_MemoryStats.m_DramParallel;
                        current.m_Weights.m_MemoryStats.m_DramNonParallel = 0;
                    }
                }
                ++numCascadingNodes;
            }

            previousNode = &node;
        }

        // It has finished going through all the nodes, update the last node statistics if it has been cascaded.
        if (numCascadingNodes > 0)
        {
            PassStats& previous = previousNode->m_Stats;

            // Update input statistics
            previous.m_Input.m_MemoryStats.m_Sram =
                previous.m_Input.m_MemoryStats.m_DramNonParallel + previous.m_Input.m_MemoryStats.m_DramParallel;

            // Update weights statistics
            previous.m_Weights.m_MemoryStats.m_DramParallel =
                previous.m_Weights.m_MemoryStats.m_DramNonParallel + previous.m_Weights.m_MemoryStats.m_DramParallel;
            previous.m_Weights.m_MemoryStats.m_DramNonParallel = 0;

            // Update outputs statistics
            const uint32_t dramNonParallel = previous.m_Output.m_MemoryStats.m_DramNonParallel;
            const uint32_t dramParallel    = previous.m_Output.m_MemoryStats.m_DramParallel;

            previous.m_Output.m_MemoryStats.m_DramNonParallel =
                static_cast<uint32_t>((dramParallel + dramNonParallel) * factor);
            previous.m_Output.m_MemoryStats.m_DramParallel =
                static_cast<uint32_t>((dramParallel + dramNonParallel) * (1 - factor));
        }

        m_PerformanceStream.m_Stream = perfStream;
    }
}

void Compiler::Estimate()
{
    std::vector<Node*> sorted = m_Graph.GetNodesSorted();

    for (Node* n : sorted)
    {
        if (!n->IsPrepared())
        {
            std::stringstream result;
            for (auto id : n->GetCorrespondingOperationIds())
            {
                result << " " << id;
            }
            std::cerr << "Failed to prepare operation:" << result.str() << "\n";
        }
        n->Estimate(m_PerformanceStream, m_EstimationOptions);
    }

    EstimateCascading(m_EstimationOptions.m_Current);

    DumpGraph("GraphFinal.dot");
}

void Compiler::DumpGraph(const std::string& filename)
{
    if (m_DumpDebugFiles)
    {
        std::ofstream dotStream(m_DebugDir + '/' + filename);
        m_Graph.DumpToDotFormat(dotStream);
    }
}

CompiledNetworkImpl::CompiledNetworkImpl(const std::vector<uint8_t>& constantDmaData,
                                         const std::vector<uint8_t>& constantControlUnitData,
                                         const std::map<uint32_t, CompilerBufferInfo>& buffers,
                                         const std::set<uint32_t>& operationIds)
    : m_ConstantDmaData(constantDmaData)
    , m_ConstantControlUnitData(constantControlUnitData)
    , m_OperationIds(operationIds)
{
    // Convert the set of buffers from the BufferManager into the format that CompiledNetwork exposes.
    for (auto internalBufferIt : buffers)
    {
        uint32_t bufferId = internalBufferIt.first;

        const CompilerBufferInfo& compilerBuffer = internalBufferIt.second;
        if (compilerBuffer.m_Location != BufferLocation::Dram)
        {
            // Sram buffers do not need to be exposed.
            continue;
        }

        BufferInfo buffer(bufferId, compilerBuffer.m_Offset, compilerBuffer.m_Size);
        switch (compilerBuffer.m_Type)
        {
            case BufferType::Input:
            {
                InputBufferInfo inputbuffer(buffer.m_Id, buffer.m_Offset, buffer.m_Size,
                                            compilerBuffer.m_SourceOperationId,
                                            compilerBuffer.m_SourceOperationOutputIndex);
                m_InputBufferInfos.push_back(inputbuffer);
                break;
            }
            case BufferType::Output:
            {
                OutputBufferInfo outputbuffer(bufferId, compilerBuffer.m_Offset, compilerBuffer.m_Size,
                                              compilerBuffer.m_SourceOperationId,
                                              compilerBuffer.m_SourceOperationOutputIndex);
                m_OutputBufferInfos.push_back(outputbuffer);
                break;
            }
            case BufferType::Intermediate:
            {
                m_IntermediateDataBufferInfos.push_back(buffer);
                break;
            }
            case BufferType::ConstantControlUnit:
            {
                m_ConstantControlUnitDataBufferInfos.push_back(buffer);
                break;
            }
            case BufferType::ConstantDma:
            {
                m_ConstantDmaDataBufferInfos.push_back(buffer);
                break;
            }
            default:
                assert(false);
        }
    }
}

uint32_t CompiledNetworkImpl::GetIntermediateDataSize() const
{
    if (m_IntermediateDataBufferInfos.empty())
    {
        return 0;
    }
    auto GetLastBufferAddress = [](const auto& buf) { return buf.m_Offset + buf.m_Size; };
    auto maxBuffer            = std::max_element(
        m_IntermediateDataBufferInfos.begin(), m_IntermediateDataBufferInfos.end(),
        [&](const auto& a, const auto& b) { return GetLastBufferAddress(a) < GetLastBufferAddress(b); });
    return GetLastBufferAddress(*maxBuffer);
}

template <typename T>
void CompiledNetworkImpl::Serialize(std::ostream& out, const std::vector<T>& data) const
{
    static_assert(std::is_trivially_copyable<T>::value, "Type must be trivially copyable");

    size_t size = data.size();
    out.write(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    for (size_t i = 0; i < size; ++i)
    {
        out.write(reinterpret_cast<const char*>(&data[i]), sizeof(T));
    }
}

void CompiledNetworkImpl::Serialize(std::ostream& out) const
{
    // Serialize the library version
    const std::string version = GetLibraryVersion().ToString();
    size_t size               = version.size();
    assert(size < 100);
    out.write(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    out.write(version.c_str(), size);

    // Serialize the vectors
    Serialize(out, m_ConstantDmaData);
    Serialize(out, m_ConstantControlUnitData);
    Serialize(out, m_InputBufferInfos);
    Serialize(out, m_OutputBufferInfos);
    Serialize(out, m_ConstantControlUnitDataBufferInfos);
    Serialize(out, m_ConstantDmaDataBufferInfos);
    Serialize(out, m_IntermediateDataBufferInfos);
}

template <typename T>
void CompiledNetworkImpl::Deserialize(std::istream& in, std::vector<T>& data)
{
    static_assert(std::is_trivially_copyable<T>::value, "Type must be trivially copyable");

    auto size = Read<uint32_t>(in);

    for (uint32_t i = 0; i < size; ++i)
    {
        T item = T();
        char data_buffer[sizeof(T)];
        in.read(data_buffer, sizeof(T));
        std::memcpy(&item, data_buffer, sizeof(T));
        data.emplace_back(item);
    }
}

void CompiledNetworkImpl::Deserialize(std::istream& in)
{
    // Check that input stream was serialized with the same version of the support library
    auto size = Read<uint32_t>(in);
    assert(size < 100);

    char versionString[100];
    in.read(versionString, size);
    Version version(versionString);
    Version libraryVersion = GetLibraryVersion();

    if (libraryVersion.Major != version.Major || libraryVersion.Minor < version.Minor)
    {
        std::stringstream str;
        str << "Compiled Network was serialized with Support Library version " << versionString
            << ". Attempting to de-serialize with version " << GetLibraryVersion().ToString() << std::endl;

        throw VersionMismatchException(str.str().c_str());
    }

    // Deserialize vectors
    Deserialize(in, m_ConstantDmaData);
    Deserialize(in, m_ConstantControlUnitData);
    Deserialize(in, m_InputBufferInfos);
    Deserialize(in, m_OutputBufferInfos);
    Deserialize(in, m_ConstantControlUnitDataBufferInfos);
    Deserialize(in, m_ConstantDmaDataBufferInfos);
    Deserialize(in, m_IntermediateDataBufferInfos);
}

template <typename T>
T CompiledNetworkImpl::Read(std::istream& in)
{
    T data;
    in.read(reinterpret_cast<char*>(&data), sizeof(data));
    return data;
}

}    // namespace support_library
}    // namespace ethosn
