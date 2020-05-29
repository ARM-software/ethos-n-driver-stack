//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "GraphNodes.hpp"

#include "BufferManager.hpp"
#include "ConversionPass.hpp"
#include "McePlePass.hpp"
#include "Pass.hpp"
#include "Utils.hpp"

#include <ethosn_command_stream/PleOperation.hpp>

using namespace ethosn::support_library::utils;

namespace ethosn
{
namespace support_library
{

namespace
{

command_stream::MceAlgorithm ConvertAlgorithmCompilerToCommand(CompilerMceAlgorithm algorithm)
{
    if (algorithm == CompilerMceAlgorithm::Direct)
    {
        return command_stream::MceAlgorithm::DIRECT;
    }
    else if (algorithm == CompilerMceAlgorithm::Winograd)
    {
        return command_stream::MceAlgorithm::WINOGRAD;
    }
    else
    {
        assert(false);
        return command_stream::MceAlgorithm::DIRECT;
    }
}

MceOperationNode* CreateIdentityMceOpNode(Graph& graph, Node* previousNode)
{
    const uint32_t numIfm   = previousNode->GetShape()[3];
    const float weightScale = 0.5f;
    const float biasScale   = weightScale * previousNode->GetQuantizationInfo().m_Scale;

    std::vector<uint8_t> weightsData(1 * 1 * 1 * numIfm, 2);
    std::vector<int32_t> biasData(numIfm, 0);

    TensorInfo weightInfo{ { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale } };
    TensorInfo biasInfo{ { 1, 1, 1, numIfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };

    MceOperationNode* result = graph.CreateAndAddNode<MceOperationNode>(
        previousNode->GetShape(), previousNode->GetShape(), previousNode->GetQuantizationInfo(), weightInfo,
        weightsData, biasInfo, biasData, Stride{ 1, 1 }, 1, 0, 0,
        ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION, CompilerDataFormat::NHWCB,
        previousNode->GetCorrespondingOperationIds());

    return result;
}

void InsertIdentityNode(Graph& graph, Edge* edge)
{
    MceOperationNode* convNode = CreateIdentityMceOpNode(graph, edge->GetSource());
    graph.SplitEdge(edge, convNode);
}

}    // namespace

InputNode::InputNode(NodeId id, const TensorInfo& outputTensorInfo, std::set<uint32_t> correspondingOperationIds)
    : Node(id,
           outputTensorInfo.m_Dimensions,
           outputTensorInfo.m_QuantizationInfo,
           ConvertExternalToCompilerDataFormat(outputTensorInfo.m_DataFormat),
           correspondingOperationIds)
{
    Reset();
}

bool InputNode::IsPrepared()
{
    return true;
}

void InputNode::Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam)
{
    Node::Generate(cmdStream, bufferManager, dumpRam);

    // Calculate buffer size based on input format
    const uint32_t inputSize = CalculateBufferSize(GetShape(), GetBufferFormat());

    // The InputNode can only ever be associated with one input network operation.
    assert(m_CorrespondingOperationIds.size() == 1);

    SetBufferId(bufferManager.AddDramInput(inputSize, *m_CorrespondingOperationIds.begin()));
}

DotAttributes InputNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "InputNode\n" + result.m_Label;
    return result;
}

void InputNode::Reset()
{
    m_Location = BufferLocation::Dram;
}

bool ConstantNode::IsPrepared()
{
    // Constant can only be merged with other nodes or removed from the graph if unconnected.
    return false;
}

DotAttributes ConstantNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "ConstantNode\n" + result.m_Label;
    return result;
}

const std::vector<uint8_t>& ConstantNode::GetConstantData() const
{
    return m_ConstantData;
}

const DataType& ConstantNode::GetConstantDataType() const
{
    return m_ConstantDataType;
}

MceOperationNode::MceOperationNode(NodeId id,
                                   const TensorShape& uninterleavedInputTensorShape,
                                   const TensorShape& outputTensorShape,
                                   const QuantizationInfo& outputQuantizationInfo,
                                   const TensorInfo& weightsInfo,
                                   std::vector<uint8_t> weightsData,
                                   const TensorInfo& biasInfo,
                                   std::vector<int32_t> biasData,
                                   Stride stride,
                                   uint32_t upscaleFactor,
                                   uint32_t padTop,
                                   uint32_t padLeft,
                                   command_stream::MceOperation op,
                                   CompilerDataFormat format,
                                   std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, outputQuantizationInfo, format, correspondingOperationIds)
    , m_UninterleavedInputShape(uninterleavedInputTensorShape)
    , m_WeightsInfo(weightsInfo)
    , m_WeightsData(std::move(weightsData))
    , m_BiasInfo(biasInfo)
    , m_BiasData(std::move(biasData))
    , m_Stride(stride)
    , m_UpscaleFactor(upscaleFactor)
    , m_PadTop(padTop)
    , m_PadLeft(padLeft)
    , m_Operation(op)
    , m_AlgorithmHint(AlgorithmHint::AllowWinograd)
    , m_FixGraphAlgorithmHint(AlgorithmHint::None)
{
    Reset();
}

const ethosn::support_library::TensorShape& MceOperationNode::GetUninterleavedInputShape() const
{
    return m_UninterleavedInputShape;
}

const ethosn::support_library::TensorInfo& MceOperationNode::GetWeightsInfo() const
{
    return m_WeightsInfo;
}

const std::vector<uint8_t>& MceOperationNode::GetWeightsData() const
{
    return m_WeightsData;
}

const ethosn::support_library::TensorInfo& MceOperationNode::GetBiasInfo() const
{
    return m_BiasInfo;
}

const std::vector<int32_t>& MceOperationNode::GetBiasData() const
{
    return m_BiasData;
}

Stride MceOperationNode::GetStride() const
{
    return m_Stride;
}

void MceOperationNode::SetStride(Stride s)
{
    m_Stride = s;
}

uint32_t MceOperationNode::GetUpscaleFactor() const
{
    return m_UpscaleFactor;
}

void MceOperationNode::SetUpscaleFactor(uint32_t upscaleFactor)
{
    m_UpscaleFactor = upscaleFactor;
}

ethosn::command_stream::MceOperation MceOperationNode::GetOperation() const
{
    return m_Operation;
}

void MceOperationNode::SetOperation(ethosn::command_stream::MceOperation op)
{
    m_Operation = op;
}

void MceOperationNode::SetAlgorithm(CompilerMceAlgorithm a)
{
    m_Algorithm = a;
}

CompilerMceAlgorithm MceOperationNode::GetAlgorithm() const
{
    return m_Algorithm;
}

void MceOperationNode::SetAlgorithmHint(AlgorithmHint a)
{
    m_AlgorithmHint = a;
}

AlgorithmHint MceOperationNode::GetAlgorithmHint() const
{
    return m_AlgorithmHint;
}

void MceOperationNode::SetFixGraphAlgorithmHint(AlgorithmHint a)
{
    m_FixGraphAlgorithmHint = a;
}

AlgorithmHint MceOperationNode::GetFixGraphAlgorithmHint() const
{
    return m_FixGraphAlgorithmHint;
}

ethosn::command_stream::MceData MceOperationNode::GetMceData() const
{
    ethosn::command_stream::MceData result;
    result.m_Stride().m_X()            = m_Stride.m_X;
    result.m_Stride().m_Y()            = m_Stride.m_Y;
    result.m_PadTop()                  = m_PadTop;
    result.m_PadLeft()                 = m_PadLeft;
    result.m_Operation()               = m_Operation;
    result.m_Algorithm()               = ConvertAlgorithmCompilerToCommand(m_Algorithm);
    result.m_OutputZeroPoint()         = static_cast<int16_t>(m_QuantizationInfo.m_ZeroPoint);
    result.m_OutputRescaleMultiplier() = 1U;
    result.m_OutputRescaleShift()      = 0U;
    return result;
}

bool MceOperationNode::IsPrepared()
{
    return m_Pass != nullptr;
}

DotAttributes MceOperationNode::GetDotAttributes()
{
    DotAttributes result    = Node::GetDotAttributes();
    std::string labelPrefix = "MceOperationNode\n";
    labelPrefix += ToString(m_Operation) + "\n";
    switch (m_Algorithm)
    {
        case CompilerMceAlgorithm::None:
            labelPrefix += "NONE\n";
            break;
        case CompilerMceAlgorithm::Direct:
            labelPrefix += "DIRECT\n";
            break;
        case CompilerMceAlgorithm::Winograd:
            labelPrefix += "WINOGRAD\n";
            break;
        default:
            assert(false);
    }
    result.m_Label = labelPrefix + result.m_Label;
    return result;
}

bool MceOperationNode::FixGraph(Graph& graph, FixGraphSeverity severity)
{
    bool changed = Node::FixGraph(graph, severity);
    if (m_Pass == nullptr && GetFixGraphAlgorithmHint() != AlgorithmHint::None &&
        GetAlgorithmHint() != GetFixGraphAlgorithmHint())
    {
        SetAlgorithmHint(AlgorithmHint::RequireDirect);
        SetFixGraphAlgorithmHint(AlgorithmHint::None);
        changed = true;
    }
    return changed;
}

void MceOperationNode::Reset()
{
    Node::Reset();
    m_Algorithm = CompilerMceAlgorithm::None;
}

utils::ShapeMultiplier MceOperationNode::GetShapeMultiplier() const
{
    return { m_UpscaleFactor, m_UpscaleFactor, 1 };
}

McePostProcessOperationNode::McePostProcessOperationNode(NodeId id,
                                                         const TensorShape& outputTensorShape,
                                                         const QuantizationInfo& outputQuantizationInfo,
                                                         uint8_t lowerBound,
                                                         uint8_t upperBound,
                                                         CompilerDataFormat format,
                                                         std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, outputQuantizationInfo, format, correspondingOperationIds)
    , m_LowerBound(lowerBound)
    , m_UpperBound(upperBound)
{}

void McePostProcessOperationNode::Apply(ethosn::command_stream::MceData& mceData) const
{
    mceData.m_ActivationMin() = std::max(mceData.m_ActivationMin(), m_LowerBound);
    mceData.m_ActivationMax() = std::min(mceData.m_ActivationMax(), m_UpperBound);
}

bool McePostProcessOperationNode::IsPrepared()
{
    return m_Pass != nullptr;
}

DotAttributes McePostProcessOperationNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "McePostProcessOperationNode\n" + result.m_Label;
    return result;
}

bool McePostProcessOperationNode::FixGraph(Graph& graph, FixGraphSeverity severity)
{
    bool changed = Node::FixGraph(graph, severity);
    // If we couldn't be assigned into a pass then it may be because there is no convolution node before for us to
    // be assigned to. In this case make an identity convolution node.
    if (m_Pass == nullptr && (dynamic_cast<MceOperationNode*>(GetInput(0)->GetSource()) == nullptr ||
                              GetInput(0)->GetSource()->GetOutputs().size() > 1))
    {
        InsertIdentityNode(graph, GetInput(0));
        changed = true;
    }
    return changed;
}

FuseOnlyPleOperationNode::FuseOnlyPleOperationNode(NodeId id,
                                                   const TensorShape& outputTensorShape,
                                                   const QuantizationInfo& outputQuantizationInfo,
                                                   command_stream::PleOperation k,
                                                   CompilerDataFormat format,
                                                   ShapeMultiplier shapeMultiplier,
                                                   std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, outputQuantizationInfo, format, correspondingOperationIds)
    , m_KernelOperation(k)
    , m_ShapeMultiplier(shapeMultiplier)
{}

command_stream::PleOperation FuseOnlyPleOperationNode::GetKernelOperation() const
{
    return m_KernelOperation;
}

bool FuseOnlyPleOperationNode::IsAgnosticToRequantisation() const
{
    using namespace command_stream;
    PleOperation op = GetKernelOperation();
    return op == PleOperation::MAXPOOL_2X2_2_2 || op == PleOperation::INTERLEAVE_2X2_2_2 ||
           op == PleOperation::MAXPOOL_3X3_2_2_EVEN || op == PleOperation::MAXPOOL_3X3_2_2_ODD ||
           op == PleOperation::MEAN_XY_7X7 || op == PleOperation::MEAN_XY_8X8 || op == PleOperation::PASSTHROUGH;
}

bool FuseOnlyPleOperationNode::IsPrepared()
{
    return m_Pass != nullptr;
}

DotAttributes FuseOnlyPleOperationNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "FuseOnlyPleOperationNode\n" + result.m_Label;
    return result;
}

bool FuseOnlyPleOperationNode::FixGraph(Graph& graph, FixGraphSeverity severity)
{
    bool changed = Node::FixGraph(graph, severity);
    // If we couldn't be assigned into a pass then it may be because there is no convolution node before for us to
    // be assigned to. In this case make an identity convolution node.
    if (m_Pass == nullptr && (dynamic_cast<MceOperationNode*>(GetInput(0)->GetSource()) == nullptr ||
                              GetInput(0)->GetSource()->GetOutputs().size() > 1))
    {
        InsertIdentityNode(graph, GetInput(0));
        changed = true;
    }
    return changed;
}

StandalonePleOperationNode::StandalonePleOperationNode(NodeId id,
                                                       const TensorShape& outputTensorShape,
                                                       const QuantizationInfo& outputQuantizationInfo,
                                                       command_stream::PleOperation k,
                                                       CompilerDataFormat format,
                                                       std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, outputQuantizationInfo, format, correspondingOperationIds)
    , m_KernelOperation(k)
{}

command_stream::PleOperation StandalonePleOperationNode::GetKernelOperation() const
{
    return m_KernelOperation;
}

bool StandalonePleOperationNode::IsPrepared()
{
    return m_Pass != nullptr;
}

DotAttributes StandalonePleOperationNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "StandalonePleOperationNode\n" + result.m_Label;
    return result;
}

bool StandalonePleOperationNode::FixGraph(Graph& graph, FixGraphSeverity severity)
{
    bool changed = Node::FixGraph(graph, severity);
    if (m_Pass == nullptr && GetInputs().size() > 1)
    {
        for (uint32_t i = 0; i < GetInputs().size(); ++i)
        {
            if (GetInput(i)->GetSource()->GetLocationHint() != LocationHint::RequireDram)
            {
                GetInput(i)->GetSource()->SetLocationHint(LocationHint::RequireDram);
                changed = true;
            }
        }
    }
    return changed;
}

FormatConversionNode::FormatConversionNode(NodeId id,
                                           const TensorShape& outputTensorShape,
                                           const QuantizationInfo& outputQuantizationInfo,
                                           CompilerDataFormat format,
                                           std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, outputQuantizationInfo, format, correspondingOperationIds)
{}

bool FormatConversionNode::IsPrepared()
{
    return m_Pass != nullptr &&
           (!GetInputCompressed(0) || (GetInputCompressedFormat(0) != CompilerDataFormat::FCAF_DEEP &&
                                       GetInputCompressedFormat(0) != CompilerDataFormat::FCAF_WIDE));
}

DotAttributes FormatConversionNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "FormatConversionNode\n" + result.m_Label;
    return result;
}

bool FormatConversionNode::FixGraph(Graph& graph, FixGraphSeverity severity)
{
    bool changed = Node::FixGraph(graph, severity);
    if (m_Pass == nullptr && GetInput(0)->GetSource()->GetLocationHint() != LocationHint::RequireDram)
    {
        // Try forcing our input into DRAM (e.g. If reshape is last layer and the preceding McePlePass gets left in SRAM)
        GetInput(0)->GetSource()->SetLocationHint(LocationHint::RequireDram);
        changed = true;
    }

    // Conversion nodes does not support FCAF formats as input
    if (GetInputCompressed(0) && (GetInputCompressedFormat(0) == CompilerDataFormat::FCAF_DEEP ||
                                  GetInputCompressedFormat(0) == CompilerDataFormat::FCAF_WIDE))
    {
        GetInput(0)->GetSource()->SetCompressionHint(CompressionHint::RequiredUncompressed);
        changed = true;
    }
    return changed;
}

ReinterpretNode::ReinterpretNode(NodeId id,
                                 const TensorShape& outputTensorShape,
                                 const QuantizationInfo& outputQuantizationInfo,
                                 CompilerDataFormat format,
                                 std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, outputQuantizationInfo, format, correspondingOperationIds)
{}

bool ReinterpretNode::IsPrepared()
{
    return true;
}

void ReinterpretNode::Generate(command_stream::CommandStreamBuffer& cmdStream,
                               BufferManager& bufferManager,
                               bool dumpRam)
{
    Node::Generate(cmdStream, bufferManager, dumpRam);

    if (!m_Pass)
    {
        // Map this node's output buffer to the same as its input
        SetBufferId(GetInput(0)->GetSource()->GetBufferId());
    }
}

DotAttributes ReinterpretNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "ReinterpretNode\n" + result.m_Label;
    return result;
}

void ReinterpretNode::PrepareAfterPassAssignment(SramAllocator& sramAllocator)
{
    Node::PrepareAfterPassAssignment(sramAllocator);
    if (m_Pass == nullptr)
    {
        // This is called if there is no pass for us. Necessary so future passes can see our location.
        // If we are in a pass then the pass will handle this for us.
        SetLocation(GetInputLocation(0));
    }
}

ConcatNode::ConcatNode(NodeId id,
                       const TensorShape& outputTensorShape,
                       const QuantizationInfo& outputQuantizationInfo,
                       CompilerDataFormat format,
                       uint32_t axis,
                       std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, outputQuantizationInfo, format, correspondingOperationIds)
    , m_Axis(axis)
{}

bool ConcatNode::IsPrepared()
{
    for (uint32_t i = 0; i < GetInputs().size(); ++i)
    {
        // Concat inputs are required to be in DRAM
        if (GetInput(i)->GetSource()->GetLocation() != BufferLocation::Dram)
        {
            return false;
        }
        // Concat inputs are required to be uncompressed. This is because the data written into
        // the supertensor may not be the full width and depth. Ideally we would perform this check
        // in the same place as the existing compression checks but the information about supertensors is
        // not available at that point.
        if (GetInput(i)->GetSource()->GetCompressed())
        {
            return false;
        }
        // Concats are handled by the preceding Passes writing directly into the concat output buffer.
        // Therefore all our inputs need to be in a pass that supports this, which is currently just McePlePasses
        if (dynamic_cast<McePlePass*>(GetInput(i)->GetSource()->GetPass()) == nullptr &&
            dynamic_cast<ConversionPass*>(GetInput(i)->GetSource()->GetPass()) == nullptr)
        {
            return false;
        }
    }
    return true;
}

DotAttributes ConcatNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "ConcatNode\n" + result.m_Label;
    return result;
}

bool ConcatNode::FixGraph(Graph& graph, FixGraphSeverity severity)
{
    bool changed = Node::FixGraph(graph, severity);
    for (uint32_t i = 0; i < GetInputs().size(); ++i)
    {
        if (GetInput(i)->GetSource()->GetLocationHint() != LocationHint::RequireDram)
        {
            GetInput(i)->GetSource()->SetLocationHint(LocationHint::RequireDram);
            changed = true;
        }
        // See IsPrepared() above for explanation
        if (GetInput(i)->GetSource()->GetCompressionHint() != CompressionHint::RequiredUncompressed)
        {
            GetInput(i)->GetSource()->SetCompressionHint(CompressionHint::RequiredUncompressed);
            changed = true;
        }
        // See IsPrepared for context.
        // We can force an McePlePass pass to be created on our input by adding a convolution there.
        // This counts as a more severe change because adding an extra node to the graph may be suboptimal in the case
        // that other fixes to the graph are possible. For example the preceding node may be able to fix the graph itself.
        const bool mceOperationRequired =
            dynamic_cast<McePlePass*>(GetInput(i)->GetSource()->GetPass()) == nullptr &&
            dynamic_cast<ConversionPass*>(GetInput(i)->GetSource()->GetPass()) == nullptr &&
            // Make sure that it's not adding another Identity node for every iteration.
            dynamic_cast<MceOperationNode*>(GetInput(i)->GetSource()) == nullptr;

        if (severity == FixGraphSeverity::High && mceOperationRequired)
        {
            InsertIdentityNode(graph, GetInput(i));
            if (GetFormat() == CompilerDataFormat::NHWC)
            {
                // Set the location hint of the Identity Node to be in DRAM
                // If it chooses put the output in SRAM we cannot fuse the format conversion.
                GetInput(i)->GetSource()->SetLocationHint(LocationHint::RequireDram);
                Node* reformat = graph.CreateAndAddNode<FormatConversionNode>(
                    GetInputShape(i), GetInputQuantizationInfo(i), GetFormat(), GetCorrespondingOperationIds());
                graph.SplitEdge(GetInput(i), reformat);
            }
            changed = true;
        }
    }
    return changed;
}

void ConcatNode::Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam)
{
    Node::Generate(cmdStream, bufferManager, dumpRam);
    uint32_t bufferId = GetInput(0)->GetSource()->GetBufferId();
    for (uint32_t i = 0; i < GetInputs().size(); ++i)
    {
        assert(bufferId == GetInput(i)->GetSource()->GetBufferId());
        assert(m_Format == GetInputFormat(i));
    }
    SetBufferId(GetInput(0)->GetSource()->GetBufferId());
}

void ConcatNode::PrepareAfterPassAssignment(SramAllocator& sramAllocator)
{
    Node::PrepareAfterPassAssignment(sramAllocator);
    SetLocation(BufferLocation::Dram);
}

ExtractSubtensorNode::ExtractSubtensorNode(NodeId id,
                                           const TensorShape& supertensorOffset,
                                           const TensorShape& outputTensorShape,
                                           const QuantizationInfo& outputQuantizationInfo,
                                           CompilerDataFormat format,
                                           std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, outputQuantizationInfo, format, correspondingOperationIds)
    , m_SupertensorOffset(supertensorOffset)
{}

DotAttributes ExtractSubtensorNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "ExtractSubtensorNode\n" + result.m_Label;
    return result;
}

bool ExtractSubtensorNode::IsPrepared()
{
    return m_Pass != nullptr;
}

bool ExtractSubtensorNode::FixGraph(Graph& graph, FixGraphSeverity)
{
    // It may be that the we cannot be placed into an McePlePass, so if there isn't one directly after us
    // then add an identity depthwise!
    bool hasSingleOutputToMceOperation =
        GetOutputs().size() == 1 && dynamic_cast<MceOperationNode*>(GetOutput(0)->GetDestination()) != nullptr;
    if (m_Pass == nullptr && !hasSingleOutputToMceOperation)
    {
        MceOperationNode* identityNode = CreateIdentityMceOpNode(graph, this);
        graph.InsertNodeAfter(this, identityNode);

        // May need to convert back to the format we were originally outputting in order not to inadvertently change
        // the meaning of the graph.
        if (identityNode->GetFormat() != GetFormat())
        {
            Node* reformat = graph.CreateAndAddNode<FormatConversionNode>(identityNode->GetShape(),
                                                                          identityNode->GetQuantizationInfo(),
                                                                          GetFormat(), GetCorrespondingOperationIds());
            graph.InsertNodeAfter(identityNode, reformat);
        }
    }
    return false;
}

TensorShape ExtractSubtensorNode::GetSupertensorOffset()
{
    return m_SupertensorOffset;
}

bool SoftmaxNode::IsPrepared()
{
    return false;
}

SoftmaxNode::SoftmaxNode(NodeId id,
                         const TensorShape& outputTensorShape,
                         const QuantizationInfo& outputQuantizationInfo,
                         CompilerDataFormat format,
                         std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, outputQuantizationInfo, format, correspondingOperationIds)
{}

RequantizeNode::RequantizeNode(NodeId id,
                               const TensorShape& outputTensorShape,
                               const QuantizationInfo& outputQuantizationInfo,
                               CompilerDataFormat format,
                               std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, outputQuantizationInfo, format, correspondingOperationIds)
{}

bool RequantizeNode::IsPrepared()
{
    return m_Pass != nullptr;
}

bool RequantizeNode::FixGraph(Graph& graph, FixGraphSeverity severity)
{
    bool changed = Node::FixGraph(graph, severity);
    // If we couldn't be assigned into a pass then it may be because there is no convolution node before for us to
    // be assigned to. In this case make an identity convolution node.
    // This counts as a more severe change because adding an extra node to the graph may be suboptimal in the case
    // that other fixes to the graph are possible. For example the preceding node may be able to fix the graph itself.
    if (severity == FixGraphSeverity::High && m_Pass == nullptr &&
        (dynamic_cast<MceOperationNode*>(GetInput(0)->GetSource()) == nullptr ||
         GetInput(0)->GetSource()->GetOutputs().size() > 1))
    {
        InsertIdentityNode(graph, GetInput(0));
        changed = true;
    }
    return changed;
}

DotAttributes RequantizeNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "RequantizeNode\n" + result.m_Label;
    return result;
}

void RequantizeNode::Apply(ethosn::command_stream::MceData& mceData,
                           const QuantizationInfo& inputQuantizationInfo) const
{
    // Dequantize then requantize the upper and lower bounds
    float dequantizedMin = static_cast<float>(mceData.m_ActivationMin() - inputQuantizationInfo.m_ZeroPoint) *
                           inputQuantizationInfo.m_Scale;
    float dequantizedMax = static_cast<float>(mceData.m_ActivationMax() - inputQuantizationInfo.m_ZeroPoint) *
                           inputQuantizationInfo.m_Scale;

    float requantizedMin =
        (dequantizedMin / m_QuantizationInfo.m_Scale) + static_cast<float>(m_QuantizationInfo.m_ZeroPoint);
    float requantizedMax =
        (dequantizedMax / m_QuantizationInfo.m_Scale) + static_cast<float>(m_QuantizationInfo.m_ZeroPoint);

    constexpr auto max = std::numeric_limits<uint8_t>::max();
    constexpr auto min = std::numeric_limits<uint8_t>::min();

    float clampedQuantizedMin = utils::Clamp(requantizedMin, static_cast<float>(min), static_cast<float>(max));
    float clampedQuantizedMax = utils::Clamp(requantizedMax, static_cast<float>(min), static_cast<float>(max));

    mceData.m_ActivationMin() = static_cast<uint8_t>(clampedQuantizedMin);
    mceData.m_ActivationMax() = static_cast<uint8_t>(clampedQuantizedMax);
}

bool OutputNode::IsPrepared()
{
    return GetInputLocation(0) == BufferLocation::Dram && !GetInputCompressed(0);
}

bool OutputNode::FixGraph(Graph& graph, FixGraphSeverity severity)
{
    bool changed = Node::FixGraph(graph, severity);
    if (GetInput(0)->GetSource()->GetLocationHint() != LocationHint::RequireDram)
    {
        GetInput(0)->GetSource()->SetLocationHint(LocationHint::RequireDram);
        changed = true;
    }
    if (GetInput(0)->GetSource()->GetCompressionHint() != CompressionHint::RequiredUncompressed)
    {
        GetInput(0)->GetSource()->SetCompressionHint(CompressionHint::RequiredUncompressed);
        changed = true;
    }

    return changed;
}

void OutputNode::Generate(command_stream::CommandStreamBuffer&, BufferManager& bufferManager, bool)
{
    // Modify output buffer descriptor to be an output
    uint32_t bufferId = GetInput(0)->GetSource()->GetBufferId();

    if (bufferManager.GetBuffers().at(bufferId).m_Type == BufferType::Input)
    {
        throw NotSupportedException(std::string("Unable to change input buffer to output buffer").c_str());
    }

    // The OutputNode can only ever be associated with one input network operation.
    assert(m_CorrespondingOperationIds.size() == 1);

    bufferManager.ChangeToOutput(bufferId, *m_CorrespondingOperationIds.begin(), m_SourceOperationOutputIndex);
}

DotAttributes OutputNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "OutputNode\n" + result.m_Label;
    return result;
}

EstimateOnlyNode::EstimateOnlyNode(NodeId id,
                                   const TensorShape& outputTensorShape,
                                   const QuantizationInfo& outputQuantizationInfo,
                                   CompilerDataFormat format,
                                   std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, outputQuantizationInfo, format, correspondingOperationIds)
{}

bool EstimateOnlyNode::IsPrepared()
{
    return false;
}

void EstimateOnlyNode::Estimate(NetworkPerformanceData& perfData, const EstimationOptions&)
{
    for (const auto it : GetCorrespondingOperationIds())
    {
        perfData.m_OperationIdFailureReasons.emplace(
            it, "Could not be estimated: Please provide a mapping file entry for this operation");
    }
}

DotAttributes EstimateOnlyNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "EstimateOnlyNode\n" + result.m_Label;
    return result;
}

}    // namespace support_library
}    // namespace ethosn
