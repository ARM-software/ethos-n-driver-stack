//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "GraphNodes.hpp"

#include "DebuggingContext.hpp"
#include "SramAllocator.hpp"
#include "Utils.hpp"
#include "nonCascading/BufferManager.hpp"
#include "nonCascading/ConversionPass.hpp"
#include "nonCascading/Pass.hpp"

#include <ethosn_command_stream/PleOperation.hpp>
#include <ethosn_utils/Quantization.hpp>

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

void InsertCopyNode(Graph& graph, Edge* edge)
{
    Node* prevNode     = edge->GetSource();
    CopyNode* copyNode = graph.CreateAndAddNodeWithDebug<CopyNode>(
        "InsertCopyNode", prevNode->GetShape(), prevNode->GetDataType(), prevNode->GetQuantizationInfo(),
        prevNode->GetFormat(), prevNode->GetCorrespondingOperationIds());
    graph.SplitEdge(edge, copyNode);
}

bool ContainsPass(Node* node)
{
    auto NodeContainsPass = [](Node* node) -> Node* {
        if (node->GetPass() != nullptr)
        {
            return node;
        }
        return nullptr;
    };
    return SearchDependencies(node, NodeContainsPass) != nullptr;
}

}    // namespace

InputNode::InputNode(NodeId id, const TensorInfo& outputTensorInfo, std::set<uint32_t> correspondingOperationIds)
    : Node(id,
           outputTensorInfo.m_Dimensions,
           outputTensorInfo.m_DataType,
           outputTensorInfo.m_QuantizationInfo,
           ConvertExternalToCompilerDataFormat(outputTensorInfo.m_DataFormat),
           std::move(correspondingOperationIds))
{}

bool InputNode::IsPrepared()
{
    return true;
}

NodeType InputNode::GetNodeType()
{
    return NodeType::InputNode;
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

void InputNode::ResetPreparation()
{
    Node::ResetPreparation();
    m_Location = BufferLocation::Dram;
}

void ConstantNode::PrepareAfterPassAssignment(SramAllocator&)
{
    m_Location = BufferLocation::Dram;
}

bool ConstantNode::IsPrepared()
{
    return true;
}

NodeType ConstantNode::GetNodeType()
{
    return NodeType::ConstantNode;
}

void ConstantNode::Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam)
{
    Node::Generate(cmdStream, bufferManager, dumpRam);

    SetBufferId(bufferManager.AddDramConstant(BufferType::ConstantDma, m_ConstantData));
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
                                   DataType dataType,
                                   const QuantizationInfo& outputQuantizationInfo,
                                   const TensorInfo& weightsInfo,
                                   std::vector<uint8_t> weightsData,
                                   const TensorInfo& biasInfo,
                                   std::vector<int32_t> biasData,
                                   const Stride& stride,
                                   uint32_t padTop,
                                   uint32_t padLeft,
                                   command_stream::MceOperation op,
                                   CompilerDataFormat format,
                                   std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, std::move(correspondingOperationIds))
    , m_UninterleavedInputShape(uninterleavedInputTensorShape)
    , m_WeightsInfo(weightsInfo)
    , m_WeightsData(std::make_shared<std::vector<uint8_t>>(std::move(weightsData)))
    , m_BiasInfo(biasInfo)
    , m_BiasData(std::move(biasData))
    , m_Stride(stride)
    , m_UpscaleFactor(1U)
    , m_UpsampleType(command_stream::UpsampleType::OFF)
    , m_PadTop(padTop)
    , m_PadLeft(padLeft)
    , m_Operation(op)
    , m_Algorithm(CompilerMceAlgorithm::None)
    , m_AlgorithmHint(AlgorithmHint::AllowWinograd)
    , m_FixGraphAlgorithmHint(AlgorithmHint::None)
{}

const ethosn::support_library::TensorShape& MceOperationNode::GetUninterleavedInputShape() const
{
    return m_UninterleavedInputShape;
}

const ethosn::support_library::TensorInfo& MceOperationNode::GetWeightsInfo() const
{
    return m_WeightsInfo;
}

std::shared_ptr<const std::vector<uint8_t>> MceOperationNode::GetWeightsData() const
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

uint32_t MceOperationNode::GetPadTop() const
{
    return m_PadTop;
}

uint32_t MceOperationNode::GetPadLeft() const
{
    return m_PadLeft;
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

ethosn::command_stream::UpsampleType MceOperationNode::GetUpsampleType() const
{
    return m_UpsampleType;
}

void MceOperationNode::SetUpsampleParams(const uint32_t upscaleFactor,
                                         const ethosn::command_stream::UpsampleType upsampleType)
{
    // Check that upscaleFactor and upscaleType are coherent.
    assert((upscaleFactor != 1U) == (upsampleType != ethosn::command_stream::UpsampleType::OFF));
    m_UpscaleFactor = upscaleFactor;
    m_UpsampleType  = upsampleType;
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

CompilerMceAlgorithm MceOperationNode::GetEffectiveAlgorithm(HardwareCapabilities capabilities,
                                                             bool isWinogradEnabled) const
{
    const TensorShape& weightsShape = m_WeightsInfo.m_Dimensions;
    if (GetAlgorithmHint() == AlgorithmHint::AllowWinograd && isWinogradEnabled &&
        GetOperation() == command_stream::MceOperation::CONVOLUTION && GetStride() == Stride{ 1, 1 } &&
        // Winograd and upscaling cannot be performed at the same time
        GetUpsampleType() == command_stream::UpsampleType::OFF)
    {
        return FindBestConvAlgorithm(capabilities, weightsShape[0], weightsShape[1]);
    }

    return CompilerMceAlgorithm::Direct;
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
    result.m_Stride().m_X()    = m_Stride.m_X;
    result.m_Stride().m_Y()    = m_Stride.m_Y;
    result.m_PadTop()          = m_PadTop;
    result.m_PadLeft()         = m_PadLeft;
    result.m_Operation()       = m_Operation;
    result.m_Algorithm()       = ConvertAlgorithmCompilerToCommand(m_Algorithm);
    result.m_OutputZeroPoint() = static_cast<int16_t>(m_QuantizationInfo.GetZeroPoint());
    return result;
}

bool MceOperationNode::IsPrepared()
{
    return m_Pass != nullptr;
}

NodeType MceOperationNode::GetNodeType()
{
    return NodeType::MceOperationNode;
}

DotAttributes MceOperationNode::GetDotAttributes()
{
    DotAttributes result    = Node::GetDotAttributes();
    std::string labelPrefix = "MceOperationNode\n";
    labelPrefix += ToString(m_Operation) + "\n";
    labelPrefix += ToString(m_Algorithm) + "\n";
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

void MceOperationNode::ResetPreparation()
{
    Node::ResetPreparation();
    m_Algorithm = CompilerMceAlgorithm::None;
}

ShapeMultiplier MceOperationNode::GetShapeMultiplier() const
{
    return { m_UpscaleFactor, m_UpscaleFactor, 1 };
}

McePostProcessOperationNode::McePostProcessOperationNode(NodeId id,
                                                         const TensorShape& outputTensorShape,
                                                         DataType dataType,
                                                         const QuantizationInfo& outputQuantizationInfo,
                                                         int16_t lowerBound,
                                                         int16_t upperBound,
                                                         CompilerDataFormat format,
                                                         std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, std::move(correspondingOperationIds))
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

NodeType McePostProcessOperationNode::GetNodeType()
{
    return NodeType::McePostProcessOperationNode;
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
                                                   DataType dataType,
                                                   const QuantizationInfo& outputQuantizationInfo,
                                                   command_stream::PleOperation k,
                                                   CompilerDataFormat format,
                                                   ShapeMultiplier shapeMultiplier,
                                                   std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, correspondingOperationIds)
    , m_KernelOperation(k)
    , m_InsertIdentityNodeHint(false)
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
    return op == PleOperation::DOWNSAMPLE_2X2 || op == PleOperation::INTERLEAVE_2X2_2_2 ||
           op == PleOperation::MAXPOOL_2X2_2_2 || op == PleOperation::MAXPOOL_3X3_2_2_EVEN ||
           op == PleOperation::MAXPOOL_3X3_2_2_ODD || op == PleOperation::MEAN_XY_7X7 ||
           op == PleOperation::MEAN_XY_8X8 || op == PleOperation::PASSTHROUGH || op == PleOperation::TRANSPOSE_XY;
}

bool FuseOnlyPleOperationNode::IsPrepared()
{
    return m_Pass != nullptr;
}

NodeType FuseOnlyPleOperationNode::GetNodeType()
{
    return NodeType::FuseOnlyPleOperationNode;
}

void FuseOnlyPleOperationNode::SetFixGraphInsertIdentityNodeHint(bool isIdentityNode)
{
    m_InsertIdentityNodeHint = isIdentityNode;
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
    // be assigned to. In this case make an identity convolution node. We might also need to insert identity depthwise
    // if a deep convolution followed by MaxPool 3x3 and the ifm will be splitted in width, or a transpose XY PLE
    // operation to avoid it being split into stripes (transpose does not support multiple stripes).
    // Please check the comment in McePlePass.cpp

    if (m_Pass == nullptr &&
        (m_InsertIdentityNodeHint || dynamic_cast<MceOperationNode*>(GetInput(0)->GetSource()) == nullptr ||
         GetInput(0)->GetSource()->GetOutputs().size() > 1 ||
         (severity == FixGraphSeverity::High && m_KernelOperation == command_stream::PleOperation::TRANSPOSE_XY)))
    {
        InsertIdentityNode(graph, GetInput(0));
        changed                  = true;
        m_InsertIdentityNodeHint = false;
    }
    return changed;
}

void FuseOnlyPleOperationNode::SetOperationSpecificData(command_stream::McePle&) const
{}

LeakyReluNode::LeakyReluNode(NodeId id,
                             const TensorShape& outputTensorShape,
                             DataType dataType,
                             const QuantizationInfo& outputQuantizationInfo,
                             command_stream::PleOperation k,
                             CompilerDataFormat format,
                             ShapeMultiplier shapeMultiplier,
                             std::set<uint32_t> correspondingOperationIds,
                             float alpha)
    : FuseOnlyPleOperationNode(id,
                               outputTensorShape,
                               dataType,
                               outputQuantizationInfo,
                               k,
                               format,
                               shapeMultiplier,
                               correspondingOperationIds)
    , m_Alpha(alpha)
{}

float LeakyReluNode::GetAlpha() const
{
    return m_Alpha;
}

void LeakyReluNode::SetOperationSpecificData(command_stream::McePle& data) const
{
    const QuantizationInfo outQuantInfo = m_QuantizationInfo;
    const QuantizationInfo inQuantInfo  = GetInputQuantizationInfo(0);

    const double alphaRescaleFactor = m_Alpha * (inQuantInfo.GetScale() / outQuantInfo.GetScale());
    uint16_t alphaMult;
    uint16_t alphaShift;
    CalculateRescaleMultiplierAndShift(alphaRescaleFactor, alphaMult, alphaShift);

    const double inputToOutputRescaleFactor = (inQuantInfo.GetScale() / outQuantInfo.GetScale());
    uint16_t inputToOutputMult;
    uint16_t inputToOutputShift;
    CalculateRescaleMultiplierAndShift(inputToOutputRescaleFactor, inputToOutputMult, inputToOutputShift);

    data.m_PleData().m_RescaleMultiplier0() = inputToOutputMult;
    data.m_PleData().m_RescaleShift0()      = inputToOutputShift;

    data.m_PleData().m_RescaleMultiplier1() = alphaMult;
    data.m_PleData().m_RescaleShift1()      = alphaShift;
}

StandalonePleOperationNode::StandalonePleOperationNode(NodeId id,
                                                       const TensorShape& outputTensorShape,
                                                       DataType dataType,
                                                       const QuantizationInfo& outputQuantizationInfo,
                                                       command_stream::PleOperation k,
                                                       CompilerDataFormat format,
                                                       std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, std::move(correspondingOperationIds))
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

NodeType StandalonePleOperationNode::GetNodeType()
{
    return NodeType::StandalonePleOperationNode;
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
                                           DataType dataType,
                                           const QuantizationInfo& outputQuantizationInfo,
                                           CompilerDataFormat format,
                                           std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, std::move(correspondingOperationIds))
{}

bool FormatConversionNode::IsPrepared()
{
    return m_Pass != nullptr;
}

NodeType FormatConversionNode::GetNodeType()
{
    return NodeType::FormatConversionNode;
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

    // If we couldn't be assigned to a pass and our input is in FCAF format, then try forcing it to uncompressed
    // as ConversionPass doesn't support FCAF and this change might allow that to work now.
    if (m_Pass == nullptr && GetInputCompressed(0) &&
        (GetInputCompressedFormat(0) == CompilerDataCompressedFormat::FCAF_DEEP ||
         GetInputCompressedFormat(0) == CompilerDataCompressedFormat::FCAF_WIDE))
    {
        GetInput(0)->GetSource()->SetCompressionHint(CompressionHint::RequiredUncompressed);
        changed = true;
    }

    // A format conversion node using NCHW is for transpose operation.
    // If it couldn't be assigned into a pass then it may be because the convolution node before it needs
    // multi-stripe operation that is not currenntly supported for transpose.
    // Inserting a identity node when this happens so that the identity node and the format conversion node
    // will be assigned into a McePle pass. In this way, the input tensor to the McePle pass will be the
    // same as the one to the transpose operation, which in turn allows the support query to reject the
    // transpose operation that cannot avoid multipe-stripe.
    if (severity == FixGraphSeverity::High && m_Pass == nullptr && GetFormat() == CompilerDataFormat::NCHW)
    {
        InsertIdentityNode(graph, GetInput(0));
        changed = true;
    }
    return changed;
}

SpaceToDepthNode::SpaceToDepthNode(NodeId id,
                                   const TensorShape& outputTensorShape,
                                   DataType dataType,
                                   const QuantizationInfo& outputQuantizationInfo,
                                   CompilerDataFormat format,
                                   std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, std::move(correspondingOperationIds))
{}

bool SpaceToDepthNode::IsPrepared()
{
    return m_Pass != nullptr;
}

NodeType SpaceToDepthNode::GetNodeType()
{
    return NodeType::SpaceToDepthNode;
}

ReinterpretNode::ReinterpretNode(NodeId id,
                                 const TensorShape& outputTensorShape,
                                 DataType dataType,
                                 const QuantizationInfo& outputQuantizationInfo,
                                 CompilerDataFormat format,
                                 std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, std::move(correspondingOperationIds))
{}

bool ReinterpretNode::IsPrepared()
{
    // Currently, the input to ReinterpretNode must be uncompressed because
    // output quantization info used for ReinterpretQuantization is used
    // from user-provided Network. This information is based on uncompressed data.
    // But if the data is compressed its zero point changes which results
    // into wrong results as the original input quant info might not be same as the
    // compressed input quant info which can result into wrong output generation
    // by ReinterpretNode.
    // Therefore, it has to be ensured that compression and decomression happen
    // with same zero point.
    if (GetInput(0)->GetSource()->GetCompressed())
    {
        return false;
    }
    return true;
}

NodeType ReinterpretNode::GetNodeType()
{
    return NodeType::ReinterpretNode;
}

void ReinterpretNode::Generate(command_stream::CommandStreamBuffer& cmdStream,
                               BufferManager& bufferManager,
                               bool dumpRam)
{
    Node::Generate(cmdStream, bufferManager, dumpRam);

    if (!m_Pass)
    {
        uint32_t bufferId = GetInput(0)->GetSource()->GetBufferId();

        // Setting the same compression format as the input because
        // this extra information is essential to comprehend the
        // input data in correct compressed format.
        // Although, currently, we don't support compressed input
        // to a ReinterpretNode.
        SetCompressedFormat(GetInputCompressedFormat(0));

        // Map this node's output buffer to the same as its input
        SetBufferId(bufferId);

        // If this is a node that reinterprets NHWC to NHWCB,
        // then re-aligned the buffer size to 1k (1024) boundary.
        if (GetBufferFormat() == command_stream::DataFormat::NHWCB &&
            GetInput(0)->GetSource()->GetBufferFormat() == command_stream::DataFormat::NHWC)
        {
            bufferManager.ChangeBufferAlignment(bufferId, g_NhwcbBufferAlignment);
        }
    }
}

bool ReinterpretNode::FixGraph(Graph& graph, FixGraphSeverity severity)
{
    bool changed = Node::FixGraph(graph, severity);

    if (GetInput(0)->GetSource()->GetCompressionHint() != CompressionHint::RequiredUncompressed)
    {
        // This sets the hints for previous node such that the ReinterpretNode always receives
        // uncompressed inputs.
        GetInput(0)->GetSource()->SetCompressionHint(CompressionHint::RequiredUncompressed);
        changed = true;
    }

    return changed;
}

DotAttributes ReinterpretNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "ReinterpretNode\n" + result.m_Label;
    return result;
}

void ReinterpretNode::PrepareAfterPassAssignment(SramAllocator& sramAllocator)
{
    if (m_Pass == nullptr)
    {
        const BufferLocation& bufferLocation = GetInputLocation(0);
        if (bufferLocation == BufferLocation::Sram)
        {
            sramAllocator.IncrementReferenceCount(m_Id, GetInputSramOffset(0));
            SetOutputSramOffset(GetInputSramOffset(0));
        }
        // This is called if there is no pass for us. Necessary so future passes can see our location.
        // If we are in a pass then the pass will handle this for us.
        SetLocation(bufferLocation);
    }

    // Call the parent function's implementation after the node had the chance to increment the SRAM reference count.
    Node::PrepareAfterPassAssignment(sramAllocator);
}

ConcatNode::ConcatNode(NodeId id,
                       const TensorShape& outputTensorShape,
                       DataType dataType,
                       const QuantizationInfo& outputQuantizationInfo,
                       CompilerDataFormat format,
                       uint32_t axis,
                       std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, std::move(correspondingOperationIds))
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
    }
    return true;
}

NodeType ConcatNode::GetNodeType()
{
    return NodeType::ConcatNode;
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
    SetBufferId(bufferId);

    if (dumpRam)
    {
        // Add dump especially for this concat node, otherwise we just get partial dumps from the input subtensors
        ethosn::command_stream::DumpDram cmdStrDumpDram =
            utils::GetDumpDramCommand(GetShape(), GetBufferId(), GetDataType(), GetQuantizationInfo().GetZeroPoint(),
                                      ToString(GetBufferFormat()).c_str());
        cmdStream.EmplaceBack(cmdStrDumpDram);
    }
}

void ConcatNode::PrepareAfterPassAssignment(SramAllocator& sramAllocator)
{
    Node::PrepareAfterPassAssignment(sramAllocator);
    SetLocation(BufferLocation::Dram);
}

ExtractSubtensorNode::ExtractSubtensorNode(NodeId id,
                                           const TensorShape& supertensorOffset,
                                           const TensorShape& outputTensorShape,
                                           DataType dataType,
                                           const QuantizationInfo& outputQuantizationInfo,
                                           CompilerDataFormat format,
                                           std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, std::move(correspondingOperationIds))
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

NodeType ExtractSubtensorNode::GetNodeType()
{
    return NodeType::ExtractSubtensorNode;
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
            Node* reformat = graph.CreateAndAddNodeWithDebug<FormatConversionNode>(
                "ExtractSubtensorNode identity conv format fixup", identityNode->GetShape(),
                identityNode->GetDataType(), identityNode->GetQuantizationInfo(), GetFormat(),
                GetCorrespondingOperationIds());
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

NodeType SoftmaxNode::GetNodeType()
{
    return NodeType::SoftmaxNode;
}

SoftmaxNode::SoftmaxNode(NodeId id,
                         const TensorShape& outputTensorShape,
                         DataType dataType,
                         const QuantizationInfo& outputQuantizationInfo,
                         CompilerDataFormat format,
                         std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, std::move(correspondingOperationIds))
{}

CopyNode::CopyNode(NodeId id,
                   const TensorShape& outputTensorShape,
                   DataType dataType,
                   const QuantizationInfo& outputQuantizationInfo,
                   CompilerDataFormat format,
                   std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, std::move(correspondingOperationIds))
{}

bool CopyNode::IsPrepared()
{
    return m_Pass != nullptr;
}

NodeType CopyNode::GetNodeType()
{
    return NodeType::CopyNode;
}

bool CopyNode::FixGraph(Graph& graph, FixGraphSeverity severity)
{
    bool changed = Node::FixGraph(graph, severity);

    // We don't support a ConversionPass that goes from Sram into Dram, so we may need to force our input
    // back to Dram in order for a pass to be created.
    if (m_Pass == nullptr && GetInputLocation(0) == BufferLocation::Sram)
    {
        GetInput(0)->GetSource()->SetLocationHint(LocationHint::RequireDram);
        changed = true;
    }
    return changed;
}

DotAttributes CopyNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "CopyNode\n" + result.m_Label;
    return result;
}

RequantizeNode::RequantizeNode(NodeId id,
                               const TensorShape& outputTensorShape,
                               DataType dataType,
                               const QuantizationInfo& outputQuantizationInfo,
                               CompilerDataFormat format,
                               std::set<uint32_t> correspondingOperationIds)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, std::move(correspondingOperationIds))
{}

bool RequantizeNode::IsPrepared()
{
    return m_Pass != nullptr;
}

NodeType RequantizeNode::GetNodeType()
{
    return NodeType::RequantizeNode;
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
    float dequantizedMin = ethosn::utils::Dequantize(mceData.m_ActivationMin(), inputQuantizationInfo.GetScale(),
                                                     inputQuantizationInfo.GetZeroPoint());
    float dequantizedMax = ethosn::utils::Dequantize(mceData.m_ActivationMax(), inputQuantizationInfo.GetScale(),
                                                     inputQuantizationInfo.GetZeroPoint());

    switch (m_DataType)
    {
        case DataType::UINT8_QUANTIZED:
            mceData.m_ActivationMin() = ethosn::utils::Quantize<uint8_t>(dequantizedMin, m_QuantizationInfo.GetScale(),
                                                                         m_QuantizationInfo.GetZeroPoint());
            mceData.m_ActivationMax() = ethosn::utils::Quantize<uint8_t>(dequantizedMax, m_QuantizationInfo.GetScale(),
                                                                         m_QuantizationInfo.GetZeroPoint());
            break;
        case DataType::INT8_QUANTIZED:
            mceData.m_ActivationMin() = ethosn::utils::Quantize<int8_t>(dequantizedMin, m_QuantizationInfo.GetScale(),
                                                                        m_QuantizationInfo.GetZeroPoint());
            mceData.m_ActivationMax() = ethosn::utils::Quantize<int8_t>(dequantizedMax, m_QuantizationInfo.GetScale(),
                                                                        m_QuantizationInfo.GetZeroPoint());
            break;
        default:
            ETHOSN_FAIL_MSG("Not implemented");
    }
}

bool OutputNode::IsPrepared()
{
    if (GetInputLocation(0) != BufferLocation::Dram)
    {
        return false;
    }
    if (GetInputCompressed(0))
    {
        return false;
    }
    // The input to an output node cannot be used as both an intermediate and output buffer.
    if (GetInput(0)->GetSource()->GetOutputs().size() != 1)
    {
        return false;
    }
    // Walk the graph to the inputs, a path with at least one pass in it is required
    // If there isn't one, it means an input goes straight to an output
    // which would make the input buffer the same as the output buffer, which is not supported by our API.
    if (!ContainsPass(this))
    {
        return false;
    }
    return true;
}

NodeType OutputNode::GetNodeType()
{
    return NodeType::OutputNode;
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
    if (severity == FixGraphSeverity::High)
    {
        // Walk the graph to the inputs, a path with at least one pass in it is required If there isn't one, it means an
        // input goes straight to an output which would make the input buffer the same as the output buffer, which is
        // not supported by our API.
        // Another case that isn't supported is when the input to the output node is used by another node because a
        // buffer cannot both be an intermediate and output buffer at the same time.
        //
        // Both these cases are handled by inserting a copy node so that the input and output uses different buffers.
        //
        // This counts as a more severe change because adding an extra node to the graph may be suboptimal in the case
        // that other fixes to the graph are possible. For example the preceding node may be able to fix the graph itself.
        if (!ContainsPass(this) || GetInput(0)->GetSource()->GetOutputs().size() != 1)
        {
            InsertCopyNode(graph, GetInput(0));
            changed = true;
        }
    }
    return changed;
}

void OutputNode::Generate(command_stream::CommandStreamBuffer&, BufferManager& bufferManager, bool)
{
    // Modify output buffer descriptor to be an output
    uint32_t bufferId = GetInput(0)->GetSource()->GetBufferId();

    assert(bufferManager.GetBuffers().at(bufferId).m_Type == BufferType::Intermediate);

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
                                   DataType dataType,
                                   const QuantizationInfo& outputQuantizationInfo,
                                   CompilerDataFormat format,
                                   std::set<uint32_t> correspondingOperationIds,
                                   const char* reasons)
    : Node(id, outputTensorShape, dataType, outputQuantizationInfo, format, std::move(correspondingOperationIds))
{
    assert(reasons != nullptr);

    m_ReasonForEstimateOnly = reasons;

    if (m_ReasonForEstimateOnly.size() == 0)
    {
        g_Logger.Warning("Reason is missing for estimate only node");
        m_ReasonForEstimateOnly.assign("Unknown.");
    }

    if (m_ReasonForEstimateOnly.back() != '.')
    {
        m_ReasonForEstimateOnly += ".";
    }
}

bool EstimateOnlyNode::IsPrepared()
{
    return false;
}

NodeType EstimateOnlyNode::GetNodeType()
{
    return NodeType::EstimateOnlyNode;
}

void EstimateOnlyNode::Estimate(NetworkPerformanceData& perfData, const EstimationOptions&)
{
    for (const auto it : GetCorrespondingOperationIds())
    {
        perfData.m_OperationIdFailureReasons.emplace(
            it, "Could not be estimated and has zero performance impact. Reason: " + m_ReasonForEstimateOnly);
    }
}

DotAttributes EstimateOnlyNode::GetDotAttributes()
{
    DotAttributes result = Node::GetDotAttributes();
    result.m_Label       = "EstimateOnlyNode\n" + result.m_Label;
    return result;
}

MceOperationNode* CreateIdentityMceOpNode(Graph& graph, Node* previousNode)
{
    const uint32_t numIfm   = previousNode->GetShape()[3];
    const float weightScale = g_IdentityWeightScale;
    const float biasScale   = weightScale * previousNode->GetQuantizationInfo().GetScale();

    std::vector<uint8_t> weightsData(numIfm, g_IdentityWeightValue);
    std::vector<int32_t> biasData(numIfm, 0);

    TensorInfo weightInfo{ { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale } };
    TensorInfo biasInfo{ { 1, 1, 1, numIfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };

    MceOperationNode* result = graph.CreateAndAddNodeWithDebug<MceOperationNode>(
        "CreateIdentityMceOpNode", previousNode->GetShape(), previousNode->GetShape(), previousNode->GetDataType(),
        previousNode->GetQuantizationInfo(), weightInfo, weightsData, biasInfo, biasData, Stride{ 1, 1 }, 0, 0,
        ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION, CompilerDataFormat::NHWCB,
        previousNode->GetCorrespondingOperationIds());

    return result;
}

void InsertIdentityNode(Graph& graph, Edge* edge)
{
    MceOperationNode* convNode = CreateIdentityMceOpNode(graph, edge->GetSource());
    graph.SplitEdge(edge, convNode);
}

}    // namespace support_library
}    // namespace ethosn
