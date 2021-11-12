//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "PartV1.hpp"

#include "../Graph.hpp"
#include "../Utils.hpp"
#include "CapabilitiesInternal.hpp"
#include "GraphNodes.hpp"
#include "InputPart.hpp"
#include "OutputPart.hpp"
#include "Plan.hpp"
#include "ReshapePart.hpp"
#include "WeightEncoder.hpp"
#include "WeightEncoderCache.hpp"

#include <unordered_map>

using namespace std;
using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{

using namespace utils;

namespace
{
uint32_t GetWeightStripeDepth(const TensorInfo& weightInfo, MceOp* mceOp)
{
    if (weightInfo.m_DataFormat == DataFormat::HWIO)
    {
        return mceOp->m_WeightsStripeShape[3];
    }
    else if (weightInfo.m_DataFormat == DataFormat::HWIM)
    {
        return mceOp->m_WeightsStripeShape[2] * mceOp->m_WeightsStripeShape[3] /
               (mceOp->m_Stride.m_X * mceOp->m_Stride.m_Y);
    }
    else
    {
        assert(false);
        return 0;
    }
}

/// Generates a stripe shape given an encoding and an input tensor
/// Tries to create a stripe with the stripe shape in the encoding, if the dimension is 0 then it uses the full length of that dimension.
TensorShape CreateStripe(TensorShape input, TensorShape inputEncoding, const HardwareCapabilities& caps)
{
    TensorShape inputStripeShape;
    for (uint32_t i = 0; i < input.size(); ++i)
    {
        inputStripeShape[i] = inputEncoding[i] != 0 ? inputEncoding[i] : input[i];
        inputStripeShape[i] = std::min(inputStripeShape[i], input[i]);
    }
    inputStripeShape    = utils::RoundUpHeightAndWidthToBrickGroup(inputStripeShape);
    inputStripeShape[3] = utils::RoundUpToNearestMultiple(inputStripeShape[3], caps.GetNumberOfSrams());
    return inputStripeShape;
}

}    // namespace

bool PartV1::MceStripesInfo::operator<(const MceStripesInfo& rhs) const
{
    if (m_Input < rhs.m_Input)
        return true;
    if (rhs.m_Input < m_Input)
        return false;
    if (m_Output < rhs.m_Output)
        return true;
    if (rhs.m_Output < m_Output)
        return false;
    if (m_Weight < rhs.m_Weight)
        return true;
    if (rhs.m_Weight < m_Weight)
        return false;
    if (m_BlockConfig.m_BlockWidth() < rhs.m_BlockConfig.m_BlockWidth())
        return true;
    if (rhs.m_BlockConfig.m_BlockWidth() < m_BlockConfig.m_BlockWidth())
        return false;
    if (m_BlockConfig.m_BlockHeight() < rhs.m_BlockConfig.m_BlockHeight())
        return true;
    if (rhs.m_BlockConfig.m_BlockHeight() < m_BlockConfig.m_BlockHeight())
        return false;
    return false;
}

bool PartV1::PleStripesInfo::operator<(const PleStripesInfo& rhs) const
{
    if (m_Input < rhs.m_Input)
        return true;
    if (rhs.m_Input < m_Input)
        return false;
    if (m_Output < rhs.m_Output)
        return true;
    if (rhs.m_Output < m_Output)
        return false;
    if (m_BlockConfig.m_BlockWidth() < rhs.m_BlockConfig.m_BlockWidth())
        return true;
    if (rhs.m_BlockConfig.m_BlockWidth() < m_BlockConfig.m_BlockWidth())
        return false;
    if (m_BlockConfig.m_BlockHeight() < rhs.m_BlockConfig.m_BlockHeight())
        return true;
    if (rhs.m_BlockConfig.m_BlockHeight() < m_BlockConfig.m_BlockHeight())
        return false;
    return false;
}

bool PartV1::MceAndPleInfo::operator<(const MceAndPleInfo& rhs) const
{
    if (m_MceCompute < rhs.m_MceCompute)
        return true;
    if (rhs.m_MceCompute < m_MceCompute)
        return false;
    if (m_PleCompute < rhs.m_PleCompute)
        return true;
    if (rhs.m_PleCompute < m_PleCompute)
        return false;
    if (m_Memory < rhs.m_Memory)
        return true;
    if (rhs.m_Memory < m_Memory)
        return false;
    return false;
}

bool PartV1::MceOnlyInfo::operator<(const MceOnlyInfo& rhs) const
{
    if (m_MceCompute < rhs.m_MceCompute)
        return true;
    if (rhs.m_MceCompute < m_MceCompute)
        return false;
    if (m_Memory < rhs.m_Memory)
        return true;
    if (rhs.m_Memory < m_Memory)
        return false;
    return false;
}

bool PartV1::PleOnlyInfo::operator<(const PleOnlyInfo& rhs) const
{
    if (m_PleCompute < rhs.m_PleCompute)
        return true;
    if (rhs.m_PleCompute < m_PleCompute)
        return false;
    if (m_Memory < rhs.m_Memory)
        return true;
    if (rhs.m_Memory < m_Memory)
        return false;
    return false;
}

std::unique_ptr<Op> CreateOpFromNode(const Node* node,
                                     const BlockConfig& blockConfig,
                                     const CompilationOptions& compOpt,
                                     const HardwareCapabilities& caps)
{
    const MceOperationNode* mceOperationNode = dynamic_cast<const MceOperationNode*>(node);
    const McePostProcessOperationNode* mcePostProcessOperationNode =
        dynamic_cast<const McePostProcessOperationNode*>(node);
    const FuseOnlyPleOperationNode* fuseOnlyPleOperationNode = dynamic_cast<const FuseOnlyPleOperationNode*>(node);
    const StandalonePleOperationNode* standalonePleOperationNode =
        dynamic_cast<const StandalonePleOperationNode*>(node);
    const FormatConversionNode* formatConversionNode = dynamic_cast<const FormatConversionNode*>(node);
    const EstimateOnlyNode* estimateOnlyNode         = dynamic_cast<const EstimateOnlyNode*>(node);
    const ReinterpretNode* reinterpretNode           = dynamic_cast<const ReinterpretNode*>(node);

    if (mceOperationNode)
    {
        uint32_t kernelHeight   = mceOperationNode->GetWeightsInfo().m_Dimensions[0];
        uint32_t kernelWidth    = mceOperationNode->GetWeightsInfo().m_Dimensions[1];
        const bool isWinograd2d = (kernelHeight > 1) && (kernelWidth > 1);
        const CompilerMceAlgorithm effectiveAlgo =
            mceOperationNode->GetEffectiveAlgorithm(caps, !compOpt.m_DisableWinograd);

        std::vector<command_stream::BlockConfig> res =
            FilterAlgoBlockConfigs(effectiveAlgo, isWinograd2d, { blockConfig }, caps);
        const CompilerMceAlgorithm mceOpAlgo = res.empty() ? CompilerMceAlgorithm::Direct : effectiveAlgo;

        MceOp op(Lifetime::Cascade, mceOperationNode->GetOperation(), mceOpAlgo, blockConfig, TensorShape{},
                 TensorShape{}, TensorShape{}, TraversalOrder::Xyz, mceOperationNode->GetStride(),
                 mceOperationNode->GetPadLeft(), mceOperationNode->GetPadTop());
        return std::make_unique<MceOp>(std::move(op));
    }
    else if (mcePostProcessOperationNode)
    {
        return std::make_unique<MceOp>();
    }
    else if (fuseOnlyPleOperationNode)
    {
        PleOp op(Lifetime::Cascade, fuseOnlyPleOperationNode->GetKernelOperation(), blockConfig,
                 static_cast<uint32_t>(fuseOnlyPleOperationNode->GetInputs().size()), std::vector<TensorShape>{},
                 TensorShape{});
        return std::make_unique<PleOp>(std::move(op));
    }
    else if (standalonePleOperationNode)
    {
        PleOp op(Lifetime::Cascade, standalonePleOperationNode->GetKernelOperation(), BlockConfig{ 16U, 16U },
                 static_cast<uint32_t>(standalonePleOperationNode->GetInputs().size()), std::vector<TensorShape>{},
                 TensorShape{});
        return std::make_unique<PleOp>(std::move(op));
    }
    else if (formatConversionNode)
    {
        return std::make_unique<DmaOp>();
    }
    else if (estimateOnlyNode || reinterpretNode)
    {
        return std::make_unique<DummyOp>();
    }

    std::cout
        << "Warning: Unsupported node type received during the plan generation. A dummy operation will be inserted."
        << std::endl;
    return std::make_unique<DummyOp>();
}

int GetStripePosition(TraversalOrder order)
{
    switch (order)
    {
        case TraversalOrder::Xyz:
            return 1;
        case TraversalOrder::Zxy:
            return 3;
        default:
            throw NotSupportedException("Unknown traversal order");
    }
}

TensorShape GetShapeRoundedToBrickGroup(TensorShape shape)
{
    shape    = utils::RoundUpHeightAndWidthToBrickGroup(shape);
    shape[3] = utils::RoundUpToNearestMultiple(shape[3], 16);
    return shape;
}

TensorInfo GetWeightsInfo(const Node* node)
{
    const MceOperationNode* mceOpNode = dynamic_cast<const MceOperationNode*>(node);
    if (mceOpNode)
    {
        return mceOpNode->GetWeightsInfo();
    }

    return TensorInfo();
}

TensorShape GetWeightsShape(const Node* node)
{
    return GetWeightsInfo(node).m_Dimensions;
}

std::vector<const Edge*> PartV1::GetInputs() const
{
    assert(m_SubGraph.size());
    std::vector<const Edge*> result;

    for (uint32_t n = 0; n < m_SubGraph.size(); ++n)
    {
        bool found        = false;
        const Node& nodeA = *m_SubGraph.at(n);
        for (uint32_t i = 0; i < nodeA.GetInputs().size(); ++i)
        {
            const Edge* in = nodeA.GetInput(i);
            for (uint32_t m = 0; m < m_SubGraph.size(); ++m)
            {
                if (m == n)
                {
                    continue;
                }
                const Node& nodeB = *m_SubGraph.at(m);
                for (uint32_t o = 0; o < nodeB.GetOutputs().size(); ++o)
                {
                    const Edge* out = nodeB.GetOutput(o);
                    if (in == out)
                    {
                        found = true;
                        break;
                    }
                    found = false;
                }
                if (found)
                {
                    break;
                }
            }
            if (!found)
            {
                result.push_back(in);
            }
        }
    }
    return result;
}

std::vector<const Edge*> PartV1::GetOutputs() const
{
    assert(m_SubGraph.size());
    std::vector<const Edge*> result;

    for (uint32_t n = 0; n < m_SubGraph.size(); ++n)
    {
        bool found        = false;
        const Node& nodeA = *m_SubGraph.at(n);
        for (uint32_t o = 0; o < nodeA.GetOutputs().size(); ++o)
        {
            const Edge* out = nodeA.GetOutput(o);
            for (uint32_t m = 0; m < m_SubGraph.size(); ++m)
            {
                if (m == n)
                {
                    continue;
                }
                const Node& nodeB = *m_SubGraph.at(m);
                for (uint32_t i = 0; i < nodeB.GetInputs().size(); ++i)
                {
                    const Edge* in = nodeB.GetInput(i);
                    if (in == out)
                    {
                        found = true;
                        break;
                    }
                    found = false;
                }
                if (found)
                {
                    break;
                }
            }
            if (!found)
            {
                result.push_back(out);
            }
        }
    }
    return result;
}

Plans PartV1::GetPlans(CascadeType cascadeType,
                       BlockConfig blockConfig,
                       Buffer* sramBuffer,
                       uint32_t numWeightStripes) const
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    HardwareCapabilities hwCapabilities(GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    Node* node = m_SubGraph.front();
    Plans plans;

    if (IsObjectOfType<InputNode>(node))
    {
        InputPart inputPart(m_PartId, node->GetShape(), node->GetFormat(), node->GetQuantizationInfo(),
                            node->GetCorrespondingOperationIds(), estOpt, compOpt, hwCapabilities);
        plans = inputPart.GetPlans(cascadeType, blockConfig, sramBuffer, numWeightStripes);
    }
    else if (IsObjectOfType<OutputNode>(node))
    {
        TensorShape shape                     = node->GetInputs()[0]->GetSourceShape();
        CompilerDataFormat compilerDataFormat = node->GetInputs()[0]->GetSource()->GetFormat();
        QuantizationInfo quantizationInfo     = node->GetInputs()[0]->GetSource()->GetQuantizationInfo();
        OutputPart outputPart(m_PartId, shape, compilerDataFormat, quantizationInfo,
                              node->GetCorrespondingOperationIds(), estOpt, compOpt, hwCapabilities);
        plans = outputPart.GetPlans(cascadeType, blockConfig, sramBuffer, numWeightStripes);
    }
    else
    {
        WeightEncoderCache weightEncoderCache{ m_Capabilities };
        GenerateWithTraversalOrders(cascadeType, sramBuffer, numWeightStripes, node, weightEncoderCache, plans);
    }

    // Add operation ids
    std::set<uint32_t> opIds = node->GetCorrespondingOperationIds();
    for (auto&& plan : plans)
    {
        for (auto&& op : plan->m_OpGraph.GetOps())
        {
            op->m_OperationIds.insert(opIds.begin(), opIds.end());
        }
    }

    return plans;
}

void AddWeightBuffersAndDmaOpToMceOp(OwnedOpGraph& opGraph,
                                     Lifetime lifetime,
                                     const PartV1::MceStripesInfo& mceComputeInfo,
                                     const impl::NumStripesType& numMemoryWeightStripes,
                                     const TensorShape& memoryWeightStripe,
                                     TraversalOrder order,
                                     const impl::ConvData& convData,
                                     WeightEncoderCache& weightEncoderCache)
{
    const OpGraph::BufferList& buffers = opGraph.GetBuffers();
    const OpGraph::OpList& ops         = opGraph.GetOps();
    Op* op                             = ops.front();
    MceOp* mceOp                       = dynamic_cast<MceOp*>(op);

    if (!mceOp)
    {
        throw InternalErrorException("MceOp is NULL.");
    }

    CascadingBufferFormat formatInDram = impl::GetCascadingBufferFormatFromCompilerDataFormat(
        ConvertExternalToCompilerDataFormat(convData.weightInfo.m_DataFormat));
    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Dram, formatInDram, order));
    Buffer* weightsBufferInDram        = buffers.back();
    weightsBufferInDram->m_TensorShape = convData.weightInfo.m_Dimensions;
    weightsBufferInDram->m_StripeShape = memoryWeightStripe;

    CascadingBufferFormat formatInSram =
        impl::GetCascadingBufferFormatFromCompilerDataFormat(CompilerDataFormat::WEIGHT);
    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, formatInSram, order));
    Buffer* weightsBufferInSram             = buffers.back();
    weightsBufferInSram->m_TensorShape      = weightsBufferInDram->m_TensorShape;
    weightsBufferInSram->m_StripeShape      = memoryWeightStripe;
    weightsBufferInSram->m_QuantizationInfo = convData.weightInfo.m_QuantizationInfo;
    weightsBufferInSram->m_NumStripes       = numMemoryWeightStripes;

    opGraph.AddOp(std::make_unique<DmaOp>());
    Op* dmaOp                   = ops.back();
    mceOp->m_InputStripeShape   = mceComputeInfo.m_Input;
    mceOp->m_OutputStripeShape  = mceComputeInfo.m_Output;
    mceOp->m_WeightsStripeShape = weightsBufferInSram->m_StripeShape;

    opGraph.AddConsumer(weightsBufferInDram, dmaOp, 0);
    opGraph.SetProducer(weightsBufferInSram, dmaOp);
    opGraph.AddConsumer(weightsBufferInSram, op, 1);

    // Encode weights
    const uint32_t weightStripeSize  = mceOp->m_WeightsStripeShape[2];
    const uint32_t weightStripeDepth = GetWeightStripeDepth(convData.weightInfo, mceOp);

    // Encoder doesn't support multiple iterations with Winograd enabled
    if (weightStripeSize < convData.weightInfo.m_Dimensions[2])
    {
        mceOp->m_Algo = CompilerMceAlgorithm::Direct;
    }

    Buffer* mceOutput = opGraph.GetOutput(mceOp);
    Buffer* mceInput  = opGraph.GetInputs(mceOp)[0];

    WeightEncoderCache::Params wp;
    wp.weightsTensorInfo                  = convData.weightInfo;
    wp.weightsData                        = convData.weightData;
    wp.biasTensorInfo                     = convData.biasInfo;
    wp.biasData                           = convData.biasData;
    wp.inputQuantizationInfo              = mceInput->m_QuantizationInfo;
    wp.outputQuantizationInfo             = mceOutput->m_QuantizationInfo;
    wp.stripeDepth                        = weightStripeDepth;
    wp.strideY                            = mceOp->m_Stride.m_Y;
    wp.strideX                            = mceOp->m_Stride.m_X;
    wp.paddingTop                         = mceOp->m_PadTop;
    wp.paddingLeft                        = mceOp->m_PadLeft;
    wp.iterationSize                      = weightStripeSize;
    wp.operation                          = mceOp->m_Op;
    wp.algorithm                          = mceOp->m_Algo;
    weightsBufferInDram->m_EncodedWeights = weightEncoderCache.Encode(wp);

    // Use the encoded weights to determine the size of the sram and dram buffers
    weightsBufferInDram->m_SizeInBytes = static_cast<uint32_t>(weightsBufferInDram->m_EncodedWeights->m_Data.size());
    weightsBufferInSram->m_SizeInBytes = weightsBufferInDram->m_EncodedWeights->m_MaxSize * numMemoryWeightStripes;
}

std::pair<Buffer*, Buffer*> PartV1::AddIdentityMceOpForSubGraph(OwnedOpGraph& opGraph,
                                                                Lifetime lifetime,
                                                                const PartV1::MceStripesInfo& mceComputeInfo,
                                                                const impl::NumMemoryStripes& numMemoryStripes,
                                                                const impl::MemoryStripesInfo& memoryStripes,
                                                                const TensorShape& inpShape,
                                                                const QuantizationInfo& inpQuantInfo,
                                                                TraversalOrder order,
                                                                WeightEncoderCache& weightEncoderCache) const
{
    const OpGraph::BufferList& buffers = opGraph.GetBuffers();
    const OpGraph::OpList& ops         = opGraph.GetOps();

    const float weightScale = 0.5f;
    const float biasScale   = weightScale * inpQuantInfo.GetScale();
    const uint32_t numIfm   = inpShape[3];

    TensorInfo weightInfo{ { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale } };
    TensorInfo biasInfo{ { 1, 1, 1, numIfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };

    std::shared_ptr<std::vector<uint8_t>> weightsData = std::make_shared<std::vector<uint8_t>>(1 * 1 * 1 * numIfm, 2);
    std::vector<int32_t> biasData(numIfm, 0);

    // Add MceOp.
    opGraph.AddOp(std::make_unique<MceOp>(Lifetime::Cascade, MceOperation::DEPTHWISE_CONVOLUTION,
                                          CompilerMceAlgorithm::Direct, mceComputeInfo.m_BlockConfig,
                                          mceComputeInfo.m_Input, mceComputeInfo.m_Output, mceComputeInfo.m_Weight,
                                          order, Stride(1, 1), 0, 0));
    Op* idMceOp = ops.back();

    // Add input Buffer.
    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, CascadingBufferFormat::NHWCB, order));
    Buffer* idMceOpInBuff = buffers.back();

    // Add Output Buffer.
    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::PleInputSram, CascadingBufferFormat::NHWCB, order));
    Buffer* idMceOpOutBuff = buffers.back();

    opGraph.AddConsumer(idMceOpInBuff, idMceOp, 0);
    opGraph.SetProducer(idMceOpOutBuff, idMceOp);

    // Add Weight buffers and DmaOp.
    impl::ConvData convData;
    convData.weightInfo = weightInfo;
    convData.weightData = weightsData;
    convData.biasInfo   = biasInfo;
    convData.biasData   = std::move(biasData);
    AddWeightBuffersAndDmaOpToMceOp(opGraph, lifetime, mceComputeInfo, numMemoryStripes.m_Weight,
                                    memoryStripes.m_Weight.m_Shape, order, convData, weightEncoderCache);

    // Set Input & Output buffer shapes and sizes.
    idMceOpOutBuff->m_TensorShape = inpShape;
    idMceOpInBuff->m_TensorShape  = inpShape;
    idMceOpOutBuff->m_StripeShape = memoryStripes.m_PleInput.m_Shape;
    idMceOpInBuff->m_StripeShape  = memoryStripes.m_Input.m_Shape;
    idMceOpOutBuff->m_SizeInBytes = 0;    // The output buffer is in ple sram so has no size in the tile
    idMceOpInBuff->m_SizeInBytes =
        impl::CalculateTileSize(m_Capabilities, inpShape, idMceOpInBuff->m_StripeShape, numMemoryStripes.m_Input);
    idMceOpOutBuff->m_QuantizationInfo = inpQuantInfo;
    idMceOpInBuff->m_QuantizationInfo  = inpQuantInfo;
    idMceOpOutBuff->m_NumStripes       = numMemoryStripes.m_PleInput;
    idMceOpInBuff->m_NumStripes        = numMemoryStripes.m_Input;

    return { idMceOpInBuff, idMceOpOutBuff };
}

Buffer* AddPleInBuffer(OwnedOpGraph& opGraph,
                       impl::NumStripesType& numPleInputMemoryStripes,
                       const TensorShape& tensorShape,
                       const TensorShape& pleInputMemoryShape,
                       const QuantizationInfo& quantInfo,
                       Lifetime lifetime,
                       TraversalOrder order)
{
    opGraph.AddBuffer(
        std::make_unique<Buffer>(lifetime, Location::PleInputSram, impl::GetFormat(Location::PleInputSram), order));
    auto buffer = opGraph.GetBuffers().back();

    // The ple input sram doesn't care about the tensorshape
    buffer->m_TensorShape = tensorShape;
    buffer->m_StripeShape = pleInputMemoryShape;
    buffer->m_NumStripes  = numPleInputMemoryStripes;
    buffer->m_SizeInBytes = impl::CalculateBufferSize(buffer->m_TensorShape, buffer->m_Format);

    buffer->m_QuantizationInfo = quantInfo;
    return buffer;
}

std::pair<Buffer*, Op*> AddMceToOpGraph(OwnedOpGraph& opGraph,
                                        Node* node,
                                        Lifetime lifetime,
                                        TraversalOrder order,
                                        const PartV1::MceStripesInfo& mceStripeInfo,
                                        const impl::MemoryStripesInfo& memoryStripesInfo,
                                        impl::NumMemoryStripes& numMemoryStripes,
                                        std::unique_ptr<Op> mceOp,
                                        Buffer* mceOutBuffer,
                                        const TensorShape& inputShape,
                                        const QuantizationInfo& inputQuantInfo,
                                        impl::ConvData& convData,
                                        WeightEncoderCache& weightEncoderCache,
                                        const HardwareCapabilities& caps)
{
    auto& buffers  = opGraph.GetBuffers();
    Op* op         = opGraph.AddOp(std::move(mceOp));
    op->m_Lifetime = lifetime;
    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, CascadingBufferFormat::NHWCB, order));
    Buffer* inBuffer        = buffers.back();
    inBuffer->m_TensorShape = inputShape;
    inBuffer->m_StripeShape = memoryStripesInfo.m_Input.m_Shape;
    inBuffer->m_NumStripes  = numMemoryStripes.m_Input;
    inBuffer->m_SizeInBytes = impl::CalculateTileSize(node, caps, inBuffer->m_TensorShape, inBuffer->m_StripeShape,
                                                      mceOutBuffer->m_StripeShape, inBuffer->m_NumStripes);

    inBuffer->m_QuantizationInfo = inputQuantInfo;
    opGraph.AddConsumer(inBuffer, op, 0);
    opGraph.SetProducer(mceOutBuffer, op);

    AddWeightBuffersAndDmaOpToMceOp(opGraph, lifetime, mceStripeInfo, numMemoryStripes.m_Weight,
                                    memoryStripesInfo.m_Weight.m_Shape, order, convData, weightEncoderCache);

    return { inBuffer, op };
};

std::pair<Buffer*, Op*> AddPleToOpGraph(OwnedOpGraph& opGraph,
                                        Lifetime lifetime,
                                        TraversalOrder order,
                                        const TensorShape& memoryOutputShape,
                                        impl::NumMemoryStripes& numMemoryStripes,
                                        std::unique_ptr<Op> pleOp,
                                        const TensorShape& outputShape,
                                        const QuantizationInfo& outputQuantInfo)
{
    auto& buffers  = opGraph.GetBuffers();
    Op* op         = opGraph.AddOp(std::move(pleOp));
    op->m_Lifetime = lifetime;

    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, impl::GetFormat(Location::Sram), order));
    auto pleOutBuffer = buffers.back();
    opGraph.SetProducer(pleOutBuffer, op);

    pleOutBuffer->m_TensorShape = outputShape;
    pleOutBuffer->m_StripeShape = memoryOutputShape;
    pleOutBuffer->m_NumStripes  = numMemoryStripes.m_Output;
    pleOutBuffer->m_SizeInBytes = numMemoryStripes.m_Output * impl::CalculateSizeInBytes(memoryOutputShape);

    pleOutBuffer->m_QuantizationInfo = outputQuantInfo;

    return { pleOutBuffer, op };
};

void PartV1::CreateMceOnlyPlans(Node* node,
                                const MceOnlyInfo& info,
                                TraversalOrder order,
                                WeightEncoderCache& weightEncoderCache,
                                Plans& plans) const
{
    auto lifetime             = info.m_Lifetime;
    MceOperationNode* mceNode = GetObjectAs<MceOperationNode>(node);
    for (auto numInputStripes = info.m_Memory.m_Input.m_Range.m_Min;
         numInputStripes <= info.m_Memory.m_Input.m_Range.m_Max; ++numInputStripes)
    {
        for (auto numWeightStripes = info.m_Memory.m_Weight.m_Range.m_Min;
             numWeightStripes <= info.m_Memory.m_Weight.m_Range.m_Max; ++numWeightStripes)
        {
            for (auto numPleInputStripes = info.m_Memory.m_PleInput.m_Range.m_Min;
                 numPleInputStripes <= info.m_Memory.m_PleInput.m_Range.m_Max; ++numPleInputStripes)
            {
                impl::NumMemoryStripes numMemoryStripes;
                numMemoryStripes.m_Input    = numInputStripes;
                numMemoryStripes.m_Output   = 0;
                numMemoryStripes.m_Weight   = numWeightStripes;
                numMemoryStripes.m_PleInput = numPleInputStripes;
                OwnedOpGraph opGraph;
                PartInputMapping inputMappings;
                PartOutputMapping outputMappings;
                auto mceOp =
                    CreateOpFromNode(node, info.m_MceCompute.m_BlockConfig, m_CompilationOptions, m_Capabilities);
                // We need to add the output buffer first before adding mce to opgraph as it uses it.
                auto outBuffer =
                    AddPleInBuffer(opGraph, numPleInputStripes, node->GetShape(), info.m_Memory.m_PleInput.m_Shape,
                                   node->GetQuantizationInfo(), lifetime, order);
                impl::ConvData convData;
                convData.weightInfo = mceNode->GetWeightsInfo();
                convData.weightData = mceNode->GetWeightsData();
                convData.biasInfo   = mceNode->GetBiasInfo();
                convData.biasData   = mceNode->GetBiasData();
                auto inBufferAndOp =
                    AddMceToOpGraph(opGraph, node, lifetime, order, info.m_MceCompute, info.m_Memory, numMemoryStripes,
                                    std::move(mceOp), outBuffer, node->GetInputShape(0),
                                    node->GetInputQuantizationInfo(0), convData, weightEncoderCache, m_Capabilities);
                inputMappings[inBufferAndOp.first] = PartInputSlot{ m_PartId, 0 };
                outputMappings[outBuffer]          = PartOutputSlot{ m_PartId, 0 };
                AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
            }
        }
    }
}

void PartV1::CreateMceAndIdentityPlePlans(Node* node,
                                          const MceAndPleInfo& info,
                                          TraversalOrder order,
                                          WeightEncoderCache& weightEncoderCache,
                                          Plans& plans) const
{
    auto lifetime             = info.m_Lifetime;
    MceOperationNode* mceNode = GetObjectAs<MceOperationNode>(node);
    for (auto numInputStripes = info.m_Memory.m_Input.m_Range.m_Min;
         numInputStripes <= info.m_Memory.m_Input.m_Range.m_Max; ++numInputStripes)
    {
        for (auto numOutputStripes = info.m_Memory.m_Output.m_Range.m_Min;
             numOutputStripes <= info.m_Memory.m_Output.m_Range.m_Max; ++numOutputStripes)
        {
            for (auto numWeightStripes = info.m_Memory.m_Weight.m_Range.m_Min;
                 numWeightStripes <= info.m_Memory.m_Weight.m_Range.m_Max; ++numWeightStripes)
            {
                for (auto numPleInputStripes = info.m_Memory.m_PleInput.m_Range.m_Min;
                     numPleInputStripes <= info.m_Memory.m_PleInput.m_Range.m_Max; ++numPleInputStripes)
                {
                    impl::NumMemoryStripes numMemoryStripes;
                    numMemoryStripes.m_Input    = numInputStripes;
                    numMemoryStripes.m_Output   = numOutputStripes;
                    numMemoryStripes.m_Weight   = numWeightStripes;
                    numMemoryStripes.m_PleInput = numPleInputStripes;
                    OwnedOpGraph opGraph;
                    PartInputMapping inputMappings;
                    PartOutputMapping outputMappings;
                    auto mceOp =
                        CreateOpFromNode(node, info.m_MceCompute.m_BlockConfig, m_CompilationOptions, m_Capabilities);
                    auto pleInBuffer =
                        AddPleInBuffer(opGraph, numPleInputStripes, node->GetShape(), info.m_Memory.m_PleInput.m_Shape,
                                       node->GetQuantizationInfo(), lifetime, order);
                    impl::ConvData convData;
                    convData.weightInfo   = mceNode->GetWeightsInfo();
                    convData.weightData   = mceNode->GetWeightsData();
                    convData.biasInfo     = mceNode->GetBiasInfo();
                    convData.biasData     = mceNode->GetBiasData();
                    auto inBufferAndMceOp = AddMceToOpGraph(
                        opGraph, node, lifetime, order, info.m_MceCompute, info.m_Memory, numMemoryStripes,
                        std::move(mceOp), pleInBuffer, node->GetInputShape(0), node->GetInputQuantizationInfo(0),
                        convData, weightEncoderCache, m_Capabilities);
                    // Create an identity ple Op
                    std::unique_ptr<PleOp> pleOp = std::make_unique<PleOp>(
                        Lifetime::Cascade, PleOperation::PASSTHROUGH, info.m_MceCompute.m_BlockConfig, 1,
                        std::vector<TensorShape>{ info.m_PleCompute.m_Input }, info.m_PleCompute.m_Output);
                    auto outBufferAndPleOp =
                        AddPleToOpGraph(opGraph, lifetime, order, info.m_Memory.m_Output.m_Shape, numMemoryStripes,
                                        std::move(pleOp), node->GetShape(), node->GetQuantizationInfo());
                    opGraph.AddConsumer(pleInBuffer, outBufferAndPleOp.second, 0);
                    inputMappings[inBufferAndMceOp.first]   = PartInputSlot{ m_PartId, 0 };
                    outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
                    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
                }
            }
        }
    }
}

void PartV1::CreateIdentityMceAndFusedPlePlans(Node* node,
                                               const MceAndPleInfo& info,
                                               TraversalOrder order,
                                               WeightEncoderCache& weightEncoderCache,
                                               Plans& plans) const
{
    auto lifetime = info.m_Lifetime;
    // Create plan with identity mce op and ple op
    for (auto numInputStripes = info.m_Memory.m_Input.m_Range.m_Min;
         numInputStripes <= info.m_Memory.m_Input.m_Range.m_Max; ++numInputStripes)
    {
        for (auto numOutputStripes = info.m_Memory.m_Output.m_Range.m_Min;
             numOutputStripes <= info.m_Memory.m_Output.m_Range.m_Max; ++numOutputStripes)
        {
            for (auto numWeightStripes = info.m_Memory.m_Weight.m_Range.m_Min;
                 numWeightStripes <= info.m_Memory.m_Weight.m_Range.m_Max; ++numWeightStripes)
            {
                for (auto numPleInputStripes = info.m_Memory.m_PleInput.m_Range.m_Min;
                     numPleInputStripes <= info.m_Memory.m_PleInput.m_Range.m_Max; ++numPleInputStripes)
                {
                    impl::NumMemoryStripes numMemoryStripes;
                    numMemoryStripes.m_Input    = numInputStripes;
                    numMemoryStripes.m_Output   = numOutputStripes;
                    numMemoryStripes.m_Weight   = numWeightStripes;
                    numMemoryStripes.m_PleInput = numPleInputStripes;
                    OwnedOpGraph opGraph;
                    PartInputMapping inputMappings;
                    PartOutputMapping outputMappings;
                    auto mceInAndOutBuffer = AddIdentityMceOpForSubGraph(
                        opGraph, lifetime, info.m_MceCompute, numMemoryStripes, info.m_Memory, node->GetInputShape(0),
                        node->GetInputQuantizationInfo(0), order, weightEncoderCache);
                    auto op =
                        CreateOpFromNode(node, info.m_MceCompute.m_BlockConfig, m_CompilationOptions, m_Capabilities);
                    PleOp* pleOp               = dynamic_cast<PleOp*>(op.get());
                    pleOp->m_InputStripeShapes = { info.m_PleCompute.m_Input };
                    pleOp->m_NumInputs         = 1;
                    pleOp->m_OutputStripeShape = info.m_PleCompute.m_Output;
                    auto outBufferAndPleOp =
                        AddPleToOpGraph(opGraph, lifetime, order, info.m_Memory.m_Output.m_Shape, numMemoryStripes,
                                        std::move(op), node->GetShape(), node->GetQuantizationInfo());
                    opGraph.AddConsumer(mceInAndOutBuffer.second, outBufferAndPleOp.second, 0);
                    inputMappings[mceInAndOutBuffer.first]  = PartInputSlot{ m_PartId, 0 };
                    outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
                    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
                }
            }
        }
    }
}

void AddOpToOpGraphWithInputOutputBuffers(const PartId partId,
                                          const HardwareCapabilities& capabilities,
                                          OwnedOpGraph& opGraph,
                                          Node* node,
                                          Node* outputNode,
                                          TraversalOrder order,
                                          impl::DmaOnlyInfo& info,
                                          impl::NumMemoryStripes& numMemoryStripes,
                                          Location inputBufferLocation,
                                          Location outputBufferLocation,
                                          PartInputMapping& inputMappings,
                                          PartOutputMapping& outputMappings)
{
    (void)outputMappings;    //Currently unused but expected to be used whenever multi output will be supported
    auto lifetime = info.m_Lifetime;

    assert(IsObjectOfType<ReinterpretNode>(node) || IsObjectOfType<FormatConversionNode>(node));

    if (IsObjectOfType<ReinterpretNode>(node))
    {
        opGraph.AddOp(std::make_unique<DummyOp>());
    }
    else if (IsObjectOfType<FormatConversionNode>(node))
    {
        opGraph.AddOp(std::make_unique<DmaOp>());
    }

    const OpGraph::BufferList& buffers = opGraph.GetBuffers();
    const OpGraph::OpList& ops         = opGraph.GetOps();
    Op* op                             = ops.back();
    op->m_Lifetime                     = lifetime;
    uint32_t inputIndex                = 0;
    for (Edge* edge : node->GetInputs())
    {
        opGraph.AddBuffer(
            std::make_unique<Buffer>(lifetime, inputBufferLocation, impl::GetFormat(inputBufferLocation), order));
        Buffer* inBuffer        = buffers.back();
        const Node* inputNode   = edge->GetSource();
        inBuffer->m_TensorShape = inputNode->GetShape();
        inBuffer->m_StripeShape = info.m_Input.m_Shape;
        inBuffer->m_NumStripes  = numMemoryStripes.m_Input;
        inBuffer->m_SizeInBytes =
            inputBufferLocation == Location::Sram
                ? impl::CalculateTileSize(node, capabilities, inBuffer->m_TensorShape, info.m_Input.m_Shape,
                                          info.m_Output.m_Shape, numMemoryStripes.m_Input)
                : impl::CalculateBufferSize(inBuffer->m_TensorShape, inBuffer->m_Format);

        inBuffer->m_QuantizationInfo = inputNode->GetQuantizationInfo();
        inputMappings[inBuffer]      = PartInputSlot{ partId, inputIndex };
        opGraph.AddConsumer(inBuffer, op, 0);

        PleOp* pleOp = dynamic_cast<PleOp*>(op);
        if (pleOp)
        {
            pleOp->m_InputStripeShapes.push_back(inBuffer->m_StripeShape);
        }
        inputIndex++;
    }

    opGraph.AddBuffer(
        std::make_unique<Buffer>(lifetime, outputBufferLocation, impl::GetFormat(outputBufferLocation), order));
    auto outBuffer = buffers.back();
    opGraph.SetProducer(outBuffer, op);

    outBuffer->m_TensorShape = outputNode->GetShape();
    outBuffer->m_StripeShape = info.m_Output.m_Shape;
    outBuffer->m_NumStripes  = numMemoryStripes.m_Output;
    outBuffer->m_SizeInBytes = outputBufferLocation == Location::Sram
                                   ? impl::CalculateTileSize(capabilities, outBuffer->m_TensorShape,
                                                             outBuffer->m_StripeShape, numMemoryStripes.m_Output)
                                   : impl::CalculateBufferSize(outBuffer->m_TensorShape, outBuffer->m_Format);

    outBuffer->m_QuantizationInfo = outputNode->GetQuantizationInfo();
}

void PartV1::CreateFuseOnlyPlans(Node* node, const PleOnlyInfo& info, TraversalOrder order, Plans& plans) const
{
    auto lifetime = info.m_Lifetime;
    for (auto numOutputStripes = info.m_Memory.m_Output.m_Range.m_Min;
         numOutputStripes <= info.m_Memory.m_Output.m_Range.m_Max; ++numOutputStripes)
    {
        for (auto numPleInputStripes = info.m_Memory.m_PleInput.m_Range.m_Min;
             numPleInputStripes <= info.m_Memory.m_PleInput.m_Range.m_Max; ++numPleInputStripes)
        {
            impl::NumMemoryStripes numMemoryStripes;
            numMemoryStripes.m_Input    = 0;
            numMemoryStripes.m_Output   = numOutputStripes;
            numMemoryStripes.m_Weight   = 0;
            numMemoryStripes.m_PleInput = numPleInputStripes;
            OwnedOpGraph opGraph;
            PartInputMapping inputMappings;
            PartOutputMapping outputMappings;
            auto pleInBuffer =
                AddPleInBuffer(opGraph, numPleInputStripes, node->GetInputShape(0), info.m_Memory.m_PleInput.m_Shape,
                               node->GetQuantizationInfo(), lifetime, order);
            auto op = CreateOpFromNode(node, info.m_PleCompute.m_BlockConfig, m_CompilationOptions, m_Capabilities);
            PleOp* pleOp               = dynamic_cast<PleOp*>(op.get());
            pleOp->m_InputStripeShapes = { info.m_PleCompute.m_Input };
            pleOp->m_NumInputs         = 1;
            pleOp->m_OutputStripeShape = info.m_PleCompute.m_Output;
            auto outBufferAndPleOp =
                AddPleToOpGraph(opGraph, lifetime, order, info.m_Memory.m_Output.m_Shape, numMemoryStripes,
                                std::move(op), node->GetShape(), node->GetQuantizationInfo());
            opGraph.AddConsumer(pleInBuffer, outBufferAndPleOp.second, 0);
            inputMappings[pleInBuffer]              = PartInputSlot{ m_PartId, 0 };
            outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
            AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
        }
    }
}

void PartV1::CreateFormatConversionPlans(Node* node,
                                         impl::DmaOnlyInfo& dmaInfo,
                                         impl::NumMemoryStripes& numMemoryStripes,
                                         TraversalOrder order,
                                         Location inputBufferLocaton,
                                         Location outputBufferLocation,
                                         Plans& plans) const
{
    OwnedOpGraph opGraph;
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    auto& buffers = opGraph.GetBuffers();
    AddOpToOpGraphWithInputOutputBuffers(m_PartId, m_Capabilities, opGraph, node, m_SubGraph.back(), order, dmaInfo,
                                         numMemoryStripes, inputBufferLocaton, outputBufferLocation, inputMappings,
                                         outputMappings);
    outputMappings[buffers.back()] = PartOutputSlot{ m_PartId, 0 };
    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

void PartV1::CreateVirtualSramPlans(Node* node,
                                    impl::DmaOnlyInfo& dmaInfo,
                                    impl::NumMemoryStripes& numMemoryStripes,
                                    TraversalOrder order,
                                    Plans& plans) const
{
    OwnedOpGraph opGraph;
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    auto& buffers = opGraph.GetBuffers();
    auto format   = node->GetFormat();
    switch (format)
    {
        case CompilerDataFormat::NHWCB:
            AddOpToOpGraphWithInputOutputBuffers(m_PartId, m_Capabilities, opGraph, node, m_SubGraph.back(), order,
                                                 dmaInfo, numMemoryStripes, Location::VirtualSram, Location::Sram,
                                                 inputMappings, outputMappings);
            outputMappings[buffers.back()] = PartOutputSlot{ m_PartId, 0 };
            break;
        case CompilerDataFormat::NHWC:
            AddOpToOpGraphWithInputOutputBuffers(m_PartId, m_Capabilities, opGraph, node, m_SubGraph.back(), order,
                                                 dmaInfo, numMemoryStripes, Location::Sram, Location::VirtualSram,
                                                 inputMappings, outputMappings);
            outputMappings[buffers.back()] = PartOutputSlot{ m_PartId, 0 };
            break;
        default:
            throw NotSupportedException("Unsupported compiler data format. Only NHWC and NHWCB is currently handled.");
    }
    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

std::vector<BlockConfig> GenerateBlockConfigs(Node* node)
{
    std::vector<BlockConfig> result;

    // All block configs possible
    const std::vector<BlockConfig> allBlockConfigs = { { 16u, 16u }, { 16u, 8u }, { 8u, 16u }, { 8u, 8u } };

    result = allBlockConfigs;
    if (IsObjectOfType<MceOperationNode>(node))
    {
        result = utils::FilterMceBlockConfigs(GetObjectAs<MceOperationNode>(node), allBlockConfigs);
    }
    else if (IsObjectOfType<FuseOnlyPleOperationNode>(node))
    {
        result = utils::FilterPleBlockConfigs(GetObjectAs<FuseOnlyPleOperationNode>(node), allBlockConfigs);
    }
    return result;
}

void PartV1::GenerateWithTraversalOrders(CascadeType cascadeType,
                                         Buffer* sramBuffer,
                                         uint32_t numWeightStripes,
                                         Node* node,
                                         WeightEncoderCache& weightEncoderCache,
                                         Plans& plans) const
{
    ETHOSN_UNUSED(cascadeType);

    if (node->GetInputs().size() == 0)
    {
        return;
    }
    std::vector<BlockConfig> blockConfigs = GenerateBlockConfigs(node);
    GenerateWithStripeSizes(node, blockConfigs, TraversalOrder::Xyz, weightEncoderCache, plans);
    // TODO: Add the same function call with traversal order ZXY

    auto inputStripe  = CreateStripe(node->GetInputShape(0), TensorShape{ 0, 0, 0, 0 }, m_Capabilities);
    auto outputStripe = CreateStripe(node->GetShape(), TensorShape{ 0, 0, 0, 0 }, m_Capabilities);

    if (IsObjectOfType<FormatConversionNode>(node))
    {
        impl::DmaOnlyInfo dmaInfo;
        dmaInfo.m_Lifetime = Lifetime::Cascade;
        dmaInfo.m_Input    = impl::MemoryStripeInfo{ { 1, 1 }, inputStripe };
        dmaInfo.m_Output   = impl::MemoryStripeInfo{ { 1, 1 }, outputStripe };
        impl::NumMemoryStripes numMemoryStripes;
        numMemoryStripes.m_Input  = 1;
        numMemoryStripes.m_Output = 1;
        CreateVirtualSramPlans(node, dmaInfo, numMemoryStripes, TraversalOrder::Xyz, plans);
    }
    else if (IsObjectOfType<ReinterpretNode>(node))
    {
        const EstimationOptions estOpt;
        const CompilationOptions compOpt;
        HardwareCapabilities hwCapabilities(GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

        ReshapePart reshapePart(m_PartId, node->GetInputShape(0), node->GetShape(), node->GetFormat(),
                                node->GetQuantizationInfo(), node->GetCorrespondingOperationIds(), estOpt, compOpt,
                                hwCapabilities);
        plans = reshapePart.GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, sramBuffer,
                                     numWeightStripes);
    }
}

void GenerateStripes(Node* node,
                     const HardwareCapabilities& caps,
                     const BlockConfig blockConfig,
                     PartV1::StripeInfos* outStripeInfos)
{
    using namespace utils;
    assert(outStripeInfos);

    // Note we use set rather than unordered_set to give consistent behaviour across STL implementations to make
    // debugging and testing easier.
    impl::NumStripes numStripesInput;
    impl::NumStripes numStripesOutput;
    impl::NumStripes numStripesWeights;
    impl::NumStripes numStripesPleInput;

    uint32_t strideMultiplier       = 1U;
    const MceOperationNode* mceNode = GetObjectAs<MceOperationNode>(node);
    uint32_t kernelHeight           = 0;
    uint32_t kernelWidth            = 0;
    bool isDepthwise                = false;
    TensorShape mceOutputShape      = {};
    if (mceNode)
    {
        // MceOperations output to PLE SRAM so are no "stripes"
        // At least 3 input stripes are needed because of
        // data on the top and bottom. Weights can
        // have 1 or 2 for double buffering.
        auto mceNode = GetObjectAs<MceOperationNode>(node);
        kernelHeight = mceNode->GetWeightsInfo().m_Dimensions[0];
        kernelWidth  = mceNode->GetWeightsInfo().m_Dimensions[1];
        if (kernelHeight == 1)
        {
            numStripesInput = { 1, 2 };
        }
        else
        {
            numStripesInput = { 3, 4 };
        }
        numStripesOutput   = { 1, 3 };
        numStripesWeights  = { 1, 2 };
        numStripesPleInput = { 0, 0 };
        strideMultiplier   = mceNode->GetStride().m_X * mceNode->GetStride().m_Y;
        isDepthwise        = mceNode->GetOperation() == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
        mceOutputShape     = mceNode->GetShape();
    }
    else if (IsObjectOfType<FuseOnlyPleOperationNode>(node))
    {
        numStripesInput    = { 1, 4 };
        numStripesOutput   = { 1, 3 };
        numStripesWeights  = { 1, 2 };
        numStripesPleInput = { 0, 0 };
        // For fuse only ple ops we generate plans with identity depthwises which have kernel size 1x1
        kernelHeight   = 1;
        kernelWidth    = 1;
        isDepthwise    = true;
        mceOutputShape = node->GetInputShape(0);
    }
    else if (IsObjectOfType<StandalonePleOperationNode>(node))
    {
        throw NotSupportedException("Standalone PLE operations not yet supported");
    }
    else if (IsObjectOfType<FormatConversionNode>(node) || IsObjectOfType<ReinterpretNode>(node))
    {
        // Format conversion and reinterpret need to be able to combine with the input of an MceOperation and the output of a FusedPleOperation
        numStripesInput    = { 1, 2 };
        numStripesOutput   = { 1, 3 };
        numStripesWeights  = { 0, 0 };
        numStripesPleInput = { 0, 0 };
        mceOutputShape     = node->GetShape();
    }
    else
    {
        return;
    }

    auto ApplyShapeMult = [&](TensorShape shape) {
        utils::ShapeMultiplier shapeMult = utils::g_IdentityShapeMultiplier;
        if (IsObjectOfType<FuseOnlyPleOperationNode>(node))
        {
            shapeMult = GetObjectAs<FuseOnlyPleOperationNode>(node)->GetShapeMultiplier();
        }
        return TensorShape{ shape[0], shape[1] * shapeMult.m_H, shape[2] * shapeMult.m_W, shape[3] * shapeMult.m_C };
    };

    auto AddStripeInfos = [&](const TensorShape& mceInputStripe, const TensorShape& mceOutputStripe,
                              const TensorShape& pleInputStripe, const TensorShape& pleOutputStripe,
                              const impl::NumStripes& inputRange, const impl::NumStripes& outputRange,
                              const impl::NumStripes& weightRange, const impl::NumStripes& pleInputRange,
                              const TensorShape& memoryInputStripe, const TensorShape& memoryOutputStripe,
                              const TensorShape& memoryPleInputStripe, const TensorShape& inputShape,
                              const TensorShape& outputShape) {
        // Limit the max number of stripes based on the size of the tensor - there is no point considering plans where
        // we can store more stripes in the tile than there are in the tensor!
        impl::NumStripes inputCopy = inputRange;
        inputCopy.m_Max =
            std::min(inputCopy.m_Max, DivRoundUp(GetHeight(inputShape), GetHeight(memoryInputStripe)) *
                                          DivRoundUp(GetWidth(inputShape), GetWidth(memoryInputStripe)) *
                                          DivRoundUp(GetChannels(inputShape), GetChannels(memoryInputStripe)));
        impl::NumStripes outputCopy = outputRange;
        outputCopy.m_Max =
            std::min(outputCopy.m_Max, DivRoundUp(GetHeight(outputShape), GetHeight(memoryOutputStripe)) *
                                           DivRoundUp(GetWidth(outputShape), GetWidth(memoryOutputStripe)) *
                                           DivRoundUp(GetChannels(outputShape), GetChannels(memoryOutputStripe)));

        // Prevent using stripes which have more elements than the entire tensor
        bool multipleStripes         = inputCopy.m_Max > 1 && outputCopy.m_Max > 1;
        bool stripesLargerThanTensor = utils::GetNumElements(memoryInputStripe) > utils::GetNumElements(inputShape) &&
                                       utils::GetNumElements(memoryOutputStripe) > utils::GetNumElements(outputShape);
        if (multipleStripes && stripesLargerThanTensor)
        {
            return;
        }
        TensorShape mceWeightStripe    = { kernelHeight, kernelWidth, mceInputStripe[3],
                                        isDepthwise ? 1 : mceOutputStripe[3] };
        TensorShape memoryWeightStripe = mceWeightStripe;
        impl::NumStripes weightCopy    = weightRange;
        if (isDepthwise)
        {
            if (memoryWeightStripe[2] >= node->GetInputShape(0)[3])
            {
                weightCopy.m_Max = 1;
            }
        }
        else
        {
            if (memoryWeightStripe[3] >= mceOutputShape[3])
            {
                weightCopy.m_Max = 1;
            }
        }
        {
            PartV1::MceAndPleInfo mceAndPleInfo;

            mceAndPleInfo.m_MceCompute.m_Input       = mceInputStripe;
            mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
            mceAndPleInfo.m_MceCompute.m_Weight      = mceWeightStripe;
            mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
            mceAndPleInfo.m_PleCompute.m_Input       = pleInputStripe;
            mceAndPleInfo.m_PleCompute.m_Output      = pleOutputStripe;
            mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

            mceAndPleInfo.m_Memory.m_Input    = { inputCopy, memoryInputStripe };
            mceAndPleInfo.m_Memory.m_Output   = { outputCopy, memoryOutputStripe };
            mceAndPleInfo.m_Memory.m_Weight   = { weightCopy, memoryWeightStripe };
            mceAndPleInfo.m_Memory.m_PleInput = { pleInputRange, memoryPleInputStripe };
            outStripeInfos->m_MceAndPleInfos.insert(mceAndPleInfo);
        }
        {
            PartV1::MceOnlyInfo mceOnlyInfo;

            mceOnlyInfo.m_MceCompute.m_Input       = mceInputStripe;
            mceOnlyInfo.m_MceCompute.m_Output      = mceOutputStripe;
            mceOnlyInfo.m_MceCompute.m_Weight      = mceWeightStripe;
            mceOnlyInfo.m_MceCompute.m_BlockConfig = blockConfig;

            mceOnlyInfo.m_Memory.m_Input    = { inputCopy, memoryInputStripe };
            mceOnlyInfo.m_Memory.m_Output   = { { 0, 0 }, { 0, 0, 0, 0 } };
            mceOnlyInfo.m_Memory.m_Weight   = { weightCopy, memoryWeightStripe };
            mceOnlyInfo.m_Memory.m_PleInput = { pleInputRange, memoryPleInputStripe };
            outStripeInfos->m_MceOnlyInfos.insert(mceOnlyInfo);
        }
        {
            PartV1::PleOnlyInfo pleOnlyInfo;

            pleOnlyInfo.m_PleCompute.m_Input       = pleInputStripe;
            pleOnlyInfo.m_PleCompute.m_Output      = pleOutputStripe;
            pleOnlyInfo.m_PleCompute.m_BlockConfig = blockConfig;

            pleOnlyInfo.m_Memory.m_Input    = { { 0, 0 }, { 0, 0, 0, 0 } };
            pleOnlyInfo.m_Memory.m_Output   = { outputCopy, memoryOutputStripe };
            pleOnlyInfo.m_Memory.m_Weight   = { { 0, 0 }, { 0, 0, 0, 0 } };
            pleOnlyInfo.m_Memory.m_PleInput = { pleInputRange, memoryPleInputStripe };
            outStripeInfos->m_PleOnlyInfos.insert(pleOnlyInfo);
        }
        {
            impl::DmaOnlyInfo dmaOnlyInfo;
            dmaOnlyInfo.m_Input  = { inputCopy, memoryInputStripe };
            dmaOnlyInfo.m_Output = { outputCopy, memoryOutputStripe };
            outStripeInfos->m_DmaOnlyInfos.insert(dmaOnlyInfo);
        }
    };

    // Use the minimum stripe size possible to minimize the time before processing
    // Try splitting height first
    {
        TensorShape mceInputEncoding  = { 0, blockConfig.m_BlockHeight(), 0, 0 };
        const TensorShape& inputShape = node->GetInputShape(0);
        TensorShape mceInputStripe    = CreateStripe(node->GetInputShape(0), mceInputEncoding, caps);

        TensorShape mceOutputEncoding = mceInputEncoding;
        TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, caps);

        TensorShape pleOutputEncoding          = ApplyShapeMult(mceInputEncoding);
        TensorShape pleOutputStripe            = CreateStripe(node->GetShape(), pleOutputEncoding, caps);
        const TensorShape& outputShape         = node->GetShape();
        impl::NumStripes numStripesWeightsCopy = numStripesWeights;
        numStripesWeightsCopy.m_Min            = std::min(numStripesWeights.m_Min, 1u);
        numStripesWeightsCopy.m_Max            = std::min(numStripesWeights.m_Max, 1u);

        AddStripeInfos(mceInputStripe, mceOutputStripe, mceInputStripe, pleOutputStripe, numStripesInput,
                       numStripesOutput, numStripesWeightsCopy, numStripesPleInput, mceInputStripe, pleOutputStripe,
                       mceOutputStripe, inputShape, outputShape);
    }

    // Split only input in height while the output is full tensor
    {
        TensorShape mceInputEncoding  = { 0, blockConfig.m_BlockHeight(), 0, 0 };
        const TensorShape& inputShape = node->GetInputShape(0);
        TensorShape mceInputStripe    = CreateStripe(node->GetInputShape(0), mceInputEncoding, caps);

        TensorShape mceOutputEncoding = mceInputEncoding;
        TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, caps);

        TensorShape pleOutputEncoding = ApplyShapeMult(mceInputEncoding);
        TensorShape pleOutputStripe   = CreateStripe(node->GetShape(), pleOutputEncoding, caps);

        const TensorShape& outputShape         = node->GetShape();
        TensorShape memoryOutputEncoding       = { 0, 0, 0, 0 };
        TensorShape memoryOutputStripe         = CreateStripe(outputShape, memoryOutputEncoding, caps);
        impl::NumStripes numStripesWeightsCopy = numStripesWeights;
        numStripesWeightsCopy.m_Min            = std::min(numStripesWeights.m_Min, 1u);
        numStripesWeightsCopy.m_Max            = std::min(numStripesWeights.m_Max, 1u);
        impl::NumStripes numStripesOutputCopy  = numStripesOutput;
        numStripesOutputCopy.m_Min             = std::min(numStripesOutput.m_Min, 1u);
        numStripesOutputCopy.m_Max             = std::min(numStripesOutput.m_Max, 1u);

        AddStripeInfos(mceInputStripe, mceOutputStripe, mceInputStripe, pleOutputStripe, numStripesInput,
                       numStripesOutputCopy, numStripesWeightsCopy, numStripesPleInput, mceInputStripe,
                       memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
    }

    //// Try splitting width
    {
        TensorShape mceInputEncoding  = { 0, 0, blockConfig.m_BlockWidth(), 0 };
        const TensorShape& inputShape = node->GetInputShape(0);
        TensorShape mceInputStripe    = CreateStripe(node->GetInputShape(0), mceInputEncoding, caps);

        TensorShape mceOutputEncoding = mceInputEncoding;
        TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, caps);

        TensorShape pleOutputEncoding        = ApplyShapeMult(mceInputEncoding);
        TensorShape pleOutputStripe          = CreateStripe(node->GetShape(), pleOutputEncoding, caps);
        const TensorShape& outputShape       = node->GetShape();
        impl::NumStripes numStripesInputCopy = numStripesInput;

        if (kernelWidth == 1)
        {
            numStripesInputCopy.m_Min = 1;
            numStripesInputCopy.m_Max = 2;
        }

        impl::NumStripes numStripesWeightCopy = numStripesWeights;
        numStripesWeightCopy.m_Min            = std::min(numStripesWeights.m_Min, 1u);
        numStripesWeightCopy.m_Max            = std::min(numStripesWeights.m_Max, 1u);

        AddStripeInfos(mceInputStripe, mceOutputStripe, mceInputStripe, pleOutputStripe, numStripesInputCopy,
                       numStripesOutput, numStripesWeightCopy, numStripesPleInput, mceInputStripe, pleOutputStripe,
                       mceOutputStripe, inputShape, outputShape);
    }

    // Try splitting width and height
    {
        TensorShape mceInputEncoding  = { 0, blockConfig.m_BlockHeight(), blockConfig.m_BlockWidth(), 0 };
        const TensorShape& inputShape = node->GetInputShape(0);
        TensorShape mceInputStripe    = CreateStripe(node->GetInputShape(0), mceInputEncoding, caps);

        TensorShape mceOutputEncoding = mceInputEncoding;
        TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, caps);

        TensorShape pleOutputEncoding        = ApplyShapeMult(mceInputEncoding);
        TensorShape pleOutputStripe          = CreateStripe(node->GetShape(), pleOutputEncoding, caps);
        const TensorShape& outputShape       = node->GetShape();
        impl::NumStripes numStripesInputCopy = numStripesInput;

        if (kernelWidth == 1)
        {
            numStripesInputCopy.m_Min = 1;
            numStripesInputCopy.m_Max = 2;
        }

        impl::NumStripes numStripesWeightCopy = numStripesWeights;
        numStripesWeightCopy.m_Min            = std::min(numStripesWeights.m_Min, 1u);
        numStripesWeightCopy.m_Max            = std::min(numStripesWeights.m_Max, 1u);

        AddStripeInfos(mceInputStripe, mceOutputStripe, mceInputStripe, pleOutputStripe, numStripesInputCopy,
                       numStripesOutput, numStripesWeightCopy, numStripesPleInput, mceInputStripe, pleOutputStripe,
                       mceOutputStripe, inputShape, outputShape);
    }

    // Try split input depth
    // note we have to limit the height and width to the block size
    {
        TensorShape mceInputEncoding  = { 0, blockConfig.m_BlockHeight(), blockConfig.m_BlockWidth(),
                                         caps.GetNumberOfOgs() * strideMultiplier };
        const TensorShape& inputShape = node->GetInputShape(0);
        TensorShape mceInputStripe    = CreateStripe(node->GetInputShape(0), mceInputEncoding, caps);

        TensorShape mceOutputEncoding = mceInputEncoding;
        TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, caps);

        TensorShape pleOutputEncoding  = ApplyShapeMult(mceOutputEncoding);
        TensorShape pleOutputStripe    = CreateStripe(node->GetShape(), pleOutputEncoding, caps);
        const TensorShape& outputShape = node->GetShape();

        AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe, numStripesInput,
                       numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe, pleOutputStripe,
                       mceInputStripe, inputShape, outputShape);
    }

    if (isDepthwise)
    {
        // Try split output depth
        {
            // With depthwise each only OFM needs 1 IFM
            TensorShape mceInputEncoding  = { 0, 0, 0, caps.GetNumberOfOgs() };
            const TensorShape& inputShape = node->GetInputShape(0);
            TensorShape mceInputStripe    = CreateStripe(node->GetInputShape(0), mceInputEncoding, caps);

            TensorShape mceOutputEncoding = { 0, 0, 0, caps.GetNumberOfOgs() };
            TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, caps);

            const TensorShape& outputShape = node->GetShape();
            TensorShape pleOutputEncoding  = ApplyShapeMult(mceOutputEncoding);
            TensorShape pleOutputStripe    = CreateStripe(outputShape, pleOutputEncoding, caps);

            AddStripeInfos(mceInputStripe, mceOutputStripe, mceInputStripe, pleOutputStripe, numStripesInput,
                           numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe, pleOutputStripe,
                           mceOutputStripe, inputShape, outputShape);
        }

        // Try split depth for compute but the memory buffer is the full tensor
        // e.g. strategy 1 cascading
        {
            TensorShape mceInputEncoding  = { 0, 0, 0, caps.GetNumberOfOgs() };
            const TensorShape& inputShape = node->GetInputShape(0);
            TensorShape mceInputStripe    = CreateStripe(node->GetInputShape(0), mceInputEncoding, caps);

            TensorShape mceOutputEncoding = { 0, 0, 0, caps.GetNumberOfOgs() };
            TensorShape mceOutputStripe   = CreateStripe(inputShape, mceOutputEncoding, caps);

            TensorShape pleOutputEncoding  = ApplyShapeMult(mceOutputEncoding);
            const TensorShape& outputShape = node->GetShape();
            TensorShape pleOutputStripe    = CreateStripe(outputShape, pleOutputEncoding, caps);

            TensorShape memoryOutputEncoding = { 0, 0, 0, 0 };
            TensorShape memoryOutputStripe   = CreateStripe(outputShape, memoryOutputEncoding, caps);
            AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe, numStripesInput,
                           numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe, memoryOutputStripe,
                           mceOutputStripe, inputShape, outputShape);
        }
    }
    else
    {
        // Try split output depth
        {
            TensorShape mceInputEncoding  = { 0, 0, 0, 0 };
            const TensorShape& inputShape = node->GetInputShape(0);
            TensorShape mceInputStripe    = CreateStripe(node->GetInputShape(0), mceInputEncoding, caps);

            TensorShape mceOutputEncoding = { 0, 0, 0, caps.GetNumberOfOgs() };
            TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, caps);

            TensorShape pleOutputStripe          = mceOutputStripe;
            impl::NumStripes numStripesInputCopy = numStripesInput;
            numStripesInputCopy.m_Min            = std::min(numStripesInputCopy.m_Min, 1u);
            numStripesInputCopy.m_Max            = std::min(numStripesInputCopy.m_Max, 1u);
            const TensorShape& outputShape       = node->GetShape();

            AddStripeInfos(mceInputStripe, mceOutputStripe, mceInputStripe, pleOutputStripe, numStripesInputCopy,
                           numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe, pleOutputStripe,
                           mceOutputStripe, inputShape, outputShape);
        }
        // Try split depth for compute but the memory buffer is the full tensor
        // e.g. strategy 1 cascading
        {
            TensorShape mceInputEncoding  = { 0, 0, 0, 0 };
            const TensorShape& inputShape = node->GetInputShape(0);
            TensorShape mceInputStripe    = CreateStripe(node->GetInputShape(0), mceInputEncoding, caps);

            TensorShape mceOutputEncoding = { 0, 0, 0, caps.GetNumberOfOgs() };
            TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, caps);

            TensorShape pleOutputEncoding        = ApplyShapeMult(mceOutputEncoding);
            const TensorShape& outputShape       = node->GetShape();
            TensorShape pleOutputStripe          = CreateStripe(outputShape, pleOutputEncoding, caps);
            impl::NumStripes numStripesInputCopy = numStripesInput;
            numStripesInputCopy.m_Min            = std::min(numStripesInputCopy.m_Min, 1u);
            numStripesInputCopy.m_Max            = std::min(numStripesInputCopy.m_Max, 1u);

            TensorShape memoryOutputEncoding = { 0, 0, 0, 0 };
            TensorShape memoryOutputStripe   = CreateStripe(outputShape, memoryOutputEncoding, caps);
            AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe, numStripesInputCopy,
                           numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe, memoryOutputStripe,
                           mceOutputStripe, inputShape, outputShape);
        }
    }

    // Don't split at all
    // This is needed if all of the stripes above are larger than the tensor
    // and none of them are added
    {
        TensorShape mceInputEncoding   = { 0, 0, 0, 0 };
        TensorShape mceInputStripe     = CreateStripe(node->GetInputShape(0), mceInputEncoding, caps);
        const TensorShape& inputShape  = node->GetInputShape(0);
        const TensorShape& outputShape = node->GetShape();

        TensorShape mceOutputEncoding = mceInputEncoding;
        TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, caps);

        TensorShape pleOutputStripe            = CreateStripe(node->GetShape(), mceInputEncoding, caps);
        impl::NumStripes numStripesInputCopy   = numStripesInput;
        numStripesInputCopy.m_Min              = std::min(numStripesInput.m_Min, 1u);
        numStripesInputCopy.m_Max              = std::min(numStripesInput.m_Max, 1u);
        impl::NumStripes numStripesWeightsCopy = numStripesWeights;
        numStripesWeightsCopy.m_Min            = std::min(numStripesWeights.m_Min, 1u);
        numStripesWeightsCopy.m_Max            = std::min(numStripesWeights.m_Max, 1u);
        impl::NumStripes numStripesOutputCopy  = numStripesOutput;
        numStripesOutputCopy.m_Min             = std::min(numStripesOutput.m_Min, 1u);
        numStripesOutputCopy.m_Max             = std::min(numStripesOutput.m_Max, 1u);

        AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe, numStripesInputCopy,
                       numStripesOutputCopy, numStripesWeightsCopy, numStripesPleInput, mceInputStripe, pleOutputStripe,
                       mceOutputStripe, inputShape, outputShape);
    }
}

void PartV1::GenerateWithStripeSizes(Node* node,
                                     const std::vector<BlockConfig>& blockConfigs,
                                     TraversalOrder order,
                                     WeightEncoderCache& weightEncoderCache,
                                     Plans& plans) const
{
    PartV1::StripeInfos stripeInfos;
    for (auto blockConfig : blockConfigs)
    {
        GenerateStripes(node, m_Capabilities, blockConfig, &stripeInfos);
    }

    GenerateWithNumStripes(node, order, stripeInfos, weightEncoderCache, plans);
}

void PartV1::GenerateMcePlans(Node* node,
                              TraversalOrder order,
                              PartV1::StripeInfos& stripeInfos,
                              WeightEncoderCache& weightEncoderCache,
                              Plans& plans) const
{
    for (const PartV1::MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        CreateMceAndIdentityPlePlans(node, i, order, weightEncoderCache, plans);
    }
    for (const PartV1::MceOnlyInfo& i : stripeInfos.m_MceOnlyInfos)
    {
        CreateMceOnlyPlans(node, i, order, weightEncoderCache, plans);
    }
}

void PartV1::GenerateFuseOnlyPlePlans(Node* node,
                                      TraversalOrder order,
                                      PartV1::StripeInfos& stripeInfos,
                                      WeightEncoderCache& weightEncoderCache,
                                      Plans& plans) const
{
    for (const PartV1::MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        CreateIdentityMceAndFusedPlePlans(node, i, order, weightEncoderCache, plans);
    }
    for (const PartV1::PleOnlyInfo& i : stripeInfos.m_PleOnlyInfos)
    {
        CreateFuseOnlyPlans(node, i, order, plans);
    }
}

void PartV1::GenerateFormatConversionPlans(Node* node,
                                           TraversalOrder order,
                                           StripeInfos& stripeInfos,
                                           Location inputBufferLocaton,
                                           Location outputBufferLocation,
                                           Plans& plans) const
{
    for (auto i : stripeInfos.m_DmaOnlyInfos)
    {
        if (inputBufferLocaton == Location::Dram)
        {
            i.m_Input.m_Range = { 0, 0 };
            i.m_Input.m_Shape = { 0, 0, 0, 0 };
        }
        if (outputBufferLocation == Location::Dram)
        {
            i.m_Output.m_Range = { 0, 0 };
            i.m_Output.m_Shape = { 0, 0, 0, 0 };
        }
        for (auto numInputStripes = i.m_Input.m_Range.m_Min; numInputStripes <= i.m_Input.m_Range.m_Max;
             ++numInputStripes)
        {
            for (auto numOutputStripes = i.m_Output.m_Range.m_Min; numOutputStripes <= i.m_Output.m_Range.m_Max;
                 ++numOutputStripes)
            {
                impl::NumMemoryStripes numMemoryStripes;
                numMemoryStripes.m_Input  = numInputStripes;
                numMemoryStripes.m_Output = numOutputStripes;
                numMemoryStripes.m_Weight = 0;
                CreateFormatConversionPlans(node, i, numMemoryStripes, order, inputBufferLocaton, outputBufferLocation,
                                            plans);
            }
        }
    }
}

void PartV1::GenerateWithNumStripes(Node* node,
                                    TraversalOrder order,
                                    PartV1::StripeInfos& stripeInfos,
                                    WeightEncoderCache& weightEncoderCache,
                                    Plans& plans) const
{
    if (IsObjectOfType<MceOperationNode>(node))
    {
        GenerateMcePlans(node, order, stripeInfos, weightEncoderCache, plans);
    }
    else if (IsObjectOfType<FuseOnlyPleOperationNode>(node))
    {
        GenerateFuseOnlyPlePlans(node, order, stripeInfos, weightEncoderCache, plans);
    }
    else if (IsObjectOfType<FormatConversionNode>(node))
    {
        auto format = node->GetFormat();
        switch (format)
        {
            case CompilerDataFormat::NHWC:
                GenerateFormatConversionPlans(node, order, stripeInfos, Location::Sram, Location::Dram, plans);
                break;
            case CompilerDataFormat::NHWCB:
                GenerateFormatConversionPlans(node, order, stripeInfos, Location::Dram, Location::Sram, plans);
                break;
            default:
                break;
        }
    }
}

utils::Optional<ethosn::command_stream::MceOperation> PartV1::GetMceOperation() const
{
    utils::Optional<ethosn::command_stream::MceOperation> res;
    Node* node = m_SubGraph.front();
    assert(node);
    MceOperationNode* mceNode = dynamic_cast<MceOperationNode*>(node);
    if (mceNode)
    {
        res = mceNode->GetOperation();
    }
    return res;
}

namespace
{

std::string GetLabel(InputNode*, DetailLevel)
{
    return "InputNode";
}

std::string GetLabel(OutputNode*, DetailLevel)
{
    return "OutputNode";
}

std::string GetLabel(ConstantNode*, DetailLevel)
{
    return "ConstantNode";
}

std::string GetLabel(MceOperationNode* node, DetailLevel detailLevel)
{
    std::string label = "MceOperationNode";
    if (detailLevel == DetailLevel::High)
    {
        label += "\n";
        label += ToString(node->GetOperation());
    }
    return label;
}

std::string GetLabel(FuseOnlyPleOperationNode* node, DetailLevel detailLevel)
{
    std::string label = "FuseOnlyPleOperationNode";
    if (detailLevel == DetailLevel::High)
    {
        label += "\n";
        label += ToString(node->GetKernelOperation());
    }
    return label;
}

std::string GetLabel(StandalonePleOperationNode* node, DetailLevel detailLevel)
{
    std::string label = "StandalonePleOperationNode";
    if (detailLevel == DetailLevel::High)
    {
        label += "\n";
        label += ToString(node->GetKernelOperation());
    }
    return label;
}

std::string GetLabel(McePostProcessOperationNode*, DetailLevel)
{
    return "McePostProcessOperationNode";
}

std::string GetLabel(SoftmaxNode*, DetailLevel)
{
    return "SoftmaxNode";
}

std::string GetLabel(RequantizeNode*, DetailLevel)
{
    return "RequantizeNode";
}

std::string GetLabel(FormatConversionNode*, DetailLevel)
{
    return "FormatConversionNode";
}

std::string GetLabel(ReinterpretNode*, DetailLevel)
{
    return "ReinterpretNode";
}

std::string GetLabel(ConcatNode*, DetailLevel)
{
    return "ConcatNode";
}

std::string GetLabel(ExtractSubtensorNode*, DetailLevel)
{
    return "ExtractSubtensorNode";
}

std::string GetLabel(EstimateOnlyNode*, DetailLevel)
{
    return "EstimateOnlyNode";
}

}    // namespace

ethosn::support_library::DotAttributes PartV1::GetDotAttributes(DetailLevel detailLevel) const
{
    DotAttributes result = BasePart::GetDotAttributes(detailLevel);
    result.m_Label       = "PartV1: " + result.m_Label;
    result.m_Shape       = "oval";

    std::stringstream label;
    assert(m_SubGraph.size() == 1);
    Node* node = m_SubGraph[0];
    label << "Node " + std::to_string(node->GetId()) + "\n";

    InputNode* inputNode                          = dynamic_cast<InputNode*>(node);
    OutputNode* outputNode                        = dynamic_cast<OutputNode*>(node);
    ConstantNode* constantNode                    = dynamic_cast<ConstantNode*>(node);
    MceOperationNode* mceNode                     = dynamic_cast<MceOperationNode*>(node);
    FuseOnlyPleOperationNode* fusePleNode         = dynamic_cast<FuseOnlyPleOperationNode*>(node);
    StandalonePleOperationNode* standalonePleNode = dynamic_cast<StandalonePleOperationNode*>(node);
    McePostProcessOperationNode* mcePpNode        = dynamic_cast<McePostProcessOperationNode*>(node);
    SoftmaxNode* softmaxNode                      = dynamic_cast<SoftmaxNode*>(node);
    RequantizeNode* requantNode                   = dynamic_cast<RequantizeNode*>(node);
    FormatConversionNode* formatNode              = dynamic_cast<FormatConversionNode*>(node);
    ReinterpretNode* reinterpretNode              = dynamic_cast<ReinterpretNode*>(node);
    ConcatNode* concatNode                        = dynamic_cast<ConcatNode*>(node);
    ExtractSubtensorNode* extractSubtensorNode    = dynamic_cast<ExtractSubtensorNode*>(node);
    EstimateOnlyNode* estimateNode                = dynamic_cast<EstimateOnlyNode*>(node);

    if (inputNode)
    {
        label << GetLabel(inputNode, detailLevel);
    }
    else if (outputNode)
    {
        label << GetLabel(outputNode, detailLevel);
    }
    else if (constantNode)
    {
        label << GetLabel(constantNode, detailLevel);
    }
    else if (mceNode)
    {
        label << GetLabel(mceNode, detailLevel);
    }
    else if (fusePleNode)
    {
        label << GetLabel(fusePleNode, detailLevel);
    }
    else if (standalonePleNode)
    {
        label << GetLabel(standalonePleNode, detailLevel);
    }
    else if (mcePpNode)
    {
        label << GetLabel(mcePpNode, detailLevel);
    }
    else if (softmaxNode)
    {
        label << GetLabel(softmaxNode, detailLevel);
    }
    else if (requantNode)
    {
        label << GetLabel(requantNode, detailLevel);
    }
    else if (formatNode)
    {
        label << GetLabel(formatNode, detailLevel);
    }
    else if (reinterpretNode)
    {
        label << GetLabel(reinterpretNode, detailLevel);
    }
    else if (concatNode)
    {
        label << GetLabel(concatNode, detailLevel);
    }
    else if (extractSubtensorNode)
    {
        label << GetLabel(extractSubtensorNode, detailLevel);
    }
    else if (estimateNode)
    {
        label << GetLabel(estimateNode, detailLevel);
    }

    if (detailLevel == DetailLevel::High)
    {
        label << "\n";
        label << "CorrespondingOperationIds:";
        for (auto id : node->GetCorrespondingOperationIds())
        {
            label << " " << id;
        }
        label << "\n";

        label << "Shape = " << ToString(node->GetShape()) << "\n";
        label << "Format = " << ToString(node->GetFormat()) << "\n";
        label << "CompressedFormat = " << ToString(node->GetCompressedFormat()) << "\n";
    }
    result.m_Label += "\n" + label.str();

    return result;
}

}    // namespace support_library
}    // namespace ethosn
