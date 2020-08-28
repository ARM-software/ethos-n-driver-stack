//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Part.hpp"

#include "../Graph.hpp"
#include "GraphNodes.hpp"
#include "Plan.hpp"
#include "Utils.hpp"
#include "WeightEncoder.hpp"

#include <unordered_map>

using namespace std;

namespace ethosn
{
namespace support_library
{

class WeightEncoderCache
{
public:
    WeightEncoderCache(const HardwareCapabilities& caps)
        : m_Encoder(WeightEncoder::CreateWeightEncoder(caps))
    {}

    struct Params
    {
        TensorInfo weightsTensorInfo;
        std::vector<uint8_t> weightsData;
        TensorInfo biasTensorInfo;
        std::vector<int32_t> biasData;
        QuantizationInfo inputQuantizationInfo;
        QuantizationInfo outputQuantizationInfo;
        uint32_t stripeDepth;
        uint32_t strideY;
        uint32_t strideX;
        uint32_t paddingTop;
        uint32_t paddingLeft;
        uint32_t iterationSize;
        ethosn::command_stream::MceOperation operation;
        CompilerMceAlgorithm algorithm;

        bool operator==(const Params& r) const
        {
            return weightsTensorInfo == r.weightsTensorInfo && weightsData == r.weightsData &&
                   biasTensorInfo == r.biasTensorInfo && biasData == r.biasData &&
                   inputQuantizationInfo == r.inputQuantizationInfo &&
                   outputQuantizationInfo == r.outputQuantizationInfo && stripeDepth == r.stripeDepth &&
                   strideY == r.strideY && strideX == r.strideX && paddingTop == r.paddingTop &&
                   paddingLeft == r.paddingLeft && iterationSize == r.iterationSize && operation == r.operation &&
                   algorithm == r.algorithm;
        }
    };

    EncodedWeights Encode(const Params& params)
    {
        auto it = m_Entries.find(params);
        if (it == m_Entries.end())
        {
            EncodedWeights w =
                m_Encoder->Encode(params.weightsTensorInfo, params.weightsData.data(), params.biasTensorInfo,
                                  params.biasData.data(), params.inputQuantizationInfo, params.outputQuantizationInfo,
                                  params.stripeDepth, params.strideY, params.strideX, params.paddingTop,
                                  params.paddingLeft, params.iterationSize, params.operation, params.algorithm);
            m_Entries[params] = w;
            return w;
        }
        else
        {
            return it->second;
        }
    }

private:
    struct Hasher
    {
        size_t operator()(const Params& p) const
        {
            // This hash function is deliberately very simple and therefore you might think would lead to lots of
            // collisions. However, because we are only using it in the context of a single Part, the differences
            // between each set of encoding params will be mostly (wholly?) in these fields that we are hashing,
            // and not the exact weight values
            size_t h = 17;
            h        = h * 37 + std::hash<size_t>()(p.weightsData.size());
            h        = h * 37 + std::hash<size_t>()(p.biasData.size());
            h        = h * 37 + std::hash<uint32_t>()(p.stripeDepth);
            h        = h * 37 + std::hash<uint32_t>()(p.iterationSize);
            // Note we cast the enum to an integral type, as some compilers (e.g. aarch64-linux-gnu-g++ 5.3.1)
            // don't support using the enum type directly, even though the spec indicates that they should.
            h = h * 37 + std::hash<uint32_t>()(static_cast<uint32_t>(p.algorithm));
            return h;
        }
    };

    std::unique_ptr<WeightEncoder> m_Encoder;
    std::unordered_map<Params, EncodedWeights, Hasher> m_Entries;
};

bool Part::NumStripes::operator<(const NumStripes& rhs) const
{
    if (minInputStripes < rhs.minInputStripes)
        return true;
    if (rhs.minInputStripes < minInputStripes)
        return false;
    if (maxInputStripes < rhs.maxInputStripes)
        return true;
    if (rhs.maxInputStripes < maxInputStripes)
        return false;
    if (minOutputStripes < rhs.minOutputStripes)
        return true;
    if (rhs.minOutputStripes < minOutputStripes)
        return false;
    if (maxOutputStripes < rhs.maxOutputStripes)
        return true;
    if (rhs.maxOutputStripes < maxOutputStripes)
        return false;
    if (minWeightStripes < rhs.minWeightStripes)
        return true;
    if (rhs.minWeightStripes < minWeightStripes)
        return false;
    if (maxWeightStripes < rhs.maxWeightStripes)
        return true;
    if (rhs.maxWeightStripes < maxWeightStripes)
        return false;
    return false;
}

bool Part::StripeInfos::operator<(const StripeInfos& rhs) const
{
    if (m_InputStripeShape < rhs.m_InputStripeShape)
        return true;
    if (rhs.m_InputStripeShape < m_InputStripeShape)
        return false;
    if (m_OutputStripeShape < rhs.m_OutputStripeShape)
        return true;
    if (rhs.m_OutputStripeShape < m_OutputStripeShape)
        return false;
    if (m_NumStripes < rhs.m_NumStripes)
        return true;
    if (rhs.m_NumStripes < m_NumStripes)
        return false;
    return false;
}

std::unique_ptr<Op> CreateOpFromNode(const Node* node)
{
    if (IsObjectOfType<MceOperationNode>(node))
    {
        const MceOperationNode* mceOperationNode = dynamic_cast<const MceOperationNode*>(node);
        MceOp op(Lifetime::Atomic, mceOperationNode->GetOperation(), CompilerMceAlgorithm::Direct,
                 BlockConfig{ 8U, 8U }, TensorShape{}, TensorShape{}, TensorShape{}, TraversalOrder::Xyz,
                 mceOperationNode->GetStride(), mceOperationNode->GetPadLeft(), mceOperationNode->GetPadTop());
        return std::make_unique<MceOp>(std::move(op));
    }
    else if (IsObjectOfType<McePostProcessOperationNode>(node))
    {
        return std::make_unique<MceOp>();
    }
    else if (IsObjectOfType<FuseOnlyPleOperationNode>(node))
    {
        const FuseOnlyPleOperationNode* fuseOnlyPleOperationNode = dynamic_cast<const FuseOnlyPleOperationNode*>(node);
        PleOp op(Lifetime::Atomic, fuseOnlyPleOperationNode->GetKernelOperation(), BlockConfig{ 8U, 8U },
                 static_cast<uint32_t>(fuseOnlyPleOperationNode->GetInputs().size()), std::vector<TensorShape>{},
                 TensorShape{});
        return std::make_unique<PleOp>(std::move(op));
    }
    else if (IsObjectOfType<StandalonePleOperationNode>(node))
    {
        const StandalonePleOperationNode* standalonePleOperationNode =
            dynamic_cast<const StandalonePleOperationNode*>(node);
        PleOp op(Lifetime::Atomic, standalonePleOperationNode->GetKernelOperation(), BlockConfig{ 8U, 8U },
                 static_cast<uint32_t>(standalonePleOperationNode->GetInputs().size()), std::vector<TensorShape>{},
                 TensorShape{});
        return std::make_unique<PleOp>(std::move(op));
    }
    else if (IsObjectOfType<FormatConversionNode>(node))
    {
        return std::make_unique<DmaOp>();
    }
    else if (IsObjectOfType<EstimateOnlyNode>(node) || IsObjectOfType<ReinterpretNode>(node))
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

CompilerDataFormat GetFormat(Location location)
{
    switch (location)
    {
        case Location::Dram:
            return CompilerDataFormat::NHWC;
        case Location::PleInputSram:
        case Location::Sram:
            return CompilerDataFormat::NHWCB;
        case Location::VirtualSram:
            return CompilerDataFormat::NHWC;
        default:
            throw NotSupportedException("Unkwnown location");
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
    if (IsObjectOfType<MceOperationNode>(node))
    {
        return dynamic_cast<const MceOperationNode*>(node)->GetWeightsInfo();
    }

    return TensorInfo();
}

TensorShape GetWeightsShape(const Node* node)
{
    return GetWeightsInfo(node).m_Dimensions;
}

uint32_t CalculateBufferSize(const TensorShape& shape, CompilerDataFormat f)
{
    switch (f)
    {
        case CompilerDataFormat::NHWCB:
            return utils::TotalSizeBytesNHWCB(shape);
        case CompilerDataFormat::NHWC:
            return utils::TotalSizeBytes(shape);
        default:
            assert(false);
            return 0;
    }
}

uint32_t CalculateSizeInBytes(const TensorShape& shape)
{
    return utils::TotalSizeBytesNHWCB(shape);
}

uint32_t CalculateTileSize(const HardwareCapabilities& caps,
                           const TensorShape& tensorShape,
                           const TensorShape& stripeShape,
                           uint32_t numStripes)
{
    // Restrict the tile max size to be the full tensor so we don't waste space when we have partial stripes
    const uint32_t inputFullStripeSize = numStripes * CalculateSizeInBytes(stripeShape);
    const uint32_t inputTileSize       = utils::MaxTileSize(tensorShape, caps);

    return std::min(inputTileSize, inputFullStripeSize);
}

bool IsPlanValid(const Plan& plan)
{
    (void)plan;
    return true;
}

const Plan& Part::GetPlan(const PlanId id) const
{
    assert(id < m_Plans.size());
    return *m_Plans.at(id).get();
}

size_t Part::GetNumPlans() const
{
    return m_Plans.size();
}

std::vector<const Edge*> Part::GetInputs() const
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

std::vector<const Edge*> Part::GetOutputs() const
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

InPart GraphOfParts::GetInputPart(const Edge& e) const
{
    const size_t numParts = m_Parts.size();

    for (PartId p = 0; p < numParts; ++p)
    {
        const Part& part = GetPart(p);

        for (uint32_t i = 0; i < part.GetInputs().size(); ++i)
        {
            const Edge* frEdge = part.GetInputs().at(i);
            if (&e == frEdge)
            {
                return std::make_pair(true, p);
            }
        }
    }
    return std::make_pair(false, 0);
}

OutPart GraphOfParts::GetOutputPart(const Edge& e) const
{
    const size_t numParts = m_Parts.size();

    for (PartId p = 0; p < numParts; ++p)
    {
        const Part& part = GetPart(p);

        for (uint32_t i = 0; i < part.GetOutputs().size(); ++i)
        {
            const Edge* bkEdge = part.GetOutputs().at(i);
            if (&e == bkEdge)
            {
                return std::make_pair(true, p);
            }
        }
    }
    return std::make_pair(false, 0);
}

size_t GraphOfParts::GetNumParts() const
{
    return m_Parts.size();
}

const Part& GraphOfParts::GetPart(const PartId id) const
{
    assert(id < m_Parts.size());
    return *m_Parts.at(id).get();
}

const Parts& GraphOfParts::GetParts() const
{
    return m_Parts;
}

void Part::CreatePlans(const HardwareCapabilities& caps)
{
    using DataFormats                      = std::list<CompilerDataFormat>;
    const DataFormats supportedDataFormats = { CompilerDataFormat::NHWC, CompilerDataFormat::NHWCB };

    Node* node = m_SubGraph.front();
    if (IsObjectOfType<InputNode>(node))
    {
        CreatePlanForInputNode(node, Lifetime::Atomic, TraversalOrder::Xyz);
    }
    else if (IsObjectOfType<OutputNode>(node))
    {
        CreatePlanForOutputNode(node, Lifetime::Atomic, TraversalOrder::Xyz);
    }
    else
    {
        WeightEncoderCache weightEncoderCache(caps);
        GenerateWithTraversalOrders(node, caps, weightEncoderCache);
    }

    if (m_Plans.empty())
    {
        throw NotSupportedException("No plans generated for this part");
    }
}

void Part::AddNewPlan(Plan::InputMapping&& inputMappings, Plan::OutputMapping&& outputMappings, OwnedOpGraph&& opGraph)
{
    auto plan       = std::make_unique<Plan>(std::move(inputMappings), std::move(outputMappings));
    plan->m_OpGraph = std::move(opGraph);
    if (IsPlanValid(*plan))
    {
        m_Plans.push_back(std::move(plan));
    }
}

void Part::CreatePlanForInputNode(Node* node, Lifetime lifetime, TraversalOrder order)
{
    Plan::InputMapping inputMappings;
    Plan::OutputMapping outputMappings;
    OwnedOpGraph opGraph;

    auto buffer                  = std::make_unique<Buffer>(lifetime, Location::Dram, node->GetFormat(), order);
    buffer->m_TensorShape        = node->GetShape();
    buffer->m_SizeInBytes        = CalculateBufferSize(node->GetShape(), node->GetFormat());
    buffer->m_QuantizationInfo   = node->GetQuantizationInfo();
    outputMappings[buffer.get()] = node;
    opGraph.AddBuffer(std::move(buffer));
    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph));
}

void Part::CreatePlanForOutputNode(Node* node, Lifetime lifetime, TraversalOrder order)
{
    Plan::InputMapping inputMappings;
    Plan::OutputMapping outputMappings;
    OwnedOpGraph opGraph;

    assert(node->GetInputs().size() > 0);
    for (Edge* edge : node->GetInputs())
    {
        std::unique_ptr<Buffer> buffer =
            std::make_unique<Buffer>(lifetime, Location::Dram, edge->GetSource()->GetFormat(), order);
        buffer->m_TensorShape       = edge->GetSourceShape();
        buffer->m_SizeInBytes       = CalculateBufferSize(edge->GetSourceShape(), edge->GetSource()->GetFormat());
        buffer->m_QuantizationInfo  = edge->GetSource()->GetQuantizationInfo();
        inputMappings[buffer.get()] = edge;
        opGraph.AddBuffer(std::move(buffer));
    }
    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph));
}

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

TensorShape CalculateWeightStripeShape(const TensorInfo& weightInfo, TensorShape inputStripe, TensorShape outputStripe)
{
    assert(weightInfo.m_DataFormat == DataFormat::HWIO || weightInfo.m_DataFormat == DataFormat::HWIM);

    TensorShape result = weightInfo.m_Dimensions;
    result[2]          = inputStripe[3];
    // For HWIO the weight stripe "O" dimension needs to match the output tensor stripe size
    // For HWIM it should be 1 as we don't support channel multiplier > 1.
    if (weightInfo.m_DataFormat == DataFormat::HWIO)
    {
        result[3] = outputStripe[3];
    }
    else
    {
        assert(result[3] == 1);
    }
    return result;
}

/// Generates a stripe shape given an encoding and an input tensor
/// Tries to create a stripe with the stripe shape in the encoding, if the dimension is 0 then it uses the full length of that dimension.
TensorShape CreateStripe(TensorShape input, TensorShape inputEncoding, const HardwareCapabilities& caps)
{
    TensorShape inputStripeShape;
    for (uint32_t i = 0; i < input.size(); ++i)
    {
        inputStripeShape[i] = inputEncoding[i] != 0 ? inputEncoding[i] : input[i];
    }
    inputStripeShape    = utils::RoundUpHeightAndWidthToBrickGroup(inputStripeShape);
    inputStripeShape[3] = utils::RoundUpToNearestMultiple(inputStripeShape[3], caps.GetNumberOfSrams());
    return inputStripeShape;
}

}    // namespace

void AddWeightBuffersAndDmaOpToMceOp(OwnedOpGraph& opGraph,
                                     const TensorShape& inpStripeShape,
                                     const TensorShape& outStripeShape,
                                     const uint32_t numWeightStripes,
                                     const TensorInfo& weightInfo,
                                     const std::vector<uint8_t>& weightData,
                                     const TensorInfo& biasInfo,
                                     const std::vector<int32_t>& biasData,
                                     Lifetime lifetime,
                                     TraversalOrder order,
                                     WeightEncoderCache& weightEncoderCache)
{
    const OpGraph::BufferList& buffers = opGraph.GetBuffers();
    const OpGraph::OpList& ops         = opGraph.GetOps();
    Op* op                             = ops.front();

    assert(dynamic_cast<MceOp*>(op) != nullptr);
    MceOp* mceOp = dynamic_cast<MceOp*>(op);

    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Dram,
                                               ConvertExternalToCompilerDataFormat(weightInfo.m_DataFormat), order));
    Buffer* weightsBufferInDram        = buffers.back();
    weightsBufferInDram->m_TensorShape = weightInfo.m_Dimensions;
    weightsBufferInDram->m_StripeShape = CalculateWeightStripeShape(weightInfo, inpStripeShape, outStripeShape);

    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, CompilerDataFormat::NHWCB, order));
    Buffer* weightsBufferInSram        = buffers.back();
    weightsBufferInSram->m_TensorShape = weightsBufferInDram->m_TensorShape;

    weightsBufferInSram->m_StripeShape      = CalculateWeightStripeShape(weightInfo, inpStripeShape, outStripeShape);
    weightsBufferInSram->m_Format           = CompilerDataFormat::WEIGHT;
    weightsBufferInSram->m_QuantizationInfo = weightInfo.m_QuantizationInfo;
    weightsBufferInSram->m_NumStripes       = numWeightStripes;

    opGraph.AddOp(std::make_unique<DmaOp>());
    Op* dmaOp                   = ops.back();
    mceOp->m_InputStripeShape   = inpStripeShape;
    mceOp->m_OutputStripeShape  = outStripeShape;
    mceOp->m_WeightsStripeShape = weightsBufferInSram->m_StripeShape;

    opGraph.AddConsumer(weightsBufferInDram, dmaOp, 0);
    opGraph.SetProducer(weightsBufferInSram, dmaOp);
    opGraph.AddConsumer(weightsBufferInSram, op, 1);

    // Encode weights
    const uint32_t weightStripeSize  = mceOp->m_WeightsStripeShape[2];
    const uint32_t weightStripeDepth = GetWeightStripeDepth(weightInfo, mceOp);

    Buffer* mceOutput = opGraph.GetOutput(mceOp);
    Buffer* mceInput  = opGraph.GetInputs(mceOp)[0];

    WeightEncoderCache::Params wp;
    wp.weightsTensorInfo                  = weightInfo;
    wp.weightsData                        = weightData;
    wp.biasTensorInfo                     = biasInfo;
    wp.biasData                           = biasData;
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
    weightsBufferInDram->m_EncodedWeights = std::make_unique<EncodedWeights>(weightEncoderCache.Encode(wp));

    // Use the encoded weights to determine the size of the sram and dram buffers
    weightsBufferInDram->m_SizeInBytes = static_cast<uint32_t>(weightsBufferInDram->m_EncodedWeights->m_Data.size());
    weightsBufferInSram->m_SizeInBytes = weightsBufferInDram->m_EncodedWeights->m_MaxSize * numWeightStripes;
}

Buffer* Part::AddIdentityMceOpForSubGraph(OwnedOpGraph& opGraph,
                                          const TensorShape& inpShape,
                                          const QuantizationInfo& inpQuantInfo,
                                          Lifetime lifetime,
                                          const HardwareCapabilities& caps,
                                          TraversalOrder order,
                                          TensorShape inputStripe,
                                          TensorShape outputStripe,
                                          NumStripesType numInputStripes,
                                          NumStripesType numWeightStripes,
                                          WeightEncoderCache& weightEncoderCache)
{
    const OpGraph::BufferList& buffers = opGraph.GetBuffers();
    const OpGraph::OpList& ops         = opGraph.GetOps();

    const float weightScale = 0.5f;
    const float biasScale   = weightScale * inpQuantInfo.GetScale();
    const uint32_t numIfm   = inpShape[3];

    TensorInfo weightInfo{ { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale } };
    TensorInfo biasInfo{ { 1, 1, 1, numIfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };

    TensorShape weightStripe = CalculateWeightStripeShape(weightInfo, inputStripe, outputStripe);

    std::vector<uint8_t> weightsData(1 * 1 * 1 * numIfm, 2);
    std::vector<int32_t> biasData(numIfm, 0);

    // Add MceOp.
    opGraph.AddOp(std::make_unique<MceOp>(Lifetime::Atomic, MceOperation::DEPTHWISE_CONVOLUTION,
                                          CompilerMceAlgorithm::Direct, BlockConfig{ 8U, 8U }, inputStripe,
                                          outputStripe, weightStripe, order, Stride(1, 1), 0, 0));
    Op* idMceOp = ops.back();

    // Add input Buffer.
    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, CompilerDataFormat::NHWCB, order));
    Buffer* idMceOpInBuff = buffers.back();

    // Add Output Buffer.
    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::PleInputSram, CompilerDataFormat::NHWCB, order));
    Buffer* idMceOpOutBuff = buffers.back();

    opGraph.AddConsumer(idMceOpInBuff, idMceOp, 0);
    opGraph.SetProducer(idMceOpOutBuff, idMceOp);

    // Add Weight buffers and DmaOp.
    AddWeightBuffersAndDmaOpToMceOp(opGraph, inputStripe, outputStripe, numWeightStripes, weightInfo, weightsData,
                                    biasInfo, biasData, lifetime, order, weightEncoderCache);

    // Set Input & Output buffer shapes and sizes.
    idMceOpOutBuff->m_TensorShape = inpShape;
    idMceOpInBuff->m_TensorShape  = inpShape;
    idMceOpOutBuff->m_StripeShape = outputStripe;
    idMceOpInBuff->m_StripeShape  = inputStripe;
    idMceOpOutBuff->m_SizeInBytes = 0;    // The output buffer is in ple sram so has no size in the tile
    idMceOpInBuff->m_SizeInBytes  = CalculateTileSize(caps, inpShape, idMceOpInBuff->m_StripeShape, numInputStripes);
    idMceOpOutBuff->m_QuantizationInfo = inpQuantInfo;
    idMceOpInBuff->m_QuantizationInfo  = inpQuantInfo;
    idMceOpOutBuff->m_NumStripes       = 0;
    idMceOpInBuff->m_NumStripes        = numInputStripes;

    return idMceOpOutBuff;
}

void Part::CreatePlanWithIdentityMceOp(FuseOnlyPleOperationNode* node,
                                       Lifetime lifetime,
                                       const HardwareCapabilities& caps,
                                       TraversalOrder order,
                                       TensorShape inputStripe,
                                       TensorShape outputStripe,
                                       NumStripesType numOutputStripes,
                                       WeightEncoderCache& weightEncoderCache)
{
    // By definition MceOp are single input single output
    assert(node->GetInputs().size() == 1);
    assert(node->GetOutputs().size() == 1);

    Node* inputNode  = node->GetInput(0)->GetSource();
    Node* outputNode = m_SubGraph.back();
    assert(inputNode);
    assert(outputNode);

    const TensorShape outShape = outputNode->GetShape();

    // Generate the input stripes needed for the mce op
    TensorShape encoding          = { 0, inputStripe[1], inputStripe[2], inputStripe[3] };
    const TensorShape& inputShape = node->GetInputShape(0);
    TensorShape mceInputStripe    = CreateStripe(inputShape, encoding, caps);
    // Identity MceOps have a kernel size of 1x1 so only require 1-2 stripes
    // They output into PLE SRAM so there are no output stripes
    NumStripes numStripes = { 1U, 2U, 0U, 0U, 1U, 2U };

    for (auto numInputStripes = numStripes.minInputStripes; numInputStripes <= numStripes.maxInputStripes;
         ++numInputStripes)
    {
        for (auto numWeightStripes = numStripes.minWeightStripes; numWeightStripes <= numStripes.maxWeightStripes;
             ++numWeightStripes)
        {
            OwnedOpGraph opGraph;
            Plan::InputMapping inputMappings;
            Plan::OutputMapping outputMappings;
            // Add Identity MCeOp.
            auto mceOpOutputBuffer = AddIdentityMceOpForSubGraph(opGraph, inputShape, inputNode->GetQuantizationInfo(),
                                                                 lifetime, caps, order, mceInputStripe, mceInputStripe,
                                                                 numInputStripes, numWeightStripes, weightEncoderCache);
            // Add PleOp
            opGraph.AddOp(CreateOpFromNode(node));
            Op* op                             = opGraph.GetOps().back();
            const OpGraph::BufferList& buffers = opGraph.GetBuffers();

            assert(dynamic_cast<PleOp*>(op) != nullptr);
            op->m_Lifetime = lifetime;

            Buffer* outBuffer       = opGraph.AddBuffer(std::make_unique<Buffer>(
                lifetime, Location::Sram, CompilerDataFormat::NHWCB, outShape, outputStripe, order,
                CalculateTileSize(caps, outShape, outputStripe, numOutputStripes), outputNode->GetQuantizationInfo()));
            outBuffer->m_NumStripes = numOutputStripes;

            opGraph.AddConsumer(mceOpOutputBuffer, op, 0);
            opGraph.SetProducer(outBuffer, op);

            inputMappings[buffers.front()] = node->GetInput(0);
            outputMappings[outBuffer]      = outputNode;

            PleOp* pleOp = dynamic_cast<PleOp*>(op);
            pleOp->m_InputStripeShapes.push_back(mceOpOutputBuffer->m_StripeShape);
            pleOp->m_OutputStripeShape = outBuffer->m_StripeShape;

            AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph));
        }
    }
}

void Part::CreatePlanWithIdentityPleOp(Node* node,
                                       Lifetime lifetime,
                                       const HardwareCapabilities& caps,
                                       TraversalOrder order,
                                       TensorShape inputStripe,
                                       TensorShape outputStripe,
                                       NumStripesType numInputStripes,
                                       NumStripesType numOutputStripes,
                                       NumStripesType numWeightStripes,
                                       Location inputBufferLocation,
                                       Location outputBufferLocation,
                                       WeightEncoderCache& weightEncoderCache)
{
    assert(node->GetInputs().size() > 0);

    // We need to generate 1-4 output stripes because we may need boundary data + double buffering
    NumStripes numStripes = { 0U, 0U, 1U, 4U, 0U, 0U };

    for (auto numPleOutputStripes = numStripes.minOutputStripes; numPleOutputStripes <= numStripes.maxOutputStripes;
         ++numPleOutputStripes)
    {
        Plan::InputMapping inputMappings;
        Plan::OutputMapping outputMappings;
        OwnedOpGraph opGraph;
        const OpGraph::BufferList& buffers = opGraph.GetBuffers();
        const OpGraph::OpList& ops         = opGraph.GetOps();

        AddOpToOpGraphWithInputOutputBuffers(opGraph, node, lifetime, caps, order, inputStripe, outputStripe,
                                             numInputStripes, numOutputStripes, inputBufferLocation,
                                             outputBufferLocation, inputMappings, outputMappings);
        Buffer* mceOutputBuff = buffers.back();
        Buffer* mceInputBuff  = buffers.front();

        // Add weights
        MceOperationNode* mceNode = GetObjectAs<MceOperationNode>(node);
        AddWeightBuffersAndDmaOpToMceOp(opGraph, mceInputBuff->m_StripeShape, mceOutputBuff->m_StripeShape,
                                        numWeightStripes, mceNode->GetWeightsInfo(), mceNode->GetWeightsData(),
                                        mceNode->GetBiasInfo(), mceNode->GetBiasData(), lifetime, order,
                                        weightEncoderCache);

        // Add Passthrough PleOp.node
        opGraph.AddOp(std::make_unique<PleOp>(Lifetime::Atomic, PleOperation::PASSTHROUGH, BlockConfig{ 1U, 1U }, 1,
                                              std::vector<TensorShape>{}, TensorShape{}));
        Op* op = ops.back();

        // Add output Buffer.
        opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, CompilerDataFormat::NHWCB,
                                                   mceOutputBuff->m_TensorShape, mceOutputBuff->m_StripeShape, order,
                                                   mceOutputBuff->m_SizeInBytes, mceOutputBuff->m_QuantizationInfo));
        Buffer* idPleOpOutBuff       = buffers.back();
        idPleOpOutBuff->m_NumStripes = numPleOutputStripes;

        opGraph.AddConsumer(mceOutputBuff, op, 0);
        opGraph.SetProducer(idPleOpOutBuff, op);

        PleOp* idPleOp = dynamic_cast<PleOp*>(op);
        assert(idPleOp);
        idPleOp->m_InputStripeShapes.push_back(mceOutputBuff->m_StripeShape);
        idPleOp->m_OutputStripeShape = mceOutputBuff->m_StripeShape;

        outputMappings[idPleOpOutBuff] = this->m_SubGraph.back();

        AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph));
    }
}

void Part::AddOpToOpGraphWithInputOutputBuffers(OwnedOpGraph& opGraph,
                                                Node* node,
                                                Lifetime lifetime,
                                                const HardwareCapabilities& caps,
                                                TraversalOrder order,
                                                TensorShape inputStripe,
                                                TensorShape outputStripe,
                                                NumStripesType numInputStripes,
                                                NumStripesType numOutputStripes,
                                                Location inputBufferLocation,
                                                Location outputBufferLocation,
                                                Plan::InputMapping& inputMappings,
                                                Plan::OutputMapping& outputMappings)
{
    (void)outputMappings;    //Currently unused but expected to be used whenever multi output will be supported

    opGraph.AddOp(CreateOpFromNode(node));

    const OpGraph::BufferList& buffers = opGraph.GetBuffers();
    const OpGraph::OpList& ops         = opGraph.GetOps();
    Op* op                             = ops.back();
    op->m_Lifetime                     = lifetime;
    for (Edge* edge : node->GetInputs())
    {
        opGraph.AddBuffer(
            std::make_unique<Buffer>(lifetime, inputBufferLocation, GetFormat(inputBufferLocation), order));
        Buffer* inBuffer        = buffers.back();
        const Node* inputNode   = edge->GetSource();
        inBuffer->m_TensorShape = inputNode->GetShape();
        inBuffer->m_StripeShape = inputStripe;
        inBuffer->m_NumStripes  = numInputStripes;
        inBuffer->m_SizeInBytes =
            inputBufferLocation == Location::Sram
                ? CalculateTileSize(caps, inBuffer->m_TensorShape, inBuffer->m_StripeShape, numInputStripes)
                : CalculateBufferSize(inBuffer->m_TensorShape, inBuffer->m_Format);
        inBuffer->m_QuantizationInfo = inputNode->GetQuantizationInfo();
        inputMappings[inBuffer]      = edge;
        opGraph.AddConsumer(inBuffer, op, 0);

        if (IsObjectOfType<PleOp>(op))
        {
            PleOp* pleOp = dynamic_cast<PleOp*>(op);
            pleOp->m_InputStripeShapes.push_back(inBuffer->m_StripeShape);
        }
    }

    if (IsObjectOfType<FormatConversionNode>(node) &&
        (inputBufferLocation == Location::VirtualSram || outputBufferLocation == Location::VirtualSram))
    {
        GetObjectAs<DmaOp>(op)->m_Location = Location::VirtualSram;
    }

    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, outputBufferLocation, GetFormat(outputBufferLocation), order));
    auto outBuffer = buffers.back();
    opGraph.SetProducer(outBuffer, op);

    auto outputNode          = m_SubGraph.back();
    outBuffer->m_TensorShape = outputNode->GetShape();
    outBuffer->m_StripeShape = outputStripe;
    outBuffer->m_NumStripes  = numOutputStripes;
    outBuffer->m_SizeInBytes =
        outputBufferLocation == Location::Sram
            ? CalculateTileSize(caps, outBuffer->m_TensorShape, outBuffer->m_StripeShape, numOutputStripes)
            : CalculateBufferSize(outBuffer->m_TensorShape, outBuffer->m_Format);
    outBuffer->m_QuantizationInfo = outputNode->GetQuantizationInfo();
}

void Part::CreatePlanForNode(Node* node,
                             Lifetime lifetime,
                             const HardwareCapabilities& caps,
                             TraversalOrder order,
                             TensorShape inputStripe,
                             TensorShape outputStripe,
                             NumStripesType numInputStripes,
                             NumStripesType numOutputStripes,
                             NumStripesType numWeightStripes,
                             Location inputBufferLocaton,
                             Location outputBufferLocation,
                             WeightEncoderCache& weightEncoderCache)
{
    assert(node->GetInputs().size() > 0);

    Plan::InputMapping inputMappings;
    Plan::OutputMapping outputMappings;
    OwnedOpGraph opGraph;
    auto& buffers = opGraph.GetBuffers();
    auto& ops     = opGraph.GetOps();

    AddOpToOpGraphWithInputOutputBuffers(opGraph, node, lifetime, caps, order, inputStripe, outputStripe,
                                         numInputStripes, numOutputStripes, inputBufferLocaton, outputBufferLocation,
                                         inputMappings, outputMappings);

    auto outBuffer = buffers.back();
    auto op        = ops.back();

    if (this->m_SubGraph.size() > 1 && IsObjectOfType<McePostProcessOperationNode>(this->m_SubGraph[1]))
    {
        opGraph.AddOp(CreateOpFromNode(this->m_SubGraph[1]));
        auto mcePpOp = ops.back();
        opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, CompilerDataFormat::NHWCB, order));
        auto mcePpOpBuffer = buffers.back();
        opGraph.AddConsumer(outBuffer, mcePpOp, 0);
        opGraph.SetProducer(mcePpOpBuffer, mcePpOp);
    }

    outputMappings[buffers.back()] = this->m_SubGraph.back();

    if (IsObjectOfType<MceOp>(op))
    {
        MceOperationNode* mceNode     = GetObjectAs<MceOperationNode>(node);
        const TensorInfo& weightsInfo = GetWeightsInfo(node);
        if (utils::GetNumElements(weightsInfo.m_Dimensions) > 0)
        {
            for (auto pair : inputMappings)
            {
                Buffer* inBuffer = pair.first;
                AddWeightBuffersAndDmaOpToMceOp(opGraph, inBuffer->m_StripeShape, outBuffer->m_StripeShape,
                                                numWeightStripes, mceNode->GetWeightsInfo(), mceNode->GetWeightsData(),
                                                mceNode->GetBiasInfo(), mceNode->GetBiasData(), lifetime, order,
                                                weightEncoderCache);
            }
        }
    }
    else if (IsObjectOfType<PleOp>(op))
    {
        PleOp* pleOp               = dynamic_cast<PleOp*>(op);
        pleOp->m_OutputStripeShape = outBuffer->m_StripeShape;
    }

    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph));

    // Check for Only FuseOnlyOperationNode exists  in this part, then add identity MceOp.
    bool idMceOpRequired = IsObjectOfType<FuseOnlyPleOperationNode>(node) && (this->m_SubGraph.size() == 1);
    if (idMceOpRequired)
    {
        // Add another plan with IdMceOp.
        CreatePlanWithIdentityMceOp(GetObjectAs<FuseOnlyPleOperationNode>(node), lifetime, caps, order, inputStripe,
                                    outputStripe, numOutputStripes, weightEncoderCache);
    }

    // Check for only MceOperationNode exists in this part.
    bool idPleOpRequired = IsObjectOfType<MceOperationNode>(node) && (this->m_SubGraph.size() == 1);
    if (idPleOpRequired)
    {
        // Add another plan with IdMceOp
        CreatePlanWithIdentityPleOp(node, lifetime, caps, order, inputStripe, outputStripe, numInputStripes,
                                    numOutputStripes, numWeightStripes, inputBufferLocaton, outputBufferLocation,
                                    weightEncoderCache);
    }
}

std::vector<BlockConfig> GenerateBlockConfigs(Node* node)
{
    std::vector<BlockConfig> result;

    // All block configs possible
    const std::vector<BlockConfig> allBlockConfigs = { { 16u, 16u }, { 32u, 8u }, { 8u, 32u },
                                                       { 16u, 8u },  { 8u, 16u }, { 8u, 8u } };

    result = allBlockConfigs;
    if (IsObjectOfType<MceOperationNode>(node))
    {
        if (GetObjectAs<MceOperationNode>(node)->GetOperation() == command_stream::MceOperation::FULLY_CONNECTED)
        {
            result = utils::Filter(allBlockConfigs, [](BlockConfig c) { return c == BlockConfig{ 8u, 8u }; });
        }
    }
    return result;
}

void Part::GenerateWithTraversalOrders(Node* node,
                                       const HardwareCapabilities& caps,
                                       WeightEncoderCache& weightEncoderCache)
{
    std::vector<BlockConfig> blockConfigs = GenerateBlockConfigs(node);
    GenerateWithStripeSizes(node, caps, blockConfigs, TraversalOrder::Xyz, weightEncoderCache);
    // TODO: Add the same function call with traversal order ZXY

    auto inputStripe  = CreateStripe(node->GetInputShape(0), TensorShape{ 0, 0, 0, 0 }, caps);
    auto outputStripe = CreateStripe(node->GetShape(), TensorShape{ 0, 0, 0, 0 }, caps);
    if (IsObjectOfType<FormatConversionNode>(node))
    {
        auto format = node->GetFormat();
        switch (format)
        {
            case CompilerDataFormat::NHWCB:
                CreatePlanForNode(node, Lifetime::Atomic, caps, TraversalOrder::Xyz, inputStripe, outputStripe, 1U, 1U,
                                  0u, Location::VirtualSram, Location::Sram, weightEncoderCache);
                break;
            case CompilerDataFormat::NHWC:
                CreatePlanForNode(node, Lifetime::Atomic, caps, TraversalOrder::Xyz, inputStripe, outputStripe, 1U, 1U,
                                  0u, Location::Sram, Location::VirtualSram, weightEncoderCache);
                break;
            default:
                throw NotSupportedException(
                    "Unsupported compiler data format. Only NHWC and NHWCB is currently handled.");
        }
    }
    else if (IsObjectOfType<ReinterpretNode>(node))
    {
        CreatePlanForNode(node, Lifetime::Atomic, caps, TraversalOrder::Xyz, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, 0U, 0U, 0u,
                          Location::Dram, Location::Dram, weightEncoderCache);
        CreatePlanForNode(node, Lifetime::Atomic, caps, TraversalOrder::Xyz, inputStripe, outputStripe, 1U, 1U, 0u,
                          Location::VirtualSram, Location::VirtualSram, weightEncoderCache);
    }
}

std::set<Part::StripeInfos> GenerateStripes(Node* node, const HardwareCapabilities& caps, const BlockConfig blockConfig)
{
    using namespace utils;

    // Note we use set rather than unordered_set to give consistent behaviour across STL implementations to make
    // debugging and testing easier.
    std::set<Part::StripeInfos> result;

    Part::NumStripes numStripes;

    if (IsObjectOfType<MceOperationNode>(node))
    {
        // MceOperations output to PLE SRAM so are no "stripes"
        // At least 3 input stripes are needed because of
        // data on the top and bottom. Weights can
        // have 1 or 2 for double buffering.
        numStripes        = { 3U, 4U, 0U, 0U, 1U, 2U };
        auto mceNode      = GetObjectAs<MceOperationNode>(node);
        auto kernelHeight = mceNode->GetWeightsInfo().m_Dimensions[0];
        if (kernelHeight == 1)
        {
            numStripes.minInputStripes = 1;
            numStripes.maxInputStripes = 2;
        }
    }
    else if (IsObjectOfType<FuseOnlyPleOperationNode>(node))
    {
        // The input to fuse only ple operations are in ple sram and have no "stripes"
        numStripes = { 0U, 0U, 1U, 4U, 0U, 0U };
    }
    else if (IsObjectOfType<StandalonePleOperationNode>(node))
    {
        throw NotSupportedException("Standalone PLE operations not yet supported");
    }
    else
    {
        // Format conversion and reinterpret need to be able to combine with the input of an MceOperation and the output of a FusedPleOperation
        numStripes = { 1U, 4U, 1U, 4U, 0U, 0U };
    }

    auto ApplyShapeMult = [&](TensorShape shape) {
        utils::ShapeMultiplier shapeMult = utils::g_IdentityShapeMultiplier;
        if (IsObjectOfType<FuseOnlyPleOperationNode>(node))
        {
            shapeMult = GetObjectAs<FuseOnlyPleOperationNode>(node)->GetShapeMultiplier();
        }
        return TensorShape{ shape[0], shape[1] * shapeMult.m_H, shape[2] * shapeMult.m_W, shape[3] * shapeMult.m_C };
    };

    auto AddStripeInfos = [&result](const TensorShape& inputStripeShape, const TensorShape& outputStripeShape,
                                    Part::NumStripes numStripes, const TensorShape& inputShape,
                                    const TensorShape& outputShape) {
        // Limit the max number of stripes based on the size of the tensor - there is no point considering plans where
        // we can store more stripes in the tile than there are in the tensor!
        numStripes.maxInputStripes  = std::min(numStripes.maxInputStripes,
                                              DivRoundUp(GetHeight(inputShape), GetHeight(inputStripeShape)) *
                                                  DivRoundUp(GetWidth(inputShape), GetWidth(inputStripeShape)) *
                                                  DivRoundUp(GetChannels(inputShape), GetChannels(inputStripeShape)));
        numStripes.maxOutputStripes = std::min(
            numStripes.maxOutputStripes, DivRoundUp(GetHeight(outputShape), GetHeight(outputStripeShape)) *
                                             DivRoundUp(GetWidth(outputShape), GetWidth(outputStripeShape)) *
                                             DivRoundUp(GetChannels(outputShape), GetChannels(outputStripeShape)));

        // Prevent using stripes which have more elements than the entire tensor
        if (utils::GetNumElements(inputStripeShape) < utils::GetNumElements(inputShape) ||
            utils::GetNumElements(outputStripeShape) < utils::GetNumElements(outputShape))
        {
            result.insert(Part::StripeInfos{ inputStripeShape, outputStripeShape, numStripes });
        }
    };

    // Use the minimum stripe size possible to minimize the time before processing
    // Try splitting height first
    {
        TensorShape inputEncoding     = { 0, blockConfig.m_BlockHeight(), 0, 0 };
        const TensorShape& inputShape = node->GetInputShape(0);
        TensorShape inputStripe       = CreateStripe(node->GetInputShape(0), inputEncoding, caps);

        TensorShape outputEncoding      = ApplyShapeMult(inputEncoding);
        TensorShape outputStripe        = CreateStripe(node->GetShape(), outputEncoding, caps);
        const TensorShape& outputShape  = node->GetShape();
        Part::NumStripes numStripesCopy = numStripes;
        numStripesCopy.maxWeightStripes = std::min(numStripes.maxWeightStripes, 1u);

        AddStripeInfos(inputStripe, outputStripe, numStripesCopy, inputShape, outputShape);
    }

    // Try splitting width
    {
        TensorShape inputEncoding     = { 0, 0, blockConfig.m_BlockWidth(), 0 };
        const TensorShape& inputShape = node->GetInputShape(0);
        TensorShape inputStripe       = CreateStripe(node->GetInputShape(0), inputEncoding, caps);

        TensorShape outputEncoding      = ApplyShapeMult(inputEncoding);
        TensorShape outputStripe        = CreateStripe(node->GetShape(), outputEncoding, caps);
        const TensorShape& outputShape  = node->GetShape();
        Part::NumStripes numStripesCopy = numStripes;
        numStripesCopy.maxWeightStripes = std::min(numStripes.maxWeightStripes, 1u);
        AddStripeInfos(inputStripe, outputStripe, numStripesCopy, inputShape, outputShape);
    }

    // Try splitting width and height
    {
        TensorShape inputEncoding     = { 0, blockConfig.m_BlockHeight(), blockConfig.m_BlockWidth(), 0 };
        const TensorShape& inputShape = node->GetInputShape(0);
        TensorShape inputStripe       = CreateStripe(node->GetInputShape(0), inputEncoding, caps);

        TensorShape outputEncoding      = ApplyShapeMult(inputEncoding);
        TensorShape outputStripe        = CreateStripe(node->GetShape(), outputEncoding, caps);
        const TensorShape& outputShape  = node->GetShape();
        Part::NumStripes numStripesCopy = numStripes;
        numStripesCopy.maxWeightStripes = std::min(numStripes.maxWeightStripes, 1u);
        AddStripeInfos(inputStripe, outputStripe, numStripesCopy, inputShape, outputShape);
    }

    if (IsObjectOfType<MceOperationNode>(node))
    {
        // Try split output depth
        {
            TensorShape inputEncoding     = { 0, 0, 0, 0 };
            const TensorShape& inputShape = node->GetInputShape(0);
            TensorShape inputStripe       = CreateStripe(node->GetInputShape(0), inputEncoding, caps);

            TensorShape outputEncoding = { 0, 0, 0, caps.GetNumberOfOfm() };
            TensorShape outputStripe   = CreateStripe(node->GetShape(), outputEncoding, caps);
            // We have the full input tensor so we only have 1 stripe
            Part::NumStripes numStripesCopy = numStripes;
            numStripesCopy.maxInputStripes  = std::min(numStripes.maxInputStripes, 1u);
            const TensorShape& outputShape  = node->GetShape();
            AddStripeInfos(inputStripe, outputStripe, numStripesCopy, inputShape, outputShape);
        }

        // Try split input depth
        // note we have to limit the height and width to the block size
        {
            TensorShape encoding          = { 0, blockConfig.m_BlockHeight(), blockConfig.m_BlockWidth(),
                                     caps.GetNumberOfOfm() };
            const TensorShape& inputShape = node->GetInputShape(0);
            TensorShape inputStripe       = CreateStripe(node->GetInputShape(0), encoding, caps);

            TensorShape outputStripe       = CreateStripe(node->GetShape(), encoding, caps);
            const TensorShape& outputShape = node->GetShape();
            AddStripeInfos(inputStripe, outputStripe, numStripes, inputShape, outputShape);
        }
    }
    else if (IsObjectOfType<FuseOnlyPleOperationNode>(node))
    {
        // Assume that the ple operations has a 1-1 mapping between input and output
        // Try split depth
        {
            TensorShape inputEncoding     = { 0, 0, 0, caps.GetNumberOfOfm() };
            const TensorShape& inputShape = node->GetInputShape(0);
            TensorShape inputStripe       = CreateStripe(node->GetInputShape(0), inputEncoding, caps);

            TensorShape outputEncoding = ApplyShapeMult(inputEncoding);
            TensorShape outputStripe   = CreateStripe(node->GetShape(), outputEncoding, caps);
            // We have the full input tensor so we only have 1 stripe
            Part::NumStripes numStripesCopy = numStripes;
            numStripesCopy.maxInputStripes  = std::min(numStripes.maxInputStripes, 1u);
            const TensorShape& outputShape  = node->GetShape();
            AddStripeInfos(inputStripe, outputStripe, numStripesCopy, inputShape, outputShape);
        }
        // Height width and depth for PLE operations assuming that they have a 1-1 mapping between inputs and outputs.
        // e.g. relu
        {
            TensorShape inputEncoding     = { 0, blockConfig.m_BlockHeight(), blockConfig.m_BlockWidth(),
                                          caps.GetNumberOfOfm() };
            const TensorShape& inputShape = node->GetInputShape(0);
            TensorShape inputStripe       = CreateStripe(node->GetInputShape(0), inputEncoding, caps);

            TensorShape outputEncoding     = ApplyShapeMult(inputEncoding);
            TensorShape outputStripe       = CreateStripe(node->GetShape(), outputEncoding, caps);
            const TensorShape& outputShape = node->GetShape();
            AddStripeInfos(inputStripe, outputStripe, numStripes, inputShape, outputShape);
        }
    }

    // Don't split at all
    // This is needed if all of the stripes above are larger than the tensor
    // and none of them are added
    {
        TensorShape encoding    = { 0, 0, 0, 0 };
        TensorShape inputStripe = CreateStripe(node->GetInputShape(0), encoding, caps);

        TensorShape outputStripe        = CreateStripe(node->GetShape(), encoding, caps);
        Part::NumStripes numStripesCopy = numStripes;
        numStripesCopy.maxWeightStripes = std::min(numStripes.maxWeightStripes, 1u);
        numStripesCopy.maxInputStripes  = std::min(numStripes.maxInputStripes, 1u);
        numStripesCopy.maxOutputStripes = std::min(numStripes.maxOutputStripes, 1u);
        result.insert(Part::StripeInfos{ inputStripe, outputStripe, numStripesCopy });
    }
    return result;
}

void Part::GenerateWithStripeSizes(Node* node,
                                   const HardwareCapabilities& caps,
                                   const std::vector<BlockConfig>& blockConfigs,
                                   TraversalOrder order,
                                   WeightEncoderCache& weightEncoderCache)
{
    std::set<Part::StripeInfos> stripeInfos;
    for (auto blockConfig : blockConfigs)
    {
        auto mceStripes = GenerateStripes(node, caps, blockConfig);
        stripeInfos.insert(mceStripes.begin(), mceStripes.end());
    }

    GenerateWithNumStripes(node, caps, order, stripeInfos, weightEncoderCache);
}

void Part::GenerateWithNumStripes(Node* node,
                                  const HardwareCapabilities& caps,
                                  TraversalOrder order,
                                  const std::set<Part::StripeInfos>& stripeInfos,
                                  WeightEncoderCache& weightEncoderCache)
{
    if (IsObjectOfType<MceOperationNode>(node))
    {
        GenerateWithNumStripesForLocation(node, caps, order, stripeInfos, Location::Sram, Location::PleInputSram,
                                          weightEncoderCache);
    }
    else if (IsObjectOfType<FuseOnlyPleOperationNode>(node))
    {
        GenerateWithNumStripesForLocation(node, caps, order, stripeInfos, Location::PleInputSram, Location::Sram,
                                          weightEncoderCache);
    }
    else if (IsObjectOfType<FormatConversionNode>(node))
    {
        auto format = node->GetFormat();
        switch (format)
        {
            case CompilerDataFormat::NHWC:
                GenerateWithNumStripesForLocation(node, caps, order, stripeInfos, Location::Sram, Location::Dram,
                                                  weightEncoderCache);
                break;
            case CompilerDataFormat::NHWCB:
                GenerateWithNumStripesForLocation(node, caps, order, stripeInfos, Location::Dram, Location::Sram,
                                                  weightEncoderCache);
                break;
            default:
                break;
        }
    }
}

void Part::GenerateWithNumStripesForLocation(Node* node,
                                             const HardwareCapabilities& caps,
                                             TraversalOrder order,
                                             const std::set<Part::StripeInfos>& stripeInfos,
                                             Location inputBufferLocaton,
                                             Location outputBufferLocation,
                                             WeightEncoderCache& weightEncoderCache)
{
    for (Part::StripeInfos stripeInfosI : stripeInfos)
    {
        if (inputBufferLocaton == Location::Dram)
        {
            stripeInfosI.m_InputStripeShape           = { 0, 0, 0, 0 };
            stripeInfosI.m_NumStripes.minInputStripes = 0;
            stripeInfosI.m_NumStripes.maxInputStripes = 0;
        }
        if (outputBufferLocation == Location::Dram)
        {
            stripeInfosI.m_OutputStripeShape           = { 0, 0, 0, 0 };
            stripeInfosI.m_NumStripes.minOutputStripes = 0;
            stripeInfosI.m_NumStripes.maxOutputStripes = 0;
        }

        for (auto numInputStripes = stripeInfosI.m_NumStripes.minInputStripes;
             numInputStripes <= stripeInfosI.m_NumStripes.maxInputStripes; ++numInputStripes)
        {
            for (auto numOutputStripes = stripeInfosI.m_NumStripes.minOutputStripes;
                 numOutputStripes <= stripeInfosI.m_NumStripes.maxOutputStripes; ++numOutputStripes)
            {
                for (auto numWeightStripes = stripeInfosI.m_NumStripes.minWeightStripes;
                     numWeightStripes <= stripeInfosI.m_NumStripes.maxWeightStripes; ++numWeightStripes)
                {
                    CreatePlanForNode(node, Lifetime::Atomic, caps, order, stripeInfosI.m_InputStripeShape,
                                      stripeInfosI.m_OutputStripeShape, numInputStripes, numOutputStripes,
                                      numWeightStripes, inputBufferLocaton, outputBufferLocation, weightEncoderCache);
                }
            }
        }
    }
}

}    // namespace support_library
}    // namespace ethosn
