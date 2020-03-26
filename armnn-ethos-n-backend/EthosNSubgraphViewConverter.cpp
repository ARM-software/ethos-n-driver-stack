//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNSubgraphViewConverter.hpp"

#include "EthosNConfig.hpp"
#include "EthosNTensorUtils.hpp"
#include "LayersFwd.hpp"
#include "workloads/EthosNPreCompiledWorkload.hpp"

#include <armnn/Logging.hpp>
#include <armnn/Optional.hpp>
#include <armnnUtils/Permute.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <boost/assert.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <boost/polymorphic_pointer_cast.hpp>
#include <ethosn_driver_library/Network.hpp>

#include <algorithm>

namespace armnn
{

using namespace ethosntensorutils;

uint32_t EthosNSubgraphViewConverter::ms_NextInstanceId = 0;

EthosNSubgraphViewConverter::EthosNSubgraphViewConverter(const SubgraphView& subgraph)
    : m_InstanceId(ms_NextInstanceId++)
    , m_Subgraph(subgraph)
    , m_EthosNConfig(GetEthosNConfig())
{}

template <typename Layer>
EthosNConstantPtr EthosNSubgraphViewConverter::AddBiases(const Layer& layer, bool biasEnabled)
{
    void* biasData = nullptr;
    ethosn_lib::TensorInfo ethosnBiasInfo;

    auto inputInfo  = layer.GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
    auto outputInfo = layer.GetOutputSlot(0).GetTensorInfo();
    auto weightInfo = layer.m_Weight->GetTensorInfo();

    // use the actual bias, if provided by the layer
    std::vector<int32_t> dummyBiasData;
    if (biasEnabled)
    {
        BOOST_ASSERT(layer.m_Bias != nullptr);
        ethosnBiasInfo = BuildEthosNBiasesInfo(layer.m_Bias->GetTensorInfo(), inputInfo, weightInfo);
        biasData       = layer.m_Bias->template GetTensor<void>();
    }
    else
    {
        // create zero bias values
        ethosn_lib::TensorShape ethosnOutputShape = BuildEthosNTensorShape(outputInfo.GetShape());
        const unsigned int numBiasElements        = ethosnOutputShape[3];
        dummyBiasData.resize(numBiasElements, 0);
        ethosnBiasInfo = BuildEthosNBiasesInfo(numBiasElements, inputInfo, weightInfo);
        biasData       = reinterpret_cast<void*>(dummyBiasData.data());
    }

    return ethosn_lib::AddConstant(m_Network, ethosnBiasInfo, biasData).tensor;
}

template <typename Layer>
EthosNConstantPtr EthosNSubgraphViewConverter::AddWeights(const Layer& layer)
{
    BOOST_ASSERT(layer.m_Weight != nullptr);

    const TensorInfo& tensorInfo       = layer.m_Weight->GetTensorInfo();
    ethosn_lib::TensorInfo weightsInfo = BuildEthosNConvolutionWeightsInfo(
        tensorInfo, layer.GetParameters().m_DataLayout, layer.GetType() == LayerType::DepthwiseConvolution2d);

    const TensorShape& tensorShape = tensorInfo.GetShape();

    std::vector<uint8_t> swizzledWeightsData(tensorShape.GetNumElements(), 0);
    SwizzleConvolutionWeightsData<uint8_t>(layer.m_Weight->template GetTensor<void>(), swizzledWeightsData.data(),
                                           tensorShape, layer.GetParameters().m_DataLayout,
                                           layer.GetType() == LayerType::DepthwiseConvolution2d);

    return ethosn_lib::AddConstant(m_Network, weightsInfo, swizzledWeightsData.data()).tensor;
}

template <>
EthosNConstantPtr EthosNSubgraphViewConverter::AddWeights(const FullyConnectedLayer& layer)
{
    BOOST_ASSERT(layer.m_Weight != nullptr);

    const TensorInfo& weightsInfo = layer.m_Weight->GetTensorInfo();

    const bool transposeWeights              = layer.GetParameters().m_TransposeWeightMatrix;
    ethosn_lib::TensorInfo ethosnWeightsInfo = BuildEthosNFullyConnectedWeightsInfo(weightsInfo, transposeWeights);

    void* weightsData = layer.m_Weight->template GetTensor<void>();

    if (transposeWeights)
    {
        const TensorShape& weightsShape = weightsInfo.GetShape();
        const bool isWeightsTensor2d    = weightsInfo.GetNumDimensions() == 2;

        // Transpose weight data: [HW]OI -> [HW]IO
        const TensorShape transposedWeightsShape =
            isWeightsTensor2d ? TensorShape({ weightsShape[1], weightsShape[0] })
                              : TensorShape({ weightsShape[0], weightsShape[1], weightsShape[3], weightsShape[2] });

        const PermutationVector permutationVector =
            isWeightsTensor2d ? PermutationVector({ 1, 0 }) : PermutationVector({ 0, 1, 3, 2 });

        std::vector<uint8_t> transposedWeightsData(weightsInfo.GetNumElements(), 0);
        armnnUtils::Permute(transposedWeightsShape, permutationVector, reinterpret_cast<const uint8_t*>(weightsData),
                            transposedWeightsData.data(), sizeof(uint8_t));

        return ethosn_lib::AddConstant(m_Network, ethosnWeightsInfo,
                                       reinterpret_cast<void*>(transposedWeightsData.data()))
            .tensor;
    }

    return ethosn_lib::AddConstant(m_Network, ethosnWeightsInfo, weightsData).tensor;
}

void EthosNSubgraphViewConverter::AddInput(uint32_t inputSlotIdx)
{
    const InputSlot& inputSlot      = *m_Subgraph.GetInputSlot(inputSlotIdx);
    const OutputSlot* connectedSlot = inputSlot.GetConnectedOutputSlot();
    BOOST_ASSERT(connectedSlot != nullptr);

    // Add input to the Ethos-N network
    ethosn_lib::TensorInfo EthosNTensorInfo = BuildEthosNTensorInfo(connectedSlot->GetTensorInfo(), DataLayout::NHWC);
    EthosNAddOperationResult inputOperandAndId = ethosn_lib::AddInput(m_Network, EthosNTensorInfo);

    // Store the mapping from our input slot index to the Ethos-N's input ID, which is defined as a pair of
    // the operation ID that produces the input and the specific output index from that layer.
    // In this case the layer that produces the input is the input operation itself which always has a single
    // output, so our index is zero.
    m_EthosNInputIdToInputSlot[{ inputOperandAndId.operationId, 0 }] = inputSlotIdx;

    // Inputs have exactly one output that maps neatly to the NPU
    m_ConvertedOutputSlots[connectedSlot] = { inputOperandAndId.operationId, inputOperandAndId.tensor, 0 };
    m_EthosNOperationNameMapping[inputOperandAndId.operationId] =
        "Input from " + connectedSlot->GetOwningLayer().GetNameStr();
}

void EthosNSubgraphViewConverter::AddOutput(uint32_t outputSlotIdx)
{
    const OutputSlot& outputSlot = *m_Subgraph.GetOutputSlot(outputSlotIdx);

    // Get the Ethos-N operand that should connect to this output
    auto input = AddOrRetrieveEthosNOperand(&outputSlot);

    // Add output operand to Ethos-N network
    ethosn_lib::TensorAndId<ethosn_lib::Output> output = ethosn_lib::AddOutput(m_Network, *input.tensor);

    // Store the mapping from our output slot index to the Ethos-N's output ID, which is defined as a pair of
    // the operation ID that produces the output and the specific output index from that layer.
    m_EthosNOutputIdToOutputSlot[{ input.operationId, input.outputIndex }] = outputSlotIdx;
    m_EthosNOperationNameMapping[output.operationId] = "Output from " + outputSlot.GetOwningLayer().GetNameStr();
}

void EthosNSubgraphViewConverter::AddActivationLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::Activation);

    ActivationLayer& activationLayer = *boost::polymorphic_pointer_downcast<ActivationLayer>(layer);

    auto input1 = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnectedOutputSlot());

    EthosNAddOperationResult newOperand;
    switch (activationLayer.GetParameters().m_Function)
    {
        case ActivationFunction::ReLu:
        case ActivationFunction::BoundedReLu:
        {
            const ethosn_lib::ReluInfo reluInfo =
                BuildEthosNReluInfo(activationLayer.GetParameters(), activationLayer.GetOutputSlot(0).GetTensorInfo());

            newOperand = ethosn_lib::AddRelu(m_Network, *input1.tensor, reluInfo);

            break;
        }
        case ActivationFunction::Sigmoid:
        {
            newOperand = ethosn_lib::AddSigmoid(m_Network, *input1.tensor);
            break;
        }
        default:
        {
            if (m_EthosNConfig.m_PerfOnly)
            {
                ethosn_lib::EstimateOnlyInfo estimateInfo(
                    { BuildEthosNTensorInfo(activationLayer.GetOutputSlot(0).GetTensorInfo(), DataLayout::NHWC) });
                auto tensorsAndId = ethosn_lib::AddEstimateOnly(m_Network, { input1.tensor.get() }, estimateInfo);
                newOperand        = { tensorsAndId.tensors[0], tensorsAndId.operationId };
                break;
            }
            throw Exception("Unsupported activation function");
        }
    }

    // All activation functions have exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(layer, newOperand);
}

void EthosNSubgraphViewConverter::AddAdditionLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::Addition);

    auto input1                  = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnectedOutputSlot());
    auto input2                  = AddOrRetrieveEthosNOperand(layer->GetInputSlot(1).GetConnectedOutputSlot());
    const TensorInfo& outputInfo = layer->GetOutputSlot(0).GetTensorInfo();
    ethosn_lib::QuantizationInfo outputQuantInfo(outputInfo.GetQuantizationOffset(), outputInfo.GetQuantizationScale());

    // Addition has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(
        layer, ethosn_lib::AddAddition(m_Network, *input1.tensor, *input2.tensor, outputQuantInfo));
}

void EthosNSubgraphViewConverter::AddConstantLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::Constant);

    ethosn_lib::TensorInfo tensorInfo =
        BuildEthosNTensorInfo(layer->GetOutputSlot(0).GetTensorInfo(), DataLayout::NHWC);
    void* data = boost::polymorphic_pointer_downcast<ConstantLayer>(layer)->m_LayerOutput->GetTensor<void>();

    auto constantAndId = ethosn_lib::AddConstant(m_Network, tensorInfo, data);

    EthosNOperandPtr operand      = ethosn_lib::GetOperand(constantAndId.tensor);
    EthosNOperationId operationId = constantAndId.operationId;

    // Constant has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(layer, EthosNAddOperationResult{ operand, operationId });
}

void EthosNSubgraphViewConverter::AddConvolution2dLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::Convolution2d);

    Convolution2dLayer& convolutionLayer = *boost::polymorphic_pointer_downcast<Convolution2dLayer>(layer);
    Convolution2dDescriptor descriptor   = convolutionLayer.GetParameters();

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnectedOutputSlot());

    auto biases  = AddBiases(convolutionLayer, descriptor.m_BiasEnabled);
    auto weights = AddWeights(convolutionLayer);

    auto& outputInfo = convolutionLayer.GetOutputSlot(0).GetTensorInfo();
    ethosn_lib::ConvolutionInfo convolutionInfo =
        BuildEthosNConvolutionInfo(descriptor, outputInfo.GetQuantizationOffset(), outputInfo.GetQuantizationScale());

    // Convolution has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(
        layer, ethosn_lib::AddConvolution(m_Network, *input.tensor, *biases, *weights, convolutionInfo));
}

void EthosNSubgraphViewConverter::AddDepthwiseConvolution2dLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::DepthwiseConvolution2d);

    DepthwiseConvolution2dLayer& depthwiseConvolution2dLayer =
        *boost::polymorphic_pointer_downcast<DepthwiseConvolution2dLayer>(layer);
    DepthwiseConvolution2dDescriptor descriptor = depthwiseConvolution2dLayer.GetParameters();

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnectedOutputSlot());

    auto biases  = AddBiases(depthwiseConvolution2dLayer, descriptor.m_BiasEnabled);
    auto weights = AddWeights(depthwiseConvolution2dLayer);

    auto outputInfo = layer->GetOutputSlot(0).GetTensorInfo();
    auto convolutionInfo =
        BuildEthosNConvolutionInfo(descriptor, outputInfo.GetQuantizationOffset(), outputInfo.GetQuantizationScale());
    // Depthwise Convolution has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(
        layer, ethosn_lib::AddDepthwiseConvolution(m_Network, *input.tensor, *biases, *weights, convolutionInfo));
}

void EthosNSubgraphViewConverter::AddTransposeConvolution2dLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::TransposeConvolution2d);

    TransposeConvolution2dLayer& transposeConvolution2dLayer =
        *boost::polymorphic_pointer_downcast<TransposeConvolution2dLayer>(layer);
    TransposeConvolution2dDescriptor descriptor = transposeConvolution2dLayer.GetParameters();

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnectedOutputSlot());

    auto biases  = AddBiases(transposeConvolution2dLayer, descriptor.m_BiasEnabled);
    auto weights = AddWeights(transposeConvolution2dLayer);

    auto outputInfo = layer->GetOutputSlot(0).GetTensorInfo();
    auto convolutionInfo =
        BuildEthosNConvolutionInfo(descriptor, outputInfo.GetQuantizationOffset(), outputInfo.GetQuantizationScale());
    // Transpose Convolution has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(
        layer, ethosn_lib::AddTransposeConvolution(m_Network, *input.tensor, *biases, *weights, convolutionInfo));
}

void EthosNSubgraphViewConverter::AddFullyConnectedLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::FullyConnected);

    auto inputInfo  = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
    auto inputShape = inputInfo.GetShape();

    EthosNOperand input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnectedOutputSlot());

    // The Ethos-N input tensor will be of shape N x C1 x C2 x C3 where the later channels dimensions will be 1 if
    // not specified in the Arm NN tensor (due to the way we pad 2D tensors up to 4D for the Ethos-N, see BuildEthosNTensorShape).
    // However the Ethos-N FC layer takes input in NHWC so we need to add a trivial reshape.
    ethosn_lib::TensorShape targetShape{ inputShape[0], 1, 1, inputShape.GetNumElements() / inputShape[0] };
    EthosNAddOperationResult reshape = ethosn_lib::AddReshape(m_Network, *input.tensor, targetShape);
    input                            = { reshape.operationId, reshape.tensor, 0 };

    FullyConnectedLayer& fullyConnectedLayer = *boost::polymorphic_pointer_downcast<FullyConnectedLayer>(layer);
    FullyConnectedDescriptor descriptor      = fullyConnectedLayer.GetParameters();

    auto biases  = AddBiases(fullyConnectedLayer, descriptor.m_BiasEnabled);
    auto weights = AddWeights(fullyConnectedLayer);

    const TensorInfo& outputInfo                      = layer->GetOutputSlot(0).GetTensorInfo();
    ethosn_lib::FullyConnectedInfo fullyConnectedInfo = BuildEthosNFullyConnectedLayerInfo(
        descriptor, outputInfo.GetQuantizationOffset(), outputInfo.GetQuantizationScale());

    EthosNAddOperationResult fc =
        ethosn_lib::AddFullyConnected(m_Network, *input.tensor, *biases, *weights, fullyConnectedInfo);

    // Add a reshape to convert back to the tensor shape the rest of the backend expects.
    // If we don't do this then the IsSupported checks will pass a tensor shape that doesn't match
    // what will actually be input to that layer.
    ethosn_lib::TensorShape targetShape2 = BuildEthosNTensorShape(outputInfo.GetShape());
    EthosNAddOperationResult reshape2    = ethosn_lib::AddReshape(m_Network, *fc.tensor, targetShape2);

    // Fully Connected has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(layer, reshape2);
}

void EthosNSubgraphViewConverter::AddConcatLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::Concat);
    ConcatLayer& concatLayer = *boost::polymorphic_pointer_downcast<ConcatLayer>(layer);

    unsigned int numInputSlots = layer->GetNumInputSlots();
    BOOST_ASSERT(numInputSlots >= 2u);

    std::vector<ethosn_lib::Operand*> inputLayers;
    for (unsigned int i = 0u; i < numInputSlots; i++)
    {
        EthosNOperand input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(i).GetConnectedOutputSlot());
        inputLayers.push_back(input.tensor.get());
    }

    const TensorInfo& outputTensorInfo = layer->GetOutputSlot(0).GetTensorInfo();
    ethosn_lib::QuantizationInfo outputQuantInfo(outputTensorInfo.GetQuantizationOffset(),
                                                 outputTensorInfo.GetQuantizationScale());

    // The Ethos-N's concat axis is the same as Arm NN's even if the tensor shapes have been padded to 4D,
    // because we pad on the right hand side of the dimensions.
    uint32_t ethosnConcatAxis = concatLayer.GetParameters().GetConcatAxis();

    // Concatenation has exactly one output that maps neatly to the NPU
    // Note we ignore the "view origins" contained in OriginsDescriptor and use just the "concat axis".
    // This is a known issue/confusion in the Arm NN API - see Github Issue #234.
    InsertConvertedLayerSingleOutput(
        layer, ethosn_lib::AddConcatenation(m_Network, inputLayers,
                                            ethosn_lib::ConcatenationInfo(ethosnConcatAxis, outputQuantInfo)));
}

void EthosNSubgraphViewConverter::AddPooling2dLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::Pooling2d);

    Pooling2dLayer& pooling2dLayer = *boost::polymorphic_pointer_downcast<Pooling2dLayer>(layer);
    Pooling2dDescriptor descriptor = pooling2dLayer.GetParameters();

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnectedOutputSlot());

    // Pooling has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(
        layer, ethosn_lib::AddPooling(m_Network, *input.tensor, BuildEthosNPoolingLayerInfo(descriptor)));
}

void EthosNSubgraphViewConverter::AddReshapeLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::Reshape);

    ReshapeLayer& reshapeLayer   = *boost::polymorphic_pointer_downcast<ReshapeLayer>(layer);
    ReshapeDescriptor descriptor = reshapeLayer.GetParameters();

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnectedOutputSlot());

    // Reshape has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(
        layer, ethosn_lib::AddReshape(m_Network, *input.tensor, BuildEthosNTensorShape(descriptor.m_TargetShape)));
}

void EthosNSubgraphViewConverter::AddSoftmaxLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::Softmax);

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnectedOutputSlot());

    // Softmax has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(layer, ethosn_lib::AddSoftmax(m_Network, *input.tensor));
}

void EthosNSubgraphViewConverter::AddSplitterLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::Splitter);
    SplitterLayer& splitterLayer = *boost::polymorphic_pointer_downcast<SplitterLayer>(layer);

    auto input             = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnectedOutputSlot());
    TensorShape inputShape = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo().GetShape();

    // Note it's safe to assume that BuildEthosNSplitInfo succeeds, because we checked this in IsSplitterSupported.
    ethosn_lib::SplitInfo ethosnSplitInfo = BuildEthosNSplitInfo(inputShape, splitterLayer.GetParameters()).value();

    InsertConvertedLayerMultipleOutput(layer, ethosn_lib::AddSplit(m_Network, *input.tensor, ethosnSplitInfo));
}

void EthosNSubgraphViewConverter::AddDepthToSpaceLayer(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);
    BOOST_ASSERT(layer->GetType() == LayerType::DepthToSpace);
    DepthToSpaceLayer& depthToSpaceLayer = *boost::polymorphic_pointer_downcast<DepthToSpaceLayer>(layer);

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnectedOutputSlot());

    ethosn_lib::DepthToSpaceInfo info(depthToSpaceLayer.GetParameters().m_BlockSize);

    // DepthToSpace has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(layer, ethosn_lib::AddDepthToSpace(m_Network, *input.tensor, info));
}

void EthosNSubgraphViewConverter::AddEstimateOnly(Layer* layer)
{
    BOOST_ASSERT(layer != nullptr);

    std::vector<ethosn_lib::Operand*> inputOperands;
    inputOperands.reserve(layer->GetNumInputSlots());
    for (uint32_t i = 0; i < layer->GetNumInputSlots(); ++i)
    {
        auto ethosnOperand = AddOrRetrieveEthosNOperand(layer->GetInputSlot(i).GetConnectedOutputSlot());
        inputOperands.push_back(ethosnOperand.tensor.get());
    }

    std::vector<ethosn_lib::TensorInfo> ethosnOutputInfos;
    ethosnOutputInfos.reserve(layer->GetNumOutputSlots());
    for (uint32_t i = 0; i < layer->GetNumOutputSlots(); ++i)
    {
        ethosnOutputInfos.push_back(BuildEthosNTensorInfo(layer->GetOutputSlot(i).GetTensorInfo(), DataLayout::NHWC));
    }
    ethosn_lib::EstimateOnlyInfo estimateInfo(ethosnOutputInfos);
    InsertConvertedLayerMultipleOutput(layer, ethosn_lib::AddEstimateOnly(m_Network, inputOperands, estimateInfo));
}
EthosNOperand EthosNSubgraphViewConverter::AddOrRetrieveEthosNOperand(const OutputSlot* outputSlot)
{
    BOOST_ASSERT(outputSlot != nullptr);

    // Check if the layer has already been converted and added
    auto locationInMap = m_ConvertedOutputSlots.find(outputSlot);
    if (locationInMap != m_ConvertedOutputSlots.end())
    {
        // Layer already present in the network => retrieve it
        return locationInMap->second;
    }

    // Layer not added yet => add it
    Layer* layer = &outputSlot->GetOwningLayer();
    switch (layer->GetType())
    {
        case LayerType::Activation:
            AddActivationLayer(layer);
            break;
        case LayerType::Addition:
            AddAdditionLayer(layer);
            break;
        case LayerType::Constant:
            AddConstantLayer(layer);
            break;
        case LayerType::Convolution2d:
            AddConvolution2dLayer(layer);
            break;
        case LayerType::DepthwiseConvolution2d:
            AddDepthwiseConvolution2dLayer(layer);
            break;
        case LayerType::TransposeConvolution2d:
            AddTransposeConvolution2dLayer(layer);
            break;
        case LayerType::FullyConnected:
            AddFullyConnectedLayer(layer);
            break;
        case LayerType::Concat:
            AddConcatLayer(layer);
            break;
        case LayerType::Pooling2d:
            AddPooling2dLayer(layer);
            break;
        case LayerType::Reshape:
            AddReshapeLayer(layer);
            break;
        case LayerType::Softmax:
            AddSoftmaxLayer(layer);
            break;
        case LayerType::Splitter:
            AddSplitterLayer(layer);
            break;
        case LayerType::DepthToSpace:
            AddDepthToSpaceLayer(layer);
            break;
        default:
            if (m_EthosNConfig.m_PerfOnly)
            {
                ARMNN_LOG(info) << "\"" << layer->GetNameStr() << "\" is replaced with an estimate only node "
                                << "LayerType: " << GetLayerTypeAsCString(layer->GetType());
                AddEstimateOnly(layer);
                break;
            }
            else
            {
                throw Exception("Conversion not supported for layer type " +
                                std::string(GetLayerTypeAsCString(layer->GetType())));
            }
    }

    // Return the Ethos-N operand that should now have been added
    locationInMap = m_ConvertedOutputSlots.find(outputSlot);
    BOOST_ASSERT(locationInMap != m_ConvertedOutputSlots.end());
    return locationInMap->second;
}

void EthosNSubgraphViewConverter::InsertConvertedLayerSingleOutput(const Layer* layer,
                                                                   EthosNAddOperationResult ethosnAddOperationResult)
{
    BOOST_ASSERT(layer->GetNumOutputSlots() == 1);
    m_ConvertedOutputSlots[&layer->GetOutputSlot(0)]                   = { ethosnAddOperationResult.operationId,
                                                         ethosnAddOperationResult.tensor, 0 };
    m_EthosNOperationNameMapping[ethosnAddOperationResult.operationId] = layer->GetNameStr();
}

void EthosNSubgraphViewConverter::InsertConvertedLayerMultipleOutput(const Layer* layer,
                                                                     ethosn_lib::TensorsAndId ethosnAddOperationResult)
{
    BOOST_ASSERT(layer->GetNumOutputSlots() == ethosnAddOperationResult.tensors.size());
    for (uint32_t i = 0; i < layer->GetNumOutputSlots(); ++i)
    {
        m_ConvertedOutputSlots[&layer->GetOutputSlot(i)] = { ethosnAddOperationResult.operationId,
                                                             ethosnAddOperationResult.tensors[i], i };
    }
    m_EthosNOperationNameMapping[ethosnAddOperationResult.operationId] = layer->GetNameStr();
}

void EthosNSubgraphViewConverter::ResetNextInstanceId()
{
    ms_NextInstanceId = 0;
}

void EthosNSubgraphViewConverter::CreateUncompiledNetwork()
{
    if (m_Network)
    {
        // Network already created
        return;
    }

    // Initialize a new network
    m_Network = m_EthosNConfig.m_PerfOnly ? ethosn_lib::CreateEstimationNetwork() : ethosn_lib::CreateNetwork();

    // Add inputs
    for (uint32_t inputSlotIdx = 0; inputSlotIdx < m_Subgraph.GetNumInputSlots(); ++inputSlotIdx)
    {
        AddInput(inputSlotIdx);
    }

    // Add outputs.
    // This will recurse through the graph converting layers until we end up connecting to the input operations
    // added to the Ethos-N graph above.
    for (uint32_t outputSlotIdx = 0; outputSlotIdx < m_Subgraph.GetNumOutputSlots(); ++outputSlotIdx)
    {
        AddOutput(outputSlotIdx);
    }
}

namespace
{
template <typename T>
void DeleteAsType(const void* const blob)
{
    delete static_cast<const T*>(blob);
}
}    // namespace

std::vector<CompiledBlobPtr>
    EthosNSubgraphViewConverter::Estimate(const ethosn_lib::CompilationOptions& ethosnCompilationOpts)
{
    ethosn_lib::EstimationOptions ethosnEstimationOpts;
    ethosnEstimationOpts.m_ActivationCompressionSaving  = m_EthosNConfig.m_PerfActivationCompressionSaving;
    ethosnEstimationOpts.m_UseWeightCompressionOverride = m_EthosNConfig.m_PerfUseWeightCompressionOverride;
    ethosnEstimationOpts.m_WeightCompressionSaving      = m_EthosNConfig.m_PerfWeightCompressionSaving;
    ethosnEstimationOpts.m_Current                      = m_EthosNConfig.m_PerfCurrent;
    EthosNPreCompiledObject::PerfData perfData;

    perfData.m_PerfOutFile               = ethosnCompilationOpts.m_DebugDir + "/report.json";
    perfData.m_PerfVariant               = m_EthosNConfig.m_PerfVariant;
    perfData.m_PerfSramSizeBytesOverride = m_EthosNConfig.m_PerfSramSizeBytesOverride;
    perfData.m_EstimationOptions         = ethosnEstimationOpts;

    perfData.m_Data = ethosn_lib::EstimatePerformance(*m_Network, ethosnCompilationOpts, ethosnEstimationOpts);

    auto preCompiledObj = std::make_unique<EthosNPreCompiledObject>(std::move(perfData), m_EthosNOperationNameMapping);

    std::vector<CompiledBlobPtr> compiledBlobs;

    // Convert the EthosNPreCompiledObject into a "blob" (void) object and attach the custom blob deleter
    compiledBlobs.emplace_back(preCompiledObj.release(), DeleteAsType<EthosNPreCompiledObject>);

    return compiledBlobs;
}

std::vector<CompiledBlobPtr> EthosNSubgraphViewConverter::CompileNetwork()
{
    // Get the capabilities from the driver library if this is running on real HW, or get representative ones
    // if we are running perf-only.
    std::vector<char> caps = m_EthosNConfig.m_PerfOnly
                                 ? ethosn_lib::GetPerformanceEstimatorFwAndHwCapabilities(
                                       m_EthosNConfig.m_PerfVariant, m_EthosNConfig.m_PerfSramSizeBytesOverride)
                                 : ethosn::driver_library::GetFirmwareAndHardwareCapabilities();

    ethosn_lib::CompilationOptions ethosnCompilationOpts(caps);
    ethosnCompilationOpts.m_DumpDebugFiles = m_EthosNConfig.m_DumpDebugFiles;
    ethosnCompilationOpts.m_DebugDir       = m_EthosNConfig.m_PerfOutDir + "/subgraph_" + std::to_string(m_InstanceId);
    // Disable RAM dump that is enabled by default by the Ethos-N support library
    // and litters the execution folder with sizable HEX files
    ethosnCompilationOpts.m_DumpRam = false;

    boost::filesystem::create_directories(ethosnCompilationOpts.m_DebugDir);

    // Compile the network into a list of generic type-agnostic "blobs"
    std::vector<CompiledBlobPtr> compiledBlobs;

    try
    {
        // Create a new network to be compiled by the Ethos-N backend
        CreateUncompiledNetwork();

        compiledBlobs = m_EthosNConfig.m_PerfOnly ? Estimate(ethosnCompilationOpts) : Compile(ethosnCompilationOpts);
    }
    catch (const std::exception&)
    {
        // An exception has been thrown when either trying to build the uncompiled network, or by the compiler.
        // Swallow it, as this API is not expected to throw, and return an empty list of compiled blobs instead,
        // it will be handled by the caller (the OptimizeSubgraphView method in the backend) when putting together
        // the OptimizationViews object to return as the optimization result
        compiledBlobs.clear();
    }

    return compiledBlobs;
}

std::vector<CompiledBlobPtr>
    EthosNSubgraphViewConverter::Compile(const ethosn_lib::CompilationOptions& ethosnCompilationOpts)
{
    std::vector<CompiledBlobPtr> compiledBlobs;

    std::vector<EthosNCompiledNetworkPtr> compiledNetworks = ethosn_lib::Compile(*m_Network, ethosnCompilationOpts);

    // Create a list of generic type-agnostic compiled "blobs"
    for (EthosNCompiledNetworkPtr& compiledNetwork : compiledNetworks)
    {
        // Map Arm NN input slots to Ethos-N input indices, based on the data we gathered while adding the Ethos-N operations.
        std::unordered_map<uint32_t, uint32_t> inputSlotsToEthosNInputs;
        for (uint32_t ethosnInputIdx = 0; ethosnInputIdx < compiledNetwork->GetInputBufferInfos().size();
             ++ethosnInputIdx)
        {
            const ethosn_lib::InputBufferInfo& inputBufferInfo = compiledNetwork->GetInputBufferInfos()[ethosnInputIdx];
            uint32_t inputSlotIdx                              = m_EthosNInputIdToInputSlot.at(
                { inputBufferInfo.m_SourceOperationId, inputBufferInfo.m_SourceOperationOutputIndex });
            inputSlotsToEthosNInputs[inputSlotIdx] = ethosnInputIdx;
        }

        // Map Arm NN output slots to Ethos-N output indices, based on the data we gathered while adding the Ethos-N operations.
        std::unordered_map<uint32_t, uint32_t> outputSlotsToEthosNOutputs;
        for (uint32_t ethosnOutputIdx = 0; ethosnOutputIdx < compiledNetwork->GetOutputBufferInfos().size();
             ++ethosnOutputIdx)
        {
            const ethosn_lib::OutputBufferInfo& outputBufferInfo =
                compiledNetwork->GetOutputBufferInfos()[ethosnOutputIdx];
            uint32_t outputSlotIdx = m_EthosNOutputIdToOutputSlot.at(
                { outputBufferInfo.m_SourceOperationId, outputBufferInfo.m_SourceOperationOutputIndex });
            outputSlotsToEthosNOutputs[outputSlotIdx] = ethosnOutputIdx;
        }

        // Construct a EthosNPreCompiledObject containing the ethosn_lib::CompiledNetwork along with
        // other data needed by the workload.
        auto preCompiledObject = std::make_unique<EthosNPreCompiledObject>(
            EthosNPreCompiledObject::Network(std::move(compiledNetwork), inputSlotsToEthosNInputs,
                                             outputSlotsToEthosNOutputs),
            m_EthosNOperationNameMapping);

        // Convert the EthosNPreCompiledObject into a "blob" (void) object and attach the custom blob deleter
        compiledBlobs.emplace_back(preCompiledObject.release(), DeleteAsType<EthosNPreCompiledObject>);
    }

    return compiledBlobs;
}

}    // namespace armnn
