//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNSubgraphViewConverter.hpp"

#include "EthosNBackend.hpp"
#include "EthosNConfig.hpp"
#include "EthosNTensorUtils.hpp"
#include "workloads/EthosNPreCompiledWorkload.hpp"

#include <armnn/Logging.hpp>
#include <armnn/Optional.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnnUtils/Filesystem.hpp>
#include <armnnUtils/Permute.hpp>
#include <ethosn_utils/VectorStream.hpp>

#include <algorithm>
#include <utility>

namespace armnn
{

using namespace ethosntensorutils;

ARMNN_DLLEXPORT std::unique_ptr<EthosNSupportLibraryInterface> g_EthosNSupportLibraryInterface =
    std::make_unique<EthosNSupportLibraryInterface>();

EthosNSubgraphViewConverter::EthosNSubgraphViewConverter(const SubgraphView& subgraphToCompile,
                                                         uint32_t subgraphIdx,
                                                         ModelOptions modelOptions,
                                                         const EthosNConfig& config,
                                                         const std::vector<char>& capabilities)
    : m_SubgraphIdx(subgraphIdx)
    , m_Subgraph(subgraphToCompile)
    , m_EthosNConfig(config)
    , m_Capabilities(capabilities)
{
    try
    {
        m_CompilationOptions = GetCompilationOptions(m_EthosNConfig, modelOptions, m_SubgraphIdx);
    }
    catch (const InvalidArgumentException& e)
    {
        ARMNN_LOG(error) << "Failed to parse backend options - " << e.what();
        throw;
    }
}

std::pair<const ConstTensorHandle*, const ConstTensorHandle&> GetBiasAndWeightsHandle(const IConnectableLayer& layer)
{

    // Get the weights tensor from the layer connected to input slot #1
    if (layer.GetNumInputSlots() < 2)
    {
        throw armnn::Exception("Layer doesn't have a second input slot");
    }
    const IOutputSlot* weightsSlot = layer.GetInputSlot(1).GetConnection();
    if (weightsSlot == nullptr)
    {
        throw armnn::Exception("Layer's weight slot not connected");
    }
    const IConnectableLayer& weightsLayer = weightsSlot->GetOwningIConnectableLayer();
    if (weightsLayer.GetType() != armnn::LayerType::Constant)
    {
        throw armnn::Exception("Layer's weight slot connected to non-constant layer");
    }
    const ConstTensorHandle& weightsHandle =
        *armnn::PolymorphicDowncast<const ConstantLayer*>(&weightsLayer)->m_LayerOutput;

    // Get the bias tensor (if any) from the layer connected to input slot #2
    const ConstTensorHandle* biasHandle = nullptr;
    if (layer.GetNumInputSlots() >= 3)    // Bias input is optional
    {
        const IOutputSlot* biasSlot = layer.GetInputSlot(2).GetConnection();
        if (biasSlot == nullptr)
        {
            throw armnn::Exception("Layer's bias slot not connected");
        }
        const IConnectableLayer& biasLayer = biasSlot->GetOwningIConnectableLayer();
        if (biasLayer.GetType() != armnn::LayerType::Constant)
        {
            throw armnn::Exception("Layer's bias slot connected to non-constant layer");
        }
        biasHandle = armnn::PolymorphicDowncast<const ConstantLayer*>(&biasLayer)->m_LayerOutput.get();
    }

    return { biasHandle, weightsHandle };
}

EthosNConstantPtr EthosNSubgraphViewConverter::AddBiases(const IConnectableLayer& layer,
                                                         const ConstTensorHandle* bias,
                                                         const TensorInfo& weightInfo,
                                                         bool biasEnabled)
{
    const void* biasData = nullptr;
    ethosn_lib::TensorInfo ethosnBiasInfo;

    auto inputInfo  = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
    auto outputInfo = layer.GetOutputSlot(0).GetTensorInfo();

    // use the actual bias, if provided by the layer
    std::vector<int32_t> dummyBiasData;
    if (biasEnabled)
    {
        ARMNN_ASSERT(bias != nullptr);
        ethosnBiasInfo = BuildEthosNBiasesInfo(bias->GetTensorInfo(), inputInfo, weightInfo);
        biasData       = bias->GetConstTensor<void>();
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

EthosNConstantPtr EthosNSubgraphViewConverter::AddWeights(const IConnectableLayer& layer,
                                                          DataLayout dataLayout,
                                                          const ConstTensorHandle& weights)
{
    const TensorInfo& inputInfo        = layer.GetInputSlot(0).GetConnection()->GetTensorInfo();
    const TensorInfo& tensorInfo       = weights.GetTensorInfo();
    ethosn_lib::TensorInfo weightsInfo = BuildEthosNConvolutionWeightsInfo(
        tensorInfo, inputInfo, dataLayout, layer.GetType() == LayerType::DepthwiseConvolution2d);

    const TensorShape& tensorShape = tensorInfo.GetShape();

    const TensorInfo& outputInfo = layer.GetOutputSlot(0).GetTensorInfo();
    unsigned int depthMultiplier = outputInfo.GetShape()[3] / inputInfo.GetShape()[3];

    std::vector<uint8_t> swizzledWeightsData(tensorShape.GetNumElements(), 0);
    SwizzleConvolutionWeightsData<uint8_t>(weights.GetConstTensor<void>(), swizzledWeightsData.data(), tensorShape,
                                           dataLayout, layer.GetType() == LayerType::DepthwiseConvolution2d,
                                           depthMultiplier);

    return ethosn_lib::AddConstant(m_Network, weightsInfo, swizzledWeightsData.data()).tensor;
}

EthosNConstantPtr EthosNSubgraphViewConverter::AddWeights(bool transposeWeights, const ConstTensorHandle& weights)
{
    const TensorInfo& weightsInfo = weights.GetTensorInfo();

    ethosn_lib::TensorInfo ethosnWeightsInfo = BuildEthosNFullyConnectedWeightsInfo(weightsInfo, transposeWeights);

    const void* weightsData = weights.GetConstTensor<void>();

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
    const IInputSlot& inputSlot      = *m_Subgraph.GetIInputSlot(inputSlotIdx);
    const IOutputSlot* connectedSlot = inputSlot.GetConnection();
    ARMNN_ASSERT(connectedSlot != nullptr);

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
        "Input from " + std::string(connectedSlot->GetOwningIConnectableLayer().GetName());
}

void EthosNSubgraphViewConverter::AddOutput(uint32_t outputSlotIdx)
{
    const IOutputSlot& outputSlot = *m_Subgraph.GetIOutputSlot(outputSlotIdx);

    // Get the Ethos-N operand that should connect to this output
    auto input = AddOrRetrieveEthosNOperand(&outputSlot);

    // Add output operand to Ethos-N network
    ethosn_lib::TensorAndId<ethosn_lib::Output> output = ethosn_lib::AddOutput(m_Network, *input.tensor);

    // Store the mapping from our output slot index to the Ethos-N's output ID, which is defined as a pair of
    // the operation ID that produces the output and the specific output index from that layer.
    m_EthosNOutputIdToOutputSlot[{ input.operationId, input.outputIndex }] = outputSlotIdx;
    m_EthosNOperationNameMapping[output.operationId] =
        "Output from " + std::string(outputSlot.GetOwningIConnectableLayer().GetName());
}

void EthosNSubgraphViewConverter::AddActivationLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Activation);

    const ActivationLayer& activationLayer = *PolymorphicPointerDowncast<const ActivationLayer>(layer);

    auto input1 = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    EthosNAddOperationResult newOperand;
    switch (activationLayer.GetParameters().m_Function)
    {
        case ActivationFunction::ReLu:
        case ActivationFunction::BoundedReLu:
        {
            const Optional<ethosn_lib::ReluInfo> reluInfo = BuildEthosNReluInfo(
                activationLayer.GetParameters(), layer->GetInputSlot(0).GetConnection()->GetTensorInfo());
            if (!reluInfo.has_value())
            {
                throw Exception("Unsupported relu configuration");
            }

            newOperand = ethosn_lib::AddRelu(m_Network, *input1.tensor, reluInfo.value());

            break;
        }
        case ActivationFunction::LeakyReLu:
        {
            const ethosn_lib::LeakyReluInfo leakyReluInfo = BuildEthosNLeakyReluInfo(
                activationLayer.GetParameters(), activationLayer.GetOutputSlot(0).GetTensorInfo());

            newOperand = ethosn_lib::AddLeakyRelu(m_Network, *input1.tensor, leakyReluInfo);

            break;
        }
        case ActivationFunction::Sigmoid:
        {
            newOperand = ethosn_lib::AddSigmoid(m_Network, *input1.tensor);
            break;
        }
        case ActivationFunction::TanH:
        {
            newOperand = ethosn_lib::AddTanh(m_Network, *input1.tensor);
            break;
        }
        default:
        {
            if (m_EthosNConfig.m_PerfOnly)
            {
                std::string reasonForEstimateOnly = "Unsupported activation function.";
                ethosn_lib::EstimateOnlyInfo estimateInfo(
                    { BuildEthosNTensorInfo(activationLayer.GetOutputSlot(0).GetTensorInfo(), DataLayout::NHWC) },
                    reasonForEstimateOnly);
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

void EthosNSubgraphViewConverter::AddAdditionLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Addition);

    auto input1                  = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());
    auto input2                  = AddOrRetrieveEthosNOperand(layer->GetInputSlot(1).GetConnection());
    const TensorInfo& outputInfo = layer->GetOutputSlot(0).GetTensorInfo();
    ethosn_lib::QuantizationInfo outputQuantInfo(outputInfo.GetQuantizationOffset(), outputInfo.GetQuantizationScale());

    // Addition has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(
        layer, ethosn_lib::AddAddition(m_Network, *input1.tensor, *input2.tensor, outputQuantInfo));
}

void EthosNSubgraphViewConverter::AddConstantLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Constant);

    ethosn_lib::TensorInfo tensorInfo =
        BuildEthosNTensorInfo(layer->GetOutputSlot(0).GetTensorInfo(), DataLayout::NHWC);
    const void* data = PolymorphicPointerDowncast<const ConstantLayer>(layer)->m_LayerOutput->GetConstTensor<void>();

    auto constantAndId = ethosn_lib::AddConstant(m_Network, tensorInfo, data);

    EthosNOperandPtr operand      = ethosn_lib::GetOperand(constantAndId.tensor);
    EthosNOperationId operationId = constantAndId.operationId;

    // Constant has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(layer, EthosNAddOperationResult{ operand, operationId });
}

void EthosNSubgraphViewConverter::AddConvolution2dLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Convolution2d);

    const Convolution2dLayer& convolutionLayer = *PolymorphicPointerDowncast<const Convolution2dLayer>(layer);
    Convolution2dDescriptor descriptor         = convolutionLayer.GetParameters();

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    auto biasAndWeightsHandle = GetBiasAndWeightsHandle(convolutionLayer);

    auto biases = AddBiases(convolutionLayer, biasAndWeightsHandle.first, biasAndWeightsHandle.second.GetTensorInfo(),
                            descriptor.m_BiasEnabled);
    auto weights =
        AddWeights(convolutionLayer, convolutionLayer.GetParameters().m_DataLayout, biasAndWeightsHandle.second);

    auto& outputInfo                                      = convolutionLayer.GetOutputSlot(0).GetTensorInfo();
    Optional<ethosn_lib::ConvolutionInfo> convolutionInfo = BuildEthosNConvolutionInfo(
        descriptor, outputInfo.GetQuantizationOffset(), outputInfo.GetQuantizationScale(), {});
    if (!convolutionInfo.has_value())
    {
        throw Exception("Not supported");
    }

    // Convolution has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(
        layer, ethosn_lib::AddConvolution(m_Network, *input.tensor, *biases, *weights, convolutionInfo.value()));
}

void EthosNSubgraphViewConverter::AddMeanXyLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Mean);

    const MeanLayer& meanLayer = *PolymorphicPointerDowncast<const MeanLayer>(layer);
    MeanDescriptor descriptor  = meanLayer.GetParameters();

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    // Mean has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(layer, ethosn_lib::AddMeanXy(m_Network, *input.tensor));
}

void EthosNSubgraphViewConverter::AddDepthwiseConvolution2dLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::DepthwiseConvolution2d);

    const DepthwiseConvolution2dLayer& depthwiseConvolution2dLayer =
        *PolymorphicPointerDowncast<const DepthwiseConvolution2dLayer>(layer);
    DepthwiseConvolution2dDescriptor descriptor = depthwiseConvolution2dLayer.GetParameters();

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    auto biasAndWeightsHandle = GetBiasAndWeightsHandle(depthwiseConvolution2dLayer);

    auto biases  = AddBiases(depthwiseConvolution2dLayer, biasAndWeightsHandle.first,
                            biasAndWeightsHandle.second.GetTensorInfo(), descriptor.m_BiasEnabled);
    auto weights = AddWeights(depthwiseConvolution2dLayer, depthwiseConvolution2dLayer.GetParameters().m_DataLayout,
                              biasAndWeightsHandle.second);

    auto outputInfo      = layer->GetOutputSlot(0).GetTensorInfo();
    auto convolutionInfo = BuildEthosNConvolutionInfo(descriptor, outputInfo.GetQuantizationOffset(),
                                                      outputInfo.GetQuantizationScale(), {});
    if (!convolutionInfo.has_value())
    {
        throw Exception("Not supported");
    }

    // Depthwise Convolution has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(layer, ethosn_lib::AddDepthwiseConvolution(m_Network, *input.tensor, *biases,
                                                                                *weights, convolutionInfo.value()));
}

void EthosNSubgraphViewConverter::AddTransposeConvolution2dLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::TransposeConvolution2d);

    const TransposeConvolution2dLayer& transposeConvolution2dLayer =
        *PolymorphicPointerDowncast<const TransposeConvolution2dLayer>(layer);
    TransposeConvolution2dDescriptor descriptor = transposeConvolution2dLayer.GetParameters();

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    // Arm NN is transitioning from having weights/bias as intrinsic properties of the layer to having them
    // as separate layers with connections. For now, TransposeConvolution2d is not converted and use the old
    // way with m_Weight and m_Bias

    ARMNN_ASSERT(transposeConvolution2dLayer.m_Weight);
    auto biases  = AddBiases(transposeConvolution2dLayer, transposeConvolution2dLayer.m_Bias.get(),
                            transposeConvolution2dLayer.m_Weight->GetTensorInfo(), descriptor.m_BiasEnabled);
    auto weights = AddWeights(transposeConvolution2dLayer, transposeConvolution2dLayer.GetParameters().m_DataLayout,
                              *transposeConvolution2dLayer.m_Weight);

    auto outputInfo = layer->GetOutputSlot(0).GetTensorInfo();
    auto convolutionInfo =
        BuildEthosNConvolutionInfo(descriptor, outputInfo.GetQuantizationOffset(), outputInfo.GetQuantizationScale());
    // Transpose Convolution has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(
        layer, ethosn_lib::AddTransposeConvolution(m_Network, *input.tensor, *biases, *weights, convolutionInfo));
}

void EthosNSubgraphViewConverter::AddFullyConnectedLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::FullyConnected);

    auto inputInfo  = layer->GetInputSlot(0).GetConnection()->GetTensorInfo();
    auto inputShape = inputInfo.GetShape();

    EthosNOperand input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    // The Ethos-N input tensor will be of shape N x C1 x C2 x C3 where the later channels dimensions will be 1 if
    // not specified in the Arm NN tensor (due to the way we pad 2D tensors up to 4D for the Ethos-N, see BuildEthosNTensorShape).
    // However the Ethos-N FC layer takes input in NHWC so we need to add a trivial reshape.
    ethosn_lib::TensorShape targetShape{ inputShape[0], 1, 1, inputShape.GetNumElements() / inputShape[0] };
    EthosNAddOperationResult reshape = ethosn_lib::AddReshape(m_Network, *input.tensor, targetShape);
    input                            = { reshape.operationId, reshape.tensor, 0 };

    const BaseDescriptor& baseDescriptor       = layer->GetParameters();
    const FullyConnectedDescriptor& descriptor = *PolymorphicDowncast<const FullyConnectedDescriptor*>(&baseDescriptor);
    bool transposeWeights                      = descriptor.m_TransposeWeightMatrix;

    auto biasAndWeightsHandle = GetBiasAndWeightsHandle(*layer);

    auto biases  = AddBiases(*layer, biasAndWeightsHandle.first, biasAndWeightsHandle.second.GetTensorInfo(),
                            descriptor.m_BiasEnabled);
    auto weights = AddWeights(transposeWeights, biasAndWeightsHandle.second);

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

void EthosNSubgraphViewConverter::AddConcatLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Concat);
    const ConcatLayer& concatLayer = *PolymorphicPointerDowncast<const ConcatLayer>(layer);

    unsigned int numInputSlots = layer->GetNumInputSlots();
    ARMNN_ASSERT(numInputSlots >= 2u);

    std::vector<ethosn_lib::Operand*> inputLayers;
    for (unsigned int i = 0u; i < numInputSlots; i++)
    {
        EthosNOperand input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(i).GetConnection());
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

void EthosNSubgraphViewConverter::AddPooling2dLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Pooling2d);

    const Pooling2dLayer& pooling2dLayer = *PolymorphicPointerDowncast<const Pooling2dLayer>(layer);
    Pooling2dDescriptor descriptor       = pooling2dLayer.GetParameters();

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    // Pooling has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(
        layer, ethosn_lib::AddPooling(m_Network, *input.tensor, BuildEthosNPoolingLayerInfo(descriptor)));
}

void EthosNSubgraphViewConverter::AddReshapeLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Reshape);

    const ReshapeLayer& reshapeLayer = *PolymorphicPointerDowncast<const ReshapeLayer>(layer);
    ReshapeDescriptor descriptor     = reshapeLayer.GetParameters();

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    // Reshape has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(
        layer, ethosn_lib::AddReshape(m_Network, *input.tensor, BuildEthosNTensorShape(descriptor.m_TargetShape)));
}

void EthosNSubgraphViewConverter::AddSoftmaxLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Softmax);

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    // Softmax has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(layer, ethosn_lib::AddSoftmax(m_Network, *input.tensor));
}

void EthosNSubgraphViewConverter::AddSplitterLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Splitter);
    const SplitterLayer& splitterLayer = *PolymorphicPointerDowncast<const SplitterLayer>(layer);

    auto input             = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());
    TensorShape inputShape = layer->GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape();

    // Note it's safe to assume that BuildEthosNSplitInfo succeeds, because we checked this in IsSplitterSupported.
    ethosn_lib::SplitInfo ethosnSplitInfo = BuildEthosNSplitInfo(inputShape, splitterLayer.GetParameters()).value();

    InsertConvertedLayerMultipleOutput(layer, ethosn_lib::AddSplit(m_Network, *input.tensor, ethosnSplitInfo));
}

void EthosNSubgraphViewConverter::AddDepthToSpaceLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::DepthToSpace);
    const DepthToSpaceLayer& depthToSpaceLayer = *PolymorphicPointerDowncast<const DepthToSpaceLayer>(layer);

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    ethosn_lib::DepthToSpaceInfo info(depthToSpaceLayer.GetParameters().m_BlockSize);

    // DepthToSpace has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(layer, ethosn_lib::AddDepthToSpace(m_Network, *input.tensor, info));
}

void EthosNSubgraphViewConverter::AddSpaceToDepthLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::SpaceToDepth);
    const SpaceToDepthLayer& spaceToDepthLayer = *PolymorphicPointerDowncast<const SpaceToDepthLayer>(layer);

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    ethosn_lib::SpaceToDepthInfo info(spaceToDepthLayer.GetParameters().m_BlockSize);

    // SpaceToDepth has exactly one output that maps neatly to the NPU
    InsertConvertedLayerSingleOutput(layer, ethosn_lib::AddSpaceToDepth(m_Network, *input.tensor, info));
}

void EthosNSubgraphViewConverter::AddTransposeLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Transpose);
    const TransposeLayer& transposeLayer = *PolymorphicPointerDowncast<const TransposeLayer>(layer);

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    // Note it's safe to assume that BuildEthosNTransposeInfo succeeds, because we checked this in IsTransposeSupported.
    auto transposeInfo = BuildEthosNTransposeInfo(transposeLayer.GetPermutation());

    InsertConvertedLayerSingleOutput(layer, ethosn_lib::AddTranspose(m_Network, *input.tensor, transposeInfo.value()));
}

void EthosNSubgraphViewConverter::AddQuantizeLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Quantize);
    const QuantizeLayer& quantizeLayer = *PolymorphicPointerDowncast<const QuantizeLayer>(layer);

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    // Note it's safe to assume that BuildEthosNRequantizeInfo succeeds, because we checked this in IsRequantizeSupported.
    auto requantizeInfo = BuildEthosNRequantizeInfo(quantizeLayer.GetOutputSlot(0).GetTensorInfo());

    InsertConvertedLayerSingleOutput(layer, ethosn_lib::AddRequantize(m_Network, *input.tensor, requantizeInfo));
}

void EthosNSubgraphViewConverter::AddResizeLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::Resize);
    const ResizeLayer& resizeLayer = *PolymorphicPointerDowncast<const ResizeLayer>(layer);

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    // Note it's safe to assume that BuildEthosNResizeInfo succeeds, because we checked this in IsResizeSupported.
    auto resizeInfo = BuildEthosNResizeInfo(resizeLayer.GetParameters(), resizeLayer.GetOutputSlot(0).GetTensorInfo());

    InsertConvertedLayerSingleOutput(layer, ethosn_lib::AddResize(m_Network, *input.tensor, resizeInfo));
}

void EthosNSubgraphViewConverter::AddEstimateOnly(const IConnectableLayer* layer,
                                                  const std::string& reasonForEstimateOnly)
{
    ARMNN_ASSERT(layer != nullptr);

    std::vector<ethosn_lib::Operand*> inputOperands;
    inputOperands.reserve(layer->GetNumInputSlots());
    for (uint32_t i = 0; i < layer->GetNumInputSlots(); ++i)
    {
        auto ethosnOperand = AddOrRetrieveEthosNOperand(layer->GetInputSlot(i).GetConnection());
        inputOperands.push_back(ethosnOperand.tensor.get());
    }

    std::vector<ethosn_lib::TensorInfo> ethosnOutputInfos;
    ethosnOutputInfos.reserve(layer->GetNumOutputSlots());
    for (uint32_t i = 0; i < layer->GetNumOutputSlots(); ++i)
    {
        ethosnOutputInfos.push_back(BuildEthosNTensorInfo(layer->GetOutputSlot(i).GetTensorInfo(), DataLayout::NHWC));
    }
    ethosn_lib::EstimateOnlyInfo estimateInfo(ethosnOutputInfos, reasonForEstimateOnly);
    InsertConvertedLayerMultipleOutput(layer, ethosn_lib::AddEstimateOnly(m_Network, inputOperands, estimateInfo));
}

// StandIn Layer is to be used as a generic layer whenever a layer is not defined in the Arm NN.
void EthosNSubgraphViewConverter::AddStandInLayer(const IConnectableLayer* layer)
{
    ARMNN_ASSERT(layer != nullptr);
    ARMNN_ASSERT(layer->GetType() == LayerType::StandIn);
    const StandInLayer& standInLayer = *PolymorphicPointerDowncast<const StandInLayer>(layer);

    auto input = AddOrRetrieveEthosNOperand(layer->GetInputSlot(0).GetConnection());

    // StandIn layer being used to add a ReinterpretQuantization operation.
    if ((std::string(layer->GetName()) == "EthosNBackend:ReplaceScalarMulWithReinterpretQuantization") &&
        (standInLayer.GetParameters().m_NumInputs == 1U) && (standInLayer.GetParameters().m_NumOutputs == 1U))
    {
        auto reinterpretQuantizeInfo =
            BuildEthosNReinterpretQuantizationInfo(standInLayer.GetOutputSlot(0).GetTensorInfo());

        InsertConvertedLayerSingleOutput(
            layer, ethosn_lib::AddReinterpretQuantization(m_Network, *input.tensor, reinterpretQuantizeInfo));
    }
    else if ((std::string(layer->GetName()) == "EthosNBackend:ReplaceScalarAddWithReinterpretQuantization") &&
             (standInLayer.GetParameters().m_NumInputs == 1U) && (standInLayer.GetParameters().m_NumOutputs == 1U))
    {
        auto reinterpretQuantizeInfo =
            BuildEthosNReinterpretQuantizationInfo(standInLayer.GetOutputSlot(0).GetTensorInfo());
        InsertConvertedLayerSingleOutput(
            layer, ethosn_lib::AddReinterpretQuantization(m_Network, *input.tensor, reinterpretQuantizeInfo));
    }
    else
    {
        std::string reason = "StandIn layer is not supported.";
        HandleUnknownLayer(layer, reason);
    }
}

void EthosNSubgraphViewConverter::HandleUnknownLayer(const IConnectableLayer* layer, const std::string& reason)
{
    if (m_EthosNConfig.m_PerfOnly)
    {
        ARMNN_LOG(info) << "\"" << std::string(layer->GetName()) << "\" is replaced with an estimate only node "
                        << "LayerType: " << GetLayerTypeAsCString(layer->GetType());
        AddEstimateOnly(layer, reason);
    }
    else
    {
        throw Exception("Conversion not supported for layer type " +
                        std::string(GetLayerTypeAsCString(layer->GetType())));
    }
}

EthosNOperand EthosNSubgraphViewConverter::AddOrRetrieveEthosNOperand(const IOutputSlot* outputSlot)
{
    ARMNN_ASSERT(outputSlot != nullptr);

    // Check if the layer has already been converted and added
    auto locationInMap = m_ConvertedOutputSlots.find(outputSlot);
    if (locationInMap != m_ConvertedOutputSlots.end())
    {
        // Layer already present in the network => retrieve it
        return locationInMap->second;
    }

    // Layer not added yet => add it
    const IConnectableLayer& layer = outputSlot->GetOwningIConnectableLayer();
    ARMNN_LOG(trace) << "Converting layer " << layer.GetGuid();
    switch (layer.GetType())
    {
        case LayerType::Activation:
            AddActivationLayer(&layer);
            break;
        case LayerType::Addition:
            AddAdditionLayer(&layer);
            break;
        case LayerType::Constant:
            AddConstantLayer(&layer);
            break;
        case LayerType::Convolution2d:
            AddConvolution2dLayer(&layer);
            break;
        case LayerType::DepthwiseConvolution2d:
            AddDepthwiseConvolution2dLayer(&layer);
            break;
        case LayerType::TransposeConvolution2d:
            AddTransposeConvolution2dLayer(&layer);
            break;
        case LayerType::FullyConnected:
            AddFullyConnectedLayer(&layer);
            break;
        case LayerType::Concat:
            AddConcatLayer(&layer);
            break;
        case LayerType::Pooling2d:
            AddPooling2dLayer(&layer);
            break;
        case LayerType::Reshape:
            AddReshapeLayer(&layer);
            break;
        case LayerType::Softmax:
            AddSoftmaxLayer(&layer);
            break;
        case LayerType::Splitter:
            AddSplitterLayer(&layer);
            break;
        case LayerType::DepthToSpace:
            AddDepthToSpaceLayer(&layer);
            break;
        case LayerType::SpaceToDepth:
            AddSpaceToDepthLayer(&layer);
            break;
        case LayerType::Transpose:
            AddTransposeLayer(&layer);
            break;
        case LayerType::Quantize:
            AddQuantizeLayer(&layer);
            break;
        case LayerType::StandIn:
            AddStandInLayer(&layer);
            break;
        case LayerType::Resize:
            AddResizeLayer(&layer);
            break;
        case LayerType::Mean:
            AddMeanXyLayer(&layer);
            break;
        default:
            std::string string(layer.GetName());
            std::string reason = string + " is not currently supported.";
            HandleUnknownLayer(&layer, reason);
    }

    // Return the Ethos-N operand that should now have been added
    locationInMap = m_ConvertedOutputSlots.find(outputSlot);
    ARMNN_ASSERT(locationInMap != m_ConvertedOutputSlots.end());
    return locationInMap->second;
}

void EthosNSubgraphViewConverter::InsertConvertedLayerSingleOutput(const IConnectableLayer* layer,
                                                                   EthosNAddOperationResult ethosnAddOperationResult)
{
    ARMNN_ASSERT(layer->GetNumOutputSlots() == 1);
    m_ConvertedOutputSlots[&layer->GetOutputSlot(0)]                   = { ethosnAddOperationResult.operationId,
                                                         ethosnAddOperationResult.tensor, 0 };
    m_EthosNOperationNameMapping[ethosnAddOperationResult.operationId] = layer->GetName();
}

void EthosNSubgraphViewConverter::InsertConvertedLayerMultipleOutput(const IConnectableLayer* layer,
                                                                     ethosn_lib::TensorsAndId ethosnAddOperationResult)
{
    ARMNN_ASSERT(layer->GetNumOutputSlots() == ethosnAddOperationResult.tensors.size());
    for (uint32_t i = 0; i < layer->GetNumOutputSlots(); ++i)
    {
        m_ConvertedOutputSlots[&layer->GetOutputSlot(i)] = { ethosnAddOperationResult.operationId,
                                                             ethosnAddOperationResult.tensors[i], i };
    }
    m_EthosNOperationNameMapping[ethosnAddOperationResult.operationId] = layer->GetName();
}

void EthosNSubgraphViewConverter::CreateUncompiledNetwork()
{
    if (m_Network)
    {
        // Network already created
        return;
    }

    // Initialize a new network
    m_Network = m_EthosNConfig.m_PerfOnly ? ethosn_lib::CreateEstimationNetwork(m_Capabilities)
                                          : ethosn_lib::CreateNetwork(m_Capabilities);

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

std::vector<PreCompiledObjectPtr> EthosNSubgraphViewConverter::Estimate()
{
    ethosn_lib::EstimationOptions ethosnEstimationOpts;
    ethosnEstimationOpts.m_ActivationCompressionSaving  = m_EthosNConfig.m_PerfActivationCompressionSaving;
    ethosnEstimationOpts.m_UseWeightCompressionOverride = m_EthosNConfig.m_PerfUseWeightCompressionOverride;
    ethosnEstimationOpts.m_WeightCompressionSaving      = m_EthosNConfig.m_PerfWeightCompressionSaving;
    ethosnEstimationOpts.m_Current                      = m_EthosNConfig.m_PerfCurrent;
    EthosNPreCompiledObject::PerfData perfData;

    perfData.m_PerfOutFile               = m_CompilationOptions.m_DebugInfo.m_DebugDir + "/report.json";
    perfData.m_PerfVariant               = m_EthosNConfig.m_PerfVariant;
    perfData.m_PerfSramSizeBytesOverride = m_EthosNConfig.m_PerfSramSizeBytesOverride;
    perfData.m_EstimationOptions         = ethosnEstimationOpts;

    ARMNN_LOG(debug) << "Estimating Ethos-N network";
    perfData.m_Data = ethosn_lib::EstimatePerformance(*m_Network, m_CompilationOptions, ethosnEstimationOpts);

    auto preCompiledObj =
        std::make_unique<EthosNPreCompiledObject>(std::move(perfData), m_EthosNOperationNameMapping, m_SubgraphIdx);

    std::vector<PreCompiledObjectPtr> compiledBlobs;

    // Convert the EthosNPreCompiledObject into a "blob" (void) object and attach the custom blob deleter
    compiledBlobs.emplace_back(preCompiledObj.release(), DeleteAsType<EthosNPreCompiledObject>);

    return compiledBlobs;
}

std::vector<PreCompiledObjectPtr> EthosNSubgraphViewConverter::CompileNetwork()
{
    fs::create_directories(m_CompilationOptions.m_DebugInfo.m_DebugDir);

    // Compile the network into a list of generic type-agnostic "blobs"
    std::vector<PreCompiledObjectPtr> compiledBlobs;

    try
    {
        // Create a new network to be compiled by the Ethos-N backend
        CreateUncompiledNetwork();

        compiledBlobs = m_EthosNConfig.m_PerfOnly ? Estimate() : Compile();
    }
    catch (const std::exception& e)
    {
        // An exception has been thrown when either trying to build the uncompiled network, or by the compiler.
        // Swallow it, as this API is not expected to throw, and return an empty list of compiled blobs instead,
        // it will be handled by the caller (the OptimizeSubgraphView method in the backend) when putting together
        // the OptimizationViews object to return as the optimization result
        ARMNN_LOG(warning) << "Exception thrown in EthosNSubgraphViewConverter::CompileNetwork: " << e.what();
        compiledBlobs.clear();
    }

    return compiledBlobs;
}

std::vector<PreCompiledObjectPtr> EthosNSubgraphViewConverter::Compile()
{
    std::vector<PreCompiledObjectPtr> compiledBlobs;

    auto caching = EthosNCachingService::GetInstance().GetEthosNCachingPtr();

    if (caching->IsLoading())
    {
        // Currently one compiled network is supported.
        std::pair<std::vector<char>, uint32_t> networkAndIntermediate =
            caching->GetCompiledNetworkAndIntermediateSize(m_SubgraphIdx);

        auto preCompiledObject = std::make_unique<EthosNPreCompiledObject>(
            EthosNPreCompiledObject::Network(std::move(networkAndIntermediate.first)), m_EthosNOperationNameMapping,
            m_EthosNConfig.m_InferenceTimeout, m_SubgraphIdx, networkAndIntermediate.second);

        // Convert the EthosNPreCompiledObject into a "blob" (void) object and attach the custom blob deleter
        compiledBlobs.emplace_back(preCompiledObject.release(), DeleteAsType<EthosNPreCompiledObject>);
    }
    else
    {
        ARMNN_LOG(debug) << "Compiling Ethos-N network (subgraph " << m_SubgraphIdx << ")";
        std::vector<EthosNCompiledNetworkPtr> compiledNetworks =
            g_EthosNSupportLibraryInterface->Compile(*m_Network, m_CompilationOptions);

        // Create a list of generic type-agnostic compiled "blobs"
        for (EthosNCompiledNetworkPtr& compiledNetwork : compiledNetworks)
        {
            // Ensure the Arm NN input slots match the Ethos-N input indices, based on the data we gathered while adding the Ethos-N operations.
            for (uint32_t ethosnInputIdx = 0; ethosnInputIdx < compiledNetwork->GetInputBufferInfos().size();
                 ++ethosnInputIdx)
            {
                const ethosn_lib::InputBufferInfo& inputBufferInfo =
                    compiledNetwork->GetInputBufferInfos()[ethosnInputIdx];
                uint32_t inputSlotIdx = m_EthosNInputIdToInputSlot.at(
                    { inputBufferInfo.m_SourceOperationId, inputBufferInfo.m_SourceOperationOutputIndex });

                if (inputSlotIdx != ethosnInputIdx)
                {
                    throw armnn::InvalidArgumentException(
                        "Arm NN input slot indice does not match Ethos-N input slot indice.", CHECK_LOCATION());
                }
            }

            // Ensure the Arm NN output slots match the Ethos-N output indices, based on the data we gathered while adding the Ethos-N operations.
            for (uint32_t ethosnOutputIdx = 0; ethosnOutputIdx < compiledNetwork->GetOutputBufferInfos().size();
                 ++ethosnOutputIdx)
            {
                const ethosn_lib::OutputBufferInfo& outputBufferInfo =
                    compiledNetwork->GetOutputBufferInfos()[ethosnOutputIdx];
                uint32_t outputSlotIdx = m_EthosNOutputIdToOutputSlot.at(
                    { outputBufferInfo.m_SourceOperationId, outputBufferInfo.m_SourceOperationOutputIndex });

                if (outputSlotIdx != ethosnOutputIdx)
                {
                    throw armnn::InvalidArgumentException(
                        "Arm NN output slot indice does not match Ethos-N output slot indice.", CHECK_LOCATION());
                }
            }

            auto intermediateBufSize = compiledNetwork->GetIntermediateBufferSize();

            // Construct a EthosNPreCompiledObject containing the serialized ethosn_lib::CompiledNetwork along with
            // other data needed by the workload.
            std::vector<char> compiledNetworkData;
            {
                ethosn::utils::VectorStream compiledNetworkStream(compiledNetworkData);
                compiledNetwork->Serialize(compiledNetworkStream);
            }
            compiledNetwork.release();    // No longer need this, so save the memory

            // If saving options are specified, add to stored map to save once complete.
            if (caching->IsSaving())
            {
                caching->AddCompiledNetwork(compiledNetworkData, intermediateBufSize);
                caching->IncrementSubgraphCount();
            }

            auto preCompiledObject = std::make_unique<EthosNPreCompiledObject>(
                EthosNPreCompiledObject::Network(std::move(compiledNetworkData)), m_EthosNOperationNameMapping,
                m_EthosNConfig.m_InferenceTimeout, m_SubgraphIdx, intermediateBufSize);

            // Convert the EthosNPreCompiledObject into a "blob" (void) object and attach the custom blob deleter
            compiledBlobs.emplace_back(preCompiledObject.release(), DeleteAsType<EthosNPreCompiledObject>);
        }
    }

    return compiledBlobs;
}

ethosn_lib::CompilationOptions
    GetCompilationOptions(const EthosNConfig& config, const ModelOptions& modelOptions, uint32_t instanceId)
{
    ethosn_lib::CompilationOptions result;
    result.m_DebugInfo.m_DumpDebugFiles    = config.m_DumpDebugFiles;
    result.m_DebugInfo.m_DumpRam           = config.m_DumpRam;
    result.m_DebugInfo.m_DebugDir          = config.m_PerfOutDir + "/subgraph_" + std::to_string(instanceId);
    result.m_EnableIntermediateCompression = config.m_IntermediateCompression;

    for (const auto& optionsGroup : modelOptions)
    {
        if (optionsGroup.GetBackendId() == EthosNBackend::GetIdStatic())
        {
            for (size_t i = 0; i < optionsGroup.GetOptionCount(); i++)
            {
                const BackendOptions::BackendOption& option = optionsGroup.GetOption(i);

                if (option.GetName() == "DisableWinograd")
                {
                    if (option.GetValue().IsBool())
                    {
                        result.m_DisableWinograd = option.GetValue().AsBool();
                    }
                    else
                    {
                        throw armnn::InvalidArgumentException(
                            "Invalid option type for DisableWinograd - must be bool.");
                    }
                }
                else if (option.GetName() == "StrictPrecision")
                {
                    if (option.GetValue().IsBool())
                    {
                        result.m_StrictPrecision = option.GetValue().AsBool();
                    }
                    else
                    {
                        throw armnn::InvalidArgumentException(
                            "Invalid option type for StrictPrecision - must be bool.");
                    }
                }
                else if (option.GetName() == "Device")
                {
                    // Device option is allowed, but not used here.
                }
                else if (option.GetName() == "SaveCachedNetwork" || option.GetName() == "CachedNetworkFilePath")
                {
                    // SaveCachedNetwork and CachedNetworkFilePath option is allowed, but not used here.
                }
                else
                {
                    throw armnn::InvalidArgumentException("Invalid option - " + option.GetName());
                }
            }
        }
    }

    return result;
}

}    // namespace armnn
