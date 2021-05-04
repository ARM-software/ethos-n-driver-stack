//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNBackend.hpp"

#include "EthosNBackendId.hpp"
#include "EthosNBackendProfilingContext.hpp"
#include "EthosNBackendUtils.hpp"
#include "EthosNLayerSupport.hpp"
#include "EthosNMapping.hpp"
#include "EthosNSubgraphViewConverter.hpp"
#include "EthosNWorkloadFactory.hpp"

#include <Optimizer.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/Assert.hpp>
#include <backendsCommon/IBackendContext.hpp>
#include <backendsCommon/IMemoryManager.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>
#include <ethosn_driver_library/Network.hpp>
#include <ethosn_support_library/Support.hpp>

namespace armnn
{

ARMNN_DLLEXPORT EthosNConfig g_EthosNConfig;
ARMNN_DLLEXPORT EthosNMappings g_EthosNMappings;

namespace ethosnbackend
{

constexpr bool IsLibraryVersionSupported(const uint32_t& majorVer, const uint32_t& maxVer, const uint32_t& minVer)
{
    return (majorVer <= maxVer) && (majorVer >= minVer);
}

bool VerifyLibraries()
{
    constexpr bool IsDriverLibSupported = IsLibraryVersionSupported(ETHOSN_DRIVER_LIBRARY_VERSION_MAJOR,
                                                                    MAX_ETHOSN_DRIVER_LIBRARY_MAJOR_VERSION_SUPPORTED,
                                                                    MIN_ETHOSN_DRIVER_LIBRARY_MAJOR_VERSION_SUPPORTED);
    static_assert(IsDriverLibSupported, "Driver library version is not supported by the backend");

    constexpr bool IsSupportLibSupported = IsLibraryVersionSupported(
        ETHOSN_SUPPORT_LIBRARY_VERSION_MAJOR, MAX_ETHOSN_SUPPORT_LIBRARY_MAJOR_VERSION_SUPPORTED,
        MIN_ETHOSN_SUPPORT_LIBRARY_MAJOR_VERSION_SUPPORTED);
    static_assert(IsSupportLibSupported, "Support library version is not supported by the backend");

    return IsDriverLibSupported && IsSupportLibSupported;
}

template <typename T>
T NextEnumValue(T current)
{
    return static_cast<T>(static_cast<uint32_t>(current) + 1);
}

BackendRegistry::StaticRegistryInitializer g_RegisterHelper{ BackendRegistryInstance(), EthosNBackend::GetIdStatic(),
                                                             []() {
                                                                 return IBackendInternalUniquePtr(new EthosNBackend);
                                                             } };

Graph CloneGraph(const SubgraphView& originalSubgraph)
{
    Graph newGraph = Graph();

    std::unordered_map<const Layer*, Layer*> originalToClonedLayerMap;
    std::list<armnn::Layer*> originalSubgraphLayers = originalSubgraph.GetLayers();

    for (auto&& originalLayer : originalSubgraphLayers)
    {
        Layer* const layer = originalLayer->Clone(newGraph);
        originalToClonedLayerMap.emplace(originalLayer, layer);
    }

    LayerBindingId slotCount = 0;

    // SubstituteSubgraph() currently cannot be called on a Graph that contains only one layer.
    // CloneGraph() and ReinterpretGraphToSubgraph() are used to work around this.

    // creating new layers for the input slots, adding them to the new graph and connecting them
    for (auto originalSubgraphInputSlot : originalSubgraph.GetInputSlots())
    {
        Layer& originalSubgraphLayer = originalSubgraphInputSlot->GetOwningLayer();
        Layer* const clonedLayer     = originalToClonedLayerMap[&originalSubgraphLayer];

        const std::string& originalLayerName =
            originalSubgraphInputSlot->GetConnectedOutputSlot()->GetOwningLayer().GetNameStr();

        // add it as an input layer into the new graph
        InputLayer* const newInputLayer = newGraph.AddLayer<InputLayer>(slotCount, originalLayerName.c_str());
        InputSlot& clonedLayerIS        = clonedLayer->GetInputSlot(originalSubgraphInputSlot->GetSlotIndex());
        newInputLayer->GetOutputSlot(0).Connect(clonedLayerIS);
        newInputLayer->GetOutputSlot(0).SetTensorInfo(
            originalSubgraphInputSlot->GetConnectedOutputSlot()->GetTensorInfo());

        ++slotCount;
    }

    std::list<Layer*>::iterator it;
    for (it = originalSubgraphLayers.begin(); it != originalSubgraphLayers.end(); ++it)
    {
        Layer* originalSubgraphLayer = *it;
        Layer* const clonedLayer     = originalToClonedLayerMap[originalSubgraphLayer];

        //connect all cloned layers as per original subgraph
        auto outputSlot = clonedLayer->BeginOutputSlots();
        for (auto&& originalOutputSlot : originalSubgraphLayer->GetOutputSlots())
        {
            for (auto&& connection : originalOutputSlot.GetConnections())
            {
                const Layer& otherTgtLayer = connection->GetOwningLayer();
                // in the case that the connection is a layer outside the subgraph, it will not have a corresponding connection
                if (originalToClonedLayerMap.find(&otherTgtLayer) != originalToClonedLayerMap.end())
                {
                    Layer* const newGrTgtLayer = originalToClonedLayerMap[&otherTgtLayer];

                    InputSlot& inputSlot = newGrTgtLayer->GetInputSlot(connection->GetSlotIndex());
                    outputSlot->Connect(inputSlot);
                }
            }
            outputSlot->SetTensorInfo(originalOutputSlot.GetTensorInfo());
            ++outputSlot;
        }
    }

    // creating new layers for the output slots, adding them to the new graph and connecting them
    for (auto os : originalSubgraph.GetOutputSlots())
    {
        Layer& originalSubgraphLayer = os->GetOwningLayer();
        Layer* const clonedLayer     = originalToClonedLayerMap[&originalSubgraphLayer];

        uint32_t i = 0;
        for (; i < originalSubgraphLayer.GetNumOutputSlots(); ++i)
        {
            if (os == &originalSubgraphLayer.GetOutputSlot(i))
            {
                break;
            }
        }

        ARMNN_ASSERT(i < originalSubgraphLayer.GetNumOutputSlots());

        const std::string& originalLayerName = os->GetConnection(0)->GetOwningLayer().GetNameStr();

        OutputSlot* outputSlotOfLayer     = &clonedLayer->GetOutputSlot(i);
        OutputLayer* const newOutputLayer = newGraph.AddLayer<OutputLayer>(slotCount, originalLayerName.c_str());
        ++slotCount;

        outputSlotOfLayer->Connect(newOutputLayer->GetInputSlot(0));
        outputSlotOfLayer->SetTensorInfo(originalSubgraphLayer.GetOutputSlot(i).GetTensorInfo());
    }

    return newGraph;
}

// This is different to creating a subgraph directly from the Graph
// and is needed to obtain a subgraph that does not contain the input and output layers
SubgraphView ReinterpretGraphToSubgraph(Graph& newGraph)
{
    std::list<Layer*> graphLayers(newGraph.begin(), newGraph.end());
    std::list<Layer*> subgrLayers;

    std::vector<InputLayer*> inputLayersNewGr;
    std::vector<OutputLayer*> outputLayersNewGr;

    for (auto layer : graphLayers)
    {
        switch (layer->GetType())
        {
            case LayerType::Input:
                inputLayersNewGr.push_back(PolymorphicDowncast<InputLayer*>(layer));
                break;
            case LayerType::Output:
                outputLayersNewGr.push_back(PolymorphicDowncast<OutputLayer*>(layer));
                break;
            default:
                subgrLayers.push_back(layer);
                break;
        }
    }

    std::vector<InputSlot*> inSlotsPointers;
    for (InputLayer* is : inputLayersNewGr)
    {
        inSlotsPointers.push_back(is->GetOutputSlot(0).GetConnection(0));
    }

    std::vector<OutputSlot*> outSlotsPointers;
    for (OutputLayer* os : outputLayersNewGr)
    {
        outSlotsPointers.push_back(os->GetInputSlot(0).GetConnectedOutputSlot());
    }

    return SubgraphView(std::move(inSlotsPointers), std::move(outSlotsPointers), std::move(subgrLayers));
}

char const* GetLayerTypeAsCStringWrapper(LayerType type)
{
    switch (type)
    {
        // workaround because these LayerTypes have not been added to the function in Arm NN yet
        case LayerType::MemImport:
            return "MemImport";
        case LayerType::Quantize:
            return "Quantize";
        case LayerType::SpaceToDepth:
            return "SpaceToDepth";
        default:
            return GetLayerTypeAsCString(type);
    }
}

std::map<std::string, LayerType> GetMapStringToLayerType()
{
    std::map<std::string, LayerType> mapStringToType;

    for (LayerType type = LayerType::FirstLayer; type <= LayerType::LastLayer; type = NextEnumValue(type))
    {
        mapStringToType.emplace(GetLayerTypeAsCStringWrapper(type), type);
    }

    return mapStringToType;
}

std::map<std::string, ActivationFunction> GetMapStringToActivationFunction()
{
    std::map<std::string, ActivationFunction> mapStringToType;

    for (ActivationFunction type = ActivationFunction::Sigmoid; type <= ActivationFunction::HardSwish;
         type                    = NextEnumValue(type))
    {
        mapStringToType.emplace(GetActivationFunctionAsCString(type), type);
    }

    return mapStringToType;
}

std::map<std::string, PoolingAlgorithm> GetMapStringToPoolingAlgorithm()
{

    std::map<std::string, PoolingAlgorithm> mapStringToType;

    for (PoolingAlgorithm type = PoolingAlgorithm::Max; type <= PoolingAlgorithm::L2; type = NextEnumValue(type))
    {
        mapStringToType.emplace(GetPoolingAlgorithmAsCString(type), type);
    }

    return mapStringToType;
}

template <class ConvLayerClass, class ConvLayerDescriptor>
Layer* CreateConvolutionLayer(
    std::string layerName, ConvLayerDescriptor& desc, Graph& graph, TensorInfo& weight, TensorInfo& bias)
{
    ConvLayerClass* convLayer = graph.AddLayer<ConvLayerClass>(desc, layerName.c_str());
    SetWeightAndBias(convLayer, weight, bias);

    return convLayer;
}

template <class ConvLayerDescriptor>
void FillStride(ConvLayerDescriptor& desc, AdditionalLayerParams& params)
{
    auto stride    = GetLayerParameterValue(params, "stride");
    desc.m_StrideX = stride[STRIDE_X];
    desc.m_StrideY = stride[STRIDE_Y];
}

template <class ConvLayerDescriptor>
void FillPadding(ConvLayerDescriptor& desc, AdditionalLayerParams& params)
{
    auto padding     = GetLayerParameterValue(params, "padding");
    desc.m_PadBottom = padding[PAD_BOTTOM];
    desc.m_PadLeft   = padding[PAD_LEFT];
    desc.m_PadRight  = padding[PAD_RIGHT];
    desc.m_PadTop    = padding[PAD_TOP];
}

template <class ConvLayerDescriptor>
void FillDilation(ConvLayerDescriptor& desc, AdditionalLayerParams& params)
{
    auto dilation    = GetLayerParameterValue(params, "dilation");
    desc.m_DilationX = dilation[DILATION_X];
    desc.m_DilationY = dilation[DILATION_Y];
}

template <class ConvLayerDescriptor>
void FillPoolSize(ConvLayerDescriptor& desc, AdditionalLayerParams& params)
{
    // For Pooling layers, the user specifies the pool width and height
    // in additional parameter "kernel".
    // ie ((kernel=poolWidthxpoolHeight))
    auto kernel       = GetLayerParameterValue(params, "kernel");
    desc.m_PoolHeight = kernel[KERNEL_HEIGHT];
    desc.m_PoolWidth  = kernel[KERNEL_WIDTH];
}

template <class ConvLayerDescriptor>
void SetDataLayoutAndBias(ConvLayerDescriptor& desc)
{
    desc.m_DataLayout  = DataLayout::NHWC;
    desc.m_BiasEnabled = true;
}

Convolution2dDescriptor CreateConv2dDescriptor(AdditionalLayerParams params)
{
    Convolution2dDescriptor desc;

    FillStride<Convolution2dDescriptor>(desc, params);
    FillDilation<Convolution2dDescriptor>(desc, params);
    FillPadding<Convolution2dDescriptor>(desc, params);
    SetDataLayoutAndBias<Convolution2dDescriptor>(desc);

    return desc;
}

TransposeConvolution2dDescriptor CreateTransConv2dDescriptor(AdditionalLayerParams params)
{
    TransposeConvolution2dDescriptor desc;

    FillStride<TransposeConvolution2dDescriptor>(desc, params);
    FillPadding<TransposeConvolution2dDescriptor>(desc, params);
    SetDataLayoutAndBias<TransposeConvolution2dDescriptor>(desc);

    return desc;
}

DepthwiseConvolution2dDescriptor CreateDepthConvDescriptor(AdditionalLayerParams params)
{
    DepthwiseConvolution2dDescriptor desc;

    FillStride<DepthwiseConvolution2dDescriptor>(desc, params);
    FillDilation<DepthwiseConvolution2dDescriptor>(desc, params);
    FillPadding<DepthwiseConvolution2dDescriptor>(desc, params);
    SetDataLayoutAndBias<DepthwiseConvolution2dDescriptor>(desc);

    return desc;
}

Layer* CreateConvolutionLayer(LayerType type,
                              Graph& graph,
                              unsigned int inputChannels,
                              AdditionalLayerParams additionalLayerParams,
                              DataType weightDatatype,
                              DataType biasDataType)
{
    Layer* newLayer       = nullptr;
    auto kernel           = GetLayerParameterValue(additionalLayerParams, "kernel");
    auto kernelHeight     = kernel[KERNEL_HEIGHT];
    auto kernelWidth      = kernel[KERNEL_WIDTH];
    std::string layerName = additionalLayerParams["name"];

    // For all the convolutions, the basic assumption is that the outputChannels
    // is same as inputChannels, so that the output tensor shape is same as input
    // tensor shape.
    if (type == LayerType::Convolution2d)
    {
        ARMNN_LOG(info) << "The replacement is Convolution2d \n";

        // The weightDimensions are of the format OHWI
        const unsigned int weightDimensions[4]{ inputChannels, kernelHeight, kernelWidth, inputChannels };
        TensorInfo weight(4, weightDimensions, weightDatatype, 0.5f, 0);

        // The bias is of the format NHWC
        TensorInfo bias = TensorInfo({ 1, kernelHeight, kernelWidth, inputChannels }, biasDataType, 0.9f, 0);

        auto desc = CreateConv2dDescriptor(additionalLayerParams);

        newLayer =
            CreateConvolutionLayer<Convolution2dLayer, Convolution2dDescriptor>(layerName, desc, graph, weight, bias);
    }
    // The notable exception to this is TransposeConvolution2d, where the output tensor's width and
    // height is double of that of the input tensor.
    else if (type == LayerType::TransposeConvolution2d)
    {
        ARMNN_LOG(info) << "The replacement is TransposeConvolution2d \n";

        // The weightDimensions are of the format OHWI
        const unsigned int weightDimensions[4]{ inputChannels, kernelHeight, kernelWidth, inputChannels };
        TensorInfo weight(4, weightDimensions, weightDatatype, 0.5f, 0);

        // The bias is of the format NHWC
        TensorInfo bias = TensorInfo({ 1, kernelHeight, kernelWidth, inputChannels }, biasDataType, 0.9f, 0);

        auto desc = CreateTransConv2dDescriptor(additionalLayerParams);

        newLayer = CreateConvolutionLayer<TransposeConvolution2dLayer, TransposeConvolution2dDescriptor>(
            layerName, desc, graph, weight, bias);
    }
    else if (type == LayerType::DepthwiseConvolution2d)
    {
        unsigned int channelMultiplier = 1;

        ARMNN_LOG(info) << "The replacement is DepthwiseConvolution2d \n";

        // The weightDimensions are of the format MIHW
        const unsigned int weightDimensions[4]{ channelMultiplier, inputChannels, kernelHeight, kernelWidth };
        TensorInfo weight(4, weightDimensions, weightDatatype, 0.5f, 0);

        // The bias is of the format NHWC
        TensorInfo bias =
            TensorInfo({ 1, kernelHeight, kernelWidth, (inputChannels * channelMultiplier) }, biasDataType, 0.9f, 0);

        auto desc = CreateDepthConvDescriptor(additionalLayerParams);

        newLayer = CreateConvolutionLayer<DepthwiseConvolution2dLayer, DepthwiseConvolution2dDescriptor>(
            layerName, desc, graph, weight, bias);
    }
    return newLayer;
}

Layer* CreateActivationLayer(Graph& graph, std::string activationFunc, std::string layerName)
{
    std::map<std::string, ActivationFunction> mapStringToActivationFunction = GetMapStringToActivationFunction();

    auto func = mapStringToActivationFunction.find(activationFunc)->second;
    ActivationDescriptor desc;
    desc.m_Function = func;

    auto* layer = graph.AddLayer<ActivationLayer>(desc, layerName.c_str());

    return layer;
}

void ValidatePoolingLayerParameters(SimpleLayer& layer)
{
    std::string errors;
    auto poolAlgo   = GetMapStringToPoolingAlgorithm();
    auto m_PoolType = poolAlgo.find(layer.m_LayerParams["function"])->second;

    // Currently, only AveragePooling is supported.
    if (m_PoolType != PoolingAlgorithm::Average)
    {
        errors = "Invalid Value: Only Average Pooling is supported\n";
        throw armnn::InvalidArgumentException(errors);
    }
}

void ValidateActivationLayerParameters(SimpleLayer& layer)
{
    std::map<std::string, ActivationFunction> mapStringToActivationFunction = GetMapStringToActivationFunction();
    auto funcName                                                           = layer.m_LayerParams.find("function");
    auto func = mapStringToActivationFunction.find(funcName->second);
    std::string errors;

    // Currently we support only Sigmoid, ReLu and LeakyReLu Activation functions
    switch (func->second)
    {
        case ActivationFunction::Sigmoid:
        case ActivationFunction::ReLu:
        case ActivationFunction::LeakyReLu:
            break;
        default:
            errors = "Invalid Value: Activation functions other than Sigmoid, ReLu and LeakyRelu are not supported\n";
            throw armnn::InvalidArgumentException(errors);
            break;
    }
}

void ValidateConvolutionLayerParameters(SimpleLayer& layer)
{
    std::string errors;

    if (layer.m_Inputs[0].m_Shape.size() != 4)
    {
        errors = "Invalid Value: The number of dimensions for input/output tensor has to be 4\n";
        throw armnn::InvalidArgumentException(errors);
    }
}

Layer* CreateFullyConnectedLayer(Graph& graph,
                                 const TensorInfo& inputTensor,
                                 const TensorInfo& outputTensor,
                                 AdditionalLayerParams& params)
{
    FullyConnectedDescriptor desc;
    auto name = params["name"];

    uint32_t numInputs  = inputTensor.GetShape()[inputTensor.GetNumDimensions() - 1];
    uint32_t numOutputs = outputTensor.GetShape()[outputTensor.GetNumDimensions() - 1];
    // One needs to ensure that inputTensor.GetQuantizationScale() * weightInfo.GetQuantizationScale() / outputTensor.GetQuantizationScale()
    // should be in the range of [0, 1)
    float weightQuantizationScale = 0.5f * outputTensor.GetQuantizationScale() / inputTensor.GetQuantizationScale();

    const TensorInfo weightInfo({ numInputs, numOutputs }, inputTensor.GetDataType(), weightQuantizationScale, 0);
    float biasQuantizationScale = inputTensor.GetQuantizationScale() * weightInfo.GetQuantizationScale();
    const TensorInfo biasesInfo({ 1, numOutputs }, DataType::Signed32, biasQuantizationScale, 0);

    desc.m_BiasEnabled = true;

    ARMNN_LOG(info) << "Creating a Fully Connected layer \n";
    auto layer = graph.AddLayer<FullyConnectedLayer>(desc, name.c_str());
    SetWeightAndBias(layer, weightInfo, biasesInfo);

    return layer;
}

Layer* CreatePooling2dLayer(Graph& graph, AdditionalLayerParams& params)
{
    Pooling2dDescriptor poolDesc;
    auto name = params["name"];

    poolDesc.m_DataLayout = DataLayout::NHWC;
    FillStride<Pooling2dDescriptor>(poolDesc, params);
    FillPadding<Pooling2dDescriptor>(poolDesc, params);
    FillPoolSize<Pooling2dDescriptor>(poolDesc, params);

    auto poolAlgo       = GetMapStringToPoolingAlgorithm();
    poolDesc.m_PoolType = poolAlgo.find(params["function"])->second;

    return graph.AddLayer<Pooling2dLayer>(poolDesc, name.c_str());
}

void SubstituteLayer(
    Mapping& mapping, Layer* layer, const TensorInfo& inputTensor, const TensorInfo& outputTensor, Graph& newGraph)
{
    Layer* newLayer = nullptr;
    std::string errors;
    std::string replacementLayerTypeName = mapping.m_ReplacementLayers[0].m_LayerTypeName;
    AdditionalLayerParams params         = mapping.m_ReplacementLayers[0].m_LayerParams;
    LayerType type                       = GetLayerType(replacementLayerTypeName);

    ARMNN_LOG(info) << "Replacement layer type name is " << replacementLayerTypeName << "\n";

    if (type == LayerType::Activation)
    {
        auto funcName = mapping.m_ReplacementLayers[0].m_LayerParams.find("function");

        ARMNN_LOG(info) << "The replacement is activation function " << funcName->second << "\n";

        ValidateActivationLayerParameters(mapping.m_ReplacementLayers[0]);
        newLayer =
            CreateActivationLayer(newGraph, funcName->second, mapping.m_ReplacementLayers[0].m_LayerParams["name"]);
    }
    else if ((type == LayerType::Convolution2d) || (type == LayerType::TransposeConvolution2d) ||
             (type == LayerType::DepthwiseConvolution2d))
    {
        unsigned int inputChannels = inputTensor.GetShape()[3];

        ValidateConvolutionLayerParameters(mapping.m_ReplacementLayers[0]);
        newLayer = CreateConvolutionLayer(type, newGraph, inputChannels, params, inputTensor.GetDataType(),
                                          DataType::Signed32);
    }
    else if (type == LayerType::FullyConnected)
    {
        newLayer = CreateFullyConnectedLayer(newGraph, inputTensor, outputTensor, params);
    }
    else if (type == LayerType::Pooling2d)
    {
        ValidatePoolingLayerParameters(mapping.m_ReplacementLayers[0]);
        newLayer = CreatePooling2dLayer(newGraph, params);
    }
    else
    {
        errors = "Invalid Argument: The Replacement layer type \"";
        errors += replacementLayerTypeName;
        errors += "\" is not yet supported\n";

        throw armnn::InvalidArgumentException(errors);
    }

    ARMNN_ASSERT((newLayer != nullptr));

    SubgraphView newSubgraphFromLayer = SubgraphView(layer);

    // SubstituteSubgraph() currently cannot be called on a Graph that contains only one layer.
    // CloneGraph() and ReinterpretGraphToSubgraph() are used to work around this.
    newGraph.SubstituteSubgraph(newSubgraphFromLayer, newLayer);
}

// Check for additional parameters required for certain layer types
// 1. Activation layer needs ((function=someActionfunction))
// 2. Pooling2d layer needs ((function=somePoolingAlgo))
// 3. StandIn layer needs ((name=someName))
// The mappedlayer's additional paramaters should be matching to that
// of the layer to be replaced
void CheckParamValuesForLayer(SimpleLayer layer)
{
    // Excluded is a word defined by us, it is not a standard layer type
    if (!(layer.m_LayerTypeName.compare("Excluded")))
    {
        return;
    }

    LayerType type = GetLayerType(layer.m_LayerTypeName);
    std::string errors;

    if ((type == LayerType::Activation) || (type == LayerType::Pooling2d))
    {
        // We need function=<someFunctionName> for this layer
        if ((layer.m_LayerParams.find("function") == layer.m_LayerParams.end()) ||
            ((layer.m_LayerParams.find("function")->second).empty()))
        {
            errors += "Invalid Argument: ((function=somefunction)) is needed ";
            errors += "for mapping Activation or Pooling2d layers \n";
        }
        else
        {
            auto func = layer.m_LayerParams.find("function")->second;

            if (type == LayerType::Activation)
            {
                auto nameFuncPair = GetMapStringToActivationFunction();

                if (nameFuncPair.find(func) == nameFuncPair.end())
                {
                    errors += "Invalid Value: ";
                    errors += func + "is not a valid Activation Function\n";
                    errors += "valid activation functions are as follows - ";
                    errors += " Sigmoid, TanH, Linear, ReLu, BoundedReLu,";
                    errors += " SoftReLu, LeakyReLu, Abs, Sqrt, Square, Elu, HardSwish \n";
                }
            }
            else
            {
                auto nameFuncPair = GetMapStringToPoolingAlgorithm();

                if (nameFuncPair.find(func) == nameFuncPair.end())
                {
                    errors += "Invalid Value: ";
                    errors += func + "is not a valid pooling algorithm\n";
                    errors += "Average, Max, L2 \n";
                }
            }
        }
    }

    if (!errors.empty())
    {
        throw armnn::InvalidArgumentException(errors);
    }
}

bool IsAdditionalParamsMatching(Layer* layer, SimpleLayer& mappingLayer)
{
    // For Activation layers, the "function" value should match
    if (layer->GetType() == LayerType::Activation)
    {
        ActivationLayer* actLayer = dynamic_cast<ActivationLayer*>(layer);

        auto actFunc = std::string(GetActivationFunctionAsCString(actLayer->GetParameters().m_Function));

        if (actFunc.compare(mappingLayer.m_LayerParams["function"]))
        {
            return false;
        }
    }
    // For Pooling2d layers as well, the "function" value should match
    else if (layer->GetType() == LayerType::Pooling2d)
    {
        Pooling2dLayer* poolLayer = dynamic_cast<Pooling2dLayer*>(layer);
        auto algo                 = std::string(GetPoolingAlgorithmAsCString(poolLayer->GetParameters().m_PoolType));

        if (algo.compare(mappingLayer.m_LayerParams["function"]))
        {
            return false;
        }
    }
    // If the layer name is provided, then it should match
    else if (!mappingLayer.m_LayerParams["name"].empty())
    {
        if (std::string(layer->GetName()).compare(mappingLayer.m_LayerParams["name"]))
        {
            return false;
        }
    }

    return true;
}

LayerType GetLayerType(std::string layerTypeName)
{
    std::map<std::string, LayerType> mapStringToLayerType = GetMapStringToLayerType();

    auto type = mapStringToLayerType.find(layerTypeName);
    if (type == mapStringToLayerType.end())
    {
        std::string errors = "layername \"" + layerTypeName + "\" is not valid";
        if (!errors.empty())
        {
            throw armnn::InvalidArgumentException(errors);
        }
    }

    return type->second;
}

// Check if the layerTypeName can be mapped to one of the armnn::LayerType
bool IsLayerType(std::string layerTypeName)
{
    std::string errors;

    // loop through layer types, call GetLayerTypeAsCString() and create a map, then get the type from the string from the map
    std::map<std::string, LayerType> mapStringToLayerType = GetMapStringToLayerType();

    auto type = mapStringToLayerType.find(layerTypeName);

    return (type != mapStringToLayerType.end());
}

// Check if the layer additional parameters are of known types.
// Note:- We do not check the values at this point. The values
// will be validated when they are retrieved during layer creation.
void ValidateAdditionalParameters(SimpleLayer layer)
{
    auto layerParams = layer.m_LayerParams;
    std::string errors;

    for (auto param : layerParams)
    {
        auto name = param.first;

        if (name.compare("function") && name.compare("stride") && name.compare("kernel") && name.compare("name") &&
            name.compare("padding") && name.compare("dilation"))
        {
            errors = "Invalid Argument: Layer Parameter \"";
            errors += name;
            errors += "\"is unknown. \n";
            errors += "Known parameters are \"function\", \"stride\", \"kernel\", ";
            errors += " \"name\", \"padding\", \"dilation\" ";

            throw armnn::InvalidArgumentException(errors);
        }
    }

    CheckParamValuesForLayer(layer);
}

void ValidateMappingParameters(Mapping mapping)
{
    std::string errors;
    std::string pattern     = mapping.m_PatternLayers[0].m_LayerTypeName;
    std::string replacement = mapping.m_ReplacementLayers[0].m_LayerTypeName;

    if (mapping.m_PatternLayers.size() != 1)
    {
        errors = "Invalid Argument: N:1 mapping is not supported\n";
    }
    else if (!IsLayerType(pattern))
    {
        errors = "Invalid Argument: Pattern Layer Type is invalid\n";
        errors += pattern;
        errors += "\n";
    }
    else if ((!IsLayerType(replacement)) && (replacement.compare("Excluded")))
    {
        errors = "Invalid Argument: Replacement Layer Type is invalid\n";
        errors += replacement;
        errors += "\n";
    }
    else
    {
        ValidateAdditionalParameters(mapping.m_PatternLayers[0]);
        ValidateAdditionalParameters(mapping.m_ReplacementLayers[0]);
    }

    if (!errors.empty())
    {
        throw armnn::InvalidArgumentException(errors);
    }
}

void ApplyMappings(std::vector<Mapping> mappings, Graph& newGraph)
{
    // substitute the layers as per the mapping
    std::list<Layer*> newGraphLayers(newGraph.begin(), newGraph.end());

    for (Mapping mapping : mappings)
    {
        ValidateMappingParameters(mapping);

        // Create copy of newGraphLayers and remove substituted layers from original list.
        const std::list<Layer*> newGraphLayersCopy(newGraphLayers.begin(), newGraphLayers.end());
        for (Layer* layer : newGraphLayersCopy)
        {
            if (layer->GetType() == GetLayerType(mapping.m_PatternLayers[0].m_LayerTypeName))
            {
                if (!mapping.m_ReplacementLayers[0].m_LayerTypeName.compare("Excluded"))
                {
                    continue;
                }

                auto inputTensorsCnt  = layer->GetNumOutputSlots();
                auto outputTensorsCnt = layer->GetNumOutputSlots();

                // The original layer has single tensor input / single tensor output
                if ((inputTensorsCnt == 1) && (outputTensorsCnt == 1))
                {
                    TensorInfo inputTensor = layer->GetInputSlots()[0].GetConnectedOutputSlot()->GetTensorInfo();
                    TensorInfo outputTensor =
                        layer->GetOutputSlots()[0].GetConnection(0)->GetConnectedOutputSlot()->GetTensorInfo();

                    // In the mapping, the count of replacement layers and pattern layers has to be 1.
                    // This is because we are interested in replacing a single layer at a time.
                    // This will change when we implement N:1 mapping scheme.
                    // Also, the mapping's pattern and replacement layer's input/output tensor
                    // shape has to be the same.
                    if ((mapping.m_ReplacementLayers.size() == 1) && (mapping.m_PatternLayers.size() == 1) &&
                        (mapping.m_PatternLayers[0].m_Inputs == mapping.m_ReplacementLayers[0].m_Inputs) &&
                        (mapping.m_PatternLayers[0].m_Outputs == mapping.m_ReplacementLayers[0].m_Outputs))
                    {
                        // The original layer has input tensor shape same as output tensor shape
                        if (inputTensor.GetShape() == outputTensor.GetShape())
                        {
                            ARMNN_LOG(info) << "Input and Output tensors are of same shape\n";
                        }
                        else
                        {
                            ARMNN_LOG(info) << "Input and Output tensors are of different shape\n";
                        }

                        // For some layer types like Activation, Pooling2d we need to match
                        // not only the layer types but also the function (should be present in
                        // m_LayerParams).
                        // Also if name is provided as part of m_LayerParams, it needs to be matched
                        // as well.
                        bool ret = IsAdditionalParamsMatching(layer, mapping.m_PatternLayers[0]);

                        if (!ret)
                        {
                            continue;
                        }

                        SubstituteLayer(mapping, layer, inputTensor, outputTensor, newGraph);
                        newGraphLayers.remove(layer);
                    }
                }
            }
        }
    }
}

namespace
{
// Replaces the pattern Constant-Multiplication with an optimized DepthwiseConvolution2d operation.
// Original pattern:
// Input    ->
//              Multiplication -> Output
// Constant ->
// Expected modified pattern:
// Input -> DepthwiseConvolution2d -> Output
//
bool ReplaceConstantMultiplicationWithDepthwise(Graph& graph, Layer* layer)
{
    if (layer->GetType() == LayerType::Multiplication)
    {
        InputSlot* patternSubgraphInput = &layer->GetInputSlot(0);

        Layer* inputLayer    = &patternSubgraphInput->GetConnectedOutputSlot()->GetOwningLayer();
        Layer* constantLayer = &layer->GetInputSlots()[1].GetConnectedOutputSlot()->GetOwningLayer();

        if (constantLayer->GetType() != LayerType::Constant)
        {
            patternSubgraphInput = &layer->GetInputSlot(1);
            std::swap(inputLayer, constantLayer);
        }

        if (constantLayer->GetType() == LayerType::Constant)
        {
            const TensorInfo& inputInfo = inputLayer->GetOutputSlot().GetTensorInfo();
            const TensorInfo& constInfo = constantLayer->GetOutputSlot().GetTensorInfo();

            // Add a Depthwise only where the constant input is a scalar that takes the form { 1, 1, 1, C }.
            // The scalar is used as weights for the convolution.
            if (constInfo.GetShape() == TensorShape({ 1, 1, 1, inputInfo.GetShape()[3] }))
            {
                Graph replacementGraph;

                DepthwiseConvolution2dDescriptor desc;
                desc.m_DataLayout = DataLayout::NHWC;

                const auto depthwiseLayer =
                    replacementGraph.AddLayer<DepthwiseConvolution2dLayer>(desc, "DepthwiseConv2d");

                TensorInfo weightInfo = constInfo;
                weightInfo.SetShape({ 1, constInfo.GetShape()[3], 1, 1 });

                const void* weightData = PolymorphicPointerDowncast<const ConstantLayer>(constantLayer)
                                             ->m_LayerOutput->GetConstTensor<void>();

                const ConstTensor weights(weightInfo, weightData);

                depthwiseLayer->m_Weight = std::make_unique<ScopedTensorHandle>(weights);

                SubgraphView patternSubgraph({ patternSubgraphInput }, { &layer->GetOutputSlot() },
                                             { layer, constantLayer });

                graph.SubstituteSubgraph(patternSubgraph, SubgraphView{ depthwiseLayer });

                return true;
            }
        }
    }
    return false;
}
}    // namespace

void ReplaceUnsupportedLayers(Graph& graph)
{
    using ReplacementFunc                    = bool (*)(Graph&, Layer*);
    const ReplacementFunc replacementFuncs[] = {
        &ReplaceConstantMultiplicationWithDepthwise,
    };

    bool madeChange;
    do
    {
        madeChange = false;
        for (Layer* layer : graph)
        {
            for (const ReplacementFunc f : replacementFuncs)
            {
                madeChange = f(graph, layer);
                if (madeChange)
                {
                    goto nextIteration;
                }
            }
        }
    nextIteration:;
    } while (madeChange);
}
}    // namespace ethosnbackend

void CreatePreCompiledLayerInGraph(OptimizationViews& optimizationViews,
                                   const SubgraphView& subgraph,
                                   const EthosNMappings& mappings,
                                   const ModelOptions& modelOptions)
{
    SubgraphView subgraphToCompile = subgraph;
    g_EthosNConfig                 = GetEthosNConfig();

    // Graph is needed here to keep ownership of the layers
    Graph newGraph = ethosnbackend::CloneGraph(subgraph);

    // If we're in Performance Estimator mode, we might want to replace some of the layers we do not support with
    // layers we do, for performance estimation purposes
    if (g_EthosNConfig.m_PerfOnly && !mappings.empty())
    {
        // apply the mapping to the subgraph to replace nodes in EstimatorOnly mode
        ethosnbackend::ApplyMappings(mappings, newGraph);
    }

    // Constant configuration to always replace unsupported layer patterns
    ethosnbackend::ReplaceUnsupportedLayers(newGraph);

    subgraphToCompile = ethosnbackend::ReinterpretGraphToSubgraph(newGraph);

    std::vector<CompiledBlobPtr> compiledNetworks;

    try
    {
        // Attempt to convert and compile the sub-graph
        compiledNetworks = EthosNSubgraphViewConverter(subgraphToCompile, modelOptions).CompileNetwork();
    }
    catch (std::exception&)
    {
        // Failed to compile the network
        // compiledNetworks will be empty and the condition below will apply
    }

    if (compiledNetworks.empty())
    {
        // The compiler returned an empty list of compiled objects
        optimizationViews.AddFailedSubgraph(SubgraphView(subgraph));
        return;
    }

    // Only the case of a single compiled network is currently supported
    ARMNN_ASSERT(compiledNetworks.size() == 1);

    // Wrap the precompiled layer into a graph
    PreCompiledLayer& preCompiledLayer = *optimizationViews.GetGraph().AddLayer<PreCompiledLayer>(
        PreCompiledDescriptor(subgraph.GetNumInputSlots(), subgraph.GetNumOutputSlots()), "pre-compiled");

    // Copy the output tensor infos from sub-graph
    for (unsigned int i = 0; i < subgraph.GetNumOutputSlots(); i++)
    {
        preCompiledLayer.GetOutputSlot(i).SetTensorInfo(subgraph.GetOutputSlot(i)->GetTensorInfo());
    }

    // Assign the pre-compiled object to layer
    // Pass only the first compiled network for the moment, as Arm NN does not handle
    // multiple pre-compiled objects in a single pre-compiled layer just yet
    preCompiledLayer.SetPreCompiledObject(std::move(compiledNetworks.at(0)));

    // Set the backend-id for the pre-compiled layer
    preCompiledLayer.SetBackendId(EthosNBackendId());

    optimizationViews.AddSubstitution({ std::move(subgraph), SubgraphView(&preCompiledLayer) });
}

const BackendId& EthosNBackend::GetIdStatic()
{
    static const BackendId s_Id{ EthosNBackendId() };
    return s_Id;
}

IBackendInternal::IWorkloadFactoryPtr
    EthosNBackend::CreateWorkloadFactory(const IBackendInternal::IMemoryManagerSharedPtr&) const
{
    return std::make_unique<EthosNWorkloadFactory>();
}

IBackendInternal::IBackendContextPtr EthosNBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::IBackendProfilingContextPtr
    EthosNBackend::CreateBackendProfilingContext(const IRuntime::CreationOptions& options,
                                                 IBackendProfilingPtr& backendProfiling)
{
    if (!options.m_ProfilingOptions.m_EnableProfiling)
    {
        return nullptr;
    }
    std::shared_ptr<profiling::EthosNBackendProfilingContext> context =
        std::make_shared<profiling::EthosNBackendProfilingContext>(backendProfiling);
    EthosNBackendProfilingService::Instance().SetProfilingContextPtr(context);
    return context;
}

IBackendInternal::IMemoryManagerUniquePtr EthosNBackend::CreateMemoryManager() const
{
    return IMemoryManagerUniquePtr{};
}

IBackendInternal::ILayerSupportSharedPtr EthosNBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{ new EthosNLayerSupport };
    return layerSupport;
}

OptimizationViews EthosNBackend::OptimizeSubgraphView(const SubgraphView& subgraph) const
{
    return EthosNBackend::OptimizeSubgraphView(subgraph, {});
}

OptimizationViews EthosNBackend::OptimizeSubgraphView(const SubgraphView& subgraph,
                                                      const ModelOptions& modelOptions) const
{
    if (!ethosnbackend::VerifyLibraries())
    {
        throw RuntimeException("Driver or support library version is not supported by the backend");
    }
    OptimizationViews optimizationViews;
    g_EthosNConfig   = GetEthosNConfig();
    g_EthosNMappings = GetMappings(g_EthosNConfig.m_PerfMappingFile);

    // Create a pre-compiled layer
    armnn::CreatePreCompiledLayerInGraph(optimizationViews, subgraph, g_EthosNMappings, modelOptions);

    return optimizationViews;
}

}    // namespace armnn
