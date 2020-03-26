//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNBackend.hpp"

#include "EthosNBackendId.hpp"
#include "EthosNBackendProfilingContext.hpp"
#include "EthosNLayerSupport.hpp"
#include "EthosNMapping.hpp"
#include "EthosNSubgraphViewConverter.hpp"
#include "EthosNWorkloadFactory.hpp"

#include <Optimizer.hpp>
#include <armnn/BackendRegistry.hpp>
#include <backendsCommon/IBackendContext.hpp>
#include <backendsCommon/IMemoryManager.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>
#include <boost/cast.hpp>

namespace armnn
{

EthosNConfig g_EthosNConfig;
EthosNMappings g_EthosNMappings;

namespace ethosnbackend
{

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

        Layer* const clonedLayer = originalToClonedLayerMap[&originalSubgraphLayer];

        // add it as an input layer into the new graph
        InputLayer* const newInputLayer = newGraph.AddLayer<InputLayer>(slotCount, "input");
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

        BOOST_ASSERT(i < originalSubgraphLayer.GetNumOutputSlots());

        OutputSlot* outputSlotOfLayer     = &clonedLayer->GetOutputSlot(i);
        OutputLayer* const newOutputLayer = newGraph.AddLayer<OutputLayer>(slotCount, "output");
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
                inputLayersNewGr.push_back(boost::polymorphic_downcast<InputLayer*>(layer));
                break;
            case LayerType::Output:
                outputLayersNewGr.push_back(boost::polymorphic_downcast<OutputLayer*>(layer));
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

    for (ActivationFunction type = ActivationFunction::Sigmoid; type <= ActivationFunction::Square;
         type                    = NextEnumValue(type))
    {
        mapStringToType.emplace(GetActivationFunctionAsCString(type), type);
    }

    return mapStringToType;
}

template <class ConvLayerClass, class ConvLayerDescriptor>
Layer* CreateConvolutionLayer(Graph& graph,
                              unsigned int inputChannels,
                              unsigned int kernelWidth,
                              unsigned int kernelHeight,
                              unsigned int strideX,
                              unsigned int strideY,
                              DataType weightDataType,
                              DataType biasDataType)
{
    ConvLayerDescriptor convolutionDescriptor;
    const TensorInfo weightInfo =
        TensorInfo({ inputChannels, kernelHeight, kernelWidth, inputChannels }, weightDataType);
    const TensorInfo bias               = TensorInfo({ 1, 1, 1, inputChannels }, biasDataType, 0.9f, 0);
    convolutionDescriptor.m_StrideX     = strideX;
    convolutionDescriptor.m_StrideY     = strideY;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    ConvLayerClass* convLayer = graph.AddLayer<ConvLayerClass>(convolutionDescriptor, "Convolution2d");
    SetWeightAndBias(convLayer, weightInfo, bias);

    return convLayer;
}

Layer* CreateConvolutionLayer(std::string layerName,
                              Graph& graph,
                              unsigned int inputChannels,
                              unsigned int kernelWidth,
                              unsigned int kernelHeight,
                              unsigned int strideX,
                              unsigned int strideY,
                              DataType weightDataType,
                              DataType biasDataType)
{
    Layer* newLayer = nullptr;

    if (layerName == "Convolution2d")
    {
        std::cout << "The replacement is Convolution2d \n";

        newLayer = CreateConvolutionLayer<Convolution2dLayer, Convolution2dDescriptor>(
            graph, inputChannels, kernelWidth, kernelHeight, strideX, strideY, weightDataType, biasDataType);
    }
    else
    {
        std::cout << "The replacement is TransposeConvolution2d \n";

        newLayer = CreateConvolutionLayer<TransposeConvolution2dLayer, TransposeConvolution2dDescriptor>(
            graph, inputChannels, kernelWidth, kernelHeight, strideX, strideY, weightDataType, biasDataType);
    }

    return newLayer;
}

Layer* CreateActivationLayer(Graph& graph, std::string activationFunc)
{
    std::map<std::string, ActivationFunction> mapStringToActivationFunction = GetMapStringToActivationFunction();

    auto func = mapStringToActivationFunction.find(activationFunc)->second;
    ActivationDescriptor desc;
    desc.m_Function = func;

    auto* layer = graph.AddLayer<ActivationLayer>(desc, activationFunc.c_str());

    return layer;
}

bool ValidateActivationLayerParameters(SimpleLayer& layer)
{
    std::map<std::string, ActivationFunction> mapStringToActivationFunction = GetMapStringToActivationFunction();
    auto extraArg                                                           = layer.m_ExtraArgs.find("function");

    if (extraArg == layer.m_ExtraArgs.end())
    {
        // if no activation function is provided
        return false;
    }

    auto func = mapStringToActivationFunction.find(extraArg->second);

    if (func == mapStringToActivationFunction.end())
    {
        // if activation function is invalid
        return false;
    }

    // Currently we support only Sigmoid and ReLu Activation functions
    if ((func->second == ActivationFunction::Sigmoid) || (func->second == ActivationFunction::ReLu))
    {
        return true;
    }
    else
    {
        return false;
    }
}

void SubstituteLayer(Mapping& mapping, Layer* layer, Graph& newGraph)
{
    Layer* newLayer                  = NULL;
    std::string replacementLayerName = mapping.m_ReplacementLayers[0].m_Name;

    if (replacementLayerName == "Activation")
    {
        auto extraArg = mapping.m_ReplacementLayers[0].m_ExtraArgs.find("function");

        std::cout << "The replacement is activation function " << extraArg->second << "\n";

        if (!ValidateActivationLayerParameters(mapping.m_ReplacementLayers[0]))
        {
            return;
        }

        newLayer = CreateActivationLayer(newGraph, extraArg->second);
    }
    else if ((replacementLayerName == "Convolution2d") || (replacementLayerName == "TransposeConvolution2d"))
    {

        Layer& previousLayer    = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
        const TensorInfo& tinfo = previousLayer.GetOutputSlot(0).GetTensorInfo();
        // Need to validate that input tensor is of 4 dimensions
        unsigned int inputChannels = tinfo.GetShape()[3];
        unsigned int strideX = 1, strideY = 1;

        if (replacementLayerName == "TransposeConvolution2d")
        {
            strideX = 2;
            strideY = 2;
        }

        newLayer = CreateConvolutionLayer(replacementLayerName, newGraph, inputChannels, 1, 1, strideX, strideY,
                                          tinfo.GetDataType(), DataType::Signed32);
    }

    SubgraphView newSubgraphFromLayer = SubgraphView(layer);

    // SubstituteSubgraph() currently cannot be called on a Graph that contains only one layer.
    // CloneGraph() and ReinterpretGraphToSubgraph() are used to work around this.
    newGraph.SubstituteSubgraph(newSubgraphFromLayer, newLayer);

    return;
}

void ApplyMappings(std::vector<Mapping> mappings, Graph& newGraph)
{
    // substitute the layers as per the mapping
    const std::list<Layer*> newGraphLayers(newGraph.begin(), newGraph.end());

    // loop through layer types, call GetLayerTypeAsCString() and create a map, then get the type from the string from the map
    std::map<std::string, LayerType> mapStringToLayerType                   = GetMapStringToLayerType();
    std::map<std::string, ActivationFunction> mapStringToActivationFunction = GetMapStringToActivationFunction();
    for (Mapping mapping : mappings)
    {
        // Hardcoded to one pattern layer until we implement N:1
        if (mapping.m_PatternLayers.size() != 1)
        {
            continue;
        }

        auto type = mapStringToLayerType.find(mapping.m_PatternLayers[0].m_Name);

        if (type == mapStringToLayerType.end())
        {
            // skip this mapping if the mapping layer could not be found
            continue;
        }

        auto replacementType = mapStringToLayerType.find(mapping.m_ReplacementLayers[0].m_Name);

        if (replacementType == mapStringToLayerType.end())
        {
            // skip this mapping if the replacement layer could not be found
            continue;
        }

        for (Layer* layer : newGraphLayers)
        {
            if (layer->GetType() == type->second)
            {
                auto inputTensorsCnt  = layer->GetNumOutputSlots();
                auto outputTensorsCnt = layer->GetNumOutputSlots();

                // The original layer has single tensor input / single tensor output
                if ((inputTensorsCnt == 1) && (outputTensorsCnt == 1))
                {
                    TensorInfo inputTensor = layer->GetInputSlots()[0].GetConnectedOutputSlot()->GetTensorInfo();
                    TensorInfo outputTensor =
                        layer->GetOutputSlots()[0].GetConnection(0)->GetConnectedOutputSlot()->GetTensorInfo();

                    // The original layer has input tensor shape same as output tensor shape
                    if (inputTensor.GetShape() == outputTensor.GetShape())
                    {

                        // In the mapping, the count of replacement layers and pattern layers has to be 1.
                        // This is because we are interested in replacing a single layer at a time.
                        // This will change when we implement N:1 mapping scheme.
                        // Also, the mapping's pattern and replacement layer's input/output tensor
                        // shape has to be the same.
                        if ((mapping.m_ReplacementLayers.size() == 1) && (mapping.m_PatternLayers.size() == 1) &&
                            (mapping.m_PatternLayers[0].m_Inputs == mapping.m_ReplacementLayers[0].m_Inputs) &&
                            (mapping.m_PatternLayers[0].m_Outputs == mapping.m_ReplacementLayers[0].m_Outputs))
                        {
                            SubstituteLayer(mapping, layer, newGraph);
                        }
                    }
                }
            }
        }
    }
}

}    // namespace ethosnbackend

void CreatePreCompiledLayerInGraph(OptimizationViews& optimizationViews,
                                   const SubgraphView& subgraph,
                                   const EthosNMappings& mappings)
{
    SubgraphView subgraphToCompile = subgraph;
    g_EthosNConfig                 = GetEthosNConfig();

    // Graph is needed here to keep ownership of the layers
    Graph newGraph = Graph();

    // if we're in Performance Estimator mode, we might want to replace some of the layers we do not support with
    // layers we do, for performance estimation purposes
    if (g_EthosNConfig.m_PerfOnly && !mappings.empty())
    {
        // apply the mapping to the subgraph to replace nodes in EstimatorOnly mode
        newGraph = ethosnbackend::CloneGraph(subgraph);

        ethosnbackend::ApplyMappings(mappings, newGraph);
        subgraphToCompile = ethosnbackend::ReinterpretGraphToSubgraph(newGraph);
    }

    std::vector<CompiledBlobPtr> compiledNetworks;

    try
    {
        // Attempt to convert and compile the sub-graph
        compiledNetworks = EthosNSubgraphViewConverter(subgraphToCompile).CompileNetwork();
    }
    catch (std::exception&)
    {
        // Failed to compile the network
        // compiledNetworks will be empty and the condition below will apply
    }

    if (compiledNetworks.empty())
    {
        // The compiler returned an empty list of compiled objects
        optimizationViews.AddFailedSubgraph(std::move(subgraphToCompile));
        return;
    }

    // Only the case of a single compiled network is currently supported
    BOOST_ASSERT(compiledNetworks.size() == 1);

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
    if (options.m_ProfilingOptions.m_EnableProfiling)
    {
        std::shared_ptr<profiling::EthosNBackendProfilingContext> context =
            std::make_shared<profiling::EthosNBackendProfilingContext>(backendProfiling);
        EthosNBackendProfilingService::Instance().SetProfilingContextPtr(context);
        return context;
    }
    else
    {
        return nullptr;
    }
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
    OptimizationViews optimizationViews;
    g_EthosNConfig   = GetEthosNConfig();
    g_EthosNMappings = GetMappings(g_EthosNConfig.m_PerfMappingFile);

    // Create a pre-compiled layer
    armnn::CreatePreCompiledLayerInGraph(optimizationViews, subgraph, g_EthosNMappings);

    return optimizationViews;
}

}    // namespace armnn
