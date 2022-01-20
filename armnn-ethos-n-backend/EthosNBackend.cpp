//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNBackend.hpp"

#include "EthosNBackendId.hpp"
#include "EthosNBackendProfilingContext.hpp"
#include "EthosNLayerSupport.hpp"
#include "EthosNReplaceUnsupported.hpp"
#include "EthosNSubgraphViewConverter.hpp"
#include "EthosNWorkloadFactory.hpp"

#include <armnn/BackendRegistry.hpp>
#include <armnn/Logging.hpp>
#include <armnn/backends/IBackendContext.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/utility/Assert.hpp>
#include <ethosn_driver_library/Device.hpp>
#include <ethosn_driver_library/Network.hpp>
#include <ethosn_support_library/Support.hpp>

namespace armnn
{

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
    ARMNN_NO_DEPRECATE_WARN_BEGIN
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
    ARMNN_NO_DEPRECATE_WARN_END

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

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    return SubgraphView(std::move(inSlotsPointers), std::move(outSlotsPointers), std::move(subgrLayers));
    ARMNN_NO_DEPRECATE_WARN_END
}

}    // namespace ethosnbackend

void CreatePreCompiledLayerInGraph(OptimizationViews& optimizationViews,
                                   const SubgraphView& subgraph,
                                   const EthosNConfig& config,
                                   const std::vector<char>& capabilities,
                                   const ModelOptions& modelOptions)
{
    SubgraphView subgraphToCompile = subgraph;

    // Graph is needed here to keep ownership of the layers
    Graph newGraph = ethosnbackend::CloneGraph(subgraph);

    // Constant configuration to always replace unsupported layer patterns
    ethosnbackend::ReplaceUnsupportedLayers(newGraph, config, capabilities);

    subgraphToCompile = ethosnbackend::ReinterpretGraphToSubgraph(newGraph);

    std::vector<CompiledBlobPtr> compiledNetworks;

    try
    {
        // Attempt to convert and compile the sub-graph
        compiledNetworks =
            EthosNSubgraphViewConverter(subgraphToCompile, modelOptions, config, capabilities).CompileNetwork();
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
    IConnectableLayer* preCompiledLayer = optimizationViews.GetINetwork()->AddPrecompiledLayer(
        PreCompiledDescriptor(subgraph.GetNumInputSlots(), subgraph.GetNumOutputSlots()),
        std::move(compiledNetworks[0]), armnn::Optional<BackendId>(EthosNBackendId()), "pre-compiled");

    // Copy the output tensor infos from sub-graph
    for (unsigned int i = 0; i < subgraph.GetNumOutputSlots(); i++)
    {
        preCompiledLayer->GetOutputSlot(i).SetTensorInfo(subgraph.GetIOutputSlot(i)->GetTensorInfo());
    }

    optimizationViews.AddSubstitution({ std::move(subgraph), SubgraphView(preCompiledLayer) });
}

ARMNN_DLLEXPORT armnn::EthosNConfig EthosNBackend::ms_Config;
ARMNN_DLLEXPORT std::vector<char> EthosNBackend::ms_Capabilities;

EthosNBackend::EthosNBackend()
{
    // Although this EthosNBackend object is the 'main' object representing our backend, it is actually an ephemeral
    // object which Arm NN instantiates and destroys many times during various operations. Therefore it is not wise
    // to load config files and query the HW for capabilities here as it would be bad for performance and more
    // importantly could lead to different parts of the backend disagreeing about configuration settings if the
    // files on disk changed while running Arm NN. There is currently no object with an appropriate lifetime to handle
    // this, so we have to handle this in a less ideal manner - we only load these things *once*, on first instantiation
    // of this backend object. All future instantiations will use the same cached values.

    // Initialize EthosNCachingService shared pointer only once, this is used to access
    // the caching functions and cached network data held temporarily in memory.
    auto cachingService = EthosNCachingService::GetInstance().GetEthosNCachingPtr();
    if (cachingService == nullptr)
    {
        EthosNCaching cachingObject            = EthosNCaching();
        std::shared_ptr<EthosNCaching> context = std::make_shared<EthosNCaching>(cachingObject);
        EthosNCachingService::GetInstance().SetEthosNCachingPtr(context);
    }

    if (ms_Capabilities.empty())
    {
        // First-time initialization
        ms_Config       = ReadEthosNConfig();
        ms_Capabilities = ms_Config.QueryCapabilities();
    }

    // Copy the cached data into this object, for further use (passing to sub-objects etc.)
    m_Config       = ms_Config;
    m_Capabilities = ms_Capabilities;
}

const BackendId& EthosNBackend::GetIdStatic()
{
    static const BackendId s_Id{ EthosNBackendId() };
    return s_Id;
}

IBackendInternal::IWorkloadFactoryPtr
    EthosNBackend::CreateWorkloadFactory(const IBackendInternal::IMemoryManagerSharedPtr&) const
{
    return std::make_unique<EthosNWorkloadFactory>(m_Config);
}

IBackendInternal::IWorkloadFactoryPtr
    EthosNBackend::CreateWorkloadFactory(const IBackendInternal::IMemoryManagerSharedPtr&,
                                         const ModelOptions& modelOptions) const
{
    // Try to save cached subgraphs, if saving options aren't specified nothing will happen.
    // This occurs after optimization so it will be ready to save if required.
    auto caching = EthosNCachingService::GetInstance().GetEthosNCachingPtr();
    caching->Save();

    for (const auto& optionsGroup : modelOptions)
    {
        if (optionsGroup.GetBackendId() == EthosNBackend::GetIdStatic())
        {
            for (size_t i = 0; i < optionsGroup.GetOptionCount(); ++i)
            {
                const BackendOptions::BackendOption& option = optionsGroup.GetOption(i);

                if (option.GetName() == "Device")
                {
                    if (option.GetValue().IsString())
                    {
                        const std::string deviceVal = option.GetValue().AsString();

                        return std::make_unique<EthosNWorkloadFactory>(m_Config, deviceVal);
                    }
                    else
                    {
                        throw armnn::InvalidArgumentException("Invalid value type for Device - must be string.");
                    }
                }
            }
        }
    }

    return std::make_unique<EthosNWorkloadFactory>(m_Config);
}

BackendCapabilities EthosNBackend::GetCapabilities() const
{
    BackendCapabilities ethosnCap(EthosNBackend::GetIdStatic());
    ethosnCap.AddOption(
        BackendOptions::BackendOption("DeviceNamePrefix", ethosn::driver_library::GetDeviceNamePrefix()));
    ethosnCap.AddOption(BackendOptions::BackendOption(
        "DeviceBaseId", static_cast<uint32_t>(ethosn::driver_library::GetDeviceBaseId())));
    ethosnCap.AddOption(BackendOptions::BackendOption(
        "NumberOfDevices", static_cast<uint32_t>(ethosn::driver_library::GetNumberOfDevices())));

    return ethosnCap;
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
    return std::make_shared<EthosNLayerSupport>(m_Config, m_Capabilities);
}

IBackendInternal::ILayerSupportSharedPtr EthosNBackend::GetLayerSupport(const ModelOptions& modelOptions) const
{
    for (const auto& optionsGroup : modelOptions)
    {
        if (optionsGroup.GetBackendId() == EthosNBackend::GetIdStatic())
        {
            for (size_t i = 0; i < optionsGroup.GetOptionCount(); ++i)
            {
                const BackendOptions::BackendOption& option = optionsGroup.GetOption(i);

                if (option.GetName() == "Device")
                {
                    if (!option.GetValue().IsString())
                    {
                        throw armnn::InvalidArgumentException("Invalid value type for Device - must be string.");
                    }
                }
            }
        }
    }
    return std::make_shared<EthosNLayerSupport>(m_Config, m_Capabilities);
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

    // As OptimizeSubgraphView can be called multiple times we only want to set this once.
    // Set the caching options and try to load cached networks into memory only if loading was specified by the user.
    // SetEthosNCachingOptions will catch any errors in the user options.
    auto caching = EthosNCachingService::GetInstance().GetEthosNCachingPtr();
    if (caching->GetIsLoaded() == false)
    {
        caching->SetEthosNCachingOptions(modelOptions);
        caching->Load();
    }

    // Create a pre-compiled layer
    OptimizationViews optimizationViews;
    armnn::CreatePreCompiledLayerInGraph(optimizationViews, subgraph, m_Config, m_Capabilities, modelOptions);

    return optimizationViews;
}

armnn::EthosNBackendProfilingService& EthosNBackendProfilingService::Instance()
{
    static EthosNBackendProfilingService instance;
    return instance;
}

}    // namespace armnn
