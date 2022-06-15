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
#include "EthosNTensorHandleFactory.hpp"
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

BackendRegistry::StaticRegistryInitializer g_RegisterHelper{ BackendRegistryInstance(), EthosNBackend::GetIdStatic(),
                                                             []() {
                                                                 return IBackendInternalUniquePtr(new EthosNBackend);
                                                             } };

// The returned subgraph contains Input and Output layers that the
// original subgraph does not have.
SubgraphView CloneSubgraph(const SubgraphView& originalSubgraph)
{
    SubgraphView newSubgraph                = originalSubgraph.GetWorkingCopy();
    INetworkPtr newNetwork                  = INetwork::Create();
    SubgraphView::IConnectableLayers layers = newSubgraph.GetIConnectableLayers();
    SubgraphView::IInputSlots inputs        = newSubgraph.GetIInputSlots();
    SubgraphView::IOutputSlots outputs      = newSubgraph.GetIOutputSlots();
    LayerBindingId slotCount                = 0;

    // Process SubgraphView inputs
    for (uint32_t i = 0; i < originalSubgraph.GetNumInputSlots(); ++i)
    {
        // Get info about the original input layer and its output slot
        const IOutputSlot* originalOutputSlot = originalSubgraph.GetIInputSlot(i)->GetConnection();
        const std::string& layerName          = originalOutputSlot->GetOwningIConnectableLayer().GetName();
        const TensorInfo tensorInfo           = originalOutputSlot->GetTensorInfo();

        // Create an input layer and connect its output slot
        IConnectableLayer* newInputLayer = newNetwork->AddInputLayer(slotCount, layerName.c_str());
        IInputSlot* newInputSlot         = newSubgraph.GetIInputSlot(i);
        newInputLayer->GetOutputSlot(0).Connect(*newInputSlot);
        newInputLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

        layers.emplace_front(newInputLayer);
        ++slotCount;
    }

    // Process SubgraphView outputs
    for (uint32_t i = 0; i < originalSubgraph.GetNumOutputSlots(); ++i)
    {
        // Get info about the original output layer and its input slot
        const IOutputSlot* originalOutputSlot = originalSubgraph.GetIOutputSlot(i);
        const std::string& layerName = originalOutputSlot->GetConnection(0)->GetOwningIConnectableLayer().GetName();
        const TensorInfo& tensorInfo = originalOutputSlot->GetTensorInfo();

        // Create an output layer and connect its input slot
        IConnectableLayer* newOutputLayer = newNetwork->AddOutputLayer(slotCount, layerName.c_str());
        IOutputSlot* newOutputSlot        = newSubgraph.GetIOutputSlot(i);
        newOutputSlot->Connect(newOutputLayer->GetInputSlot(0));
        newOutputSlot->SetTensorInfo(tensorInfo);

        layers.emplace_back(newOutputLayer);
        ++slotCount;
    }

    return SubgraphView(std::move(layers), std::move(inputs), std::move(outputs)).GetWorkingCopy();
}

SubgraphView RemoveInputOutputLayers(SubgraphView& subgraph)
{
    SubgraphView::IConnectableLayers layers;
    subgraph.ForEachIConnectableLayer([&](IConnectableLayer* layer) {
        if (layer->GetType() != armnn::LayerType::Input && layer->GetType() != armnn::LayerType::Output)
        {
            layers.push_back(layer);
        }
    });
    SubgraphView::IInputSlots inputs   = subgraph.GetIInputSlots();
    SubgraphView::IOutputSlots outputs = subgraph.GetIOutputSlots();
    return SubgraphView(std::move(layers), std::move(inputs), std::move(outputs));
}

std::string GetDeviceOptionVal(const ModelOptions& modelOptions)
{
    for (const auto& optionsGroup : modelOptions)
    {
        if (optionsGroup.GetBackendId() != EthosNBackend::GetIdStatic())
        {
            continue;
        }

        for (size_t i = 0; i < optionsGroup.GetOptionCount(); ++i)
        {
            const BackendOptions::BackendOption& option = optionsGroup.GetOption(i);

            if (option.GetName() != "Device")
            {
                continue;
            }

            if (!option.GetValue().IsString())
            {
                throw armnn::InvalidArgumentException("Invalid value type for Device - must be string.");
            }

            return option.GetValue().AsString();
        }
    }

    return "";
}

}    // namespace ethosnbackend

void CreatePreCompiledLayerInGraph(OptimizationViews& optimizationViews,
                                   const SubgraphView& subgraph,
                                   const EthosNConfig& config,
                                   const std::vector<char>& capabilities,
                                   const ModelOptions& modelOptions)
{
    SubgraphView newSubgraph = ethosnbackend::CloneSubgraph(subgraph);

    // Constant configuration to always replace unsupported layer patterns
    ethosnbackend::ReplaceUnsupportedLayers(newSubgraph, config, capabilities);

    SubgraphView subgraphToCompile = ethosnbackend::RemoveInputOutputLayers(newSubgraph);

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
        ms_Config = ReadEthosNConfig();

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

    const std::string deviceId = ethosnbackend::GetDeviceOptionVal(modelOptions);
    if (!deviceId.empty())
    {
        return std::make_unique<EthosNWorkloadFactory>(m_Config, deviceId);
    }

    return std::make_unique<EthosNWorkloadFactory>(m_Config);
}

IBackendInternal::IWorkloadFactoryPtr
    EthosNBackend::CreateWorkloadFactory(class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
                                         const ModelOptions& modelOptions) const
{
    std::unique_ptr<ITensorHandleFactory> factory;
    std::unique_ptr<ITensorHandleFactory> importFactory;

    const std::string deviceId = ethosnbackend::GetDeviceOptionVal(modelOptions);
    if (deviceId.empty())
    {
        factory       = std::make_unique<EthosNTensorHandleFactory>(m_Config);
        importFactory = std::make_unique<EthosNImportTensorHandleFactory>(
            m_Config, static_cast<MemorySourceFlags>(MemorySource::DmaBuf),
            static_cast<MemorySourceFlags>(MemorySource::DmaBuf));
    }
    else
    {
        factory       = std::make_unique<EthosNTensorHandleFactory>(m_Config, deviceId);
        importFactory = std::make_unique<EthosNImportTensorHandleFactory>(
            m_Config, deviceId, static_cast<MemorySourceFlags>(MemorySource::DmaBuf),
            static_cast<MemorySourceFlags>(MemorySource::DmaBuf));
    }

    tensorHandleFactoryRegistry.RegisterCopyAndImportFactoryPair(factory->GetId(), importFactory->GetId());

    tensorHandleFactoryRegistry.RegisterFactory(std::move(factory));
    tensorHandleFactoryRegistry.RegisterFactory(std::move(importFactory));

    return CreateWorkloadFactory(nullptr, modelOptions);
}

IBackendInternal::IWorkloadFactoryPtr
    EthosNBackend::CreateWorkloadFactory(class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
                                         const ModelOptions& modelOptions,
                                         MemorySourceFlags importFlags,
                                         MemorySourceFlags exportFlags) const
{
    std::unique_ptr<ITensorHandleFactory> factory;
    std::unique_ptr<ITensorHandleFactory> importFactory;

    const std::string deviceId = ethosnbackend::GetDeviceOptionVal(modelOptions);
    if (deviceId.empty())
    {
        factory       = std::make_unique<EthosNTensorHandleFactory>(m_Config);
        importFactory = std::make_unique<EthosNImportTensorHandleFactory>(m_Config, importFlags, exportFlags);
    }
    else
    {
        factory       = std::make_unique<EthosNTensorHandleFactory>(m_Config, deviceId);
        importFactory = std::make_unique<EthosNImportTensorHandleFactory>(m_Config, deviceId, importFlags, exportFlags);
    }

    tensorHandleFactoryRegistry.RegisterCopyAndImportFactoryPair(factory->GetId(), importFactory->GetId());

    tensorHandleFactoryRegistry.RegisterFactory(std::move(factory));
    tensorHandleFactoryRegistry.RegisterFactory(std::move(importFactory));
    return CreateWorkloadFactory(nullptr, modelOptions);
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
    // We support Fully Connected layers having their weights and bias as separate inputs to the layer
    // and do not use the deprecated m_Weight or m_Bias members.
    ethosnCap.AddOption(BackendOptions::BackendOption("ConstantTensorsAsInputs", true));
    ethosnCap.AddOption(BackendOptions::BackendOption("AsyncExecution", true));
    ethosnCap.AddOption(BackendOptions::BackendOption("ExternallyManagedMemory", true));
    ethosnCap.AddOption(BackendOptions::BackendOption("PreImportIOTensors", true));

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

void EthosNBackend::RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry,
                                                  MemorySourceFlags importFlags,
                                                  MemorySourceFlags exportFlags)
{

    std::unique_ptr<ITensorHandleFactory> factory = std::make_unique<EthosNTensorHandleFactory>(m_Config);
    std::unique_ptr<ITensorHandleFactory> importFactory =
        std::make_unique<EthosNImportTensorHandleFactory>(m_Config, importFlags, exportFlags);

    registry.RegisterCopyAndImportFactoryPair(factory->GetId(), importFactory->GetId());

    registry.RegisterFactory(std::move(factory));
    registry.RegisterFactory(std::move(importFactory));
}

void EthosNBackend::RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry)
{
    std::unique_ptr<ITensorHandleFactory> factory       = std::make_unique<EthosNTensorHandleFactory>(m_Config);
    std::unique_ptr<ITensorHandleFactory> importFactory = std::make_unique<EthosNImportTensorHandleFactory>(
        m_Config, static_cast<MemorySourceFlags>(MemorySource::DmaBuf),
        static_cast<MemorySourceFlags>(MemorySource::DmaBuf));

    registry.RegisterCopyAndImportFactoryPair(factory->GetId(), importFactory->GetId());

    registry.RegisterFactory(std::move(factory));
    registry.RegisterFactory(std::move(importFactory));
}

armnn::EthosNBackendProfilingService& EthosNBackendProfilingService::Instance()
{
    static EthosNBackendProfilingService instance;
    return instance;
}

}    // namespace armnn
