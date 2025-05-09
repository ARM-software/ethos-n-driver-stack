//
// Copyright © 2018-2025 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNBackendProfilingContext.hpp"
#include "EthosNCaching.hpp"
#include "EthosNConfig.hpp"
#include "EthosNTensorHandleFactory.hpp"

#include <DllExport.hpp>
#include <armnn/backends/IBackendContext.hpp>
#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/OptimizationViews.hpp>

#include <ethosn_driver_library/Device.hpp>
#include <ethosn_driver_library/ProcMemAllocator.hpp>

#include <map>

#include <fmt/format.h>

namespace armnn
{

void CreatePreCompiledLayerInGraph(OptimizationViews& optimizationViews,
                                   const SubgraphView& subgraph,
                                   uint32_t subgraphIdx,
                                   const EthosNConfig& config,
                                   const std::vector<char>& capabilities,
                                   const ModelOptions& modelOptions);

class EthosNBackend : public IBackendInternal
{
public:
    EthosNBackend();
    ~EthosNBackend() = default;

    static const BackendId& GetIdStatic();
    const BackendId& GetId() const override
    {
        return GetIdStatic();
    }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override;

    IBackendInternal::IWorkloadFactoryPtr
        CreateWorkloadFactory(const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr) const override;

    IBackendInternal::IWorkloadFactoryPtr
        CreateWorkloadFactory(const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr,
                              const ModelOptions& modelOptions                               = {}) const override;

    IBackendInternal::IWorkloadFactoryPtr
        CreateWorkloadFactory(TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
                              const ModelOptions& modelOptions) const override;

    IBackendInternal::IWorkloadFactoryPtr
        CreateWorkloadFactory(TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
                              const ModelOptions& modelOptions,
                              MemorySourceFlags inputFlags,
                              MemorySourceFlags outputFlags) const override;

    BackendCapabilities GetCapabilities() const override;

    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override;

    IBackendInternal::IBackendProfilingContextPtr
        CreateBackendProfilingContext(const IRuntime::CreationOptions& creationOptions,
                                      IBackendProfilingPtr& backendProfiling) override;

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override;
    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport(const ModelOptions& modelOptions) const override;

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const override;

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph,
                                           const ModelOptions& modelOptions) const override;

    virtual void RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry,
                                               MemorySourceFlags inputFlags,
                                               MemorySourceFlags outputFlags) override;

    virtual void RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry) override;

    virtual std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const override
    {
        if (m_IsProtected)
        {
            return std::vector<ITensorHandleFactory::FactoryId>{ EthosNProtectedTensorHandleFactory::GetIdStatic() };
        }
        else
        {
            return std::vector<ITensorHandleFactory::FactoryId>{ EthosNImportTensorHandleFactory::GetIdStatic() };
        }
    }

    bool UseCustomMemoryAllocator(std::shared_ptr<armnn::ICustomAllocator> allocator,
                                  armnn::Optional<std::string&> errMsg) override
    {
        IgnoreUnused(errMsg);
        ms_IsProtected       = allocator->GetMemorySourceType() == MemorySource::DmaBufProtected;
        m_IsProtected        = ms_IsProtected;
        ms_InternalAllocator = allocator;
        m_InternalAllocator  = allocator;
        ARMNN_LOG(info) << "Using Custom Allocator for EthosNBackend";
        return true;
    }

private:
    /// 'Global' settings for this backend, loaded from config file or queried from the HW.
    /// @{
    EthosNConfig m_Config;
    std::vector<char> m_Capabilities;
    std::shared_ptr<armnn::ICustomAllocator> m_InternalAllocator;
    bool m_IsProtected;
    /// @}

    /// Subgraph counter, used to number each subgraph that we receive from Arm NN for a network.
    /// Because this backend object is re-constructed for each different network we compile, this counter
    /// gets reset for each network, which is exactly what we want.
    mutable uint32_t m_NextSubgraphIdx;

protected:
    /// Cached source for the above fields - see comments in constructor for details.
    /// Protected visibility for use in tests (see SetBackendGlobalConfig)
    /// @{
    ARMNN_DLLEXPORT static EthosNConfig ms_Config;
    ARMNN_DLLEXPORT static std::vector<char> ms_Capabilities;
    ARMNN_DLLEXPORT static std::shared_ptr<armnn::ICustomAllocator> ms_InternalAllocator;
    ARMNN_DLLEXPORT static bool ms_IsProtected;
    /// @}
};

class EthosNBackendProfilingService
{
public:
    // Getter for the singleton instance
    static EthosNBackendProfilingService& Instance();

    profiling::EthosNBackendProfilingContext* GetContext()
    {
        return m_SharedContext.get();
    }

    void SetProfilingContextPtr(std::shared_ptr<profiling::EthosNBackendProfilingContext> shared)
    {
        m_SharedContext = shared;
    }

    bool IsProfilingEnabled()
    {
        if (!m_SharedContext)
        {
            return false;
        }
        return m_SharedContext->IsProfilingEnabled();
    }

private:
    std::shared_ptr<profiling::EthosNBackendProfilingContext> m_SharedContext;
};

class EthosNBackendAllocatorService
{
public:
    // Getter for the singleton instance
    static EthosNBackendAllocatorService& GetInstance();

    void RegisterAllocator(const EthosNConfig& config, const std::string& deviceId)
    {
        using namespace ethosn::driver_library;

        if (config.m_PerfOnly || config.m_Offline)
        {
            // Performance only, allocators not needed
            return;
        }

        std::string allocString = deviceId;
        if (deviceId.empty())
        {
            allocString = GetDeviceNamePrefix() + std::to_string(GetDeviceBaseId());
        }

        bool deviceIdRegistered = m_RegisteredDeviceIds.emplace(allocString).second;
        if (m_RefCount > 0 && deviceIdRegistered)
        {
            m_Allocators.emplace(allocString, ProcMemAllocator(allocString.c_str(), m_IsProtected));
        }
    }

    ethosn::driver_library::ProcMemAllocator& GetProcMemAllocator(const std::string& deviceId)
    {
        using namespace ethosn::driver_library;

        std::string searchString = deviceId;
        if (deviceId.empty())
        {
            searchString = GetDeviceNamePrefix() + std::to_string(GetDeviceBaseId());
        }

        auto searchResult = m_Allocators.find(searchString);
        if (searchResult != m_Allocators.end())
        {
            return searchResult->second;
        }
        else
        {
            throw RuntimeException("Process memory allocator not found");
        }
    }

    void GetAllocators()
    {
        using namespace ethosn::driver_library;

        if (m_RefCount <= 0)
        {
            for (auto& deviceId : m_RegisteredDeviceIds)
            {
                m_Allocators.emplace(deviceId, ProcMemAllocator(deviceId.c_str(), m_IsProtected));
            }
        }

        m_RefCount++;
    }

    void PutAllocators()
    {
        using namespace ethosn::driver_library;

        m_RefCount--;

        if (m_RefCount <= 0)
        {
            m_Allocators.clear();
        }
    }

    void SetProtected(bool isProtected)
    {
        if (m_IsProtected != isProtected && m_RefCount > 0)
        {
            throw RuntimeException(
                fmt::format("Failed to set EthosNBackendAllocatorService to {}protected mode while in {}protected mode",
                            isProtected ? "" : "non-", m_IsProtected ? "" : "non-"));
        }

        m_IsProtected = isProtected;
    }

private:
    std::set<std::string> m_RegisteredDeviceIds;
    std::map<std::string, ethosn::driver_library::ProcMemAllocator> m_Allocators;
    uint32_t m_RefCount = 0;
    bool m_IsProtected  = false;
};

class EthosNBackendContext : public IBackendContext
{
public:
    EthosNBackendContext(const IRuntime::CreationOptions& options, const EthosNConfig& ethosNConfig)
        : IBackendContext(options)
        , m_EthosNConfig(ethosNConfig)
        , m_Options(options)
    {
        EthosNBackendAllocatorService::GetInstance().SetProtected(m_Options.m_ProtectedMode);
    }

    bool BeforeLoadNetwork(NetworkId networkId) override;

    bool AfterLoadNetwork(NetworkId networkId) override;

    bool BeforeUnloadNetwork(NetworkId networkId) override;

    bool AfterUnloadNetwork(NetworkId networkId) override;

    bool AfterEnqueueWorkload(NetworkId networkId) override;

private:
    EthosNConfig m_EthosNConfig;
    IRuntime::CreationOptions m_Options;
};

namespace ethosnbackend
{

#define MAX_ETHOSN_DRIVER_LIBRARY_MAJOR_VERSION_SUPPORTED 9
#define MIN_ETHOSN_DRIVER_LIBRARY_MAJOR_VERSION_SUPPORTED 9
#define MAX_ETHOSN_SUPPORT_LIBRARY_MAJOR_VERSION_SUPPORTED 6
#define MIN_ETHOSN_SUPPORT_LIBRARY_MAJOR_VERSION_SUPPORTED 6

constexpr bool IsLibraryVersionSupported(const uint32_t& majorVer, const uint32_t& maxVer, const uint32_t& minVer);

bool VerifyLibraries();

constexpr unsigned int STRIDE_X      = 0;
constexpr unsigned int STRIDE_Y      = 1;
constexpr unsigned int PAD_BOTTOM    = 0;
constexpr unsigned int PAD_LEFT      = 1;
constexpr unsigned int PAD_RIGHT     = 2;
constexpr unsigned int PAD_TOP       = 3;
constexpr unsigned int DILATION_X    = 0;
constexpr unsigned int DILATION_Y    = 1;
constexpr unsigned int KERNEL_HEIGHT = 0;
constexpr unsigned int KERNEL_WIDTH  = 1;

}    // namespace ethosnbackend

}    // namespace armnn
