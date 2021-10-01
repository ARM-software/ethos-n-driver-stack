//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNBackendProfilingContext.hpp"
#include "EthosNConfig.hpp"

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/OptimizationViews.hpp>

namespace armnn
{

void CreatePreCompiledLayerInGraph(OptimizationViews& optimizationViews,
                                   const SubgraphView& subgraph,
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

private:
    /// 'Global' settings for this backend, loaded from config file or queried from the HW.
    /// @{
    EthosNConfig m_Config;
    std::vector<char> m_Capabilities;
    /// @}

protected:
    /// Cached source for the above fields - see comments in constructor for details.
    /// Protected visibility for use in tests (see SetBackendGlobalConfig)
    /// @{
    ARMNN_DLLEXPORT static EthosNConfig ms_Config;
    ARMNN_DLLEXPORT static std::vector<char> ms_Capabilities;
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

namespace ethosnbackend
{

#define MAX_ETHOSN_DRIVER_LIBRARY_MAJOR_VERSION_SUPPORTED 1
#define MIN_ETHOSN_DRIVER_LIBRARY_MAJOR_VERSION_SUPPORTED 1
#define MAX_ETHOSN_SUPPORT_LIBRARY_MAJOR_VERSION_SUPPORTED 2
#define MIN_ETHOSN_SUPPORT_LIBRARY_MAJOR_VERSION_SUPPORTED 1

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
