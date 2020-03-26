//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNBackendProfilingContext.hpp"
#include "EthosNConfig.hpp"
#include "EthosNMapping.hpp"

#include <armnn/backends/IBackendInternal.hpp>
#include <backendsCommon/OptimizationViews.hpp>

namespace armnn
{

extern EthosNConfig g_EthosNConfig;
extern EthosNMappings g_EthosNMappings;

void CreatePreCompiledLayerInGraph(OptimizationViews& optimizationViews,
                                   const SubgraphView& subgraph,
                                   const EthosNMappings& mappings);

class EthosNBackend : public IBackendInternal
{
public:
    EthosNBackend()  = default;
    ~EthosNBackend() = default;

    static const BackendId& GetIdStatic();
    const BackendId& GetId() const override
    {
        return GetIdStatic();
    }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override;

    IBackendInternal::IWorkloadFactoryPtr
        CreateWorkloadFactory(const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr) const override;

    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override;

    IBackendInternal::IBackendProfilingContextPtr
        CreateBackendProfilingContext(const IRuntime::CreationOptions& creationOptions,
                                      IBackendProfilingPtr& backendProfiling) override;

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override;

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const override;
};

class EthosNBackendProfilingService
{
public:
    // Getter for the singleton instance
    static EthosNBackendProfilingService& Instance()
    {
        static EthosNBackendProfilingService instance;
        return instance;
    }

    profiling::EthosNBackendProfilingContext* GetContext()
    {
        return m_sharedContext.get();
    }

    void SetProfilingContextPtr(std::shared_ptr<profiling::EthosNBackendProfilingContext> shared)
    {
        m_sharedContext = shared;
    }

private:
    std::shared_ptr<profiling::EthosNBackendProfilingContext> m_sharedContext;
};

namespace ethosnbackend
{

std::map<std::string, LayerType> GetMapStringToLayerType();

std::map<std::string, ActivationFunction> GetMapStringToActivationFunction();

Layer* CreateConvolutionLayer(std::string layerName,
                              Graph& graph,
                              unsigned int inputChannels,
                              unsigned int kernelWidth,
                              unsigned int kernelHeight,
                              unsigned int strideX,
                              unsigned int strideY,
                              DataType weightDataType,
                              DataType biasDataType);

Layer* CreateActivationLayer(Graph& graph, std::string activationFunction);

void ApplyMappings(std::vector<Mapping> mappings, Graph& newGraph);

}    // namespace ethosnbackend

}    // namespace armnn
