//
// Copyright © 2018-2021 Arm Limited.
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

ARMNN_DLLEXPORT extern EthosNConfig g_EthosNConfig;
ARMNN_DLLEXPORT extern EthosNMappings g_EthosNMappings;

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

std::map<std::string, LayerType> GetMapStringToLayerType();

std::map<std::string, ActivationFunction> GetMapStringToActivationFunction();

std::map<std::string, PoolingAlgorithm> GetMapStringToPoolingAlgorithm();

char const* GetLayerTypeAsCStringWrapper(LayerType type);

Layer* CreateConvolutionLayer(LayerType type,
                              Graph& graph,
                              unsigned int inputChannels,
                              AdditionalLayerParams additionalLayerParams,
                              DataType weightDataType,
                              DataType biasDataType);

LayerType GetLayerType(std::string layerTypeName);

Layer* CreateActivationLayer(Graph& graph, std::string activationFunction, std::string layerName);

Layer* CreateFullyConnectedLayer(Graph& graph,
                                 const TensorInfo& inputTensor,
                                 const TensorInfo& outputTensor,
                                 AdditionalLayerParams& params);

Layer* CreatePooling2dLayer(Graph& graph, AdditionalLayerParams& params);

void ApplyMappings(std::vector<Mapping> mappings, Graph& newGraph);

void ReplaceUnsupportedLayers(Graph& graph);

}    // namespace ethosnbackend

}    // namespace armnn
