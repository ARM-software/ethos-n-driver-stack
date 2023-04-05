//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "EthosNBackend.hpp"
#include "EthosNConfig.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/backends/SubgraphView.hpp>
#include <armnnUtils/Filesystem.hpp>
#include <doctest/doctest.h>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

#define ARRAY_SIZE(X) (sizeof(X) / sizeof(X[0]))

namespace testing_utils
{

class TempDir
{
public:
    TempDir()
    {
        static std::atomic<int> g_Counter;
        int uniqueId = g_Counter++;
        // cppcheck-suppress  useInitializationList symbolName=m_Dir
        m_Dir = "TempDir-" + std::to_string(uniqueId);
        fs::create_directories(m_Dir);
    }

    ~TempDir()
    {
        fs::remove_all(m_Dir);
    }

    std::string Str() const
    {
        return m_Dir.string();
    }

private:
    fs::path m_Dir;
};

inline std::string ReadFile(const std::string& file)
{
    std::ifstream is(file);
    std::ostringstream contents;
    contents << is.rdbuf();
    return contents.str();
}

inline bool operator==(const armnn::SubgraphView& lhs, const armnn::SubgraphView& rhs)
{
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    if (lhs.GetInputSlots() != rhs.GetInputSlots())
    {
        return false;
    }

    if (lhs.GetOutputSlots() != rhs.GetOutputSlots())
    {
        return false;
    }

    auto lhsLayerI = lhs.cbegin();
    auto rhsLayerI = rhs.cbegin();

    if (std::distance(lhsLayerI, lhs.cend()) != std::distance(rhsLayerI, rhs.cend()))
    {
        return false;
    }

    while (lhsLayerI != lhs.cend() && rhsLayerI != rhs.cend())
    {
        if (*lhsLayerI != *rhsLayerI)
        {
            return false;
        }
        ++lhsLayerI;
        ++rhsLayerI;
    }

    return (lhsLayerI == lhs.cend() && rhsLayerI == rhs.cend());
    ARMNN_NO_DEPRECATE_WARN_END
}

/// Sets the globally cached backend config data, so that different tests can run with different configs.
/// Without this, the first test which instantiates an EthosNBackend object would load and set the config for all future
/// tests using EthosNBackend and there would be no way to change this. Tests can use this function to override
/// the cached values.
inline void SetBackendGlobalConfig(const armnn::EthosNConfig& config, const std::vector<char>& capabilities)
{
    class EthosNBackendEx : public armnn::EthosNBackend
    {
    public:
        void SetBackendGlobalConfigForTesting(const armnn::EthosNConfig& config, const std::vector<char>& capabilities)
        {
            ms_Config       = config;
            ms_Capabilities = capabilities;
        }
    };

    EthosNBackendEx().SetBackendGlobalConfigForTesting(config, capabilities);
}

/// Scope-controlled version of SetBackendGlobalConfig, which automatically restores
/// default settings after being destroyed. This can be used to avoid messing up the config for tests
/// that run afterwards.
class BackendGlobalConfigSetter
{
public:
    BackendGlobalConfigSetter(const armnn::EthosNConfig& config, const std::vector<char>& capabilities)
    {
        SetBackendGlobalConfig(config, capabilities);
    }
    ~BackendGlobalConfigSetter()
    {
        // Setting an empty capabilities vector will trigger a reload on next EthosNBackend instantiation
        SetBackendGlobalConfig(armnn::EthosNConfig(), {});
    }
};

inline void CreateEthosNPrecompiledWorkloadTest()
{
    using namespace armnn;

    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    // Add an input layer
    armnn::IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input layer");
    CHECK(inputLayer);

    // Arm NN weights tensor shape is OIHW (out channels, in channels, height, width) for NCHW
    // Arm NN weights tensor shape is OHWI (out channels, height, width, in channels) for NHWC
    // this test is using NHWC, so the weights shape is OHWI
    armnn::TensorInfo weightsTensorInfo(TensorShape({ 16, 1, 1, 16 }), armnn::DataType::QAsymmU8, 0.9f, 0, true);
    unsigned int weightsLength = weightsTensorInfo.GetNumElements();

    using WeightType = uint8_t;
    std::vector<WeightType> convWeightsData(weightsLength);
    for (unsigned int i = 0; i < weightsLength; ++i)
    {
        convWeightsData[i] = static_cast<WeightType>(i);
    }

    armnn::ConstTensor weights(weightsTensorInfo, convWeightsData);

    // Add a layer that can be used in the PreCompiled layer
    armnn::Convolution2dDescriptor convDesc2d;
    convDesc2d.m_StrideX     = 1;
    convDesc2d.m_StrideY     = 1;
    convDesc2d.m_BiasEnabled = false;
    convDesc2d.m_DataLayout  = armnn::DataLayout::NHWC;

    armnn::IConnectableLayer* convLayer = nullptr;
    const std::string convLayerName("conv layer");

    // Create convolution layer without biases
    convLayer                              = net->AddConvolution2dLayer(convDesc2d, convLayerName.c_str());
    armnn::IConnectableLayer* weightsLayer = net->AddConstantLayer(weights, "Conv2dWeights");
    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightsTensorInfo);
    weightsLayer->GetOutputSlot(0).Connect((*convLayer).GetInputSlot(1));

    CHECK(convLayer);

    // Add an output layer
    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output layer");
    CHECK(outputLayer);

    // set the tensors in the network (NHWC format)
    armnn::TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), DataType::QAsymmU8);
    inputTensorInfo.SetQuantizationOffset(0);
    inputTensorInfo.SetQuantizationScale(0.9f);
    inputTensorInfo.SetConstant();

    armnn::TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), DataType::QAsymmU8);
    outputTensorInfo.SetQuantizationOffset(0);
    outputTensorInfo.SetQuantizationScale(0.9f);

    // Connect the layers
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimize the network for the backend supported by the factory
    std::vector<armnn::BackendId> backends = { EthosNBackend::GetIdStatic() };
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    armnn::OptimizerOptions optimizerOptions;
    armnn::IOptimizedNetworkPtr optimizedNet =
        armnn::Optimize(*net, backends, runtime->GetDeviceSpec(), optimizerOptions);
    CHECK(optimizedNet != nullptr);

    // Load graph into runtime
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet));

    // Creates structures for inputs and outputs.
    const std::vector<uint8_t> inputData(inputTensorInfo.GetNumElements());
    std::vector<uint8_t> outputData(outputTensorInfo.GetNumElements());

    armnn::InputTensors inputTensors{ { 0, armnn::ConstTensor(inputTensorInfo, inputData.data()) } };
    armnn::OutputTensors outputTensors{ { 0, armnn::Tensor(outputTensorInfo, outputData.data()) } };

    // Execute network
    runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
}

}    // namespace testing_utils
