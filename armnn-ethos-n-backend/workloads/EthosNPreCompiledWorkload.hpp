//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../EthosNConfig.hpp"
#include "armnn/backends/Workload.hpp"

#include <ethosn_driver_library/Buffer.hpp>
#include <ethosn_driver_library/Inference.hpp>
#include <ethosn_driver_library/Network.hpp>
#include <ethosn_support_library/Support.hpp>

#include <armnn/ArmNN.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace armnn
{

/// The data type stored as the m_PreCompiledObject in the PreCompiledLayer.
/// This is the mechanism to pass data between the conversion stage (EthosNSubgraphViewConverter)
/// and the execution stage (EthosNPreCompiledWorkload).
class EthosNPreCompiledObject
{
public:
    /// The data type stored as the m_PreCompiledObject in the PreCompiledLayer.
    /// This is the mechanism to pass data between the conversion stage (EthosNSubgraphViewConverter)
    /// and the execution stage (EthosNPreCompiledWorkload).
    struct Network
    {
        Network(std::vector<char> serializedCompiledNetwork)
            : m_SerializedCompiledNetwork(std::move(serializedCompiledNetwork))
        {}

        std::vector<char> m_SerializedCompiledNetwork;
    };

    EthosNPreCompiledObject(armnn::Optional<Network> network,
                            bool isSkipInference,
                            std::map<uint32_t, std::string> ethosnOperationNameMapping,
                            int inferenceTimeout,
                            uint32_t subgraphIndex,
                            uint32_t intermediateBufSize)
        : m_IsSkipInference(isSkipInference)
        , m_InferenceTimeout(inferenceTimeout)
        , m_Network(std::move(network))
        , m_EthosNOperationNameMapping(std::move(ethosnOperationNameMapping))
        , m_SubgraphIndex(subgraphIndex)
        , m_IntermediateBufSize(intermediateBufSize)
    {}

    bool IsSkipInference() const
    {
        return m_IsSkipInference;
    }

    int GetInferenceTimeout() const
    {
        return m_InferenceTimeout;
    }

    const armnn::Optional<Network>& GetNetwork() const
    {
        return m_Network;
    }

    const std::map<uint32_t, std::string>& GetEthosNOperationNameMapping() const
    {
        return m_EthosNOperationNameMapping;
    }

    uint32_t GetSubgraphIndex() const
    {
        return m_SubgraphIndex;
    }

    uint32_t GetIntermediateBufferSize() const
    {
        return m_IntermediateBufSize;
    }

private:
    const bool m_IsSkipInference;
    const int m_InferenceTimeout;

    /// The Network may not be present if we are running in estimate-only or offline mode.
    armnn::Optional<Network> m_Network;

    /// Map from Ethos-N operation ID to the corresponding Arm NN layer name.
    std::map<uint32_t, std::string> m_EthosNOperationNameMapping;

    uint32_t m_SubgraphIndex;
    uint32_t m_IntermediateBufSize;
};

class EthosNPreCompiledWorkload : public BaseWorkload<PreCompiledQueueDescriptor>
{
public:
    EthosNPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor,
                              const WorkloadInfo& info,
                              const std::string& deviceId,
                              std::shared_ptr<armnn::ICustomAllocator> customAllocator = nullptr);
    void Execute() const override;

private:
    void Init(const EthosNPreCompiledObject::Network& network, const std::string& deviceId);

    bool SupportsTensorHandleReplacement() const override
    {
        return true;
    }

    void ReplaceInputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override
    {
        this->m_Data.m_Inputs[slot] = tensorHandle;
    }

    void ReplaceOutputTensorHandle(ITensorHandle* tensorHandle, unsigned int slot) override
    {
        this->m_Data.m_Outputs[slot] = tensorHandle;
    }

    // The workload does not own the EthosNPreCompiledObject, the ownership is still retained by the pre-compiled layer
    const EthosNPreCompiledObject* m_PreCompiledObject;

    // The workload does own the network and the inference instances
    std::unique_ptr<ethosn::driver_library::Network> m_Network;
    std::shared_ptr<armnn::ICustomAllocator> m_InternalAllocator;
};

}    //namespace armnn
