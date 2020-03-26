//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../EthosNConfig.hpp"
#include "backendsCommon/Workload.hpp"
#include "backendsCommon/WorkloadData.hpp"
#include "backendsCommon/WorkloadInfo.hpp"

#include <ethosn_driver_library/Buffer.hpp>
#include <ethosn_driver_library/Inference.hpp>
#include <ethosn_driver_library/Network.hpp>
#include <ethosn_support_library/Support.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace armnn
{

bool EthosNPreCompiledWorkloadValidate(std::string* reasonIfUnsupported);

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
        Network(std::unique_ptr<ethosn::support_library::CompiledNetwork> compiledNetwork,
                std::unordered_map<uint32_t, uint32_t> inputSlotsToEthosNInputs,
                std::unordered_map<uint32_t, uint32_t> outputSlotsToEthosNOutputs)
            : m_CompiledNetwork(std::move(compiledNetwork))
            , m_InputSlotsToEthosNInputs(std::move(inputSlotsToEthosNInputs))
            , m_OutputSlotsToEthosNOutputs(std::move(outputSlotsToEthosNOutputs))
        {}

        std::unique_ptr<ethosn::support_library::CompiledNetwork> m_CompiledNetwork;
        /// Maps from the Arm NN input/output slot index to the Ethos-N  input/output buffer index.
        /// In some cases these may be equivalent but it is not guaranteed.
        /// @{
        std::unordered_map<uint32_t, uint32_t> m_InputSlotsToEthosNInputs;
        std::unordered_map<uint32_t, uint32_t> m_OutputSlotsToEthosNOutputs;
        /// @}
    };

    struct PerfData
    {
        std::string m_PerfOutFile;
        ethosn::support_library::EthosNVariant m_PerfVariant;
        uint32_t m_PerfSramSizeBytesOverride;
        ethosn::support_library::NetworkPerformanceData m_Data;
        ethosn::support_library::EstimationOptions m_EstimationOptions;
    };

    EthosNPreCompiledObject(Network network, std::map<uint32_t, std::string> ethosnOperationNameMapping)
        : m_IsPerfEstimationOnly(false)
        , m_Network(std::move(network))
        , m_EthosNOperationNameMapping(ethosnOperationNameMapping)
    {}

    EthosNPreCompiledObject(PerfData perfData, std::map<uint32_t, std::string> ethosnOperationNameMapping)
        : m_IsPerfEstimationOnly(true)
        , m_PerfData(std::move(perfData))
        , m_EthosNOperationNameMapping(ethosnOperationNameMapping)
    {}

    ~EthosNPreCompiledObject()
    {
        if (m_IsPerfEstimationOnly)
        {
            m_PerfData.~PerfData();
        }
        else
        {
            m_Network.~Network();
        }
    }

    bool IsPerfEstimationOnly() const
    {
        return m_IsPerfEstimationOnly;
    }

    const Network* GetNetwork() const
    {
        return !m_IsPerfEstimationOnly ? &m_Network : nullptr;
    }

    const PerfData* GetPerfData() const
    {
        return m_IsPerfEstimationOnly ? &m_PerfData : nullptr;
    }

    const std::map<uint32_t, std::string>& GetEthosNOperationNameMapping() const
    {
        return m_EthosNOperationNameMapping;
    }

private:
    const bool m_IsPerfEstimationOnly;

    union
    {
        Network m_Network;
        PerfData m_PerfData;
    };

    /// Map from Ethos-N operation ID to the corresponding Arm NN layer name.
    std::map<uint32_t, std::string> m_EthosNOperationNameMapping;
};

class EthosNPreCompiledWorkload : public BaseWorkload<PreCompiledQueueDescriptor>
{
public:
    EthosNPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    void Init(const PreCompiledDescriptor& descriptor, const EthosNPreCompiledObject::Network& network);
    void SavePerformanceJson() const;

    // The workload does not own the EthosNPreCompiledObject, the ownership is still retained by the pre-compiled layer
    const EthosNPreCompiledObject* m_PreCompiledObject;

    // The workload does own the network and the inference instances
    mutable std::unique_ptr<ethosn::driver_library::Network> m_Network;

    std::vector<ethosn::driver_library::Buffer*> m_InputBuffers{};
    std::vector<ethosn::driver_library::Buffer*> m_OutputBuffers{};
};

}    //namespace armnn
