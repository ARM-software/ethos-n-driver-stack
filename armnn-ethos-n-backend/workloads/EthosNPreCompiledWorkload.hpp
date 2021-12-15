//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../EthosNConfig.hpp"
#include "backendsCommon/Workload.hpp"

#include <ethosn_driver_library/Buffer.hpp>
#include <ethosn_driver_library/Inference.hpp>
#include <ethosn_driver_library/Network.hpp>
#include <ethosn_support_library/Support.hpp>

#include <memory>
#include <string>
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
        Network(std::vector<char> serializedCompiledNetwork)
            : m_SerializedCompiledNetwork(std::move(serializedCompiledNetwork))
        {}

        std::vector<char> m_SerializedCompiledNetwork;
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
    EthosNPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor,
                              const WorkloadInfo& info,
                              const std::string& deviceId);
    void Execute() const override;

private:
    void Init(const PreCompiledDescriptor& descriptor,
              const EthosNPreCompiledObject::Network& network,
              const std::string& deviceId);
    void SavePerformanceJson() const;

    // The workload does not own the EthosNPreCompiledObject, the ownership is still retained by the pre-compiled layer
    const EthosNPreCompiledObject* m_PreCompiledObject;

    // The workload does own the network and the inference instances
    mutable std::unique_ptr<ethosn::driver_library::Network> m_Network;

    std::vector<ethosn::driver_library::Buffer*> m_InputBuffers{};
    std::vector<ethosn::driver_library::Buffer*> m_OutputBuffers{};
};

}    //namespace armnn
