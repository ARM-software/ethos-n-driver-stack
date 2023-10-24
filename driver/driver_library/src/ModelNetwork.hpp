//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "NetworkImpl.hpp"

namespace ethosn
{
namespace control_unit
{
class IModelFirmwareInterface;
}

namespace driver_library
{

class ModelNetworkImpl : public NetworkImpl
{
public:
    ModelNetworkImpl(const char* compiledNetworkData, size_t compiledNetworkSize);
    ~ModelNetworkImpl();

    Inference* ScheduleInference(Buffer* const inputBuffers[],
                                 uint32_t numInputBuffers,
                                 Buffer* const outputBuffers[],
                                 uint32_t numOutputBuffers) override;

protected:
    std::pair<const char*, size_t> MapIntermediateBuffers() override;
    void UnmapIntermediateBuffers(std::pair<const char*, size_t> mappedPtr) override;

private:
    std::unique_ptr<control_unit::IModelFirmwareInterface> m_FirmwareInterface;

    uint64_t m_IntermediateDataBaseAddress;
    std::vector<uint8_t> m_MappedIntermediateBuffer;
};

}    // namespace driver_library
}    // namespace ethosn
