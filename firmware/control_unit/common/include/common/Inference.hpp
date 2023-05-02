//
// Copyright © 2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Utils.hpp"

#include <common/FirmwareApi.hpp>

#include <ethosn_command_stream/CommandStream.hpp>

namespace ethosn
{
namespace control_unit
{

using BufferTable = utils::ArrayRange<const ethosn_buffer_desc>;

// Helper class to access binary inference data
class Inference
{
public:
    Inference(ethosn_address_t bufferArray)
        : m_BufferTable{ &reinterpret_cast<ethosn_buffer_array*>(bufferArray)->buffers[0],
                         &reinterpret_cast<ethosn_buffer_array*>(bufferArray)->buffers[0] +
                             reinterpret_cast<ethosn_buffer_array*>(bufferArray)->num_buffers }
    {}

    Inference(const Inference&) = delete;
    Inference& operator=(const Inference&) = delete;

    BufferTable GetBufferTable() const
    {
        return m_BufferTable;
    }

    command_stream::CommandStreamParser GetCommandStream() const
    {
        // The command stream is defined to be the first entry in the buffer table.
        const uint32_t* const csBegin = reinterpret_cast<const uint32_t*>(m_BufferTable[0].address);
        ASSERT_MSG(m_BufferTable[0].size % sizeof(uint32_t) == 0, "Command stream size must be multiple of 4.");
        const uint32_t* const csEnd = csBegin + (m_BufferTable[0].size / sizeof(uint32_t));
        return command_stream::CommandStreamParser({ csBegin, csEnd });
    }

private:
    BufferTable m_BufferTable;
};

}    // namespace control_unit
}    // namespace ethosn
