//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ethosn
{
namespace command_stream
{

enum class Opcode : uint8_t
{
    DUMP_DRAM,
    DUMP_SRAM,
    CASCADE
};

}    // namespace command_stream
}    // namespace ethosn
