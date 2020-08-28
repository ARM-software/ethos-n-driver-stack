//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ethosn
{
namespace command_stream
{

enum class Opcode : uint8_t
{
    OPERATION_MCE_PLE,
    OPERATION_PLE_ONLY,
    OPERATION_SOFTMAX,
    OPERATION_CONVERT,
    OPERATION_SPACE_TO_DEPTH,
    DUMP_DRAM,
    DUMP_SRAM,
    FENCE,
    SECTION,
    DELAY
};

}    // namespace command_stream
}    // namespace ethosn
