//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/Log.hpp>
#include <cstdint>

namespace ethosn
{
namespace control_unit
{

struct TaskConfig
{
    ethosn::utils::log::Severity logSeverity;
    uint32_t pleAddrExtend;
};

extern "C" __attribute__((noreturn)) void Task(const TaskConfig* config);

}    // namespace control_unit
}    // namespace ethosn
