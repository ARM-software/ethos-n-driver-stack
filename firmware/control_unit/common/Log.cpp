//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "include/common/Log.hpp"
#if defined(CONTROL_UNIT_HARDWARE)
#include "include/common/TaskSvc.hpp"
#endif

namespace ethosn
{
namespace control_unit
{

#if defined(CONTROL_UNIT_MODEL)
constexpr char ControlUnitName[] = "control_unit";
void LogSink(ethosn::utils::log::Severity severity, const char* msg)
{
    ethosn::utils::log::sinks::StdOut<ControlUnitName>(severity, msg);
}
#else
void LogSink(ethosn::utils::log::Severity severity, const char* msg)
{
    register ethosn::utils::log::Severity r0 asm("r0") = severity;
    register const char* r1 asm("r1")                  = msg;
    asm volatile("svc %[svc_num]" ::"r"(r0), "r"(r1), [svc_num] "i"(TASK_SVC_LOG_MESSAGE));
}
#endif

}    // namespace control_unit
}    // namespace ethosn
