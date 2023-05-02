//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace ethosn
{
namespace control_unit
{

namespace Dwt
{

void Reset();
void Start();
void Stop();
uint32_t GetCycleCount();

}    // namespace Dwt

namespace Interrupt
{

void SetPendSV();

}    // namespace Interrupt

namespace Cache
{

void DClean(const void* addr, ptrdiff_t dsize);

void DInvalidate(void* addr, ptrdiff_t dsize);

void DCleanInvalidate();

void IEnable();

void DEnable();

}    // namespace Cache

namespace Fault
{

uint32_t GetConfigurableFaultStatusRegister();
uint32_t GetHardFaultStatusRegister();
uint32_t GetMemManageFaultAddressRegister();
uint32_t GetBusFaultAddressRegister();

}    // namespace Fault

namespace Interrupts
{

void SetSVCallPriority(uint32_t priority);
void SetPendSVPriority(uint32_t priority);

}    // namespace Interrupts

namespace Tick
{

// Short delays, based on MCU clock. Based on the internal 24 bit timer, SysTick
void Delay(uint32_t ticks);

}    // namespace Tick

}    // namespace control_unit
}    // namespace ethosn
