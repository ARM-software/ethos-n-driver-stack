//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "HardwareHelpers.hpp"

#include "Cmsis.hpp"

namespace ethosn
{
namespace control_unit
{

namespace Dwt
{

void Reset()
{
#if defined(CONTROL_UNIT_PROFILING)
    CoreDebug->DEMCR |= 0x01000000;
    DWT->CYCCNT   = 0;    // reset the cycle counter
    DWT->SLEEPCNT = 0;    // reset the sleep counter
    DWT->CTRL     = 0;
#endif
}

void Start()
{
#if defined(CONTROL_UNIT_PROFILING)
    DWT->CTRL |= 0x00000001;    // enable the counter
#endif
}

void Stop()
{
#if defined(CONTROL_UNIT_PROFILING)
    DWT->CTRL &= 0xFFFFFFFE;    // disable the counter
#endif
}

uint32_t GetCycleCount()
{
#if defined(CONTROL_UNIT_PROFILING)
    return DWT->CYCCNT;
#else
    return 0;
#endif
}

}    // namespace Dwt

namespace Interrupt
{
void SetPendSV()
{
    SCB->ICSR = SCB_ICSR_PENDSVSET_Msk;
    __ISB();
}
}    // namespace Interrupt

namespace Cache
{
void DClean(const void* addr, ptrdiff_t dsize)
{
    auto paddr = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(addr));
    SCB_CleanDCache_by_Addr(paddr, dsize);
}

void DInvalidate(void* addr, ptrdiff_t dsize)
{
    auto paddr = reinterpret_cast<uint32_t*>(addr);
    SCB_InvalidateDCache_by_Addr(paddr, dsize);
}

void DCleanInvalidate()
{
    SCB_CleanInvalidateDCache();
}

void IEnable()
{
    SCB_EnableICache();
}

void DEnable()
{
    SCB_EnableDCache();
}

}    // namespace Cache

namespace Fault
{

uint32_t GetConfigurableFaultStatusRegister()
{
    return SCB->CFSR;
}

uint32_t GetHardFaultStatusRegister()
{
    return SCB->HFSR;
}

uint32_t GetMemManageFaultAddressRegister()
{
    return SCB->MMFAR;
}

uint32_t GetBusFaultAddressRegister()
{
    return SCB->BFAR;
}

}    // namespace Fault

namespace Interrupts
{

void SetSVCallPriority(uint32_t priority)
{
    NVIC_SetPriority(IRQn_Type::SVCall, priority);
}

void SetPendSVPriority(uint32_t priority)
{
    NVIC_SetPriority(IRQn_Type::PendSV, priority);
}

}    // namespace Interrupts

namespace Tick
{

// Short delays, based on MCU clock. Based on the internal 24 bit timer, SysTick
void Delay(uint32_t ticks)
{
    SysTick->LOAD = ticks & SysTick_LOAD_RELOAD_Msk;
    SysTick->VAL  = 0;
    SysTick->CTRL = SysTick_CTRL_ENABLE_Msk | SysTick_CTRL_CLKSOURCE_Msk;
    while ((SysTick->CTRL & SysTick_CTRL_COUNTFLAG_Msk) == 0)
        ;
    SysTick->CTRL = 0;
}

}    // namespace Tick

}    // namespace control_unit
}    // namespace ethosn
