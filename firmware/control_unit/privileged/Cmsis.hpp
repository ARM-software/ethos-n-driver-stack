//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Common Cmsis configuration for core_cm7.h, set up before including core_cm7.h
typedef enum
{
    Reset            = -15,
    Nmi              = -14,
    HardFault        = -13,
    MemoryManagement = -12,
    BusFault         = -11,
    UsageFault       = -10,
    SVCall           = -5,
    DebugMonitor     = -4,
    PendSV           = -2,
    SysTick_IRQn     = -1,
    Irq0             = 0,
} IRQn_Type;

#define __CM7_REV 0x0000U
#define __FPU_PRESENT 0
#define __MPU_PRESENT 1
#define __ICACHE_PRESENT 1
#define __DCACHE_PRESENT 1
#define __TCM_PRESENT 0
#define __NVIC_PRIO_BITS 3
#define __Vendor_SysTickConfig 0

#include <core_cm7.h>
