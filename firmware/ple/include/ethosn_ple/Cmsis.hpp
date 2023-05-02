//
// Copyright © 2018-2019 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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

#define __CM33_REV 0x0000U
#define __FPU_PRESENT 0
#define __MPU_PRESENT 0
#define __SAUREGION_PRESENT 0
#define __DSP_PRESENT 0
#define __NVIC_PRIO_BITS 3
#define __Vendor_SysTickConfig 0

#include <core_cm33.h>
