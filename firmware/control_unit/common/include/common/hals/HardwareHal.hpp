//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "HalBase.hpp"

#include <scylla_addr_fields.h>
#include <scylla_regs.h>

namespace ethosn
{
namespace control_unit
{

/// Implementation of the HAL for the real hardware.
struct HardwareHal final : public HalBase<HardwareHal>
{
    HardwareHal(LoggerType& logger)
        : HalBase(logger)
    {}

    void WriteReg(uint32_t regAddress, uint32_t value)
    {
        *reinterpret_cast<volatile uint32_t*>(regAddress) = value;
    }

    uint32_t ReadReg(uint32_t regAddress)
    {
        return *reinterpret_cast<volatile uint32_t*>(regAddress);
    }

    void WaitForEvents()
    {
        asm("WFE");
    }

    void RaiseIRQ()
    {
        // ensure that all data memory transfers and instructions area completed
        // CU has written to Mailbox before raising an interrupt
        asm("DSB");
        // Using SETIRQ_EXT
        dl1_setirq_ext_r setReg;
        // set the relevant bit to raise an edge sensitive interrupt towards Host
        setReg.set_job(1);
        // write the register
        WriteReg(TOP_REG(DL1_RP, DL1_SETIRQ_EXT), setReg.word);
    }

    void EnableDebug()
    {}
    void DisableDebug()
    {}

    void Nop()
    {
        __builtin_arm_nop();
    }
};

}    // namespace control_unit
}    // namespace ethosn
