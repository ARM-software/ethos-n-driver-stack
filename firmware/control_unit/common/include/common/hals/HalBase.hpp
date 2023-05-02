//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../FirmwareApi.hpp"
#include "../Log.hpp"
#include "../Optimize.hpp"
#include "../Utils.hpp"

#include <scylla_addr_fields.h>
#include <scylla_regs.h>

#include <cstdint>

#define SCYLLA_ARCHITECTURE_BRANCH_ETHOSN78 0x140

namespace ethosn
{
namespace control_unit
{

template <typename DerivedHal>
class HalBase
{
public:
    // Helper class interprets dl2_unit_count_r
    class UnitCountR
    {
    public:
        UnitCountR(const uint32_t value)
            : m_Value(value)
        {}

        operator dl2_unit_count_r() const
        {
            return m_Value;
        }

        Pow2 NumQuad() const
        {
            return Pow2(dl2_unit_count_r(m_Value).get_quad_count());
        }

        Pow2 CesPerQuad() const
        {
            return Pow2(dl2_unit_count_r(m_Value).get_engines_per_quad());
        }

        Pow2 EmcPerCe() const
        {
            return Pow2(dl2_unit_count_r(m_Value).get_dfc_emc_per_engine());
        }

        Pow2 NumCes() const
        {
            return Pow2(NumQuad() * CesPerQuad());
        }

        Pow2 NumEmcs() const
        {
            return Pow2(NumCes() * EmcPerCe());
        }

    private:
        uint32_t m_Value;
    };

    // Methods that DerivedHal must implement

    void WriteReg(const uint32_t regAddress, const uint32_t value)
    {
        static_cast<DerivedHal*>(this)->WriteReg(regAddress, value);
    }

    uint32_t ReadReg(const uint32_t regAddress)
    {
        return static_cast<DerivedHal*>(this)->ReadReg(regAddress);
    }

    void WaitForEvents()
    {
        static_cast<DerivedHal*>(this)->WaitForEvents();
    }

    void RaiseIRQ()
    {
        static_cast<DerivedHal*>(this)->RaiseIRQ();
    }

    // The number of bytes per beat is fixed

    constexpr Pow2 NumBytesPerBeat() const
    {
        /* The DMA controller reads 128 bit words, which is 16 bytes */
        return Pow2(16);
    }

    // Helper 2nd degree methods that use the compulsory methods

    UnitCountR UnitCount()
    {
        return ReadReg(TOP_REG(DL2_RP, DL2_UNIT_COUNT));
    }

    dl2_dfc_features_r DfcFeatures()
    {
        return ReadReg(TOP_REG(DL2_RP, DL2_DFC_FEATURES));
    }

    dl2_mce_features_r MceFeatures()
    {
        return ReadReg(TOP_REG(DL2_RP, DL2_MCE_FEATURES));
    }

    dl2_vector_engine_features_r PleFeatures()
    {
        return ReadReg(TOP_REG(DL2_RP, DL2_VECTOR_ENGINE_FEATURES));
    }

    dl2_wd_features_r WdFeatures()
    {
        return ReadReg(TOP_REG(DL2_RP, DL2_WD_FEATURES));
    }

    dl2_npu_id_r NpuId()
    {
        return ReadReg(TOP_REG(DL2_RP, DL2_NPU_ID));
    }

    // Returns true if hardware variant is N78
    bool IsEthosN78()
    {
        // clang-format off
        uint32_t archBranch = ((dl2_npu_id_r(NpuId()).get_arch_major() << 8) |
                               (dl2_npu_id_r(NpuId()).get_arch_minor() << 4));

        // clang-format on
        return (archBranch == SCYLLA_ARCHITECTURE_BRANCH_ETHOSN78);
    }
    // The total size of the CE SRAM (across all CE's)
    uint32_t SizeCeSram()
    {
        return DfcFeatures().get_dfc_mem_size_per_emc() * UnitCount().NumEmcs();
    }

    // Number of compute engines
    Pow2 NumCes()
    {
        return UnitCount().NumCes();
    }

    // Total number of DFC EMC controllers (CE-SRAMs)
    Pow2 NumEmcs()
    {
        return UnitCount().NumEmcs();
    }

    // Total number of output feature maps generated
    Pow2 NumOfms()
    {
        return NumCes() * OfmPerCe();
    }

    Pow2 IfmGeneratedPerCe()
    {
        return Pow2(MceFeatures().get_ifm_generated_per_engine());
    }

    // Input feature maps consumed per engine
    Pow2 IfmConsumedPerCe()
    {
        return IfmGeneratedPerCe() * NumCes();
    }

    // Number of ple lanes
    Pow2 NumPleLanes()
    {
        if (IsEthosN78())
        {
            return Pow2(PleFeatures().get_ple_lanes());
        }
        else
        {
            // EthosN77, EthosN57 and EthosN37 has only one Ple lane
            return Pow2(1);
        }
    }

    // Output feature maps generated per engine
    Pow2 OfmPerCe()
    {
        return Pow2(MceFeatures().get_ofm_generated_per_engine());
    }

    Pow2 EmcPerCe()
    {
        return UnitCount().EmcPerCe();
    }

    Pow2 PleCodeSramSize()
    {
        return Pow2(4096);
    }

    // Optional methods with a default empty implementation

    void DumpDram(const char*, uint64_t, uint32_t)
    {}

    void DumpSram(const char*)
    {}

    void ClearSram()
    {
        // Set PWRCTLR Active for CEs
        dl1_pwrctlr_r pwrCtl(ReadReg(TOP_REG(DL1_RP, DL1_PWRCTLR)));
        bool powerEnabled = pwrCtl.get_active();
        if (!powerEnabled)
        {
            pwrCtl.set_active(1);
            WriteReg(TOP_REG(DL1_RP, DL1_PWRCTLR), pwrCtl.word);
        }

        // Enable events such we are notified when the SRAM is cleared
        tsu_event_msk_r maskRegOriginal(ReadReg(TOP_REG(TSU_RP, TSU_TSU_EVENT_MSK)));
        bool clearEnabled = maskRegOriginal.get_udma_or_clear_done_mask() == event_mask_t::ENABLED;
        if (!clearEnabled)
        {
            tsu_event_msk_r maskRegEnabled = maskRegOriginal;
            maskRegEnabled.set_udma_or_clear_done_mask(event_mask_t::ENABLED);
            WriteReg(TOP_REG(TSU_RP, TSU_TSU_EVENT_MSK), maskRegEnabled.word);
        }

        uint32_t numEngines = NumCes();
        ce_inst_r ceInstr;
        ceInstr.set_sram_clear(1);
        for (uint32_t ce = 0; ce < numEngines; ++ce)
        {
            // Clear the CE SRAMS one by one to avoid power surge
            WriteReg(CE_REG(ce, CE_RP, CE_CE_INST), ceInstr.word);
            // HW Team recommends adding three NOPs to make sure that SRAM Clear is started at most once every fourth cycle
            // ("WriteReg + 3 * NOP") in order to lower the power ramp-up and avoid power surge
            static_cast<DerivedHal*>(this)->Nop();
            static_cast<DerivedHal*>(this)->Nop();
            static_cast<DerivedHal*>(this)->Nop();
        }

        // Wait for clear srams to finish
        bool isClearDone;
        do
        {
            tsu_event_r tsuEvent(ReadReg(TOP_REG(TSU_RP, TSU_TSU_EVENT)));
            isClearDone = tsuEvent.get_udma_or_clear_done() == event_t::TRIGGERED;
            if (!isClearDone)
            {
                WaitForEvents();
            }
        } while (!isClearDone);

        if (!clearEnabled)
        {
            WriteReg(TOP_REG(TSU_RP, TSU_TSU_EVENT_MSK), maskRegOriginal.word);
        }

        // Restore PWRCTLR Active state
        if (!powerEnabled)
        {
            dl1_pwrctlr_r pwrCtl(ReadReg(TOP_REG(DL1_RP, DL1_PWRCTLR)));
            pwrCtl.set_active(0);
            WriteReg(TOP_REG(DL1_RP, DL1_PWRCTLR), pwrCtl.word);
        }
    }

    LoggerType& m_Logger;

protected:
    // Protected because this class is only meant to be derived from
    HalBase(LoggerType& logger)
        : m_Logger(logger){};
    HalBase(const HalBase&) = default;
    HalBase(HalBase&&)      = default;
    HalBase& operator=(const HalBase&) = default;
    HalBase& operator=(HalBase&&) = default;
};

}    // namespace control_unit
}    // namespace ethosn
