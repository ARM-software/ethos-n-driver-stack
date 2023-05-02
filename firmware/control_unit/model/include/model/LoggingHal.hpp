//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/Log.hpp>
#include <common/Utils.hpp>
#include <common/hals/HalBase.hpp>

#include <scylla_regs.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <ios>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace ethosn
{
namespace control_unit
{

static constexpr uint32_t g_NumBanks = 2;

class LoggingHal final : public HalBase<LoggingHal>
{
public:
    struct Options
    {
        enum class EthosNVariant
        {
            N78_1TOPS_2PLE_RATIO,
            N78_4TOPS_4PLE_RATIO,
        };

        Options()
            : m_PleWaitsForGlobalStripeBankControl(true)
            , m_EthosNVariant(EthosNVariant::N78_4TOPS_4PLE_RATIO)
            , m_NumCE(8)
        {}
        bool m_PleWaitsForGlobalStripeBankControl;
        EthosNVariant m_EthosNVariant;
        uint32_t m_NumCE;
    };

    struct Entry
    {
        enum
        {
            WriteReg,
            ReadReg,
            WaitForEvents,
        } m_Type;
        uint32_t m_Data1;
        uint32_t m_Data2;

        bool operator==(const Entry& other) const
        {
            return m_Type == other.m_Type && m_Data1 == other.m_Data1 && m_Data2 == other.m_Data2;
        }

        bool operator!=(const Entry& other) const
        {
            return !(*this == other);
        }
    };

    LoggingHal(const Options& options)
        : HalBase(m_Logger)
        , m_Logger({ &LogSink })
        , m_Options(options)
        , m_ActiveTsuEvents(0)
        , m_StripeBankStatus(0)
        , m_DmaStatus(0)
        , m_NumStripeDoneEvents(0)
        , m_PerCeRegisters()
        , m_Entries()
        , m_ClearSramRequestStatus(0)
    {
        static_assert(g_NumBanks == 2, "Only two banks are supported");

        // Set the number of free slots for read and write to all free
        m_DmaStatus.set_rd_cmdq_free(4);
        m_DmaStatus.set_wr_cmdq_free(4);

        m_PerCeRegisters.resize(NumCes());
    }

    const Options& GetOptions() const
    {
        return m_Options;
    }

    void WriteReg(uint32_t regAddress, uint32_t value)
    {
        scylla_top_addr addr(regAddress);
        m_Entries.push_back({ Entry::WriteReg, regAddress, value });

        // We emulate roughly what a real HAL would do, by intercepting register reads and writes:
        //   * Preserve some register values, e.g. PLE scratch registers
        //   * Simulate events that would be raised, e.g. stripe_done

        // First deal with global (non-CE) registers.
        if (regAddress == TOP_REG(GLOBAL_RP, GLOBAL_STRIPE_BANK_CONTROL))
        {
            // if starting stripe, set layer/stripe/block done signals immediately
            stripe_bank_control_r stripeBankControl(value);
            if (stripeBankControl.bits.start && m_Options.m_PleWaitsForGlobalStripeBankControl)
            {
                m_ActiveTsuEvents.set_ple_layer_done(event_t::TRIGGERED);
                m_ActiveTsuEvents.set_ple_stripe_done(event_t::TRIGGERED);
                m_ActiveTsuEvents.set_ple_block_done(event_t::TRIGGERED);
                // For stripe done, we need to keep track of the number of stripes in flight to not clear
                // the bit prematurely.
                ++m_NumStripeDoneEvents;
                uint32_t currBank = m_StripeBankStatus.get_current_bank();
                // Flip the current bank
                m_StripeBankStatus.set_current_bank((currBank + 1) % g_NumBanks);
            }
        }
        else if (regAddress == TOP_REG(DMA_RP, DMA_DMA_RD_CMD))
        {
            // if starting a DMA, set dma done signal immediately
            m_ActiveTsuEvents.set_dma_done(event_t::TRIGGERED);
            dma_rd_cmd_r rdCmd(value);
            m_DmaStatus.set_last_rd_id_completed(rdCmd.get_rd_id());
        }
        else if (regAddress == TOP_REG(DMA_RP, DMA_DMA_WR_CMD))
        {
            // if starting a DMA, set dma done signal immediately
            m_ActiveTsuEvents.set_dma_done(event_t::TRIGGERED);
            dma_wr_cmd_r wrCmd(value);
            m_DmaStatus.set_last_wr_id_completed(wrCmd.get_wr_id());
        }
        else if (addr.get_reg_page() == CE_RP)
        {
            // Per-CE registers - these can be targeted at a specific CE or have the broadcast bit set.
            if (addr.get_page_offset() == CE_PLE_CONTROL_0 && addr.get_b())
            {
                // if requesting soft reset of ple, set layer/stripe/block done signals immediately
                ple_control_0_r pleControl(value);
                if (!pleControl.bits.cpuwait && !m_Options.m_PleWaitsForGlobalStripeBankControl)
                {
                    m_ActiveTsuEvents.set_ple_layer_done(event_t::TRIGGERED);
                    m_ActiveTsuEvents.set_ple_stripe_done(event_t::TRIGGERED);
                    m_ActiveTsuEvents.set_ple_block_done(event_t::TRIGGERED);
                    m_NumStripeDoneEvents += 1;
                }
            }
            else if (addr.get_page_offset() == CE_CE_INST && addr.get_b())
            {
                // if starting a clearing SRAM, set set_udma_done_clear_done signal immediately
                m_ActiveTsuEvents.set_udma_or_clear_done(event_t::TRIGGERED);
            }
            else if (addr.get_page_offset() == CE_CE_INST && (addr.get_b() == 0))
            {
                // If starting a clearing SRAM for individual CE, mark respective CE as started
                m_ClearSramRequestStatus |= (0x1u << addr.get_ce());
                if (m_ClearSramRequestStatus == (0x1u << m_Options.m_NumCE) - 1)
                {
                    // When all CE's have been "cleared" set_udma_done_clear_done signal
                    m_ActiveTsuEvents.set_udma_or_clear_done(event_t::TRIGGERED);
                }
            }
            else if (addr.get_page_offset() == CE_PLE_UDMA_LOAD_COMMAND && addr.get_b())
            {
                // if starting a uDMA, set udma done signal immediately
                m_ActiveTsuEvents.set_udma_or_clear_done(event_t::TRIGGERED);
            }
            else if (addr.get_page_offset() >= CE_PLE_SCRATCH0 && addr.get_page_offset() <= CE_PLE_SCRATCH7)
            {
                // Store PLE scratch registers so they can be read back later. Make sure to honour the broadcast flag.
                for (uint32_t ce = 0; ce < NumCes(); ce++)
                {
                    if (ce == addr.get_ce() || addr.get_b())
                    {
                        m_PerCeRegisters[ce][addr.get_page_offset()] = value;
                    }
                }
            }
            else if (addr.get_page_offset() == CE_PLE_SETIRQ && addr.get_b())
            {
                ple_setirq_r pleSetIrq(value);
                if (pleSetIrq.bits.nmi && !m_Options.m_PleWaitsForGlobalStripeBankControl)
                {
                    m_ActiveTsuEvents.set_ple_layer_done(event_t::TRIGGERED);
                    m_ActiveTsuEvents.set_ple_stripe_done(event_t::TRIGGERED);
                    m_ActiveTsuEvents.set_ple_block_done(event_t::TRIGGERED);
                    m_NumStripeDoneEvents += 1;
                }
            }
        }
    }

    uint32_t ReadReg(uint32_t regAddress)
    {
        uint32_t retValue = 0;
        scylla_top_addr addr(regAddress);
        // First deal with global (non-CE) registers.
        if (regAddress == TOP_REG(TSU_RP, TSU_TSU_EVENT))
        {
            // Reading from the event register automatically clears all outstanding events
            uint32_t events = m_ActiveTsuEvents.word;

            // For stripe done, the bit should only be cleared once the counter has gone down to zero
            if (m_NumStripeDoneEvents > 0)
            {
                --m_NumStripeDoneEvents;
            }

            // Clear all events
            m_ActiveTsuEvents.word = 0;
            // Set stripe done
            m_ActiveTsuEvents.set_ple_stripe_done(m_NumStripeDoneEvents > 0 ? event_t::TRIGGERED
                                                                            : event_t::UNTRIGGERED);

            retValue = events;
        }
        else if (regAddress == TOP_REG(TSU_RP, TSU_STRIPE_BANK_STATUS))
        {
            retValue = m_StripeBankStatus.word;
        }
        else if (regAddress == TOP_REG(DMA_RP, DMA_DMA_STATUS))
        {
            retValue = m_DmaStatus.word;
        }
        else if (regAddress == TOP_REG(DL2_RP, DL2_NPU_ID))
        {
            dl2_npu_id_r scylla_id;

            // Reflecting actual ARCH numbers
            scylla_id.set_arch_major(NPU_ARCH_VERSION_MAJOR);
            scylla_id.set_arch_minor(NPU_ARCH_VERSION_MINOR);
            scylla_id.set_arch_rev(NPU_ARCH_VERSION_PATCH);
            scylla_id.set_product_major(0);
            retValue = scylla_id.word;
        }
        else if (regAddress == TOP_REG(DL2_RP, DL2_UNIT_COUNT))
        {
            dl2_unit_count_r uCount;

            switch (m_Options.m_EthosNVariant)
            {
                case Options::EthosNVariant::N78_1TOPS_2PLE_RATIO:
                    uCount.set_quad_count(1);
                    uCount.set_engines_per_quad(2);
                    uCount.set_dfc_emc_per_engine(4);
                    break;
                case Options::EthosNVariant::N78_4TOPS_4PLE_RATIO:
                    uCount.set_quad_count(4);
                    uCount.set_engines_per_quad(2);
                    uCount.set_dfc_emc_per_engine(2);
                    break;
                default:
                    assert(false);
            }

            retValue = uCount.word;
        }
        else if (regAddress == TOP_REG(DL2_RP, DL2_VECTOR_ENGINE_FEATURES))
        {
            dl2_vector_engine_features_r features;

            switch (m_Options.m_EthosNVariant)
            {
                case Options::EthosNVariant::N78_1TOPS_2PLE_RATIO:
                    features.set_ple_lanes(1);
                    break;
                case Options::EthosNVariant::N78_4TOPS_4PLE_RATIO:
                    features.set_ple_lanes(2);
                    break;
                default:
                    assert(false);
            }

            retValue = features.word;
        }
        else if (regAddress == TOP_REG(DL2_RP, DL2_DFC_FEATURES))
        {
            dl2_dfc_features_r features;
            features.set_dfc_mem_size_per_emc(64U << 10);    // 64K
            features.set_bank_count(8);
            retValue = features.word;
        }
        else if (regAddress == TOP_REG(DL2_RP, DL2_MCE_FEATURES))
        {
            dl2_mce_features_r features;
            features.set_mce_num_macs(16);
            features.set_mce_num_acc(64);

            switch (m_Options.m_EthosNVariant)
            {
                case Options::EthosNVariant::N78_1TOPS_2PLE_RATIO:
                    features.set_ifm_generated_per_engine(4);
                    features.set_ofm_generated_per_engine(4);
                    break;
                case Options::EthosNVariant::N78_4TOPS_4PLE_RATIO:
                    features.set_ifm_generated_per_engine(2);
                    features.set_ofm_generated_per_engine(2);
                    break;

                default:
                    assert(false);
            }

            retValue = features.word;
        }
        else if (regAddress == TOP_REG(DL2_RP, DL2_INT_STATUS))
        {
            dl2_int_status_r int_status_scp1;
            int_status_scp1.set_rxev_evnt(1);

            retValue = int_status_scp1.word;
        }
        else if (addr.get_reg_page() == CE_RP)
        {
            // Per-CE registers
            if (addr.get_page_offset() == CE_PLE_CONTROL_0)
            {
                ple_control_0_r pleCtl;
                pleCtl.set_cpuwait(1);
                retValue = pleCtl.word;
            }
            else if (addr.get_page_offset() >= CE_PLE_SCRATCH0 && addr.get_page_offset() <= CE_PLE_SCRATCH7)
            {
                const std::map<uint32_t, uint32_t>& ceRegs         = m_PerCeRegisters[addr.get_ce()];
                std::map<uint32_t, uint32_t>::const_iterator regIt = ceRegs.find(addr.get_page_offset());
                if (regIt != ceRegs.end())
                {
                    retValue = regIt->second;
                }
            }
            else if (addr.get_page_offset() == CE_CE_STATUS)
            {
                const uint32_t ce     = addr.get_ce();
                const uint32_t status = m_PerCeRegisters[ce][CE_CE_STATUS];

                m_PerCeRegisters[ce][CE_CE_STATUS] = 0;

                retValue = status;
            }
        }

        m_Entries.push_back({ Entry::ReadReg, regAddress, retValue });
        return retValue;
    }

    void WaitForEvents()
    {
        m_Entries.push_back({ Entry::WaitForEvents, 0, 0 });
    }

    void RaiseIRQ()
    {}

    void SetMcuTxEv()
    {
        // This part simulate TSU received ple sev event and set mcu_txev to triggered.
        for (uint32_t ce = 0; ce < NumCes(); ++ce)
        {
            ce_status_r status;
            status.set_mcu_txev(event_t::TRIGGERED);
            m_PerCeRegisters[ce][CE_CE_STATUS] = status.word;
        }
    }

    const std::vector<Entry>& GetEntries() const
    {
        return m_Entries;
    }

    void ClearEntries()
    {
        m_Entries.clear();
    }

    uint32_t GetFinalValue(uint32_t regAddress) const
    {
        uint32_t result = 0;
        for (const Entry& e : m_Entries)
        {
            if (e.m_Type == Entry::WriteReg && e.m_Data1 == regAddress)
            {
                result = e.m_Data2;
            }
        }
        return result;
    }

    void EnableDebug()
    {}
    void DisableDebug()
    {}
    void Nop()
    {}

    LoggerType m_Logger;

private:
    Options m_Options;
    tsu_event_r m_ActiveTsuEvents;
    stripe_bank_status_r m_StripeBankStatus;
    dma_status_r m_DmaStatus;
    uint32_t m_NumStripeDoneEvents;
    std::vector<std::map<uint32_t, uint32_t>> m_PerCeRegisters;
    std::vector<Entry> m_Entries;
    uint32_t m_ClearSramRequestStatus;
};

inline std::ostream& operator<<(std::ostream& os, LoggingHal::Entry const& value)
{
    std::ios::fmtflags f(os.flags());
    switch (value.m_Type)
    {
        case LoggingHal::Entry::ReadReg:
            os << std::hex << "{ ReadReg " << utils::GetRegisterName(value.m_Data1) << " " << value.m_Data2 << " }";
            break;
        case LoggingHal::Entry::WriteReg:
            os << std::hex << "{ WriteReg " << utils::GetRegisterName(value.m_Data1) << " " << value.m_Data2 << " }";
            break;
        case LoggingHal::Entry::WaitForEvents:
            os << std::hex << "{ WaitForEvents " << value.m_Data1 << " " << value.m_Data2 << " }";
            break;
        default:
            assert(false);
    }
    os.flags(f);
    return os;
}

}    // namespace control_unit
}    // namespace ethosn
