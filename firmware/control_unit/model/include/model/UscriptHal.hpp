//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/Utils.hpp>
#include <common/hals/HalBase.hpp>

#include <scylla_addr_fields.h>
#include <scylla_regs.h>

#include <cinttypes>
#include <fstream>
#include <ios>
#include <iostream>
#include <ostream>
#include <sstream>
#include <utility>
#include <vector>

namespace ethosn
{
namespace control_unit
{

template <typename HAL>
class UscriptHal final : public HalBase<UscriptHal<HAL>>
{
public:
    UscriptHal(HAL& hal, const char* fileName, bool useFriendlyRegNames)
        : HalBase<UscriptHal<HAL>>(hal.m_Logger)
        , m_Hal(hal)
        , m_UseFriendlyRegNames(useFriendlyRegNames)
    {
        dl1_npu_id_r scyllaId = hal.ReadReg(TOP_REG(DL1_RP, DL1_NPU_ID));

        // Binary mode is required to avoid inserting \r characters on Windows
        m_FileStream.open(fileName, std::ios::binary);
        ASSERT_MSG(m_FileStream.is_open(), "File not opened");
        // uScripts should always start by specifying the arch version, product name and a reset command
        m_FileStream << "ARCH " << scyllaId.get_arch_major() << '.' << scyllaId.get_arch_minor() << '.'
                     << scyllaId.get_arch_rev() << std::endl;

        switch (scyllaId.get_product_major())
        {
            case 0:
                m_FileStream << "PRODUCT N78" << std::endl;
                break;
            default:
                ASSERT_MSG(false, "Does not recognize product id: %" PRIu32, scyllaId.get_product_major());
                break;
        }
        m_FileStream << "RESET" << std::endl;
        ASSERT_MSG(!m_FileStream.fail(), "File write failed");
    }

    ~UscriptHal()
    {
        m_FileStream.close();
        ASSERT_MSG(!m_FileStream.fail(), "File not closed");
    }

    UscriptHal(UscriptHal const& other) = delete;
    UscriptHal& operator=(UscriptHal const& other) = delete;

    /// Emits an instruction to load the given hex dump file at the given DRAM address.
    void RecordDramLoad(uint32_t dramAddress, std::string filename)
    {
        m_FileStream << "LOAD_MEM " << std::move(filename) << " " << Data2Hex(dramAddress) << std::endl;
        ASSERT_MSG(!m_FileStream.fail(), "File write failed");
    }

    /// Emits an instruction to dump the given range of DRAM to the given file.
    void DumpDram(const char* filename, uint64_t dramAddress, uint32_t dramSize)
    {
        m_FileStream << "DUMP_MEM " << Data2Hex(dramAddress) << " " << Data2Hex(dramAddress + dramSize) << " > "
                     << filename << std::endl;
        ASSERT_MSG(!m_FileStream.fail(), "File write failed");
        m_Hal.DumpDram(filename, dramAddress, dramSize);
    }

    /// Emits an instruction the entire SRAM of each CE to the given file prefix.
    void DumpSram(const char* prefix)
    {
        m_FileStream << "DUMP_SRAM > " << prefix << std::endl;
        ASSERT_MSG(!m_FileStream.fail(), "File write failed");
        m_Hal.DumpSram(prefix);
    }

    void WriteReg(uint32_t regAddress, uint32_t value)
    {
        std::string addrString = m_UseFriendlyRegNames ? utils::GetRegisterName(regAddress) : Data2Hex(regAddress);
        m_FileStream << "WRITEREG " << addrString << " " << Data2Hex(value) << std::endl;
        ASSERT_MSG(!m_FileStream.fail(), "File write failed");

        if (regAddress == TOP_REG(CE_RP, CE_PLE_UDMA_LOAD_COMMAND))
        {
            // We dont have a mechanism of waiting for the udma so we add a delay of 50us
            m_FileStream << "WAIT DELAY 50" << std::endl;
            ASSERT_MSG(!m_FileStream.fail(), "File write failed");
        }

        m_Hal.WriteReg(regAddress, value);
    }

    uint32_t ReadReg(uint32_t regAddress)
    {
        // The uScript language has no read register command; do nothing

        return m_Hal.ReadReg(regAddress);
    }

    void WaitForEvents()
    {
        // Although the bennto uscript does have a wait command, it doesn't respect the event mask flags and
        // so won't behave as we want it to

        // If we are waiting for a DMA event only then the RTL implements a special command for this
        tsu_event_msk_r dmaMaskReg(0xFFFFFFFF);
        dmaMaskReg.bits.reserved0     = 0;
        dmaMaskReg.bits.dma_done_mask = 0;
        if (m_Hal.ReadReg(TOP_REG(TSU_RP, TSU_TSU_EVENT_MSK)) == dmaMaskReg.word)
        {
            m_FileStream << "WAIT RD_DMA_DONE <UNUSED>" << std::endl;
        }
        else
        {
            m_FileStream << "WAIT POSEDGE IRQ" << std::endl;
        }
        ASSERT_MSG(!m_FileStream.fail(), "File write failed");

        m_Hal.WaitForEvents();
    }

    void RaiseIRQ()
    {}

    void EnableDebug()
    {
        m_Hal.EnableDebug();
    }
    void DisableDebug()
    {
        m_Hal.DisableDebug();
    }

private:
    /// Convert an integer to a hex-formatted string
    std::string Data2Hex(uint32_t input)
    {
        std::ostringstream ss;
        ss.fill('0');
        ss.width(8);
        ss << std::hex << input;
        ASSERT(!ss.fail());
        return ss.str();
    }

    /// Convert an integer to a hex-formatted string
    std::string Data2Hex(uint64_t input)
    {
        std::ostringstream ss;
        ss.fill('0');
        ss.width(16);
        ss << std::hex << input;
        ASSERT(!ss.fail());
        return ss.str();
    }

    HAL& m_Hal;
    std::ofstream m_FileStream;
    bool m_UseFriendlyRegNames;
};

}    // namespace control_unit
}    // namespace ethosn
