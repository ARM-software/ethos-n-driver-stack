//
// Copyright © 2020-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include <ModelFirmwareInterface.h>

#include "include/model/ModelHal.hpp"
#include "include/model/UscriptHal.hpp"

#include <Firmware.hpp>

#include <utility>

namespace ethosn
{
namespace control_unit
{

std::vector<char> GetFirmwareAndHardwareCapabilities(const char* modelOptions)
{
    ModelHal model(ModelHal::CreateWithCmdLineOptions(modelOptions ? modelOptions : ""));
    Firmware<ModelHal> fw(model, 0);
    std::pair<const char*, size_t> caps = fw.GetCapabilities();
    return std::vector<char>(caps.first, caps.first + caps.second);
}

class ModelFirmwareInterface : public IModelFirmwareInterface
{
public:
    ModelFirmwareInterface(const char* modelOptions,
                           const char* uscriptFile,
                           bool uscriptUseFriendlyRegNames,
                           uint64_t pleKernelDataAddr)
        : m_ModelHal(ModelHal::CreateWithCmdLineOptions(modelOptions ? modelOptions : ""))
        , m_UscriptHal(m_ModelHal, uscriptFile, uscriptUseFriendlyRegNames)
        , m_Firmware(m_UscriptHal, pleKernelDataAddr)
    {}

    virtual void RecordDramLoad(uint32_t dramAddress, std::string filename) final
    {
        m_UscriptHal.RecordDramLoad(dramAddress, filename);
    }

    virtual bool LoadDram(uint64_t destAddress, const uint8_t* data, uint64_t size) final
    {
        return bennto_load_mem_array(m_ModelHal.GetBenntoHandle(), data, destAddress, size) == BERROR_OK;
    }

    virtual bool LoadSram(uint32_t ceIdx,
                          uint32_t sramIdxWithinCe,
                          uint64_t destAddressWithinSram,
                          const uint8_t* data,
                          uint64_t size) final
    {
        return bennto_load_sram_array(m_ModelHal.GetBenntoHandle(), ceIdx,
                                      static_cast<bcesram_t>(BCESRAM_CE_SRAM0 + sramIdxWithinCe), data,
                                      destAddressWithinSram, size) == BERROR_OK;
    }

    virtual void ResetAndEnableProfiling(ethosn_firmware_profiling_configuration config) final
    {
        m_Firmware.ResetAndEnableProfiling(config);
    }

    virtual bool RunInference(const std::vector<uint32_t>& inferenceData) final
    {
        control_unit::Inference inference(reinterpret_cast<ethosn_address_t>(inferenceData.data()));
        return m_Firmware.RunInference(inference).success;
    }

    virtual bool DumpDram(uint8_t* dest, uint64_t srcAddress, uint64_t size) final
    {
        return bennto_dump_mem_array(m_ModelHal.GetBenntoHandle(), dest, srcAddress, size) == BERROR_OK;
    }

    virtual bool
        DumpSram(uint8_t* dest, uint32_t ceIdx, uint32_t sramIdxWithinCe, uint64_t srcAddress, uint64_t size) final
    {
        return bennto_dump_sram_array(m_ModelHal.GetBenntoHandle(), ceIdx,
                                      static_cast<bcesram_t>(BCESRAM_CE_SRAM0 + sramIdxWithinCe), dest, srcAddress,
                                      size) == BERROR_OK;
    }

    virtual void DumpSram(const char* prefix) final
    {
        m_ModelHal.DumpSram(prefix);
    }

    virtual uint64_t GetNumDramBytesRead() final
    {
        uint64_t numBytesTransferred = 0;
        if (bennto_get_stat(m_ModelHal.GetBenntoHandle(), BSTAT_DMA_DRAM_RD_BYTES, 0, BCESRAM_COUNT,
                            &numBytesTransferred) != BERROR_OK)
        {
            throw std::runtime_error("Failed to get stats");
        }
        return numBytesTransferred;
    }

private:
    ModelHal m_ModelHal;
    UscriptHal<ModelHal> m_UscriptHal;
    Firmware<UscriptHal<ModelHal>> m_Firmware;
};

std::unique_ptr<IModelFirmwareInterface> IModelFirmwareInterface::Create(const char* modelOptions,
                                                                         const char* uscriptFile,
                                                                         bool uscriptUseFriendlyRegNames,
                                                                         uint64_t pleKernelDataAddr)
{
    return std::make_unique<ModelFirmwareInterface>(modelOptions, uscriptFile, uscriptUseFriendlyRegNames,
                                                    pleKernelDataAddr);
}

}    // namespace control_unit
}    // namespace ethosn
