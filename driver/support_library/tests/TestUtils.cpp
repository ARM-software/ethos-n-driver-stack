//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "TestUtils.hpp"

#include "../src/CapabilitiesInternal.hpp"
#include "../src/Compiler.hpp"

#include <cstring>

namespace ethosn
{
namespace support_library
{

HardwareCapabilities GetEthosN78HwCapabilities()
{
    FirmwareAndHardwareCapabilities fwHwCapabilities =
        GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, 0);
    return HardwareCapabilities(fwHwCapabilities);
}

HardwareCapabilities GetEthosN78HwCapabilities(EthosNVariant variant, uint32_t sramSizeOverride)
{
    FirmwareAndHardwareCapabilities fwHwCapabilities = GetEthosN78FwHwCapabilities(variant, sramSizeOverride);
    return HardwareCapabilities(fwHwCapabilities);
}

HardwareCapabilities GetEthosN77HwCapabilities()
{
    return HardwareCapabilities(GetEthosN77FwHwCapabilities());
}

HardwareCapabilities GetEthosN57HwCapabilities()
{
    return HardwareCapabilities(GetEthosN57FwHwCapabilities());
}

HardwareCapabilities GetEthosN37HwCapabilities()
{
    return HardwareCapabilities(GetEthosN37FwHwCapabilities());
}

namespace
{

std::vector<char> GetRawCapabilities(const FirmwareAndHardwareCapabilities& fwHwCapabilities)
{
    return std::vector<char>(reinterpret_cast<const char*>(&fwHwCapabilities),
                             reinterpret_cast<const char*>(&fwHwCapabilities) + sizeof(fwHwCapabilities));
}

}    // namespace

std::vector<char> GetRawDefaultCapabilities()
{
    FirmwareAndHardwareCapabilities fwHwCapabilities =
        GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 0);
    return GetRawCapabilities(fwHwCapabilities);
}

std::vector<char> GetRawDefaultEthosN77Capabilities()
{
    FirmwareAndHardwareCapabilities fwHwCapabilities = GetEthosN77FwHwCapabilities();
    return GetRawCapabilities(fwHwCapabilities);
}

std::vector<char> GetRawDefaultEthosN78Capabilities()
{
    FirmwareAndHardwareCapabilities fwHwCapabilities =
        GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, 0);
    return GetRawCapabilities(fwHwCapabilities);
}

bool Contains(const char* string, const char* substring)
{
    return strstr(string, substring) != nullptr;
}

std::vector<uint8_t> GetCommandStreamData(const ethosn::command_stream::CommandStreamBuffer& cmdStream)
{
    std::vector<uint8_t> data;
    const uint8_t* begin = reinterpret_cast<const uint8_t*>(cmdStream.GetData().data());
    const uint8_t* end   = begin + cmdStream.GetData().size() * sizeof(cmdStream.GetData()[0]);
    data.assign(begin, end);
    return data;
}

ethosn::command_stream::CommandStream GetCommandStream(const CompiledNetwork* compiledNetwork)
{
    const CompiledNetworkImpl* cnImpl = static_cast<const CompiledNetworkImpl*>(compiledNetwork);
    auto& cuBufferInfo                = cnImpl->GetConstantControlUnitDataBufferInfos();
    // The command stream buffer id is defined to be 0.
    auto cmdStreamBufferInfo =
        std::find_if(cuBufferInfo.begin(), cuBufferInfo.end(), [](const auto& b) { return b.m_Id == 0; });
    if (cmdStreamBufferInfo == cuBufferInfo.end())
    {
        throw std::exception();
    }

    const uint32_t* begin =
        reinterpret_cast<const uint32_t*>(cnImpl->GetConstantControlUnitData().data() + cmdStreamBufferInfo->m_Offset);
    const uint32_t* end = begin + cmdStreamBufferInfo->m_Size / sizeof(uint32_t);
    return ethosn::command_stream::CommandStream(begin, end);
}

}    // namespace support_library
}    // namespace ethosn
