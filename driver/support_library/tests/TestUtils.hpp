//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "../src/Utils.hpp"

#include <ethosn_command_stream/CommandStreamBuffer.hpp>

namespace ethosn
{
namespace support_library
{

HardwareCapabilities GetEthosN78HwCapabilities();
HardwareCapabilities GetEthosN78HwCapabilities(EthosNVariant variant, uint32_t sramSizeOverride = 0);
HardwareCapabilities GetEthosN77HwCapabilities();

std::vector<char> GetRawDefaultCapabilities();
std::vector<char> GetRawDefaultEthosN78Capabilities();
std::vector<char> GetRawEthosN78Capabilities(EthosNVariant variant, uint32_t sramSizeOverride = 0);

bool Contains(const char* string, const char* substring);

std::vector<uint8_t> GetCommandStreamData(const ethosn::command_stream::CommandStreamBuffer& cmdStream);

ethosn::command_stream::CommandStream GetCommandStream(const CompiledNetwork* compiledNetwork);

}    // namespace support_library
}    // namespace ethosn
