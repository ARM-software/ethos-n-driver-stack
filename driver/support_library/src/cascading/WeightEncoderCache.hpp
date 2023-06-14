//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../../include/ethosn_support_library/Support.hpp"
#include "../Utils.hpp"
#include "../WeightEncoder.hpp"
#include "DebuggingContext.hpp"
#include <ethosn_command_stream/CommandData.hpp>

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ethosn
{
namespace support_library
{

class WeightEncoderCache
{
public:
    WeightEncoderCache(const HardwareCapabilities& caps, DebuggingContext& debuggingContext);

    std::shared_ptr<EncodedWeights> Encode(WeightEncodingRequest&& r);

private:
    struct Hasher
    {
        size_t operator()(const WeightEncodingRequest& r) const;
    };

    const HardwareCapabilities& m_Caps;
    DebuggingContext& m_DebuggingContext;
    std::unordered_map<WeightEncodingRequest, std::shared_ptr<EncodedWeights>, Hasher> m_Entries;
    uint64_t m_MaxUncompressedStripeSize;
};

}    // namespace support_library

}    // namespace ethosn
