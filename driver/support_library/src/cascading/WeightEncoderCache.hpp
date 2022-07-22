//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../../include/ethosn_support_library/Support.hpp"
#include "../Utils.hpp"
#include "../WeightEncoder.hpp"
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
    WeightEncoderCache(const HardwareCapabilities& caps, const char* id);

    struct Params
    {
        TensorInfo weightsTensorInfo;
        std::shared_ptr<const std::vector<uint8_t>> weightsData;
        TensorInfo biasTensorInfo;
        std::vector<int32_t> biasData;
        QuantizationInfo inputQuantizationInfo;
        QuantizationInfo outputQuantizationInfo;
        uint32_t stripeDepth;
        uint32_t strideY;
        uint32_t strideX;
        uint32_t paddingTop;
        uint32_t paddingLeft;
        uint32_t iterationSize;
        ethosn::command_stream::MceOperation operation;
        CompilerMceAlgorithm algorithm;

        bool operator==(const Params& r) const;
    };

    std::shared_ptr<EncodedWeights> Encode(const Params& params);

private:
    struct Hasher
    {
        size_t operator()(const Params& p) const;
    };

    const HardwareCapabilities& m_Caps;
    WeightEncoder m_Encoder;
    std::unordered_map<Params, std::shared_ptr<EncodedWeights>, Hasher> m_Entries;
    std::string m_PersistentFilename;
    uint64_t m_MaxUncompressedStripeSize;
};

}    // namespace support_library

}    // namespace ethosn
