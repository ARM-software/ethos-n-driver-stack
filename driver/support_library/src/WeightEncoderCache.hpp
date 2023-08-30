//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "DebuggingContext.hpp"
#include "Utils.hpp"
#include "WeightEncoder.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ethosn
{
namespace support_library
{

class WeightEncoderCache
{
public:
    WeightEncoderCache(const HardwareCapabilities& caps, ThreadPool& threadPool);
    WeightEncoderCache(WeightEncoderCache&&) = default;

    std::shared_ptr<EncodedWeights> Encode(WeightEncodingRequest&& r);

    /// Kick off stage 1 encoding and store the future, to speed up future requests to Encode().
    void EncodeStage1Async(WeightEncodingRequest&& request);

private:
    struct Hasher
    {
        size_t operator()(const WeightEncodingRequest& r) const;
    };

    /// Entries in the cache can be fully encoded (m_EncodedWeights non-null),
    /// or we might have kicked off the stage 1 encoding asynchronously and not yet done
    /// stage 2 yet (m_EncodedWeights null), and m_Stage1Future will be valid.
    struct CacheEntry
    {
        std::unique_ptr<IStage1ResultsFuture> m_Stage1Future;
        std::shared_ptr<EncodedWeights> m_EncodedWeights;
    };

    using TCacheMap = std::unordered_map<WeightEncodingRequest, CacheEntry, Hasher>;

    TCacheMap::iterator EncodeStage1AsyncImpl(WeightEncodingRequest&& request);

    /// Checks if the uncompressed size of the given request looks like it would result in
    /// compressed weight stripes that are too big for SRAM, and therefore not worth compressing.
    bool CheckUncompressedSize(const WeightEncodingRequest& request);

    const HardwareCapabilities& m_Caps;
    TCacheMap m_Entries;
    uint64_t m_MaxUncompressedStripeSize;

    /// A mutex is needed as this WeightEncoderCache can be accessed from different worker threads
    /// at the same time, when we are parallelising CalculateSectionsOfAllLengths (even though
    /// each parallel invocation starts from a different part, they might examine the same part
    /// at the same time).
    /// Inside a unique_ptr so it can be moved
    std::unique_ptr<std::mutex> m_Mutex;

    ThreadPool& m_ThreadPool;
};

}    // namespace support_library

}    // namespace ethosn
