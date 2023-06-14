//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "WeightEncoderCache.hpp"

#include <chrono>
#include <fstream>
#include <type_traits>

namespace ethosn
{
namespace support_library
{

WeightEncoderCache::WeightEncoderCache(const HardwareCapabilities& caps)
    : m_Caps(caps)
    , m_MaxUncompressedStripeSize(std::numeric_limits<uint64_t>::max())
    , m_Mutex(std::make_unique<std::mutex>())
{}

bool WeightEncoderCache::CheckUncompressedSize(const WeightEncodingRequest& request)
{
    // There is no point compressing weights with a stripe shape which will not fit into SRAM.
    // For example if the weights are huge and we are trying to encode them all into a single stripe,
    // then the plan that this is used for will never fit into SRAM and so it is a waste of time
    // compressing with that stripe shape. We can't know for certain the size of the compressed
    // stripe until we actually do the compression, but we make the (fairly safe) assumption that
    // there is a correlation between the uncompressed and compressed stripe sizes. Therefore
    // if we previously compressed a stripe of a smaller uncompresed size and that didn't fit,
    // then we assume that this larger uncompressed stripe won't fit either, and so don't even try.
    // We also don't bother compressing at all if the uncompressed size is so big that we don't think it could ever fit
    // (i.e. it would require an unreasonablly high compression ratio)
    const uint64_t uncompressedSize = GetUncompressedWeightStripeSize(request);
    if (uncompressedSize >= m_MaxUncompressedStripeSize || uncompressedSize / 3 > m_Caps.GetTotalSramSize())
    {
        return false;
    }
    return true;
}

std::shared_ptr<ethosn::support_library::EncodedWeights> WeightEncoderCache::Encode(WeightEncodingRequest&& request)
{
    std::lock_guard<std::mutex> lg(*m_Mutex);

    auto it = m_Entries.find(request);
    if (it == m_Entries.end())
    {
        if (!CheckUncompressedSize(request))
        {
            return {};
        }

        g_Logger.Debug("Performance warning - weights weren't pre-cached. Consider adding to PreprocessWeightsAsync.");

        // Kick off stage 1. We'll wait for it immediately below.
        it = EncodeStage1AsyncImpl(std::move(request));
    }

    if (!it->second.m_EncodedWeights)
    {
        // Wait for stage 1 to finish, and do stage 2
        assert(it->second.m_Stage1Future);
        std::unique_ptr<IStage1Results> stage1Results = it->second.m_Stage1Future->Wait();
        EncodedWeights w                              = EncodeWeightsStage2(std::move(stage1Results));

        it->second.m_EncodedWeights = std::make_shared<EncodedWeights>(w);

        // There is no point compressing weights with a stripe shape which will not fit into SRAM.
        // For example if the weights are huge and we are trying to encode them all into a single stripe,
        // then the plan that this is used for will never fit into SRAM and so it is a waste of time
        // compressing with that stripe shape. We can't know for certain the size of the compressed
        // stripe until we actually do the compression, but we make the (fairly safe) assumption that
        // there is a correlation between the uncompressed and compressed stripe sizes. Therefore
        // if we previously compressed a stripe of a smaller uncompresed size and that didn't fit,
        // then we assume that this larger uncompressed stripe won't fit either, and so don't even try.
        const uint64_t uncompressedSize = GetUncompressedWeightStripeSize(request);

        g_Logger.Verbose("Uncompressed size = %lu, compressed size = %u, compression ratio %f", uncompressedSize,
                         it->second.m_EncodedWeights->m_MaxSize,
                         (float)uncompressedSize / (float)it->second.m_EncodedWeights->m_MaxSize);

        // If the compressed stripe won't fit in SRAM, update our threshold.
        if (it->second.m_EncodedWeights->m_MaxSize > m_Caps.GetTotalSramSize())
        {
            m_MaxUncompressedStripeSize = std::min(m_MaxUncompressedStripeSize, uncompressedSize);
            return {};
        }
    }

    return it->second.m_EncodedWeights;
}

void WeightEncoderCache::EncodeStage1Async(WeightEncodingRequest&& request)
{
    std::lock_guard<std::mutex> lg(*m_Mutex);

    EncodeStage1AsyncImpl(std::move(request));
}

WeightEncoderCache::TCacheMap::iterator WeightEncoderCache::EncodeStage1AsyncImpl(WeightEncodingRequest&& request)
{
    auto it = m_Entries.find(request);
    if (it != m_Entries.end())
    {
        // Already cached
        return it;
    }

    if (!CheckUncompressedSize(request))
    {
        return m_Entries.end();
    }

    // Make a copy for storing in our map, as we're going to move the original.
    // Profiling has shown that this copy does not take significant time.
    WeightEncodingRequest requestCopy                  = request;
    std::unique_ptr<IStage1ResultsFuture> stage1Future = EncodeWeightsStage1Async(std::move(request));
    it = m_Entries.insert({ requestCopy, CacheEntry{ std::move(stage1Future), {} } }).first;

    return it;
}

size_t WeightEncoderCache::Hasher::operator()(const WeightEncodingRequest& r) const
{
    // This hash function is deliberately very simple and therefore you might think would lead to lots of
    // collisions. We may now get more hash collisions, which could be an issue as the equality comparison
    // is very expensive. However, so far we've noticed that comparing weights is less expensive than
    // copying them around for each part so this function is good enough.
    size_t h = 17;
    h        = h * 37 + std::hash<size_t>()(r.m_WeightsData->size());
    h        = h * 37 + std::hash<size_t>()(r.m_BiasData.size());
    h        = h * 37 + std::hash<uint32_t>()(r.m_StripeDepth);
    h        = h * 37 + std::hash<uint32_t>()(r.m_IterationSize);
    // Note we cast the enum to an integral type, as some compilers (e.g. aarch64-linux-gnu-g++ 5.3.1)
    // don't support using the enum type directly, even though the spec indicates that they should.
    h = h * 37 + std::hash<uint32_t>()(static_cast<uint32_t>(r.m_Algorithm));
    return h;
}

}    // namespace support_library

}    // namespace ethosn
