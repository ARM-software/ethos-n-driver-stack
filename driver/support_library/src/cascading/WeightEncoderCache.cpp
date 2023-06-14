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

WeightEncoderCache::WeightEncoderCache(const HardwareCapabilities& caps, DebuggingContext& debuggingContext)
    : m_Caps(caps)
    , m_DebuggingContext(debuggingContext)
    , m_MaxUncompressedStripeSize(std::numeric_limits<uint64_t>::max())
{}

std::shared_ptr<ethosn::support_library::EncodedWeights> WeightEncoderCache::Encode(WeightEncodingRequest&& request)
{
    auto it = m_Entries.find(request);
    if (it == m_Entries.end())
    {
        // There is no point compressing weights with a stripe shape which will not fit into SRAM.
        // For example if the weights are huge and we are trying to encode them all into a single stripe,
        // then the plan that this is used for will never fit into SRAM and so it is a waste of time
        // compressing with that stripe shape. We can't know for certain the size of the compressed
        // stripe until we actually do the compression, but we make the (fairly safe) assumption that
        // there is a correlation between the uncompressed and compressed stripe sizes. Therefore
        // if we previously compressed a stripe of a smaller uncompresed size and that didn't fit,
        // then we assume that this larger uncompressed stripe won't fit either, and so don't even try.
        const uint64_t uncompressedSize = static_cast<uint64_t>(request.m_WeightsTensorInfo.m_Dimensions[0]) *
                                          request.m_WeightsTensorInfo.m_Dimensions[1] * request.m_IterationSize *
                                          request.m_StripeDepth;
        if (uncompressedSize >= m_MaxUncompressedStripeSize)
        {
            return {};
        }

        g_Logger.Debug("Encode %lu weights, stripeDepth = %u, iterationSize = %u, algorithm = %s...",
                       request.m_WeightsData->size(), request.m_StripeDepth, request.m_IterationSize,
                       ToString(request.m_Algorithm).c_str());
        auto startTime = std::chrono::high_resolution_clock::now();

        // Make a copy for storing in our map, as we're going to move the original.
        // Profiling has shown that this copy does not take significant time.
        WeightEncodingRequest requestCopy = request;
        EncodedWeights w                  = EncodeWeights(std::move(request));
        it = m_Entries.insert({ requestCopy, std::make_shared<EncodedWeights>(w) }).first;

        auto duration = std::chrono::high_resolution_clock::now() - startTime;
        g_Logger.Debug("...%llu ms", duration.count() / (1000ULL * 1000ULL));

        m_DebuggingContext.m_TotalWeightCompressionTime += duration.count();

        // If the compressed stripe won't fit in SRAM, update our threshold.
        // Note that we do this after saving to the file cache, even though these weights won't be used,
        // because otherwise future compilations would need to repeat this encoding only to figure out
        // that it won't fit.
        if (it->second->m_MaxSize > m_Caps.GetTotalSramSize())
        {
            m_MaxUncompressedStripeSize = std::min(m_MaxUncompressedStripeSize, uncompressedSize);
            return {};
        }
    }

    return it->second;
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
