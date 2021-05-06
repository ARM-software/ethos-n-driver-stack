//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../../include/ethosn_support_library/Support.hpp"
#include "../Utils.hpp"
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
    WeightEncoderCache(const HardwareCapabilities& caps)
        : m_Encoder(WeightEncoder::CreateWeightEncoder(caps))
    {}

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

        bool operator==(const Params& r) const
        {
            return weightsTensorInfo == r.weightsTensorInfo && *weightsData == *r.weightsData &&
                   biasTensorInfo == r.biasTensorInfo && biasData == r.biasData &&
                   inputQuantizationInfo == r.inputQuantizationInfo &&
                   outputQuantizationInfo == r.outputQuantizationInfo && stripeDepth == r.stripeDepth &&
                   strideY == r.strideY && strideX == r.strideX && paddingTop == r.paddingTop &&
                   paddingLeft == r.paddingLeft && iterationSize == r.iterationSize && operation == r.operation &&
                   algorithm == r.algorithm;
        }
    };

    std::shared_ptr<EncodedWeights> Encode(const Params& params)
    {
        auto it = m_Entries.find(params);
        if (it == m_Entries.end())
        {
            EncodedWeights w =
                m_Encoder->Encode(params.weightsTensorInfo, params.weightsData->data(), params.biasTensorInfo,
                                  params.biasData.data(), params.inputQuantizationInfo, params.outputQuantizationInfo,
                                  params.stripeDepth, params.strideY, params.strideX, params.paddingTop,
                                  params.paddingLeft, params.iterationSize, params.operation, params.algorithm);
            it = m_Entries.insert({ params, std::make_shared<EncodedWeights>(w) }).first;
        }
        return it->second;
    }

private:
    struct Hasher
    {
        size_t operator()(const Params& p) const
        {
            // This hash function is deliberately very simple and therefore you might think would lead to lots of
            // collisions. We may now get more hash collisions, which could be an issue as the equality comparison
            // is very expensive. However, so far we've noticed that comparing weights is less expensive than
            // copying them around for each part so this function is good enough.
            size_t h = 17;
            h        = h * 37 + std::hash<size_t>()(p.weightsData->size());
            h        = h * 37 + std::hash<size_t>()(p.biasData.size());
            h        = h * 37 + std::hash<uint32_t>()(p.stripeDepth);
            h        = h * 37 + std::hash<uint32_t>()(p.iterationSize);
            // Note we cast the enum to an integral type, as some compilers (e.g. aarch64-linux-gnu-g++ 5.3.1)
            // don't support using the enum type directly, even though the spec indicates that they should.
            h = h * 37 + std::hash<uint32_t>()(static_cast<uint32_t>(p.algorithm));
            return h;
        }
    };

    std::unique_ptr<WeightEncoder> m_Encoder;
    std::unordered_map<Params, std::shared_ptr<EncodedWeights>, Hasher> m_Entries;
};

}    // namespace support_library

}    // namespace ethosn
