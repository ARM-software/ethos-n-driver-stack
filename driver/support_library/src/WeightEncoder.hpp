//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "GraphNodes.hpp"
#include "Network.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace ethosn
{
namespace support_library
{

class Constant;
class HardwareCapabilities;
class MceOperationNode;

struct WeightsMetadata
{
    uint32_t m_Offset;
    uint32_t m_Size;
};

struct EncodedWeights
{
    std::vector<WeightsMetadata> m_Metadata;
    uint32_t m_MaxSize;
    std::vector<uint8_t> m_Data;
};

class WeightEncoder
{
public:
    /**
     * Factory function that selects which weight encoder to use based on the hardware capabilities.
     */
    static std::unique_ptr<WeightEncoder> CreateWeightEncoder(const HardwareCapabilities& capabilities);

    WeightEncoder(const HardwareCapabilities& capabilities);

    virtual ~WeightEncoder() = default;

    EncodedWeights Encode(const MceOperationNode& mceOperation,
                          uint32_t stripeDepth,
                          uint32_t stripeSize,
                          const QuantizationInfo& outputQuantizationInfo);

    /// Override which takes data separate to the weight data in mceOperation
    /// This can be used to see the resulting meta data for different weight data
    EncodedWeights Encode(const MceOperationNode& mceOperation,
                          const std::vector<uint8_t>& weightData,
                          uint32_t stripeDepth,
                          uint32_t stripeSize,
                          const QuantizationInfo& outputQuantizationInfo);

    EncodedWeights Encode(const TensorInfo& weightsTensorInfo,
                          const uint8_t* weightsData,
                          const TensorInfo& biasTensorInfo,
                          const int32_t* biasData,
                          const QuantizationInfo& inputQuantizationInfo,
                          const QuantizationInfo& outputQuantizationInfo,
                          uint32_t stripeDepth,
                          uint32_t strideY,
                          uint32_t strideX,
                          uint32_t paddingTop,
                          uint32_t paddingLeft,
                          uint32_t iterationSize,
                          ethosn::command_stream::MceOperation operation,
                          CompilerMceAlgorithm algorithm);

protected:
    struct EncodingParams
    {
        uint16_t m_OfmScaleFactor;
        int32_t m_OfmBias;
        uint32_t m_OfmShift;
        uint32_t m_OfmZeroPoint;
        uint32_t m_FilterZeroPoint;
    };

    struct EncodedOfm
    {
        std::vector<uint8_t> m_EncodedWeights;
        uint32_t m_NumOfBits;
    };

    // Calculates the exact offset and size in DRAM of each weight stripe
    std::vector<WeightsMetadata> CalculateWeightsMetadata(const std::vector<std::vector<uint8_t>>& streamPerStripeOg,
                                                          uint32_t numOgPerStripe) const;

    /// Gets the raw (unencoded) stream for all the weights required to calculate a single OFM.
    std::vector<uint8_t> GetRawOfmStream(const uint8_t* weightData,
                                         uint32_t ofmIdx,
                                         uint32_t iteration,
                                         const TensorInfo& weightsTensorInfo,
                                         uint32_t strideY,
                                         uint32_t strideX,
                                         uint32_t paddingTop,
                                         uint32_t paddingLeft,
                                         uint32_t iterationSize,
                                         ethosn::command_stream::MceOperation operation,
                                         CompilerMceAlgorithm algorithm,
                                         bool prepareForZeroMaskCompression) const;

    /// Encodes all the weights required to calculate a single OFM.
    virtual EncodedOfm EncodeOfm(const uint8_t* weightData,
                                 uint32_t ofmIdx,
                                 uint32_t numOfmInParallel,
                                 uint32_t numIterationsOfm,
                                 uint32_t numOfmSetsPerStripe,
                                 uint32_t iteration,
                                 const TensorInfo& weightsTensorInfo,
                                 uint32_t strideY,
                                 uint32_t strideX,
                                 uint32_t paddingTop,
                                 uint32_t paddingLeft,
                                 uint32_t iterationSize,
                                 ethosn::command_stream::MceOperation operation,
                                 CompilerMceAlgorithm algorithm,
                                 const EncodingParams& params) = 0;

    /// Merges the given streams of data into 'numGroups' groups, using a round-robin allocation of streams to groups.
    /// All the streams in a group are then concatenated together.
    ///
    /// For example, the three streams below (A, B, C) are merged into numGroups=2 groups.
    ///
    ///  A:   | A1 | A2 | A3 |
    ///                                      Group 1 (streams A and C):  | A1 | A2 | A3 | C1 | C2 |
    ///  B:   | B1 | B2 | B3 | B4 |    =>
    ///                                      Group 2 (stream B):         | B1 | B2 | B3 | B4 |
    ///  C:   | C1 | C2 |
    ///
    std::vector<std::vector<uint8_t>> MergeStreams(const std::vector<std::vector<uint8_t>>& streams,
                                                   uint32_t numGroups,
                                                   uint32_t numIterations,
                                                   uint32_t numOfmsPerSram,
                                                   const uint32_t streamHeadersUpdateAlignment) const;

    std::vector<std::vector<uint8_t>> MergeStreamsOg(const std::vector<std::vector<uint8_t>>& streams,
                                                     const std::vector<uint32_t>& streamSize,
                                                     uint32_t numGroups,
                                                     const uint32_t streamHeadersUpdateAlignment) const;

    /// Interleaves the given streams of data by taking 'numBytesPerStream' bytes from each stream in turn.
    /// If some streams are shorter than others then zeroes will be used to pad these to the required length.
    ///
    /// For example, the three streams below (A, B, C) are interleaved with numBytesPerStream=2:
    ///
    ///  A:   | A1 | A2 | A3 |
    ///
    ///  B:   | B1 | B2 | B3 | B4 |    =>   | A1 | A2 | B1 | B2 | C1 | C2 | A3 | 0 | B3 | B4 | 0 | 0 |
    ///
    ///  C:   | C1 | C2 |
    ///
    std::vector<uint8_t> InterleaveStreams(const std::vector<std::vector<uint8_t>>& streams,
                                           uint32_t numBytesPerStream) const;

    /// Reset any OFM encoding parameters that may be saved from previous calls to Encode
    virtual void ResetOfmEncodingParameters() = 0;

    /// Hardware capabilities.
    const HardwareCapabilities& m_Capabilities;
};

}    // namespace support_library
}    // namespace ethosn
