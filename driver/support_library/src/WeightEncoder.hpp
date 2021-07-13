//
// Copyright © 2018-2021 Arm Limited.
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

// Currently the weights encoder for fully connected works best with multiple of 1024 input channels.
constexpr uint32_t g_WeightsChannelVecProd = 1024U;

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

class BitstreamWriter;

enum class WeightCompMode
{
    AUTO,
    UNCOMPRESSED,
    DIRECT,
    DIRECT_TRUNC,
    DIRECT_RLE,
    PALETTE,
    PALETTE_TRUNC,
    PALETTE_DIRECT,
    PALETTE_DIRECT_TRUNC,
    PALETTE_RLE,
    PALETTE_TRUNC_RLE,
    PALETTE_DIRECT_RLE,
    PALETTE_DIRECT_TRUNC_RLE,
};

class WeightEncoder
{
public:
    typedef int16_t Weight;
    typedef uint16_t WeightSymbol;
    struct EncodingParams
    {
        uint16_t m_OfmScaleFactor;
        int32_t m_OfmBias;
        uint32_t m_OfmShift;
        uint32_t m_OfmZeroPoint;
        uint32_t m_FilterZeroPoint;
    };
    enum class ZDivisor
    {
        ZDIV_0       = 0,
        ZDIV_1       = 1,
        ZDIV_2       = 2,
        ZDIV_3       = 3,
        RLE_DISABLED = 7
    };

    enum class WDivisor
    {
        WDIV_0       = 0,
        WDIV_1       = 1,
        WDIV_2       = 2,
        WDIV_3       = 3,
        WDIV_4       = 4,
        WDIV_5       = 5,
        UNCOMPRESSED = 7
    };

    struct WeightCompressionParams
    {
        WeightCompressionParams()
            : m_ReloadCompressionParams(true)
            , m_Zdiv(ZDivisor::RLE_DISABLED)
            , m_Wdiv(WDivisor::UNCOMPRESSED)
            , m_TruncationEnabled(false)
            , m_WeightOffset(0)
            , m_PaletteReload(true)
            , m_PaletteBits(7)
            , m_InitialParameters(true)
        {}

        WeightCompressionParams(const EncodingParams& encodingParams)
            : m_EncodingParams(encodingParams)
            , m_ReloadCompressionParams(true)
            , m_Zdiv(ZDivisor::RLE_DISABLED)
            , m_Wdiv(WDivisor::UNCOMPRESSED)
            , m_TruncationEnabled(false)
            , m_WeightOffset(0)
            , m_PaletteReload(true)
            , m_PaletteBits(7)
            , m_InitialParameters(false)
        {}

        EncodingParams m_EncodingParams;
        bool m_ReloadCompressionParams;
        ZDivisor m_Zdiv;
        WDivisor m_Wdiv;
        bool m_TruncationEnabled;
        uint8_t m_WeightOffset;
        bool m_PaletteReload;
        std::vector<uint16_t> m_Palette;
        std::map<Weight, uint8_t> m_InversePalette;
        uint32_t m_PaletteBits;
        bool m_InitialParameters;
    };

    WeightEncoder(const HardwareCapabilities& capabilities);
    WeightEncoder(const HardwareCapabilities& capabilities,
                  WeightCompMode mode,
                  const WeightCompressionParams& params = {});

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
    struct EncodedOfm
    {
        std::vector<uint8_t> m_EncodedWeights;
        uint32_t m_NumOfBits;
    };

    struct GRCSymbol
    {
        uint32_t m_NumQuotientBits;
        uint16_t m_Quotient;
        uint16_t m_Remainder;
    };

    struct WeightSymbolFreqInfo
    {
        std::vector<std::pair<WeightSymbol, uint32_t>> m_SymbolFreqPairs;
        WeightSymbol m_MinSymbol;
        WeightSymbol m_MaxSymbol;
    };

    struct ZeroGroupInfo
    {
        std::vector<uint32_t> m_ZeroGroups;
        uint32_t m_MinGroup;
        uint32_t m_MaxGroup;
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
                                         CompilerMceAlgorithm algorithm) const;

    /// Generate vector of weight compression parameters
    std::vector<std::unique_ptr<WeightCompressionParams>> GenerateCompressionParams(uint32_t numOfmInParallel);

    /// Encodes all the weights required to calculate a single OFM.
    EncodedOfm EncodeOfm(const uint8_t* weightData,
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
                         const EncodingParams& params,
                         std::vector<std::unique_ptr<WeightCompressionParams>>& compressionParams);

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
                                                   uint32_t numOfmsPerSram) const;

    std::vector<std::vector<uint8_t>> MergeStreamsOg(const std::vector<EncodedOfm>& streams,
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

    /// Get OfmShift based on weight encoder version
    uint32_t GetOfmShiftOffset() const;

    // Number of Ofm processed in parallel which is the minimum number of
    // weights streams that need to be loaded at the same time for all the
    // mce interfaces to start producing an Ofm each.
    uint32_t GetNumOfmInParallel(const uint32_t numOfm,
                                 const uint32_t numSrams,
                                 const uint32_t stripeDepth,
                                 const DataFormat dataFormat) const;

    /// Get HWIM encoding parameters
    std::pair<uint32_t, uint32_t> GetHwimWeightPadding(const bool usePadding,
                                                       const uint32_t ifmIdx,
                                                       const uint32_t numIfmsProcessedInParallel) const;

    uint8_t WeightOffsetClamp(WeightSymbol offset) const
    {
        constexpr uint8_t maxWeightOffset = 31;
        return static_cast<uint8_t>(offset < maxWeightOffset ? offset : maxWeightOffset);
    }

    /// Get absolute weight value.
    ///
    /// Depending on which compiler and compiler version that is being used, std::abs(short/int16_t/Weight) may
    /// return an int or a double. Therefore, to ensure a consistent behavior this function should be used to
    /// get the absolute weight value.
    ///
    /// @see https://cplusplus.github.io/LWG/issue2735
    ///
    Weight AbsWeight(Weight weight) const
    {
        return static_cast<Weight>(weight < 0 ? -weight : weight);
    }

    WeightSymbol WeightToSymbol(Weight weight) const
    {
        // See Ethos-N78 MCE specification, section 6.8.6.3.2
        return static_cast<WeightSymbol>((AbsWeight(weight) << 1u) - (weight < 0));
    }

    Weight SymbolToWeight(WeightSymbol weightSymbol) const
    {
        uint16_t sign = weightSymbol & 1;
        uint16_t mag  = static_cast<uint16_t>((weightSymbol + 1) >> 1);
        return static_cast<Weight>(sign ? -mag : mag);
    }

    /// Create vector of weight symbol frequency pairs where the DIROFS, Palette size and Palette has
    /// been applied.
    WeightSymbolFreqInfo
        CreateUncompressedSymbolFreqs(const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs,
                                      const std::map<Weight, uint8_t>& inversePalette,
                                      size_t paletteSize,
                                      uint8_t weightOffset) const;

    /// Find the optimal GRC parameters for the specified weight symbol frequency pairs.
    uint32_t FindGRCParams(WeightCompressionParams& params,
                           const WeightSymbolFreqInfo& symbolFreqPairInfo,
                           const WeightSymbolFreqInfo& noPaletteSymbolFreqPairInfo) const;

    /// Create a palette of the specified size
    void CreatePalette(WeightCompressionParams& params,
                       const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs,
                       uint8_t paletteSize,
                       bool palettePadding) const;

    /// Find Palette parameters for the specified weight symbol frequency pairs
    bool FindPaletteParams(WeightCompressionParams& params,
                           const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs) const;

    /// Find the optimal RLE parameters for the specified weights
    uint32_t FindRLEParams(WeightCompressionParams& params, const ZeroGroupInfo& zeroGroupInfo) const;

    /// Find optimal compression parameter for the specified weights
    void FindWeightCompressionParams(WeightCompressionParams& newParams,
                                     const WeightCompressionParams& prevParams,
                                     const std::vector<uint8_t>& weights,
                                     const TensorInfo& weightsTensorInfo) const;

    /// Select compression parameters based through analyzis of the weight stream.
    const WeightCompressionParams SelectWeightCompressionParams(const std::vector<uint8_t>& weights,
                                                                const TensorInfo& weightsTensorInfo,
                                                                const EncodingParams& params,
                                                                const WeightCompressionParams& prevCompParams) const;

    /// Get the size in bytes of the OFM bias.
    uint32_t GetOfmBiasSize(const TensorInfo& weightsTensorInfo) const;

    /// Determine if OFM parameters need to be reloaded.
    bool GetOfmReload(const WeightCompressionParams& compParams,
                      const WeightCompressionParams& prevCompParams,
                      const bool firstOfm) const;

    /// Convert 8-bit weights to 9-bit weights including zero point.
    std::vector<Weight> GetUncompressedWeights(const std::vector<uint8_t>& weights,
                                               const TensorInfo& weightsTensorInfo) const;

    /// Convert 9-bit signed weight to 9-bit unsigned weight symbol.
    WeightSymbol DirectEncode(const Weight weight, const WeightCompressionParams& compParams) const;

    /// Palette or direct encode the uncompressed weight symbol stream.
    void PaletteZrunEncode(const std::vector<Weight>& uncompressedWeights,
                           const WeightCompressionParams& compParams,
                           std::vector<WeightSymbol>& weightSymbols,
                           std::vector<WeightSymbol>& zeroSymbols) const;

    /// Golomb Rice code the palette/direct weight symbol stream and pack the
    /// symbols into chunks.
    void GRCCompressPackChunk(const std::vector<WeightSymbol>& weightSymbols,
                              const std::vector<WeightSymbol>& zeroSymbols,
                              const WeightCompressionParams& compParams,
                              BitstreamWriter& writer) const;

    /// Write the weight stream header. There is exactly one header per OFM.
    void WriteWeightHeader(BitstreamWriter& writer,
                           const uint32_t streamLength,
                           const uint64_t ofmBias,
                           const size_t ofmBiasLength,
                           const bool ofmReload,
                           const uint32_t ofmScaling,
                           const uint32_t ofmShift,
                           const uint32_t ofmZeroPointCorrection) const;

    /// Write the weight payload header. There may be one or multiple payload headers in the weight stream.
    void WritePayloadHeader(BitstreamWriter& writer,
                            const size_t payloadLength,
                            const WeightCompressionParams& compParams);

    const HardwareCapabilities& m_Capabilities;

    WeightCompMode m_Mode;
    WeightCompressionParams m_TestParams;
    const uint32_t m_IfmConsumedPerEnginex3d4;
    const uint32_t m_IfmConsumedPerEngined2;
};

}    // namespace support_library
}    // namespace ethosn
