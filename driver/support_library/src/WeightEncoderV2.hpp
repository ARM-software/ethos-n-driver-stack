//
// Copyright © 2020-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "GraphNodes.hpp"
#include "Network.hpp"
#include "WeightEncoder.hpp"

#include <cstdint>
#include <deque>
#include <vector>

namespace ethosn
{
namespace support_library
{

class HardwareCapabilities;
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

class WeightEncoderV2 : public WeightEncoder
{
public:
    typedef int16_t Weight;
    typedef uint16_t WeightSymbol;

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

    struct WeightCompressionParamsV2 : public WeightCompressionParams
    {
        WeightCompressionParamsV2()
            : m_ReloadCompressionParams(true)
            , m_Zdiv(ZDivisor::RLE_DISABLED)
            , m_Wdiv(WDivisor::UNCOMPRESSED)
            , m_TruncationEnabled(false)
            , m_WeightOffset(0)
            , m_PaletteReload(true)
            , m_PaletteBits(7)
            , m_InitialParameters(true)
        {}

        WeightCompressionParamsV2(const EncodingParams& encodingParams)
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

    WeightEncoderV2(const HardwareCapabilities& capabilities);

    WeightEncoderV2(const HardwareCapabilities& capabilities,
                    WeightCompMode mode,
                    const WeightCompressionParamsV2& params = {});

protected:
    struct GRCSymbol
    {
        uint32_t m_NumQuotientBits;
        uint16_t m_Quotient;
        uint16_t m_Remainder;
    };

    virtual EncodedOfm EncodeOfm(const uint8_t* weightData,
                                 uint32_t ofmIdx,
                                 uint32_t numOfmInParallel,
                                 uint32_t numIterationsOfm,
                                 uint32_t stripeDepth,
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
                                 std::vector<std::unique_ptr<WeightCompressionParams>>& compressionParams) override;

    virtual std::vector<std::unique_ptr<WeightCompressionParams>>
        GenerateCompressionParams(uint32_t numOfmInParallel) override;

    uint8_t WeightOffsetClamp(WeightSymbol offset) const
    {
        constexpr uint8_t maxWeightOffset = 31;
        return static_cast<uint8_t>(offset < maxWeightOffset ? offset : maxWeightOffset);
    }

    /**
     * Get absolute weight value.
     *
     * Depending on which compiler and compiler version that is being used, std::abs(short/int16_t/Weight) may
     * return an int or a double. Therefore, to ensure a consistent behavior this function should be used to
     * get the absolute weight value.
     *
     * @see https://cplusplus.github.io/LWG/issue2735
     */
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

    /**
     * Create vector of weight symbol frequency pairs where the DIROFS, Palette size and Palette has
     * been applied.
     */
    std::vector<std::pair<WeightSymbol, uint32_t>>
        CreateUncompressedSymbolFreqs(const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs,
                                      const std::map<Weight, uint8_t>& inversePalette,
                                      size_t paletteSize,
                                      uint8_t weightOffset) const;

    /**
     * Find the optimal GRC parameters for the specified weight symbol frequency pairs.
     */
    uint32_t FindGRCParams(WeightCompressionParamsV2& params,
                           const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs,
                           const std::vector<std::pair<WeightSymbol, uint32_t>>& noPaletteSymbolFreqPairs) const;

    /**
     * Create a palette of the specified size
     */
    void CreatePalette(WeightCompressionParamsV2& params,
                       const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs,
                       uint8_t paletteSize,
                       bool palettePadding) const;

    /**
     * Find Palette parameters for the specified weight symbol frequency pairs
     */
    bool FindPaletteParams(WeightCompressionParamsV2& params,
                           const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs) const;

    /**
     * Find the optimal RLE parameters for the specified weights
     */
    uint32_t FindRLEParams(WeightCompressionParamsV2& params, const std::deque<Weight>& weights) const;

    /**
     * Find optimal compression parameter for the specified weights
     */
    void FindWeightCompressionParams(WeightCompressionParamsV2& newParams,
                                     const WeightCompressionParamsV2& prevParams,
                                     const std::deque<Weight>& weights) const;

    /**
     * Select compression parameters based through analyzis of the weight stream.
     */
    const WeightCompressionParamsV2
        SelectWeightCompressionParams(const std::deque<Weight>& weights,
                                      const EncodingParams& params,
                                      const WeightCompressionParamsV2& prevCompParams) const;

    /**
     * Get the size in bytes of the OFM bias.
     */
    uint32_t GetOfmBiasSize(const TensorInfo& weightsTensorInfo) const;

    /**
     * Determin if OFM parameters need to be reloaded.
     */
    bool GetOfmReload(const WeightCompressionParamsV2& compParams,
                      const WeightCompressionParamsV2& prevCompParams,
                      const bool firstOfm) const;

    /**
     * Convert 8-bit weights to 9-bit weights including zero point.
     */
    std::deque<Weight> GetUncompressedWeights(const std::vector<uint8_t>& weights,
                                              const TensorInfo& weightsTensorInfo) const;

    /**
     * Convert 9-bit signed weight to 9-bit unsigned weight symbol.
     */
    WeightSymbol DirectEncode(const Weight weight, const WeightCompressionParamsV2& compParams) const;

    /**
     * Palette or direct encode the uncompressed weight symbol stream.
     */
    void PaletteZrunEncode(const std::deque<Weight>& uncompressedWeights,
                           const WeightCompressionParamsV2& compParams,
                           std::deque<WeightSymbol>& weightSymbols,
                           std::deque<WeightSymbol>& zeroSymbols) const;

    /**
     * Golomb Rice code the palette/direct weight symbol stream and pack the
     * symbols into chunks.
     */
    void GRCCompressPackChunk(const std::deque<WeightSymbol>& weightSymbols,
                              const std::deque<WeightSymbol>& zeroSymbols,
                              const WeightCompressionParamsV2& compParams,
                              BitstreamWriter& writer) const;

    /**
     * Write the weight stream header. There is exactly one header per OFM.
     */
    void WriteWeightHeader(BitstreamWriter& writer,
                           const uint32_t streamLength,
                           const uint64_t ofmBias,
                           const size_t ofmBiasLength,
                           const bool ofmReload,
                           const uint32_t ofmScaling,
                           const uint32_t ofmShift,
                           const uint32_t ofmZeroPointCorrection) const;

    /**
     * Write the weight payload header. There may be one or multiple payload headers in the weight stream.
     */
    void WritePayloadHeader(BitstreamWriter& writer,
                            const size_t payloadLength,
                            const WeightCompressionParamsV2& compParams);
    /**
     *  Weight Encoder V2 has OfmShift of 16
     */
    virtual uint32_t GetOfmShiftOffset() const override;

    /**
     * Number of Ofm processed in parallel which is the minimum number of
     * weights streams that need to be loaded at the same time for all the
     * mce interfaces to start producing an Ofm each.
     */
    virtual uint32_t GetNumOfmInParallel(const uint32_t numOfm,
                                         const uint32_t numSrams,
                                         const uint32_t stripeDepth,
                                         const DataFormat dataFormat) const override;

    /**
     * Get HWIM encoding parameters
     */
    virtual std::pair<uint32_t, uint32_t> GetHwimWeightPadding(
        const bool usePadding, const uint32_t ifmIdx, const uint32_t numIfmsProcessedInParallel) const override;

    WeightCompMode m_Mode;
    WeightCompressionParamsV2 m_TestParams;
    const uint32_t m_IfmConsumedPerEnginex3d4;
    const uint32_t m_IfmConsumedPerEngined2;
};

}    // namespace support_library
}    // namespace ethosn
