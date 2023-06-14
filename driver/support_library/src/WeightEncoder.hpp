//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <vector>

#include "../include/ethosn_support_library/Support.hpp"
#include "Utils.hpp"
#include <ethosn_command_stream/CommandData.hpp>

namespace ethosn
{
namespace support_library
{

extern uint32_t g_NumWeightEncodingsStage1;
extern uint32_t g_NumWeightEncodingsStage2;

// Currently the weights encoder for fully connected works best with multiple of 1024 input channels.
constexpr uint32_t g_WeightsChannelVecProd = 1024U;

class HardwareCapabilities;

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
    bool m_IsWideFilter;
};

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

struct WeightCompressionParams
{
    typedef int16_t Weight;

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

    WeightCompressionParams()
        : m_EncodingParams()
        , m_ReloadCompressionParams(true)
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

/// All the parameters to describe some weights that need encoding and how they should be encoded.
/// This is the input to the weight encoding functions.
struct WeightEncodingRequest
{
    WeightEncodingRequest(const HardwareCapabilities& capabilities);
    WeightEncodingRequest(TensorInfo weightsTensorInfo,
                          std::shared_ptr<const std::vector<uint8_t>> weightsData,
                          TensorInfo biasTensorInfo,
                          std::vector<int32_t> biasData,
                          QuantizationInfo inputQuantizationInfo,
                          QuantizationInfo outputQuantizationInfo,
                          uint32_t stripeDepth,
                          uint32_t strideY,
                          uint32_t strideX,
                          uint32_t paddingTop,
                          uint32_t paddingLeft,
                          uint32_t iterationSize,
                          ethosn::command_stream::MceOperation operation,
                          CompilerMceAlgorithm algorithm,
                          const HardwareCapabilities& capabilities,
                          WeightCompMode mode,
                          const WeightCompressionParams& testParams);

    HardwareCapabilities m_Capabilities;

    TensorInfo m_WeightsTensorInfo;
    std::shared_ptr<const std::vector<uint8_t>> m_WeightsData;
    TensorInfo m_BiasTensorInfo;
    std::vector<int32_t> m_BiasData;
    QuantizationInfo m_InputQuantizationInfo;
    QuantizationInfo m_OutputQuantizationInfo;
    uint32_t m_StripeDepth                           = 0;
    uint32_t m_StrideY                               = 0;
    uint32_t m_StrideX                               = 0;
    uint32_t m_PaddingTop                            = 0;
    uint32_t m_PaddingLeft                           = 0;
    uint32_t m_IterationSize                         = 0;
    ethosn::command_stream::MceOperation m_Operation = ethosn::command_stream::MceOperation::CONVOLUTION;
    CompilerMceAlgorithm m_Algorithm                 = CompilerMceAlgorithm::Direct;

    WeightCompMode m_Mode = WeightCompMode::AUTO;
    WeightCompressionParams m_TestParams;

    bool operator==(const WeightEncodingRequest& r) const;
};

uint64_t GetUncompressedWeightStripeSize(const WeightEncodingRequest& r);

// Weight encoding is split into two stages (stage 1 and stage 2). This is because
// stage 1 can be split in parallel across multiple threads but stage 2 is serial.
// This allows the caller to kick off many different stage 1 encodings in parallel
// and wait for the results later, to increase throughput.

/// Opaque representation of the results of stage 1 encoding.
class IStage1Results
{
public:
    virtual ~IStage1Results() = default;
};

/// Opaque representation of a future (similar to std::future) for the result of stage 1 encoding.
/// Call Wait() to block and obtain the results.
class IStage1ResultsFuture
{
public:
    virtual std::unique_ptr<IStage1Results> Wait() = 0;
    virtual ~IStage1ResultsFuture()                = default;
};

/// Performs both stage 1 and stage 2 encoding.
/// The stage 1 encoding is done internally in parallel using the thread pool, but this can't be parallelised
/// with other stage 1 encodings, so you may want to consider using EncodeWeightsStage1Async instead which
/// doesn't block.
EncodedWeights EncodeWeights(WeightEncodingRequest&& request);

/// Begins performing stage 1 encoding asynchronously using the global thread pool.
/// Call Wait() on the returned future to block and obtain the results, but you can do this after
/// doing other work, to maximise parallelism.
std::unique_ptr<IStage1ResultsFuture> EncodeWeightsStage1Async(WeightEncodingRequest&& request);

/// Performs stage 2 encoding, given the results of the stage 1 encoding.
EncodedWeights EncodeWeightsStage2(std::unique_ptr<IStage1Results> stage1Results);

}    // namespace support_library
}    // namespace ethosn
