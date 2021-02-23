//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "WeightEncoder.hpp"

#include "Compiler.hpp"
#include "GraphNodes.hpp"
#include "SubmapFilter.hpp"
#include "Utils.hpp"
#include "WeightEncoderV2.hpp"

#include <algorithm>
#include <deque>
#include <exception>
#include <iterator>
#include <map>
#include <utility>

namespace ethosn
{
namespace support_library
{

namespace
{
using Weight = WeightEncoderV2::Weight;
template <typename T>
std::deque<Weight>
    ConvertToUncompressedWeights(const T* const weights, const size_t numWeights, const int32_t zeroPoint)
{
    std::deque<Weight> uncompressedWeights;

    const auto correctZeroPoint = [zeroPoint](const T w) { return static_cast<Weight>(w - zeroPoint); };

    std::transform(weights, &weights[numWeights], std::back_inserter(uncompressedWeights), correctZeroPoint);

    return uncompressedWeights;
}
}    // namespace

// BitstreamWriter is a helper class that supports writing packed bitfields into a vector.

class BitstreamWriter
{
public:
    BitstreamWriter();

    // Returns the current write position in the bitstream (in bits)
    size_t GetOffset();

    // Write an element to the stream. Offset specifies where to start writing in the stream.
    void Write(uint8_t elem, int numBits, size_t offset);

    // Write an element to end of the stream.
    void Write(uint8_t elem, int numBits);

    // Write an element to the stream. Offset specifies where to start writing in the stream.
    template <class T>
    void Write(const T* elem, int numBits);

    // Reserve space in the stream by writing 0 bits
    void Reserve(size_t numBits);

    // Returns the stream as a uint8_t vector
    const std::vector<uint8_t>& GetBitstream();

    // Clears the content of the stream and resets the write position
    void Clear();

private:
    std::vector<uint8_t> m_Bitstream;
    size_t m_EndPos;
};

BitstreamWriter::BitstreamWriter()
    : m_EndPos(0)
{}

size_t BitstreamWriter::GetOffset()
{
    return m_EndPos;
}

void BitstreamWriter::Write(uint8_t elem, int numBits, size_t offset)
{
    for (int i = 0; i < numBits; ++i)
    {
        size_t idx = offset / 8;
        int bit    = offset % 8;

        if (idx >= m_Bitstream.size())
        {
            m_Bitstream.push_back(static_cast<uint8_t>((elem >> i) & 1));
        }
        else
        {
            m_Bitstream[idx] = static_cast<uint8_t>(m_Bitstream[idx] | (((elem >> i) & 1) << bit));
        }

        ++offset;
    }

    if (static_cast<size_t>(offset) > m_EndPos)
    {
        m_EndPos = offset;
    }
}

void BitstreamWriter::Write(uint8_t elem, int numBits)
{
    Write(elem, numBits, m_EndPos);
}

template <class T>
void BitstreamWriter::Write(const T* elem, int numBits)
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(elem);

    while (numBits > 0)
    {
        Write(*p, std::min(numBits, 8));

        numBits -= 8;
        ++p;
    }
}

void BitstreamWriter::Reserve(size_t numBits)
{
    size_t i = 0;

    while (i < numBits)
    {
        size_t idx = (m_EndPos + i) / 8;
        if (idx >= static_cast<size_t>(m_Bitstream.size()))
        {
            m_Bitstream.push_back(0);
        }

        i += 8 - ((m_EndPos + i) % 8);
    }

    m_EndPos += numBits;
}

const std::vector<uint8_t>& BitstreamWriter::GetBitstream()
{
    return m_Bitstream;
}

void BitstreamWriter::Clear()
{
    m_Bitstream.clear();
    m_EndPos = 0;
}

/**
 * This is the base class for the different weight compression implementations. Please refer to the MCE specification
 * for a description on how weight compression works. Note that currently only 8-bit weights are supported.
 */
class WeightCompressor
{
public:
    WeightCompressor(std::vector<uint8_t>& result);

    /**
     * Add a weight to the compressed stream. Depending on the compression algorithm, the weights are not always
     * compressed immediately when added to the stream. The user must therefore call Flush before the compressed
     * stream is used.
     */
    virtual void CompressWeight(uint8_t weight) = 0;

    /**
     * Flush the compressed stream. Causes all not yet compressed weights to be compressed and written to the stream.
     */
    virtual void Flush();

protected:
    std::vector<uint8_t>& m_Result;
};

WeightCompressor::WeightCompressor(std::vector<uint8_t>& result)
    : m_Result(result)
{}

void WeightCompressor::Flush()
{}

/**
 * Uncompressed weights.
 */
class DefaultCompressor : public WeightCompressor
{
public:
    DefaultCompressor(std::vector<uint8_t>& result);
    virtual ~DefaultCompressor()
    {}

    void CompressWeight(uint8_t weight);
};

DefaultCompressor::DefaultCompressor(std::vector<uint8_t>& result)
    : WeightCompressor(result)
{}

void DefaultCompressor::CompressWeight(uint8_t weight)
{
    m_Result.push_back(weight);
}

/**
 * Weights compressed using a LUT
 */
class IndexCompressor : public WeightCompressor
{
public:
    IndexCompressor(std::vector<uint8_t>& result, uint32_t indexSize, const std::vector<uint8_t>& lut, bool lutReload);
    virtual ~IndexCompressor()
    {}

    void CompressWeight(uint8_t weight);
    void Flush();

protected:
    uint8_t GetLutIndex(uint8_t weight);

    uint32_t m_BitsPerElement;
    std::vector<uint8_t> m_ReverseLut;

    BitstreamWriter m_Bitstream;
};

uint8_t IndexCompressor::GetLutIndex(uint8_t weight)
{
    return (m_BitsPerElement != 8) ? m_ReverseLut[weight] : weight;
}

IndexCompressor::IndexCompressor(std::vector<uint8_t>& result,
                                 uint32_t indexSize,
                                 const std::vector<uint8_t>& lut,
                                 bool lutReload)
    : WeightCompressor(result)
{
    std::vector<uint8_t> lutUsed(256, 0);

    m_ReverseLut = std::vector<uint8_t>(256);

    // Create reverse Lut for fast weight -> Lut index lookup
    for (size_t i = 0; i < lut.size(); ++i)
    {
        if (lutUsed[lut[i]] == 0)
        {
            m_ReverseLut[lut[i]] = static_cast<uint8_t>(i);
            lutUsed[lut[i]]      = 1;
        }

        if (lutReload)
        {
            m_Bitstream.Write(lut[i], 8);
        }
    }

    /* indexSize == 0 => Lut disabled. Every weight element in the stream is the actual 8-bit weight value
       indexSize == 1 => Lut enabled, each index is 3 bits
       indexSize == 2 => Lut enabled, each index is 4 bits
       indexSize == 3 => Lut enabled, each index is 5 bits */
    m_BitsPerElement = (indexSize != 0) ? indexSize + 2 : 8;
}

void IndexCompressor::CompressWeight(uint8_t weight)
{
    uint8_t index = GetLutIndex(weight);
    m_Bitstream.Write(index, m_BitsPerElement);
}

void IndexCompressor::Flush()
{
    m_Result.insert(m_Result.end(), m_Bitstream.GetBitstream().begin(), m_Bitstream.GetBitstream().end());
    m_Bitstream.Clear();
}

/**
 * Weights compressed using zero compression
 */
class ZeroCompressor : public IndexCompressor
{
public:
    ZeroCompressor(std::vector<uint8_t>& result,
                   uint32_t indexSize,
                   const std::vector<uint8_t>& lut,
                   bool lutReload,
                   const uint8_t zeroPoint,
                   int blockSize);

    virtual void CompressWeight(uint8_t weight);
    void Flush();

protected:
    const int m_BlockSize;

    uint16_t m_Mask;
    int m_NumWeights;
    size_t m_MaskOffset;
    // ZeroPoint can be signed or unsigned 8 bit value but it is always
    // stored as uint8_t.
    uint8_t m_ZeroPoint;
};

ZeroCompressor::ZeroCompressor(std::vector<uint8_t>& result,
                               uint32_t indexSize,
                               const std::vector<uint8_t>& lut,
                               bool lutReload,
                               const uint8_t zeroPoint,
                               int blockSize)
    : IndexCompressor(result, indexSize, lut, lutReload)
    , m_BlockSize(blockSize)
    , m_Mask(0)
    , m_NumWeights(0)
    , m_MaskOffset(0)
    , m_ZeroPoint(zeroPoint)
{}

void ZeroCompressor::CompressWeight(uint8_t weight)
{
    if (m_NumWeights == 0)
    {
        // Start of a new block. Reserve space for the mask
        m_MaskOffset = m_Bitstream.GetOffset();
        m_Bitstream.Reserve(m_BlockSize);
    }

    if (weight != m_ZeroPoint)
    {
        uint8_t index = GetLutIndex(weight);
        m_Bitstream.Write(index, m_BitsPerElement);
        m_Mask = static_cast<uint16_t>(m_Mask | (1 << static_cast<uint16_t>(m_NumWeights)));
    }

    ++m_NumWeights;
    if (m_NumWeights == m_BlockSize)
    {
        // Write the mask to the bitstream
        while (m_Mask != 0)
        {
            m_Bitstream.Write(static_cast<uint8_t>(m_Mask & 0xFF), 8, m_MaskOffset);
            m_MaskOffset += 8;
            m_Mask = static_cast<uint16_t>(m_Mask >> 8);
        }
        m_Mask       = 0;
        m_NumWeights = 0;
    }
}

void ZeroCompressor::Flush()
{
    /* Add zero weights until the current 16 element block has been filled which will cause
       the mask to be written to the stream */
    int numElementsToAdd = (m_BlockSize - m_NumWeights) % m_BlockSize;

    for (int i = 0; i < numElementsToAdd; ++i)
    {
        assert(m_ZeroPoint == static_cast<uint8_t>(m_ZeroPoint));
        CompressWeight(static_cast<uint8_t>(m_ZeroPoint));
    }

    m_Result.insert(m_Result.end(), m_Bitstream.GetBitstream().begin(), m_Bitstream.GetBitstream().end());
    m_Bitstream.Clear();
}

/**
 * Selects and returns a suitable compressor implementation based on the encoding parameters.
 */
static std::shared_ptr<WeightCompressor> CreateWeightCompressor(std::vector<uint8_t>& result,
                                                                uint32_t indexSize,
                                                                const std::vector<uint8_t>& lut,
                                                                bool lutReload,
                                                                bool maskEnable,
                                                                const uint8_t zeroPoint,
                                                                int blockSize)
{
    if (!maskEnable && indexSize > 0)
    {
        return std::make_shared<IndexCompressor>(result, indexSize, lut, lutReload);
    }
    else if (maskEnable)
    {
        return std::make_shared<ZeroCompressor>(result, indexSize, lut, lutReload, zeroPoint, blockSize);
    }

    return std::make_shared<DefaultCompressor>(result);
}

/**
 * Weight encoder for architecture less or equal to v1.2
 */
class WeightEncoderV1 : public WeightEncoder
{
public:
    WeightEncoderV1(const HardwareCapabilities& capabilities);

protected:
    struct WeightCompressionParamsV1 : public WeightCompressionParams
    {
        bool m_MaskEnable;
        bool m_LutReload;
        uint32_t m_IndexSize;
        std::vector<uint8_t> m_Lut;
    };

    virtual std::vector<std::unique_ptr<WeightCompressionParams>>
        GenerateCompressionParams(uint32_t numOfmInParallel) override;

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

    virtual uint32_t GetOfmShiftOffset() const override;

    virtual std::pair<uint32_t, uint32_t> GetHwimWeightPadding(
        const bool usePadding, const uint32_t ifmIdx, const uint32_t numIfmsProcessedInParallel) const override;

    virtual uint32_t GetNumOfmInParallel(const uint32_t numOfm,
                                         const uint32_t numSrams,
                                         const uint32_t stripeDepth,
                                         const DataFormat dataFormat) const override;

    // Analyze the weights for one ofm and choose appropriate compression parameters
    WeightCompressionParamsV1
        ChooseCompressionParameters(const std::vector<uint8_t>& rawWeightsForZeroMaskCompression,
                                    const std::vector<uint8_t>& rawWeightsForNoZeroMaskCompression,
                                    const TensorInfo& weightsTensorInfo) const;
};

WeightEncoderV1::WeightEncoderV1(const HardwareCapabilities& capabilities)
    : WeightEncoder(capabilities)
{}

template <class T>
void insert_back(std::vector<uint8_t>& dst, const T* src, const size_t length)
{
    const uint8_t* s = reinterpret_cast<const uint8_t*>(&src);
    dst.insert(dst.end(), s, s + length);
}

template <class T>
void insert_back(std::vector<uint8_t>& dst, const T& src)
{
    insert_back(dst, &src, sizeof(src));
}

WeightEncoderV2::WeightEncoderV2(const HardwareCapabilities& capabilities)
    : WeightEncoder(capabilities)
    , m_Mode(WeightCompMode::AUTO)
    , m_IfmConsumedPerEnginex3d4((3 * capabilities.GetIgsPerEngine() * capabilities.GetNumberOfEngines()) / 4)
    , m_IfmConsumedPerEngined2((capabilities.GetIgsPerEngine() * capabilities.GetNumberOfEngines()) / 2)
{}

WeightEncoderV2::WeightEncoderV2(const HardwareCapabilities& capabilities,
                                 WeightCompMode mode,
                                 const WeightEncoderV2::WeightCompressionParamsV2& params)
    : WeightEncoder(capabilities)
    , m_Mode(mode)
    , m_TestParams(params)
    , m_IfmConsumedPerEnginex3d4((3 * capabilities.GetIgsPerEngine() * capabilities.GetNumberOfEngines()) / 4)
    , m_IfmConsumedPerEngined2((capabilities.GetIgsPerEngine() * capabilities.GetNumberOfEngines()) / 2)
{}

std::vector<std::unique_ptr<WeightEncoder::WeightCompressionParams>>
    WeightEncoderV2::GenerateCompressionParams(uint32_t numOfmInParallel)
{
    std::vector<std::unique_ptr<WeightCompressionParams>> params(numOfmInParallel);
    std::generate(params.begin(), params.end(), std::make_unique<WeightCompressionParamsV2>);
    return params;
}

WeightEncoder::EncodedOfm
    WeightEncoderV2::EncodeOfm(const uint8_t* weightData,
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
                               std::vector<std::unique_ptr<WeightCompressionParams>>& compressionParams)
{
    uint32_t wdIdx = (ofmIdx % stripeDepth) % numOfmInParallel;

    // Grab a reference to previous compression parameters
    WeightCompressionParamsV2& prevCompParams = static_cast<WeightCompressionParamsV2&>(*compressionParams[wdIdx]);

    if (!prevCompParams.m_InitialParameters)
    {
        if (numIterationsOfm > 1)
        {
            prevCompParams.m_InitialParameters = iteration == 0;
        }

        uint32_t numOfmSetsPerStripe = utils::DivRoundUp(stripeDepth, numOfmInParallel);
        assert(numOfmSetsPerStripe >= 1);

        if ((ofmIdx % stripeDepth) == wdIdx && numOfmSetsPerStripe > 1)
        {
            prevCompParams.m_InitialParameters = true;
        }
    }

    std::vector<uint8_t> weights = GetRawOfmStream(weightData, ofmIdx, iteration, weightsTensorInfo, strideY, strideX,
                                                   paddingTop, paddingLeft, iterationSize, operation, algorithm, false);

    std::deque<Weight> uncompressedWeights = GetUncompressedWeights(weights, weightsTensorInfo);

    const WeightCompressionParamsV2 compParams =
        SelectWeightCompressionParams(uncompressedWeights, params, prevCompParams);

    const uint32_t ofmBiasSize = GetOfmBiasSize(weightsTensorInfo);

    // When using per channel quantization the reload parameter depends on the memory streaming
    // being used. At the moment this information is not available here. Always reload in this case.
    // Example:
    //
    // Number of Ofms : 4
    // Ofm number: 0 1 2 3
    // scale:      a a a b (a, b are numbers)
    // reload:     T F F T (T=True, F=False)
    //
    // Case 1
    // Ofm stripe is full height, full width and full depth
    // Streaming strategy processes Ofms in the order: 0, 1, 2, 3
    // No issue
    //
    // Case 2
    // Ofm stripe is partial height, full width and partial depth
    // Streaming strategy processes Ofms in the order: 0, 1, 0, 1, 2, 3, 2, 3
    // Reload:                                         T  F  T  F  F  T  F  T
    //                                                                   ^
    //                                                       it uses scale "b" of 3 which
    //                                                       is not correct. It should
    //                                                       have reloaded its own scale "a"
    //
    const auto isPerChannelQuantization = weightsTensorInfo.m_QuantizationInfo.GetScales().size() > 1;
    const bool ofmReload =
        isPerChannelQuantization || GetOfmReload(compParams, prevCompParams, ofmIdx < numOfmInParallel);

    BitstreamWriter writer;

    std::deque<WeightSymbol> weightSymbols, zeroSymbols;

    PaletteZrunEncode(uncompressedWeights, compParams, weightSymbols, zeroSymbols);

    // Note the weight stream length will be filled later
    WriteWeightHeader(writer, 0xffff, params.m_OfmBias, ofmBiasSize, ofmReload, params.m_OfmScaleFactor,
                      params.m_OfmShift, params.m_OfmZeroPoint);

    uint32_t pldLen = static_cast<uint32_t>(weightSymbols.size());

    WritePayloadHeader(writer, pldLen, compParams);

    GRCCompressPackChunk(weightSymbols, zeroSymbols, compParams, writer);

    // Remember current compression parameters
    prevCompParams = compParams;

    return { std::move(writer.GetBitstream()), static_cast<uint32_t>(writer.GetOffset()) };
}

uint32_t WeightEncoderV2::GetOfmShiftOffset() const
{
    return 16;
}

uint32_t WeightEncoderV2::GetNumOfmInParallel(const uint32_t numOfm,
                                              const uint32_t numSrams,
                                              const uint32_t stripeDepth,
                                              const DataFormat dataFormat) const
{
    if (dataFormat == DataFormat::HWIO)
    {
        return std::min(numOfm, stripeDepth);
    }
    else
    {
        return std::min(numSrams, stripeDepth);
    }
}

std::pair<uint32_t, uint32_t> WeightEncoderV2::GetHwimWeightPadding(const bool, const uint32_t, const uint32_t) const
{
    return std::make_pair(1, 1);
}

static uint8_t CalcBitWidth(size_t value, uint8_t minWidth)
{
    uint8_t bitwidth = minWidth;
    while ((1ull << bitwidth) <= value)
    {
        ++bitwidth;
    }
    // Nothing in the encoding can have more than 9 bits
    assert(bitwidth <= 9);
    return bitwidth;
}

std::vector<std::pair<WeightEncoderV2::WeightSymbol, uint32_t>> WeightEncoderV2::CreateUncompressedSymbolFreqs(
    const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs,
    const std::map<Weight, uint8_t>& inversePalette,
    size_t paletteSize,
    uint8_t weightOffset) const
{
    std::vector<std::pair<WeightSymbol, uint32_t>> uncompressedSymbolFreqPairs;
    uncompressedSymbolFreqPairs.reserve(symbolFreqPairs.size());

    // Populate the vector with the symbols that should be compressed. If a symbol's weight value
    // can be found in the palette, it is replaced with the palette index. Otherwise, the symbol is
    // offset to generate the final symbol value.
    for (const auto& symbolFreqPair : symbolFreqPairs)
    {
        const Weight weight                          = SymbolToWeight(symbolFreqPair.first);
        std::map<Weight, uint8_t>::const_iterator it = inversePalette.find(weight);
        WeightSymbol uncompressedSymbol;
        if (it != inversePalette.end())
        {
            uncompressedSymbol = it->second;
        }
        else
        {
            uncompressedSymbol = static_cast<WeightSymbol>(symbolFreqPair.first + paletteSize - weightOffset);
        }

        uncompressedSymbolFreqPairs.emplace_back(std::make_pair(uncompressedSymbol, symbolFreqPair.second));
    }

    return uncompressedSymbolFreqPairs;
}

uint32_t
    WeightEncoderV2::FindGRCParams(WeightCompressionParamsV2& params,
                                   const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs,
                                   const std::vector<std::pair<WeightSymbol, uint32_t>>& noPaletteSymbolFreqPairs) const
{
    constexpr uint8_t maxNumQuotientBits = 31;

    // If the no palette vector is not empty, it should be used for the uncompressed bitcost
    const auto& uncompressedSymbolFreqPairs =
        (noPaletteSymbolFreqPairs.empty() ? symbolFreqPairs : noPaletteSymbolFreqPairs);

    // Calculate the bitcost to use uncompressed symbols
    const WeightSymbol maxSymbol =
        std::max_element(uncompressedSymbolFreqPairs.begin(), uncompressedSymbolFreqPairs.end())->first;
    uint8_t symbolBitWidth       = CalcBitWidth(maxSymbol, 2);
    uint32_t uncompressedBitcost = 0;
    for (const auto& symbolFreqPair : uncompressedSymbolFreqPairs)
    {
        uncompressedBitcost += (symbolFreqPair.second * symbolBitWidth);
    }

    // Calculate the bitcost for each WDiv to find the one with the lowest overall bitcost. Use the
    // uncompressed bitcost as the intial best choice to include it in the selection process.
    uint32_t bestBitcost = uncompressedBitcost;
    WDivisor bestWDiv    = WDivisor::UNCOMPRESSED;
    bool truncated       = false;
    for (uint8_t i = 0; i <= static_cast<uint8_t>(WDivisor::WDIV_5); ++i)
    {
        uint32_t bitcost          = 0;
        uint32_t truncatedBitcost = 0;
        bool canTruncate          = (symbolFreqPairs.size() <= 3);
        for (const auto& symbolFreqPair : symbolFreqPairs)
        {
            const uint32_t numQuotientBits = (symbolFreqPair.first >> i);
            canTruncate                    = canTruncate && numQuotientBits < 3;

            if (numQuotientBits > maxNumQuotientBits)
            {
                // Too many quotient bits, skip to next WDiv
                bitcost = UINT32_MAX;
                break;
            }

            // (Number of quotient bits + (trailing zero bit) + (XDIV)) * Number of times the symbol occurs
            bitcost += (numQuotientBits + 1 + i) * symbolFreqPair.second;
            // No trailing zero bit and number of quotient bits is always 2 for truncated
            truncatedBitcost += (2 + i) * symbolFreqPair.second;
        }

        if (canTruncate)
        {
            bitcost = truncatedBitcost;
        }

        if (bitcost < bestBitcost)
        {
            bestBitcost = bitcost;
            bestWDiv    = static_cast<WDivisor>(i);
            truncated   = canTruncate;
        }
    }

    params.m_Wdiv = bestWDiv;
    // Ignore truncated if uncompressed is used
    params.m_TruncationEnabled = truncated && bestWDiv != WDivisor::UNCOMPRESSED;

    return bestBitcost;
}

void WeightEncoderV2::CreatePalette(WeightCompressionParamsV2& params,
                                    const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs,
                                    uint8_t paletteSize,
                                    bool palettePadding) const
{
    // See Ethos-N78 MCE Specification, section 6.8.6.3.4
    std::vector<uint16_t> palette(paletteSize);
    std::map<Weight, uint8_t> inversePalette;
    uint8_t noPaddingSize = static_cast<uint8_t>(paletteSize - palettePadding);

    assert(paletteSize > 0 && paletteSize <= 32);

    WeightSymbol maxSymbol = std::max_element(symbolFreqPairs.begin(), symbolFreqPairs.begin() + noPaddingSize)->first;
    const uint32_t maxWeightMag    = AbsWeight(SymbolToWeight(maxSymbol));
    const uint32_t paletteBitWidth = CalcBitWidth(maxWeightMag, 2) + (maxWeightMag > 1);
    const uint32_t signBitPos      = paletteBitWidth - 1;

    for (uint8_t i = 0; i < noPaddingSize; ++i)
    {
        const Weight weight    = SymbolToWeight(symbolFreqPairs[i].first);
        const uint16_t signMag = static_cast<uint16_t>(AbsWeight(weight) | ((weight < 0) << signBitPos));
        palette[i]             = signMag;
        inversePalette[weight] = i;
    }

    params.m_PaletteBits    = paletteBitWidth - 2;
    params.m_Palette        = std::move(palette);
    params.m_InversePalette = std::move(inversePalette);
}

bool WeightEncoderV2::FindPaletteParams(WeightCompressionParamsV2& params,
                                        const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs) const
{
    // See Ethos-N78 MCE Specification, section 6.8.6.3.4
    constexpr uint8_t maxPaletteSize            = 32;
    constexpr WeightSymbol maxWeightSymbolValue = 511;

    // Determine the initial palette size from how many symbols that are repeated at least once
    uint8_t paletteSize = 0;
    for (const auto& symbolFreqPair : symbolFreqPairs)
    {
        if (symbolFreqPair.second == 1 || ++paletteSize == maxPaletteSize)
        {
            break;
        }
    }

    // No values are repeated so there is no gain from using the palette
    if (paletteSize == 0)
    {
        return false;
    }

    bool palettePadding = false;
    if (paletteSize < 2)
    {
        // If the value is not zero and is repeated more than two times, the
        // overall bitcost will still be better by using the palette so pad the
        // palette with a zero value.
        if (symbolFreqPairs[0].first > 0 && symbolFreqPairs[0].second > 2)
        {
            palettePadding = true;
            paletteSize    = 2;
        }
        else
        {
            return false;
        }
    }

    // Adjust the palette size until all the symbols outside the palette can be represented.
    uint8_t weightOffset        = 0;
    WeightSymbol valueRangeLeft = static_cast<WeightSymbol>(maxWeightSymbolValue - paletteSize);
    do
    {
        paletteSize                        = static_cast<uint8_t>(std::min<WeightSymbol>(paletteSize, valueRangeLeft));
        const uint8_t paletteSizeNoPadding = palettePadding ? static_cast<uint8_t>(paletteSize - 1) : paletteSize;

        // Check if the palette contains all the weight values
        if (paletteSizeNoPadding == symbolFreqPairs.size())
        {
            // RLE must be taken into account when selecting the weight offset.
            weightOffset   = params.m_Zdiv != ZDivisor::RLE_DISABLED;
            valueRangeLeft = maxWeightSymbolValue;
        }
        else
        {
            // Find min and max symbol outside the palette
            const auto minMaxItPair =
                std::minmax_element(symbolFreqPairs.begin() + paletteSizeNoPadding, symbolFreqPairs.end());
            // Use the smallest symbol as offset
            weightOffset                 = WeightOffsetClamp(minMaxItPair.first->first);
            const WeightSymbol maxSymbol = minMaxItPair.second->first;
            // Calculate the value range left after the the highest symbol value outside the palette has
            // been represented
            valueRangeLeft = static_cast<WeightSymbol>(maxWeightSymbolValue - (maxSymbol - weightOffset));
        }
    } while (paletteSize > 2 && paletteSize > valueRangeLeft);

    // If the palette can't contain at least two values don't use it
    if (paletteSize < 2)
    {
        return false;
    }

    params.m_WeightOffset = weightOffset;

    CreatePalette(params, symbolFreqPairs, paletteSize, palettePadding);

    return true;
}

uint32_t WeightEncoderV2::FindRLEParams(WeightCompressionParamsV2& params, const std::deque<Weight>& weights) const
{
    constexpr uint32_t maxNumQuotientBits = 31;
    constexpr uint32_t zDiv3              = static_cast<uint32_t>(ZDivisor::ZDIV_3);

    // Find how the zeroes are grouped among the weights
    std::vector<uint32_t> zeroGroups;
    for (std::deque<Weight>::const_iterator wIt = weights.begin(); wIt != weights.end(); wIt += (wIt != weights.end()))
    {
        uint32_t numZeroes = 0;
        for (; wIt != weights.end() && *wIt == 0; ++wIt)
        {
            ++numZeroes;
        }

        zeroGroups.push_back(numZeroes);
    }

    if (weights.back() != 0)
    {
        zeroGroups.push_back(0);
    }

    // Find the ZDiv with the lowest overall bitcost
    uint32_t bestBitcost = UINT32_MAX;
    ZDivisor bestZDiv    = ZDivisor::ZDIV_0;
    for (uint32_t i = 0; i <= zDiv3; ++i)
    {

        uint32_t sumQuots  = 0;
        uint32_t sumRemain = 0;
        for (uint32_t group : zeroGroups)
        {
            const uint32_t numQuotientBits = (group >> i);
            if (numQuotientBits > maxNumQuotientBits)
            {
                // Too many quotient bits, skip to next ZDiv
                sumQuots = UINT32_MAX;
                break;
            }

            sumQuots += numQuotientBits + 1;
            sumRemain += i;
        }

        if (sumQuots == UINT32_MAX)
        {
            continue;
        }

        // Calculate the total bitcost for the RLE chunk packing with padding
        // See Ethos-N78 MCE Specification, section 6.8.6.3.5
        uint32_t packSize = i < zDiv3 ? m_IfmConsumedPerEnginex3d4 : m_IfmConsumedPerEngined2;
        uint32_t bitcost  = utils::RoundUpToNearestMultiple(sumQuots, packSize) + sumRemain;

        if (bitcost < bestBitcost)
        {
            bestBitcost = bitcost;
            bestZDiv    = static_cast<ZDivisor>(i);
        }
    }

    params.m_Zdiv = bestZDiv;

    return bestBitcost;
}

void WeightEncoderV2::FindWeightCompressionParams(WeightCompressionParamsV2& newParams,
                                                  const WeightCompressionParamsV2& prevParams,
                                                  const std::deque<Weight>& weights) const
{
    std::map<WeightSymbol, uint32_t> symbolFreq;
    for (Weight weight : weights)
    {
        const WeightSymbol symbol = WeightToSymbol(weight);
        symbolFreq[symbol]++;
    }

    // The map is no longer needed so the pairs can be moved to the vector rather than copied.
    std::vector<std::pair<WeightSymbol, uint32_t>> sortedSymbolFreqPairs(std::make_move_iterator(symbolFreq.begin()),
                                                                         std::make_move_iterator(symbolFreq.end()));
    std::sort(sortedSymbolFreqPairs.begin(), sortedSymbolFreqPairs.end(), [](const auto& a, const auto& b) {
        // If two symbols have the same frequency, place the larger symbol first to give it a better
        // chance to be placed in the palette.
        return a.second > b.second || (a.second == b.second && a.first > b.first);
    });

    auto zeroIter = find_if(sortedSymbolFreqPairs.begin(), sortedSymbolFreqPairs.end(),
                            [](const std::pair<WeightSymbol, uint32_t>& e) { return e.first == 0; });

    std::vector<std::pair<uint32_t, WeightCompressionParamsV2>> passCostParamPairs;
    // If there are zero weights, run an extra pass with RLE enabled
    uint32_t numPasses = zeroIter != sortedSymbolFreqPairs.end() ? 2 : 1;
    for (uint32_t pass = 0; pass < numPasses; ++pass)
    {
        WeightCompressionParamsV2 params = newParams;
        uint32_t bitCost                 = 0;

        // Only use RLE for the second pass
        if (pass > 0)
        {
            bitCost += FindRLEParams(params, weights);
            // If there are only zero weights, there is nothing more to do.
            if (sortedSymbolFreqPairs.size() == 1)
            {
                // There are only zero weights so only the ZDivisor will be used. All other compression
                // parameters should stay the same as the previous OFM.
                ZDivisor zDiv              = params.m_Zdiv;
                EncodingParams encParams   = params.m_EncodingParams;
                params                     = prevParams;
                params.m_Zdiv              = zDiv;
                params.m_EncodingParams    = encParams;
                params.m_InitialParameters = false;
                // The palette only needs to be written if this is the initial parameters.
                params.m_PaletteReload = prevParams.m_InitialParameters;

                // If this is not the initial parameters and the same RLE ZDivisor was used for the previous
                // OFM the compression parameters can be reused
                params.m_ReloadCompressionParams =
                    !(prevParams.m_InitialParameters == false && params.m_Zdiv == prevParams.m_Zdiv);
                passCostParamPairs.emplace_back(bitCost, params);
                break;
            }

            // Remove the zero weights from the vector as they are now handled by RLE
            sortedSymbolFreqPairs.erase(zeroIter);
        }

        // Attempt to find palette parameters that fit the weight symbols
        if (!FindPaletteParams(params, sortedSymbolFreqPairs))
        {
            // No palette will be used so find the smallest symbol to use as weight offset
            const Weight minSymbol =
                std::min_element(sortedSymbolFreqPairs.begin(), sortedSymbolFreqPairs.end())->first;
            params.m_WeightOffset = WeightOffsetClamp(minSymbol);
            params.m_PaletteBits  = 0;
        }

        // To be able to find the best GRC params, we first need to create a vector with the final
        // symbols that should be compressed.
        std::vector<std::pair<WeightSymbol, uint32_t>> uncompressedSymbolFreqs = CreateUncompressedSymbolFreqs(
            sortedSymbolFreqPairs, params.m_InversePalette, params.m_Palette.size(), params.m_WeightOffset);

        // If a palette is used and it does not contain all the values, the GRC param finder needs an
        // additional vector where the palette is not used to correctly evaluate the cost of using
        // uncompressed mode.
        std::vector<std::pair<WeightSymbol, uint32_t>> uncompressedNoPaletteSymbolFreqs;
        uint8_t noPaletteOffset = 0;
        // Inverse palette has the actual size without padding
        if (params.m_InversePalette.size() != sortedSymbolFreqPairs.size())
        {
            noPaletteOffset =
                WeightOffsetClamp(std::min_element(sortedSymbolFreqPairs.begin(), sortedSymbolFreqPairs.end())->first);
            uncompressedNoPaletteSymbolFreqs =
                CreateUncompressedSymbolFreqs(sortedSymbolFreqPairs, {}, 0, noPaletteOffset);
        }

        bitCost += FindGRCParams(params, uncompressedSymbolFreqs, uncompressedNoPaletteSymbolFreqs);
        if (params.m_Wdiv == WDivisor::UNCOMPRESSED && !uncompressedNoPaletteSymbolFreqs.empty())
        {
            params.m_Palette.clear();
            params.m_InversePalette.clear();

            // Change to offset without the palette
            params.m_WeightOffset = noPaletteOffset;
            // Calculate the uncompressed bitwidth
            const WeightSymbol maxSymbol =
                std::max_element(uncompressedNoPaletteSymbolFreqs.begin(), uncompressedNoPaletteSymbolFreqs.end())
                    ->first;
            params.m_PaletteBits = CalcBitWidth(maxSymbol, 2) - 2;
        }

        params.m_PaletteReload =
            !(prevParams.m_InitialParameters == false && params.m_Palette == prevParams.m_Palette &&
              params.m_PaletteBits == prevParams.m_PaletteBits);

        if (params.m_PaletteReload && params.m_Palette.size() > 0)
        {
            bitCost += static_cast<uint32_t>((params.m_PaletteBits + 2) * params.m_Palette.size());
        }

        params.m_ReloadCompressionParams =
            !(params.m_PaletteReload == false && params.m_Zdiv == prevParams.m_Zdiv &&
              params.m_Wdiv == prevParams.m_Wdiv && params.m_TruncationEnabled == prevParams.m_TruncationEnabled &&
              params.m_WeightOffset == prevParams.m_WeightOffset);

        passCostParamPairs.emplace_back(bitCost, params);
    }

    // Get the params with the lowest cost
    auto min_cost_cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
    newParams         = std::min_element(passCostParamPairs.begin(), passCostParamPairs.end(), min_cost_cmp)->second;
}

const WeightEncoderV2::WeightCompressionParamsV2
    WeightEncoderV2::SelectWeightCompressionParams(const std::deque<Weight>& weights,
                                                   const EncodingParams& encodingParams,
                                                   const WeightCompressionParamsV2& prevCompParams) const
{
    WeightCompressionParamsV2 params(encodingParams);

    switch (m_Mode)
    {
        case WeightCompMode::UNCOMPRESSED:
            assert(params.m_Wdiv == WDivisor::UNCOMPRESSED);
            assert(params.m_Zdiv == ZDivisor::RLE_DISABLED);
            assert(params.m_Palette.size() == 0);
            break;
        case WeightCompMode::DIRECT_RLE:
            params.m_Wdiv         = m_TestParams.m_Wdiv;
            params.m_Zdiv         = m_TestParams.m_Zdiv;
            params.m_WeightOffset = 1;
            break;
        case WeightCompMode::DIRECT_TRUNC:
            params.m_TruncationEnabled = true;
            params.m_Wdiv              = m_TestParams.m_Wdiv;
            break;
        case WeightCompMode::DIRECT:
            params.m_Wdiv = m_TestParams.m_Wdiv;
            assert(params.m_Zdiv == ZDivisor::RLE_DISABLED);
            break;
        case WeightCompMode::PALETTE_RLE:
            //FALLTHRU
        case WeightCompMode::PALETTE_DIRECT_RLE:
            params.m_WeightOffset = 1;
            //FALLTHRU
        case WeightCompMode::PALETTE:
            //FALLTHRU
        case WeightCompMode::PALETTE_DIRECT:
            params.m_Wdiv = m_TestParams.m_Wdiv;
            // sanity check WDIV != 7 for palette direct modes
            assert(params.m_Wdiv != WDivisor::UNCOMPRESSED ||
                   ((m_Mode != WeightCompMode::PALETTE_DIRECT) && (m_Mode != WeightCompMode::PALETTE_DIRECT_RLE)));
            params.m_Zdiv              = m_TestParams.m_Zdiv;
            params.m_TruncationEnabled = false;
            params.m_Palette           = m_TestParams.m_Palette;
            params.m_InversePalette    = m_TestParams.m_InversePalette;
            params.m_PaletteBits       = m_TestParams.m_PaletteBits;
            break;
        case WeightCompMode::PALETTE_DIRECT_TRUNC_RLE:
            params.m_WeightOffset = 1;
            //FALLTHRU
        case WeightCompMode::PALETTE_TRUNC_RLE:
            params.m_TruncationEnabled = true;
            //FALLTHRU
        case WeightCompMode::PALETTE_TRUNC:
            //FALLTHRU
        case WeightCompMode::PALETTE_DIRECT_TRUNC:
            params.m_Wdiv              = m_TestParams.m_Wdiv;
            params.m_Zdiv              = m_TestParams.m_Zdiv;
            params.m_TruncationEnabled = true;
            params.m_Palette           = m_TestParams.m_Palette;
            params.m_InversePalette    = m_TestParams.m_InversePalette;
            params.m_PaletteBits       = m_TestParams.m_PaletteBits;
            break;
        case WeightCompMode::AUTO:
            FindWeightCompressionParams(params, prevCompParams, weights);
            break;
        default:
            throw NotSupportedException("Unsupported weight compression mode");
            break;
    }

    return params;
}

uint32_t WeightEncoderV2::GetOfmBiasSize(const TensorInfo& weightsTensorInfo) const
{
    // See Ethos-N78 MCE Specification, section 6.8.6.2.2
    uint32_t ofmBiasSize = 3;

    switch (weightsTensorInfo.m_DataType)
    {
        case DataType::UINT8_QUANTIZED:
        case DataType::INT8_QUANTIZED:
            ofmBiasSize += 1;
            break;
        case DataType::INT32_QUANTIZED:
            ofmBiasSize += 4;
            break;
        default:
            throw NotSupportedException("Unsupported weight data type");
    }

    return ofmBiasSize;
}

bool WeightEncoderV2::GetOfmReload(const WeightCompressionParamsV2& compParams,
                                   const WeightCompressionParamsV2& prevCompParams,
                                   const bool firstOfm) const
{
    // If this is the first OFM, then we shall always reload the OFM parameters
    if (firstOfm)
    {
        return true;
    }

    // Reload OFM if the scale factor has changed
    if (compParams.m_EncodingParams.m_OfmScaleFactor != prevCompParams.m_EncodingParams.m_OfmScaleFactor)
    {
        return true;
    }

    // Reload OFM if the shift length has changed
    if (compParams.m_EncodingParams.m_OfmShift != prevCompParams.m_EncodingParams.m_OfmShift)
    {
        return true;
    }

    // Reload OFM if the zero point has changed
    if (compParams.m_EncodingParams.m_OfmZeroPoint != prevCompParams.m_EncodingParams.m_OfmZeroPoint)
    {
        return true;
    }

    return false;
}

std::deque<WeightEncoderV2::Weight> WeightEncoderV2::GetUncompressedWeights(const std::vector<uint8_t>& weights,
                                                                            const TensorInfo& weightsTensorInfo) const
{
    switch (weightsTensorInfo.m_DataType)
    {
        case DataType::UINT8_QUANTIZED:
            return ConvertToUncompressedWeights(weights.data(), weights.size(),
                                                weightsTensorInfo.m_QuantizationInfo.GetZeroPoint());
        case DataType::INT8_QUANTIZED:
            return ConvertToUncompressedWeights(reinterpret_cast<const int8_t*>(weights.data()), weights.size(),
                                                weightsTensorInfo.m_QuantizationInfo.GetZeroPoint());
        default:
        {
            std::string errorMessage = "Error in " + std::string(__func__) + ": DataType not yet supported";
            throw std::invalid_argument(errorMessage);
        }
    }
}

WeightEncoderV2::WeightSymbol WeightEncoderV2::DirectEncode(const Weight weight,
                                                            const WeightCompressionParamsV2& compParams) const
{
    WeightSymbol x = WeightToSymbol(weight);

    x = static_cast<WeightSymbol>(x + compParams.m_Palette.size());

    assert(compParams.m_WeightOffset >= 1 || compParams.m_Zdiv == ZDivisor::RLE_DISABLED);

    assert(x >= compParams.m_WeightOffset);
    x = static_cast<WeightSymbol>(x - compParams.m_WeightOffset);

    assert(x >= compParams.m_Palette.size());

    return x;
}

void WeightEncoderV2::PaletteZrunEncode(const std::deque<WeightEncoderV2::Weight>& uncompressedWeights,
                                        const WeightCompressionParamsV2& compParams,
                                        std::deque<WeightSymbol>& weightSymbols,
                                        std::deque<WeightSymbol>& zeroSymbols) const
{
    // Please refer to Ethos-N78 MCE specification, section 6.8.6.3.2
    const std::map<Weight, uint8_t>& invPalette = compParams.m_InversePalette;

    std::deque<Weight>::const_iterator wItr;
    uint32_t zeroCnt = 0;

    wItr = uncompressedWeights.begin();
    while (wItr != uncompressedWeights.end())
    {
        if (compParams.m_Zdiv != ZDivisor::RLE_DISABLED)
        {
            // RLE enabled, counts the number of consecutive 0s
            for (; wItr != uncompressedWeights.end() && *wItr == 0; ++wItr)
            {
                ++zeroCnt;
            }
        }

        Weight value = 0;
        // load next weight if not reaching the end
        if (wItr != uncompressedWeights.end())
        {
            value = *wItr++;
        }
        else
        {
            break;
        }

        if (compParams.m_Zdiv != ZDivisor::RLE_DISABLED)
        {
            // After encountering a non zero symbol, writes
            // accumulated RLE symbol then resets the RLE.
            zeroSymbols.push_back(static_cast<WeightSymbol>(zeroCnt));
            zeroCnt = 0;
        }

        // sanity check: non-zero weight if RLE
        assert(value != 0 || compParams.m_Zdiv == ZDivisor::RLE_DISABLED);

        // Search for symbol in palette (using the weight as the key)
        std::map<Weight, uint8_t>::const_iterator itr = invPalette.find(value);
        WeightSymbol x;

        // If found, then replace weight symbol with palette index
        if (itr != invPalette.end())
        {
            x = static_cast<WeightSymbol>(itr->second);
        }
        else
        {
            x = DirectEncode(value, compParams);
        }

        // writes non-zero symbol
        weightSymbols.push_back(x);
    }

    if (compParams.m_Zdiv != ZDivisor::RLE_DISABLED)
    {
        zeroSymbols.push_back(static_cast<WeightSymbol>(zeroCnt));
    }

    assert((zeroSymbols.size() == (weightSymbols.size() + 1)) || (compParams.m_Zdiv == ZDivisor::RLE_DISABLED));
}

void WeightEncoderV2::GRCCompressPackChunk(const std::deque<WeightSymbol>& weightSymbols,
                                           const std::deque<WeightSymbol>& zeroSymbols,
                                           const WeightCompressionParamsV2& compParams,
                                           BitstreamWriter& writer) const
{

    bool unCompressed = compParams.m_Wdiv == WDivisor::UNCOMPRESSED;
    bool rleEnabled   = compParams.m_Zdiv != ZDivisor::RLE_DISABLED;

    // GRC divisor for weight symbols
    int32_t wDivisor = static_cast<int32_t>(compParams.m_Wdiv);

    if (unCompressed)
    {
        if (compParams.m_Palette.size() == 0)
        {
            wDivisor = static_cast<int32_t>(compParams.m_PaletteBits + 2);
        }
        else
        {
            // <Palette vector size> - 1 because we want the bit width of the max index
            wDivisor = CalcBitWidth(compParams.m_Palette.size() - 1, 1);
        }
    }

    // GRC divisor for zero runs symbols
    int32_t zDivisor = static_cast<int32_t>(compParams.m_Zdiv);

    int32_t nWeights = static_cast<uint32_t>(weightSymbols.size());
    int32_t nZeros   = static_cast<uint32_t>(zeroSymbols.size());

    // weight and zero symbol positions used for flow control by bit stream packing
    int32_t wPos = 0;
    int32_t zPos = 0;

    int32_t wUnary0    = 0;
    int32_t wUnary1    = 0;
    int32_t wUnary1Len = 0;
    int32_t wQuot      = -1;
    int32_t wRmd       = 0;
    int32_t zUnary     = 0;
    int32_t zQuot      = -1;
    int32_t zRmd       = 0;
    int32_t zUnaryLen  = (zDivisor < 3) ? static_cast<int32_t>(m_IfmConsumedPerEnginex3d4)
                                       : static_cast<int32_t>(m_IfmConsumedPerEngined2);

    int32_t j;

    constexpr uint32_t numRmdEntries = 2;

    uint32_t rmdIdx     = 0;
    uint32_t rmdPrevIdx = 1;
    std::vector<std::vector<int32_t>> wRemain(numRmdEntries);
    std::vector<std::vector<int32_t>> zRemain(numRmdEntries);

    int32_t prevWenable = 0;
    int32_t prevZenable = 0;

    do
    {
        // See Ethos-N78 MCE specification, section 6.8.6.3.5
        int32_t balance = rleEnabled ? wPos - zPos : 0;
        bool wEnable    = (balance < static_cast<int32_t>(m_IfmConsumedPerEngined2)) && (wPos < nWeights);
        bool zEnable    = balance >= 0 && rleEnabled && zPos < nZeros;

        // maximum number of weight symbols
        int32_t maxNumWunary0Bits = (unCompressed && (wDivisor > 5)) ? static_cast<int32_t>(m_IfmConsumedPerEngined2)
                                                                     : static_cast<int32_t>(m_IfmConsumedPerEnginex3d4);

        if (wEnable)
        {
            // Encode chunk (weights)

            j          = 0;
            wUnary0    = 0;
            wUnary1    = 0;
            wUnary1Len = 0;

            assert(zRemain[rmdIdx].empty() == true);
            assert(wRemain[rmdIdx].empty() == true);

            while (j < maxNumWunary0Bits)
            {
                if (wQuot < 0)
                {
                    if (wPos < nWeights)
                    {
                        // GRC step 1: quotient and remainder

                        int32_t value = weightSymbols[wPos];

                        assert(value < 512);

                        wQuot = value >> wDivisor;
                        wRmd  = value & ((1 << wDivisor) - 1);

                        // sanity check. The search algorithm ensure quotient <= 31
                        assert(wQuot <= 31 && (!compParams.m_TruncationEnabled || wQuot <= 2));
                    }
                    else
                    {
                        wQuot = 0;
                        wRmd  = -1;    // don't send remainder
                    }
                }

                while (wQuot >= 0 && j < maxNumWunary0Bits)
                {
                    // encodes quotient and remainder

                    wUnary0 |= (wQuot > 0) ? (1 << j) : 0;

                    if (wQuot > 0)
                    {
                        // if w_unary0[j] = 1, then the next weight symbol GRC quotient bit
                        // is put in the first unused position of w_unary1
                        // otherwise in the w_nary0[j+1]
                        wUnary1 |= (wQuot > 1) ? (1 << wUnary1Len) : 0;
                        ++wUnary1Len;
                    }

                    ++j;
                    // Reduces the wQuot after emitting two bits
                    wQuot -= 2;

                    if (compParams.m_TruncationEnabled)
                    {
                        // truncation mode: no more q-bits after emitting two.
                        wQuot = -1;
                    }
                }

                if (wQuot < 0 && wRmd >= 0)
                {
                    wRemain[rmdIdx].push_back(wRmd);
                    ++wPos;
                }
            }
        }

        if (zEnable)
        {
            // Encode chunk (zero runs)

            j      = 0;
            zUnary = 0;
            assert(zRemain[rmdIdx].empty() == true);

            while (j < zUnaryLen)
            {
                if (zQuot < 0)
                {
                    if (zPos < nZeros)
                    {
                        int32_t value = zeroSymbols[zPos];
                        zQuot         = value >> zDivisor;
                        zRmd          = value & ((1 << zDivisor) - 1);
                    }
                    else
                    {
                        zQuot = 0;
                        zRmd  = -1;
                    }
                }

                // emitting zQuot bits
                while (zQuot >= 0 && j < zUnaryLen)
                {
                    zUnary |= zQuot > 0 ? (1 << j) : 0;
                    ++j;
                    --zQuot;
                }

                if (zQuot < 0 && zRmd >= 0)
                {
                    zRemain[rmdIdx].push_back(zRmd);
                    ++zPos;
                }
            }
        }

        // Write chunk to bitstream

        if (wEnable && !unCompressed)
        {
            writer.Write(reinterpret_cast<const uint8_t*>(&wUnary0), maxNumWunary0Bits);
        }

        if (zEnable)
        {
            writer.Write(reinterpret_cast<const uint8_t*>(&zUnary), zUnaryLen);
        }

        if (wEnable && !unCompressed)
        {
            writer.Write(reinterpret_cast<const uint8_t*>(&wUnary1), wUnary1Len);
        }

        if (!wRemain[rmdPrevIdx].empty())
        {
            std::vector<int32_t>::iterator it;
            for (it = wRemain[rmdPrevIdx].begin(); it != wRemain[rmdPrevIdx].end(); ++it)
            {
                assert(*it <= 31 || unCompressed);
                int32_t value = *it;
                writer.Write(reinterpret_cast<const uint8_t*>(&value), wDivisor);
            }

            wRemain[rmdPrevIdx].clear();
        }

        if (!zRemain[rmdPrevIdx].empty())
        {
            std::vector<int32_t>::iterator it;
            for (it = zRemain[rmdPrevIdx].begin(); it != zRemain[rmdPrevIdx].end(); ++it)
            {
                assert(*it <= 7);
                writer.Write(static_cast<uint8_t>(*it), zDivisor);
            }

            zRemain[rmdPrevIdx].clear();
        }

        rmdIdx     = (rmdIdx + 1) % numRmdEntries;
        rmdPrevIdx = (rmdPrevIdx + 1) % numRmdEntries;

        prevWenable = wEnable;
        prevZenable = zEnable;

    } while (prevWenable || prevZenable);
}

void WeightEncoderV2::WriteWeightHeader(BitstreamWriter& writer,
                                        const uint32_t streamLength,
                                        const uint64_t ofmBias,
                                        const size_t ofmBiasLength,
                                        const bool ofmReload,
                                        const uint32_t ofmScaling,
                                        const uint32_t ofmShift,
                                        const uint32_t ofmZeroPointCorrection) const
{
    // See Ethos-N78 MCE Specification, section 6.8.6.2.2
    writer.Write(&streamLength, 16);
    writer.Write(&ofmBias, static_cast<int>(ofmBiasLength) * 8);
    writer.Write(&ofmReload, 1);

    if (ofmReload)
    {
        writer.Write(&ofmScaling, 16);
        writer.Write(&ofmShift, 6);
        writer.Write(&ofmZeroPointCorrection, 8);
    }
}

void WeightEncoderV2::WritePayloadHeader(BitstreamWriter& writer,
                                         const size_t payloadLength,
                                         const WeightCompressionParamsV2& compParams)
{
    // See Ethos-N78 MCE Specification, section 6.8.6.3.3
    writer.Write(&payloadLength, 17);
    writer.Write(&compParams.m_ReloadCompressionParams, 1);

    if (compParams.m_ReloadCompressionParams)
    {
        writer.Write(&compParams.m_Zdiv, 3);
        writer.Write(&compParams.m_Wdiv, 3);
        writer.Write(&compParams.m_TruncationEnabled, 1);
        writer.Write(compParams.m_WeightOffset, 5);
        writer.Write(&compParams.m_PaletteReload, 1);

        if (compParams.m_PaletteReload)
        {
            const size_t paletteSize = compParams.m_Palette.empty() ? 0 : compParams.m_Palette.size() - 1;
            writer.Write(&paletteSize, 5);
            writer.Write(&compParams.m_PaletteBits, 3);

            std::vector<uint16_t>::const_iterator itr;
            for (itr = compParams.m_Palette.begin(); itr != compParams.m_Palette.end(); ++itr)
            {
                Weight value = *itr;
                writer.Write(&value, compParams.m_PaletteBits + 2);
            }
        }
    }
}

/*
 * Weight encoder base class
 */
std::unique_ptr<WeightEncoder> WeightEncoder::CreateWeightEncoder(const HardwareCapabilities& capabilities)
{
    const uint32_t version = capabilities.GetWeightCompressionVersion();

    if (version == 0)
    {
        return std::make_unique<WeightEncoderV1>(capabilities);
    }
    else if (version == 1)
    {
        return std::make_unique<WeightEncoderV2>(capabilities);
    }
    else
    {
        throw VersionMismatchException(std::string("Unsupported weight compressor version: ") +
                                       std::to_string(version));
    }
}

WeightEncoder::WeightEncoder(const HardwareCapabilities& capabilities)
    : m_Capabilities(capabilities)
{}

EncodedWeights WeightEncoder::Encode(const MceOperationNode& mceOperation,
                                     uint32_t stripeDepth,
                                     uint32_t stripeSize,
                                     const QuantizationInfo& outputQuantizationInfo)
{
    // clang-format off
    return Encode(mceOperation.GetWeightsInfo(),
                  static_cast<const uint8_t*>(mceOperation.GetWeightsData().data()),
                  mceOperation.GetBiasInfo(),
                  mceOperation.GetBiasData().data(),
                  mceOperation.GetInputQuantizationInfo(0),
                  outputQuantizationInfo,
                  stripeDepth,
                  mceOperation.GetStride().m_Y,
                  mceOperation.GetStride().m_X,
                  mceOperation.GetMceData().m_PadTop(),
                  mceOperation.GetMceData().m_PadLeft(),
                  stripeSize,
                  mceOperation.GetMceData().m_Operation(),
                  mceOperation.GetAlgorithm());
    // clang-format on
}

EncodedWeights WeightEncoder::Encode(const MceOperationNode& mceOperation,
                                     const std::vector<uint8_t>& weightData,
                                     uint32_t stripeDepth,
                                     uint32_t stripeSize,
                                     const QuantizationInfo& outputQuantizationInfo)
{
    // clang-format off
    return Encode(mceOperation.GetWeightsInfo(),
                  static_cast<const uint8_t*>(weightData.data()),
                  mceOperation.GetBiasInfo(),
                  mceOperation.GetBiasData().data(),
                  mceOperation.GetInputQuantizationInfo(0),
                  outputQuantizationInfo,
                  stripeDepth,
                  mceOperation.GetStride().m_Y,
                  mceOperation.GetStride().m_X,
                  mceOperation.GetMceData().m_PadTop(),
                  mceOperation.GetMceData().m_PadLeft(),
                  stripeSize,
                  mceOperation.GetMceData().m_Operation(),
                  mceOperation.GetAlgorithm());
    // clang-format on
}

EncodedWeights WeightEncoder::Encode(const TensorInfo& weightsTensorInfo,
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
                                     CompilerMceAlgorithm algorithm)
{
    ETHOSN_UNUSED(biasTensorInfo);
    assert(stripeDepth > 0);
    assert(iterationSize > 0);

    uint32_t numOfms = 0;
    if (weightsTensorInfo.m_DataFormat == DataFormat::HWIO)
    {
        numOfms = weightsTensorInfo.m_Dimensions[3];
    }
    else if (weightsTensorInfo.m_DataFormat == DataFormat::HWIM)
    {
        numOfms = weightsTensorInfo.m_Dimensions[2] * weightsTensorInfo.m_Dimensions[3];
    }
    else
    {
        assert(false);
    }

    // Bias dimensions should be valid
    assert((biasTensorInfo.m_Dimensions[0] * biasTensorInfo.m_Dimensions[1] * biasTensorInfo.m_Dimensions[2] == 1) &&
           biasTensorInfo.m_Dimensions[3] == numOfms);

    // Zero point value should be within allowed range
    const utils::DataTypeRange zeroPointBounds = utils::GetRangeOfDataType(weightsTensorInfo.m_DataType);
    ETHOSN_UNUSED(zeroPointBounds);
    assert(weightsTensorInfo.m_QuantizationInfo.GetZeroPoint() <= zeroPointBounds.max &&
           weightsTensorInfo.m_QuantizationInfo.GetZeroPoint() >= zeroPointBounds.min);

    uint32_t ifmChannels = weightsTensorInfo.m_Dimensions[2] * strideX * strideY;
    uint32_t numIterationsOfm =
        weightsTensorInfo.m_DataFormat == DataFormat::HWIM ? 1 : utils::DivRoundUp(ifmChannels, iterationSize);

    // Number of Ofm processed in parallel which is the minimum number of
    // weights streams that need to be loaded at the same time for all the
    // mce interfaces to start producing an Ofm each.
    uint32_t numSrams       = m_Capabilities.GetNumberOfSrams();
    uint32_t numOfmsPerSram = m_Capabilities.GetNumberOfOgs() / numSrams;

    // The number of OFMs that can be processed in parallel is limited to the stripe depth
    uint32_t numOfmInParallel =
        GetNumOfmInParallel(m_Capabilities.GetNumberOfOgs(), numSrams, stripeDepth, weightsTensorInfo.m_DataFormat);

    std::vector<std::unique_ptr<WeightCompressionParams>> compressionParams =
        GenerateCompressionParams(numOfmInParallel);

    // Encode each OFM stream independently
    std::vector<std::vector<uint8_t>> encodedStreams;
    encodedStreams.reserve(numOfms * numIterationsOfm);
    std::vector<uint32_t> encodedNumBits;
    encodedNumBits.reserve(numOfms * numIterationsOfm);
    const auto numWeightScales = weightsTensorInfo.m_QuantizationInfo.GetScales().size();

    for (uint32_t ofm = 0; ofm < (numOfms * numIterationsOfm); ++ofm)
    {
        // numIterationsOfm >= 1, fully connected
        //                   = 1, otherwise
        uint32_t iteration = ofm % numIterationsOfm;
        uint32_t ofmIdx    = ofm / numIterationsOfm;

        // Calculate encoding parameters from the various quantization infos
        EncodingParams params;
        double overallScale = (inputQuantizationInfo.GetScale() *
                               weightsTensorInfo.m_QuantizationInfo.GetScale(numWeightScales > 1 ? ofmIdx : 0)) /
                              outputQuantizationInfo.GetScale();
        utils::CalculateQuantizedMultiplierSmallerThanOne(overallScale, params.m_OfmScaleFactor, params.m_OfmShift);

        params.m_OfmShift += GetOfmShiftOffset();

        params.m_OfmBias         = biasData[ofmIdx];
        params.m_OfmZeroPoint    = outputQuantizationInfo.GetZeroPoint();
        params.m_FilterZeroPoint = weightsTensorInfo.m_QuantizationInfo.GetZeroPoint();

        EncodedOfm encodedOfm = EncodeOfm(weightsData, ofmIdx, numOfmInParallel, numIterationsOfm, stripeDepth,
                                          iteration, weightsTensorInfo, strideY, strideX, paddingTop, paddingLeft,
                                          iterationSize, operation, algorithm, params, compressionParams);

        encodedStreams.push_back(std::move(encodedOfm.m_EncodedWeights));
        encodedNumBits.push_back(encodedOfm.m_NumOfBits);
    }

    constexpr uint32_t dmaEngineAlignment = 16;

    // Merge the OFM streams together so that all the OFMs that will be processed in the same stripe
    // on the same OG are consecutive in the same stream. Here is a diagram showing how the OFM streams
    // are allocated, assuming we have 8 OGs, a stripe depth of 16 and 35 OFMs. Each row of OFM streams in
    // each stripe column correspond to a separate entry in streamPerStripeOg, reading first down the column
    // and across. i.e. the second stripe for OG 4 would be in entry 12.
    //
    //            |    STRIPE 0       |      STRIPE 1         |       STRIPE 2
    //            |-------------------|-----------------------|-------------------|
    //       0    | 0  8              | 16  24                |  32
    //       1    | 1  9              | 17  25                |  33
    //       2    | 2  10             | 18  26                |  34
    //   OG  3    | 3  11             | 19  27                |
    //       4    | 4  12             | 20  28                |
    //       5    | 5  13             | 21  29                |
    //       6    | 6  14             | 22  30                |
    //       7    | 7  15             | 23  31                |
    //
    // If numIterationsOfm > 1, then we have more entries in encodedStreams and we deal with this by pretending
    // we have more OGs.
    //
    std::vector<std::vector<uint8_t>> streamPerStripeOg;
    const uint32_t numStripes = utils::DivRoundUp(numOfms, stripeDepth);
    for (uint32_t stripeIdx = 0; stripeIdx < numStripes; ++stripeIdx)
    {
        const uint32_t firstOfmInStripe = stripeDepth * stripeIdx * numIterationsOfm;
        const uint32_t lastOfmInStripe  = std::min<uint32_t>(numOfms, stripeDepth * (stripeIdx + 1)) * numIterationsOfm;
        std::vector<std::vector<uint8_t>> encodedOfmStreamsForThisStripe(std::begin(encodedStreams) + firstOfmInStripe,
                                                                         std::begin(encodedStreams) + lastOfmInStripe);
        std::vector<std::vector<uint8_t>> streamPerOgForThisStripe;
        if (m_Capabilities.GetWeightCompressionVersion() == 0)
        {
            streamPerOgForThisStripe = MergeStreams(encodedOfmStreamsForThisStripe, numOfmInParallel * numIterationsOfm,
                                                    1, 1, dmaEngineAlignment);
        }
        else
        {
            std::vector<uint32_t> encodedOfmStreamSizesForThisStripe(std::begin(encodedNumBits) + firstOfmInStripe,
                                                                     std::begin(encodedNumBits) + lastOfmInStripe);
            streamPerOgForThisStripe =
                MergeStreamsOg(encodedOfmStreamsForThisStripe, encodedOfmStreamSizesForThisStripe,
                               numOfmInParallel * numIterationsOfm, dmaEngineAlignment);
        }
        streamPerStripeOg.insert(std::end(streamPerStripeOg), std::begin(streamPerOgForThisStripe),
                                 std::end(streamPerOgForThisStripe));
    }

    // Ensure all streams are of equal size as SRAM offsets are same on all CEs
    uint32_t maxLength = 0;
    for (const std::vector<uint8_t>& s : streamPerStripeOg)
    {
        maxLength = std::max(maxLength, static_cast<uint32_t>(s.size()));
    }
    for (std::vector<uint8_t>& s : streamPerStripeOg)
    {
        s.resize(maxLength, 0);
    }

    // Because the weights will be DMA'd in stripes, there is an alignment requirement for the start of each stripe
    // (the DMA can only transfer blocks aligned to 16-bytes).
    // Therefore we pad each stream to 16 bytes.
    for (std::vector<uint8_t>& stream : streamPerStripeOg)
    {
        if (stream.size() % dmaEngineAlignment != 0)
        {
            size_t numZeroesToAdd = dmaEngineAlignment - stream.size() % dmaEngineAlignment;
            std::fill_n(std::back_inserter(stream), numZeroesToAdd, 0);
        }
    }

    // Merge together all the stripes into groups based on the SRAM they will be loaded into.
    // Stream = group of stripes that are loaded into a particular SRAM
    assert(numOfmsPerSram >= 1);
    std::vector<std::vector<uint8_t>> mergedStreams =
        MergeStreams(streamPerStripeOg, numSrams, numIterationsOfm, numOfmsPerSram, 0);

    EncodedWeights encodedWeights;

    // Merge all the SRAM streams together by interleaving 16 bytes from each.
    // This is so the DMA will distribute the correct weight data to the correct SRAM.
    encodedWeights.m_Data     = InterleaveStreams(mergedStreams, dmaEngineAlignment);
    encodedWeights.m_Metadata = CalculateWeightsMetadata(streamPerStripeOg, numOfmInParallel);

    encodedWeights.m_MaxSize = 0;

    for (uint32_t i = 0; i < encodedWeights.m_Metadata.size(); ++i)
    {
        encodedWeights.m_MaxSize = std::max(encodedWeights.m_Metadata[i].m_Size, encodedWeights.m_MaxSize);
    }

    return encodedWeights;
}

/* Calculate the size if the weights are compressed with zero compression */
static size_t CalcZeroCompressionSize(size_t nbrElements, size_t nbrZeros, size_t numSrams)
{
    size_t elems = utils::RoundUpToNearestMultiple(nbrElements, numSrams);
    size_t totalSize;

    // totalSize = mask (1 byte per 8 weights) + elements not equal to zero
    return totalSize = (elems / 8) + (elems - nbrZeros);
}

/* Calculate the size if the weights are compressed with a Lut compressor (worst case since the Lut
   can be shared with the previous OFM which results in slightly higher compression ratio) */
static size_t CalcLutCompressionSize(size_t nbrElements, size_t nbrUniqueElements)
{
    const size_t minBitsPerIndexSupported = 3;
    const size_t maxBitsPerIndexSupported = 5;
    size_t bitsPerIndex =
        std::max(static_cast<size_t>(ceil(log2(static_cast<double>(nbrUniqueElements)))), minBitsPerIndexSupported);
    size_t totalSize;

    if (nbrUniqueElements > 0 && bitsPerIndex <= maxBitsPerIndexSupported)
    {
        // totalSize = Lut + nbrElements number of Lut indices
        totalSize = static_cast<size_t>(pow(2, static_cast<double>(bitsPerIndex))) +
                    utils::RoundUpToNearestMultiple(nbrElements * bitsPerIndex, 8) / 8;
    }
    else
    {
        // Return a very large size to disqualify this compression method
        totalSize = 0xFFFFFFFF;
    }

    return totalSize;
}

/* Calculate the size if the weights are compressed with zero and Lut compressor (worst case since
   the Lut can be shared with the previous OFM which results in slightly higher compression ratio) */
static size_t CalcZeroLutCompressionSize(size_t nbrElements, size_t nbrZeros, size_t nbrUniqueElements, size_t numSrams)
{
    size_t elems                       = utils::RoundUpToNearestMultiple(nbrElements, numSrams);
    size_t uniqueElementsExcludingZero = (nbrZeros == 0) ? nbrUniqueElements : nbrUniqueElements - 1;

    // totalSize = mask (1 byte per 8 weights) + Lut + Lut indices for elements not equal to zero
    return (elems / 8) + CalcLutCompressionSize(elems - nbrZeros, uniqueElementsExcludingZero);
}

std::vector<WeightsMetadata>
    WeightEncoder::CalculateWeightsMetadata(const std::vector<std::vector<uint8_t>>& streamPerStripeOg,
                                            uint32_t numOgPerStripe) const
{
    std::vector<WeightsMetadata> metadata;
    uint32_t runningSize = 0;
    for (size_t i = 0; i < streamPerStripeOg.size(); i += numOgPerStripe)
    {
        uint32_t stripeSize = 0;
        for (size_t j = 0; j < numOgPerStripe; ++j)
        {
            stripeSize += static_cast<uint32_t>(streamPerStripeOg[i + j].size());
        }
        metadata.push_back(WeightsMetadata{ runningSize, stripeSize });
        runningSize += stripeSize;
    }

    return metadata;
}

std::vector<uint8_t> WeightEncoder::GetRawOfmStream(const uint8_t* weightData,
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
                                                    bool prepareForZeroMaskCompression) const
{
    assert(algorithm != CompilerMceAlgorithm::None);

    const uint32_t numUninterleavedIfmsPerIteration = iterationSize / (strideX * strideY);

    utils::ConstTensorData wd(weightData, weightsTensorInfo.m_Dimensions);
    uint32_t filterX             = weightsTensorInfo.m_Dimensions[1];
    uint32_t filterY             = weightsTensorInfo.m_Dimensions[0];
    const uint32_t maxFilterSize = algorithm == CompilerMceAlgorithm::Direct ? 7 : 1;
    std::vector<SubmapFilter> subfilters =
        GetSubmapFilters(filterX, filterY, strideX, strideY, paddingLeft, paddingTop);
    const uint32_t wideKernelSize            = m_Capabilities.GetWideKernelSize();
    std::vector<SubmapFilter> wideSubfilters = GetSubmapFilters(filterX, filterY, wideKernelSize, maxFilterSize);

    uint32_t numEngines      = m_Capabilities.GetNumberOfEngines();
    uint32_t numIgsPerEngine = m_Capabilities.GetIgsPerEngine();
    // When not using zero mask compression we must tightly pack the final subfilter in the final slice
    // (where each slice is the set of weights for as many IFMs as there are IGs).
    // However when zero mask compression is enabled the HW behaves differently and requires this to be padded
    // with zeroes.
    bool tightlyPackLastSliceLastSubfilter = !prepareForZeroMaskCompression;

    std::vector<uint8_t> result;

    auto AddWeightsForIfms = [&result](auto weightCalculationFunction, uint32_t channelStart,
                                       uint32_t numChannels) -> void {
        for (uint32_t i = channelStart; i < channelStart + numChannels; ++i)
        {
            uint8_t weight = weightCalculationFunction(i);
            result.push_back(weight);
        }
    };

    if (weightsTensorInfo.m_DataFormat == DataFormat::HWIO &&
        operation != ethosn::command_stream::MceOperation::FULLY_CONNECTED && algorithm == CompilerMceAlgorithm::Direct)
    {
        const uint32_t numIfms = weightsTensorInfo.m_Dimensions[2];

        const uint32_t numIfmsProcessedInParallel = numIgsPerEngine * numEngines;

        // In the IFM depth streaming, weights need to be partitioned
        // into multiple sections per OFM.
        uint32_t chanOffset = iteration * numUninterleavedIfmsPerIteration;
        assert(chanOffset < numIfms);

        uint32_t chanEnd = std::min(chanOffset + numUninterleavedIfmsPerIteration,
                                    utils::RoundUpToNearestMultiple(numIfms, numIfmsProcessedInParallel));

        const bool isWideKernel = wideSubfilters.size() > 1;

        // Weight layout for Direct mode:
        // In wide kernel mode the base kernel is decomposed into smaller subkernels and the
        // decomposed subkernels are packed in the weight stream. The supported decomposed subkernels
        // are 1x3, 3x1 and 3x3. The wide-kernel 1xM, Nx1 and NxM will be decomposed into
        // 1x3, 3x1 and 3x3 subkernels respectively. In this mode the weight stream will have
        // a single OFM header for all the subkernel and the weight layout has weights of subkernel 0
        // across the per every channel stripe (IGs) for the whole IFM depth, followed by weights of subkernel 1, and so weights
        // of subkernel N, followed by OFM 1.
        for (SubmapFilter wideFilter : wideSubfilters)
        {
            // The weight data is grouped into slices of as many IFMs as there are IGs.
            for (uint32_t channelStart = chanOffset; channelStart < chanEnd; channelStart += numIfmsProcessedInParallel)
            {
                const uint32_t channelsInThisSlice = std::min(numIfmsProcessedInParallel, numIfms - channelStart);
                // For wide kernel the number of subfilters is 1
                for (uint32_t filterIdx = 0; filterIdx < subfilters.size(); ++filterIdx)
                {
                    const SubmapFilter& filter = subfilters[filterIdx];

                    // If there are multiple subfilters, the data in all except the last must be padded to the number of IFM
                    // channels equal to the number of IGs. The last one may be left without padding, if this is the last
                    // slice and we are not using zero compression.
                    const uint32_t numChannels =
                        (filterIdx == subfilters.size() - 1 && tightlyPackLastSliceLastSubfilter)
                            ? channelsInThisSlice
                            : numIfmsProcessedInParallel;

                    // When the dimensions of the kernel are such that cannot be decomposed in as many submap kernels as strideX * strideY
                    // it needs to elide the submapped IFM that don't need to be used.
                    // For that reason a kernel 1x1 with weight equal to zero point is created.
                    if (filter.GetFilterY() == 0 || filter.GetFilterX() == 0)
                    {
                        AddWeightsForIfms(
                            [&](uint32_t) {
                                return static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.GetZeroPoint());
                            },
                            channelStart, numChannels);
                    }
                    else
                    {
                        const uint32_t currSubKernelSizeX =
                            isWideKernel ? wideFilter.GetFilterX() : filter.GetFilterX();
                        const uint32_t currSubKernelSizeY =
                            isWideKernel ? wideFilter.GetFilterY() : filter.GetFilterY();
                        // Add weight data in row-major order, with the slice of 16 IFMs (for ethosn) tightly packed for each filter coordinate.
                        for (uint32_t h = 0; h < currSubKernelSizeY; ++h)
                        {
                            for (uint32_t w = 0; w < currSubKernelSizeX; ++w)
                            {
                                const uint32_t y       = h + wideFilter.GetOffsetY();
                                const uint32_t x       = w + wideFilter.GetOffsetX();
                                const bool isValidData = (y < filterY) && (x < filterX);
                                AddWeightsForIfms(
                                    [&](uint32_t i) {
                                        return (isValidData && i < numIfms)
                                                   ? filter.GetWeightAt(wd, y, x, i, ofmIdx)
                                                   : static_cast<uint8_t>(
                                                         weightsTensorInfo.m_QuantizationInfo.GetZeroPoint());
                                    },
                                    channelStart, numChannels);
                            }
                        }
                    }
                }
            }
        }
    }
    else if (weightsTensorInfo.m_DataFormat == DataFormat::HWIO &&
             operation != ethosn::command_stream::MceOperation::FULLY_CONNECTED &&
             algorithm == CompilerMceAlgorithm::Winograd)
    {
        // Sanity check WINOGRAD only supports non-strided convolutions
        assert(strideY == 1 && strideX == 1);

        const uint32_t numIfms = weightsTensorInfo.m_Dimensions[2];

        // Weight layout for Winograd:
        // In wide kernel mode the base kernel is decomposed into smaller subkernels and the
        // decomposed subkernels are packed in the weight stream. The supported decomposed subkernels
        // are 1x3, 3x1 and 3x3. The wide-kernel 1xM, Nx1 and NxM will be decomposed into
        // 1x3, 3x1 and 3x3 subkernels respectively. In this mode the weight stream will have
        // a single OFM header for all the subkernel and the weight layout has weights of subkernel 0
        // across the IFM depth followed by weights of subkernel 1, and so weights of subkernel N,
        // followed by OFM 1.
        for (SubmapFilter wideFilter : wideSubfilters)
        {
            uint32_t count = 0;
            for (uint32_t channel = 0; channel < numIfms; ++channel)
            {
                for (SubmapFilter filter : subfilters)
                {
                    // For WINOGRAD there can only be one submap filter since
                    // stride = 1
                    for (uint32_t h = 0; h < wideFilter.GetFilterY(); ++h)
                    {
                        for (uint32_t w = 0; w < wideFilter.GetFilterX(); ++w)
                        {
                            const uint32_t y       = h + wideFilter.GetOffsetY();
                            const uint32_t x       = w + wideFilter.GetOffsetX();
                            const bool isValidData = (y < filterY) && (x < filterX);

                            // zero padding if the index is outside the range of the original kernel
                            uint8_t weight =
                                isValidData ? filter.GetWeightAt(wd, y, x, channel, ofmIdx)
                                            : static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.GetZeroPoint());
                            result.push_back(weight);
                            ++count;
                        }
                    }
                }
            }
            // With zero compression when the number of weights per subkernel is a non-multiple of 16
            // the last subkernel will be padded with zeros.
            if (prepareForZeroMaskCompression)
            {
                for (uint32_t i = count; i < utils::RoundUpToNearestMultiple(count, m_Capabilities.GetNumberOfSrams());
                     ++i)
                {
                    result.push_back(static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.GetZeroPoint()));
                }
            }
        }
    }
    else if (weightsTensorInfo.m_DataFormat == DataFormat::HWIO &&
             operation == ethosn::command_stream::MceOperation::FULLY_CONNECTED)
    {
        // Offset in the weight data for this iteration
        const uint32_t iterationOffset = iteration * numUninterleavedIfmsPerIteration;
        const uint32_t numIfms         = weightsTensorInfo.m_Dimensions[2];
        const uint32_t numSrams        = m_Capabilities.GetNumberOfSrams();

        assert(numIfms % g_WeightsChannelVecProd == 0);

        for (SubmapFilter filter : subfilters)
        {
            for (uint32_t encodedIdx = 0; encodedIdx < numUninterleavedIfmsPerIteration; ++encodedIdx)
            {
                uint32_t rawIdx;

                uint32_t brickIdx = encodedIdx / g_WeightsChannelVecProd;
                uint32_t idxBrick = encodedIdx % g_WeightsChannelVecProd;

                const uint32_t patchSize = 16;
                assert(numSrams == 8 || numSrams == 16);

                uint32_t qbrickSize = patchSize * numSrams;
                uint32_t qbrickIdx  = idxBrick / qbrickSize;

                uint32_t numSubBricks = 16 / numSrams;
                assert(numSubBricks <= 2);

                // If the number of OFMs per engine is 1, then qbrickIdx = idxBrick / 256
                // If it is 2, then
                // qbrickIdx = 0, [0 127]
                //           = 2, [128 255]
                //           = 4, [256 383]
                //           = 6, [384 511]
                //           = 1, [512 639]
                //           = 3, [640 767]
                //           = 5, [768 893]
                //           = 7, [894 1023]
                qbrickIdx = (qbrickIdx % 4) * numSubBricks + (qbrickIdx / 4);
                assert(((qbrickIdx < 4) && (numSrams == 16)) || ((qbrickIdx < 8) && (numSrams == 8)));

                uint32_t idxQbrick   = idxBrick % qbrickSize;
                uint32_t patchIdx    = idxQbrick % numSrams;
                uint32_t patchOffset = idxQbrick / numSrams;
                uint8_t weight;

                rawIdx = iterationOffset + brickIdx * g_WeightsChannelVecProd + qbrickIdx * qbrickSize +
                         patchIdx * patchSize + patchOffset;

                if (rawIdx < numIfms)
                {
                    weight = filter.GetWeightAt(wd, 0, 0, rawIdx, ofmIdx);
                }
                else
                {
                    weight = static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.GetZeroPoint());
                }

                result.push_back(weight);
            }
        }
    }
    else if (weightsTensorInfo.m_DataFormat == DataFormat::HWIM)
    {
        // Sanity check: existing HWs don't support depth wise WINOGRAD convolution
        assert(algorithm != CompilerMceAlgorithm::Winograd);

        const uint32_t numIfms = weightsTensorInfo.m_Dimensions[2];
        // Note numIfmsProcessedInParallel is different to non-depthwise convolution weights, as in some configurations not all OGs are used.
        const uint32_t numIfmsProcessedInParallel = m_Capabilities.GetNumberOfSrams();

        // Decompose the ofm index to find which ifm it corresponds to.
        const uint32_t channelMultiplierIdx = ofmIdx / numIfms;
        const uint32_t ifmIdx               = ofmIdx % numIfms;

        // Compared to 'regular' HWIO weights, we only need to specify the weights for as many IFMs as there are IGs
        // rather than all of the IFMs.
        // Ethos-Nx7:
        // Mathematically we only need to supply 1 (as each OFM is dependent on only 1 IFM),
        // but the HW requires a full set of 16 weights so we just set the others to zero. Add weight data in row-major
        // order, with a slice of as many IFMs as there are IGs, tightly packed for each filter coordinate.
        // Ethos-N78:
        // Only packs on set of weights and the HW will insert 0s accordingly after decoding.
        for (uint32_t filterIdx = 0; filterIdx < subfilters.size(); ++filterIdx)
        {
            const SubmapFilter& filter = subfilters[filterIdx];

            // Get encoding params
            bool usePadding               = (filterIdx == subfilters.size() - 1) && tightlyPackLastSliceLastSubfilter;
            uint32_t numChannels          = 0;
            uint32_t ifmMod               = 0;
            std::tie(numChannels, ifmMod) = GetHwimWeightPadding(usePadding, ifmIdx, numIfmsProcessedInParallel);

            // Add weight data in row-major order, with the slice of as many IFMs as there are IGs, tightly packed
            // for each filter coordinate.
            for (uint32_t h = 0; h < filter.GetFilterY(); ++h)
            {
                for (uint32_t w = 0; w < filter.GetFilterX(); ++w)
                {
                    for (uint32_t i = 0; i < numChannels; ++i)
                    {
                        uint8_t weight;

                        if (i == ifmIdx % ifmMod)
                        {
                            weight = filter.GetWeightAt(wd, h, w, ifmIdx, channelMultiplierIdx);
                        }
                        else
                        {
                            weight = static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.GetZeroPoint());
                        }

                        result.push_back(weight);
                    }
                }
            }
        }
    }
    else
    {
        // Weight tensor must be HWIO or HWIM
        assert(false);
    }

    return result;
}

WeightEncoderV1::WeightCompressionParamsV1
    WeightEncoderV1::ChooseCompressionParameters(const std::vector<uint8_t>& rawWeightsForZeroMaskCompression,
                                                 const std::vector<uint8_t>& rawWeightsForNoZeroMaskCompression,
                                                 const TensorInfo& weightsTensorInfo) const
{
    // Description and working data for a single compression scheme.
    struct Scheme
    {
        // Unique ID of the scheme.
        bool m_ZeroMask;
        bool m_Lut;

        // Function to calculate the compressed size
        std::function<size_t(const Scheme& scheme)> compressedSizeCalculator;

        // Statistics of the raw weight stream used for this scheme (different schemes may use a different raw weight stream)
        std::vector<size_t> frequencies;
        size_t numElements;
        size_t numUniqueElements;
        size_t numZeroPointElements;

        // Compressed size, calculated by compressedSizeCalculator
        size_t compressedSize;
    };
    // Describe each of the four possible compression schemes
    // clang-format off
    std::array<Scheme, 4> schemes = { {
        // No compression
        {
            false, false,
            [&](const Scheme& scheme) -> size_t {
                if (weightsTensorInfo.m_DataFormat == DataFormat::HWIM)
                {
                    return 0xFFFFFFFF;    // For HWIM we cannot disable zero-mask compression
                }
                return scheme.numElements;
            },
            {},
            0, 0, 0,
            0
        },
        // LUT compression only
        {
            false, true,
            [&](const Scheme& scheme) -> size_t {
                if (weightsTensorInfo.m_DataFormat == DataFormat::HWIM)
                {
                    return 0xFFFFFFFF;    // For HWIM we cannot disable zero-mask compression
                }
                return CalcLutCompressionSize(scheme.numElements, scheme.numUniqueElements);
            },
            {},
            0, 0, 0,
            0
        },
        // Zero-mask compression only
        {
            true, false,
            [&](const Scheme& scheme) -> size_t {
                return CalcZeroCompressionSize(scheme.numElements, scheme.numZeroPointElements, m_Capabilities.GetNumberOfSrams());
            },
            {},
            0, 0, 0,
            0
        },
        // Both LUT and zero-mask compression
        {
            true, true,
            [&](const Scheme& scheme) -> size_t {
                return CalcZeroLutCompressionSize(scheme.numElements, scheme.numZeroPointElements, scheme.numUniqueElements,
                                                  m_Capabilities.GetNumberOfSrams());
            },
            {},
            0, 0, 0,
            0
        },
    } };
    // clang-format on

    // ZeroPoint must be representable in the data type (int8 or uint8 for now)
    const uint8_t zeroPoint = static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.GetZeroPoint());
    // Analyze the size for each
    for (Scheme& scheme : schemes)
    {
        const std::vector<uint8_t>& rawWeights =
            scheme.m_ZeroMask ? rawWeightsForZeroMaskCompression : rawWeightsForNoZeroMaskCompression;

        // Analyze the weight statistics and setup the compression parameters
        scheme.frequencies.resize(256, 0);

        for (uint8_t v : rawWeights)
        {
            ++scheme.frequencies[v];
        }
        scheme.numElements = rawWeights.size();
        scheme.numUniqueElements =
            count_if(scheme.frequencies.begin(), scheme.frequencies.end(), [](size_t val) { return val != 0; });
        scheme.numZeroPointElements = scheme.frequencies[zeroPoint];
        scheme.compressedSize       = scheme.compressedSizeCalculator(scheme);
    }

    const Scheme& bestScheme = *std::min_element(
        schemes.begin(), schemes.end(), [](auto a, auto b) -> bool { return a.compressedSize < b.compressedSize; });

    WeightCompressionParamsV1 params;
    params.m_LutReload  = bestScheme.m_Lut;
    params.m_MaskEnable = bestScheme.m_ZeroMask;
    params.m_IndexSize  = 0;    // 8-bit weights, Lut disabled

    if (params.m_LutReload)
    {
        // Enable Lut compression
        // IndexSize:  Bits per index (number of weights):
        //  1           3 (0 - 8 weights)
        //  2           4 (9 - 16 weights)
        //  3           5 (17 - 32 weights)
        size_t compressedUniqueElements = bestScheme.numUniqueElements;
        if (params.m_MaskEnable && bestScheme.numZeroPointElements > 0)
        {
            // Reduce the number of unique elements by one because of the mask, zero elements are not part of the LUT
            --compressedUniqueElements;
        }

        int bitsPerIndex = std::max(static_cast<int>(ceil(log2(static_cast<double>(compressedUniqueElements)))), 3);
        assert(bitsPerIndex == 3 || bitsPerIndex == 4 || bitsPerIndex == 5);
        params.m_IndexSize = bitsPerIndex - 2;
        // Make sure the Lut contains entries for 2^bitsPerIndex number of entries
        params.m_Lut = std::vector<uint8_t>(static_cast<int>(pow(2, bitsPerIndex)), 0);

        size_t index = 0;
        for (int i = 0; index < bestScheme.frequencies.size(); ++index)
        {
            if (bestScheme.frequencies[index] != 0 && !(params.m_MaskEnable && index == zeroPoint))
            {
                params.m_Lut[i] = static_cast<uint8_t>(index);
                ++i;
            }
        }
    }

    return params;
}

#pragma pack(push, 1)
// See "MCE Specification", section 6.12.6.
struct WeightHeader
{
    uint16_t m_StreamLength;
    uint16_t m_OfmScaleFactor;
    uint32_t m_OfmBiasLow;
    uint16_t m_OfmBiasHigh;
    uint32_t m_OfmShift : 5;
    uint32_t m_OfmZeroPoint : 8;
    uint32_t m_WeightLayout : 2;
    uint32_t m_WeightMaskWidth : 1;
    uint32_t m_FilterZeroPoint : 8;
    uint32_t m_MaskEnable : 1;
    uint32_t m_LutReload : 1;
    uint32_t m_IndexSize : 2;
    uint32_t m_SignExtend : 1;
    uint32_t m_Padding : 3;
};
static_assert(sizeof(WeightHeader) == 14, "WeightHeader struct has not been tightly packed.");
#pragma pack(pop)

std::vector<std::unique_ptr<WeightEncoder::WeightCompressionParams>>
    WeightEncoderV1::GenerateCompressionParams(uint32_t numOfmInParallel)
{
    std::vector<std::unique_ptr<WeightCompressionParams>> params(numOfmInParallel);
    std::generate(params.begin(), params.end(), std::make_unique<WeightCompressionParamsV1>);
    return params;
}

WeightEncoder::EncodedOfm
    WeightEncoderV1::EncodeOfm(const uint8_t* weightData,
                               uint32_t ofmIdx,
                               uint32_t numOfmInParallel,
                               uint32_t,
                               uint32_t,
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
                               std::vector<std::unique_ptr<WeightCompressionParams>>& compressionParameters)
{
    // Lookup the compression parameters for the previous OFM associated with the same CE. This is used
    // to modify the compression of this current OFM.
    WeightCompressionParamsV1& previousOfmSameCeCompressionParams =
        static_cast<WeightCompressionParamsV1&>(*compressionParameters[ofmIdx % numOfmInParallel]);

    // Get the raw (unencoded) weight stream. Note we must do this twice - once to get a stream suited
    // for zero mask compression and again to get one suited to no zero mask compression. Yuck!
    std::vector<uint8_t> rawWeightsForZeroMaskCompression =
        GetRawOfmStream(weightData, ofmIdx, iteration, weightsTensorInfo, strideY, strideX, paddingTop, paddingLeft,
                        iterationSize, operation, algorithm, true);
    std::vector<uint8_t> rawWeightsForNoZeroMaskCompression =
        GetRawOfmStream(weightData, ofmIdx, iteration, weightsTensorInfo, strideY, strideX, paddingTop, paddingLeft,
                        iterationSize, operation, algorithm, false);

    // Choose the best compression scheme
    WeightCompressionParamsV1 compressionParams = ChooseCompressionParameters(
        rawWeightsForZeroMaskCompression, rawWeightsForNoZeroMaskCompression, weightsTensorInfo);
    std::vector<uint8_t>& rawWeights =
        compressionParams.m_MaskEnable ? rawWeightsForZeroMaskCompression : rawWeightsForNoZeroMaskCompression;

    // If the Lut is the same as for previous OFM for the current CE then don't reload it
    const uint32_t numOfmsPerSram = m_Capabilities.GetNumberOfOgs() / m_Capabilities.GetNumberOfSrams();
    if (compressionParams.m_IndexSize != 0 && ofmIdx >= numOfmInParallel &&
        previousOfmSameCeCompressionParams.m_Lut == compressionParams.m_Lut &&
        // Disable for configurations with more than one OFM per SRAM, since they use a different CE OFM
        // fetching strategy
        numOfmsPerSram == 1)
    {
        compressionParams.m_LutReload = false;
    }

    EncodedOfm result{};
    previousOfmSameCeCompressionParams   = compressionParams;
    std::vector<uint8_t>& encodedWeights = result.m_EncodedWeights;

    // Add the per-OFM header.
    encodedWeights.insert(encodedWeights.end(), sizeof(WeightHeader), 0);
    WeightHeader& header = reinterpret_cast<WeightHeader&>(encodedWeights.front());

    header.m_StreamLength    = 0xFFFF;    // We'll fix this later once we know how long this stream is.
    header.m_OfmScaleFactor  = params.m_OfmScaleFactor;
    header.m_OfmBiasLow      = params.m_OfmBias;
    header.m_OfmBiasHigh     = 0;
    header.m_OfmShift        = params.m_OfmShift & 0b11111;
    header.m_OfmZeroPoint    = static_cast<uint8_t>(params.m_OfmZeroPoint);
    header.m_WeightLayout    = 0;
    header.m_WeightMaskWidth = 0;
    header.m_FilterZeroPoint = static_cast<uint8_t>(params.m_FilterZeroPoint);
    header.m_MaskEnable      = compressionParams.m_MaskEnable;
    header.m_LutReload       = compressionParams.m_LutReload;
    header.m_IndexSize       = compressionParams.m_IndexSize & 0b11;
    header.m_SignExtend      = utils::IsDataTypeSigned(weightsTensorInfo.m_DataType);
    header.m_Padding         = 0;    // Unused padding.

    // Compress each weight using the above chosen compression parameters
    std::shared_ptr<WeightCompressor> compressor =
        CreateWeightCompressor(encodedWeights, compressionParams.m_IndexSize, compressionParams.m_Lut,
                               compressionParams.m_LutReload, compressionParams.m_MaskEnable,
                               static_cast<uint8_t>(params.m_FilterZeroPoint), m_Capabilities.GetNumberOfSrams());

    for (size_t i = 0; i < rawWeights.size(); ++i)
    {
        compressor->CompressWeight(rawWeights[i]);
    }

    compressor->Flush();

    return result;
}

uint32_t WeightEncoderV1::GetOfmShiftOffset() const
{
    return 0;
}

std::pair<uint32_t, uint32_t> WeightEncoderV1::GetHwimWeightPadding(const bool usePadding,
                                                                    const uint32_t ifmIdx,
                                                                    const uint32_t numIfmsProcessedInParallel) const
{
    // If there are multiple subfilters, the data in all except the last must be padded to the number of IGs.
    // The last one may be left without padding, if we are not using zero compression.
    uint32_t numChannels = usePadding ? (ifmIdx % numIfmsProcessedInParallel) + 1 : numIfmsProcessedInParallel;

    return std::make_pair(numChannels, numIfmsProcessedInParallel);
}

uint32_t WeightEncoderV1::GetNumOfmInParallel(const uint32_t numOfm,
                                              const uint32_t numSrams,
                                              const uint32_t,
                                              const DataFormat dataFormat) const
{
    if (dataFormat == DataFormat::HWIO)
    {
        return numOfm;
    }
    else
    {
        return numSrams;
    }
}

std::vector<std::vector<uint8_t>> WeightEncoder::MergeStreams(const std::vector<std::vector<uint8_t>>& streams,
                                                              uint32_t numGroups,
                                                              uint32_t numIterations,
                                                              uint32_t numOfmPerSram,
                                                              const uint32_t streamHeadersUpdateAlignment) const
{
    // Assign each stream to a group (each group is stored as a vector of the stream indexes assigned to it).
    std::vector<std::vector<uint32_t>> groups(numGroups);
    for (uint32_t streamIdx = 0; streamIdx < streams.size(); ++streamIdx)
    {
        // when numIterations != 1
        // It is fully connected where the weight is divided into M parts per OFM
        // (0,0) (0,1), (0,2) ... (0, M-1)    --- weight 0
        // (1,0) (1,1), (1,2) ... (1, M-1)    --- weight 1
        // ....
        // (i,0) (i,1) ... (i,j) ... (i, M-1)
        //  where (i,j) is the weight of (OFM i, part j)
        // The weights belong to the same OFM are saved in the same group
        //
        // For example with NumOfmEthosN = 8
        // Group 0:
        // (0,0) (0,1), (0,2) ... (0, M-1)
        // (8,0) (8,1), (8,2) ... (8, M-1)
        // ....
        // (8*n) (8n,1)    ...
        //
        // Group 1:
        // (1,0) (1,1), (1,2) ... (1, M-1)
        // (9,0) (9,1), (9,2) ... (9, M-1)
        // ....
        // (8n+1) (8n+1,1)    ...
        //
        // Group 7:
        // (7,0) (7,1), (7,2) ... (7, M-1)
        // (15,0) (15,1), (15,2) ... (15, M-1)
        // ....
        // (8n+7) (8n+7,1)    ...
        //
        // As a result, the interleave will put the weight belong to the
        // same OFM group and iteration together
        // (0,0) (1,0) (2,0) (3, 0) ... (7,0)
        // (0,1) (1,1) (2,1) (3, 1) ... (7,1)
        //  .....
        // (i,j) (i+1, j)  ....         (i+7, j)
        // where j is the iteration id and i is the ofm id.
        uint32_t groupIdx = (streamIdx / numIterations) % numGroups;
        groups[groupIdx].push_back(streamIdx);
    }

    if (numOfmPerSram > 1 && numIterations > 1)
    {
        // Interleave the stream indices again if both the number of OFMs per SRAM
        // and number of iterations per OFM are larger than 1.

        // Sanity check (We currently only support 1 or 2 OFMs per SRAM)
        assert(numOfmPerSram == 2);

        std::vector<uint32_t> tempCopy;

        // Number of weight streams needed for two OFM produced from a SRAM bank
        uint32_t numIterationsSram = numIterations * numOfmPerSram;

        for (uint32_t groupIdx = 0; groupIdx < numGroups; ++groupIdx)
        {
            std::vector<uint32_t>& group = groups[groupIdx];

            assert(tempCopy.size() == 0);

            std::copy(group.begin(), group.end(), std::back_inserter(tempCopy));
            assert(tempCopy.size() == group.size());

            // Within a group, the indices are interleaved such that weight streams
            // belong to different OFMs are fetched to HW per iteration.
            // For example, with numOfmsPerSram = 2, we have numIterationsOfm = 4:
            // before interleaving, stream indices in group 0 are:
            // (0,0) (0,1) (0,2) (0,3)  (8,0) (8,1) (8,2) (8,3)
            // (16,0) (16,1) (16,2) (16,3)
            //
            // After interleaving:
            // (0,0) (8,0) (0,1) (8,1) (0,2) (8,2) (0,3) (8,3)
            // (16,0) (16,1) (16,2) (16,3)
            //
            // The fetch order of the weight streams is:
            // (0,0) (8,0)
            // (0,1) (8,1)
            // ...
            // (16,1)
            // (16,2)
            // (16,3)

            // sanity check: size must be multiple of numIterationsSram
            assert(group.size() % numIterationsSram == 0);
            for (uint32_t count = 0; count < group.size(); ++count)
            {
                uint32_t index0     = count / numIterationsSram;
                uint32_t localIndex = count % numIterationsSram;

                uint32_t index1 = localIndex / numOfmPerSram;
                uint32_t index2 = localIndex % numOfmPerSram;
                uint32_t index  = index0 * numIterationsSram + index2 * numIterations + index1;

                assert(index < group.size());
                group[count] = tempCopy[index];
            }

            tempCopy.clear();
        }
    }

    // For each group, merge all its streams together into one.
    std::vector<std::vector<uint8_t>> result(numGroups);
    for (uint32_t groupIdx = 0; groupIdx < numGroups; ++groupIdx)
    {
        const std::vector<uint32_t>& group = groups[groupIdx];
        std::vector<uint8_t>& mergedGroup  = result[groupIdx];

        for (uint32_t streamIdxWithinGroup = 0; streamIdxWithinGroup < group.size(); ++streamIdxWithinGroup)
        {
            uint32_t streamIdx                 = group[streamIdxWithinGroup];
            const std::vector<uint8_t>& stream = streams[streamIdx];
            uint32_t start                     = static_cast<uint32_t>(mergedGroup.size());

            std::copy(stream.begin(), stream.end(), std::back_inserter(mergedGroup));

            // If requested to update weight headers then we assume there are weight
            // headers at the start of every stream; and that they need updating.
            if (streamHeadersUpdateAlignment != 0)
            {
                // Set the stream length in the header as whole number of words that need to
                // be DMA'd in, depending on alignment.
                WeightHeader& header = reinterpret_cast<WeightHeader&>(mergedGroup[start]);
                assert(header.m_StreamLength == 0xFFFF);    // Not yet written or not a header

                const uint32_t startWord = start / streamHeadersUpdateAlignment;
                const uint32_t endWord =
                    utils::DivRoundUp(static_cast<uint32_t>(mergedGroup.size()), streamHeadersUpdateAlignment);
                const uint16_t streamLength = static_cast<uint16_t>(endWord - startWord);
                header.m_StreamLength       = streamLength;
            }
        }
    }

    return result;
}

std::vector<std::vector<uint8_t>> WeightEncoder::MergeStreamsOg(const std::vector<std::vector<uint8_t>>& streams,
                                                                const std::vector<uint32_t>& streamSize,
                                                                uint32_t numGroups,
                                                                const uint32_t streamHeadersUpdateAlignment) const
{
    // Assign each stream to a group (each group is stored as a vector of the stream indexes assigned to it).
    std::vector<std::vector<uint32_t>> groups(numGroups);
    for (uint32_t streamIdx = 0; streamIdx < streams.size(); ++streamIdx)
    {
        uint32_t groupIdx = streamIdx % numGroups;
        groups[groupIdx].push_back(streamIdx);
    }

    // For each group, merge all its streams together into one.
    std::vector<std::vector<uint8_t>> result(numGroups);
    for (uint32_t groupIdx = 0; groupIdx < numGroups; ++groupIdx)
    {
        const std::vector<uint32_t>& group = groups[groupIdx];
        std::vector<uint8_t>& mergedGroup  = result[groupIdx];

        uint32_t numBitsStream = 0;

        for (uint32_t streamIdxWithinGroup = 0; streamIdxWithinGroup < group.size(); ++streamIdxWithinGroup)
        {
            uint32_t streamIdx                 = group[streamIdxWithinGroup];
            const std::vector<uint8_t>& stream = streams[streamIdx];

            // start position in byte
            uint32_t start = numBitsStream / 8;

            // start position in word (16 bytes)
            uint32_t startWord = start / streamHeadersUpdateAlignment;

            // end position in word
            // Note Ethos-N78: weight stream header starts at the SRAM bit position
            // following the last bit of the preceding weight stream.
            uint32_t endWord = numBitsStream + streamSize[streamIdx];
            endWord          = (endWord + (streamHeadersUpdateAlignment * 8) - 1) / (streamHeadersUpdateAlignment * 8);
            uint16_t headerLength = static_cast<uint16_t>(endWord - startWord);
            uint8_t* headerPtr    = reinterpret_cast<uint8_t*>(&headerLength);

            if ((numBitsStream % 8) == 0)
            {
                // if the last bit stream's end position is byte aligned
                // then replaces the first two bytes with ofm stream length
                // in word.
                mergedGroup.push_back(headerPtr[0]);
                mergedGroup.push_back(headerPtr[1]);
                std::copy(stream.begin() + 2, stream.end(), std::back_inserter(mergedGroup));
            }
            else
            {
                //otherwise, merging the first byte of the new bit stream
                // with the last byte of the new bit stream.

                std::vector<uint8_t> tempStream;
                // take the last element of the previous ofm in the same OG.
                uint32_t elemByte = static_cast<uint32_t>(mergedGroup.back());

                // remove the last element which will be merged with the new stream
                mergedGroup.pop_back();

                // current bit position in the merged bit stream
                uint32_t bitPos     = numBitsStream & 7;
                uint32_t remNumBits = streamSize[streamIdx];

                for (uint32_t i = 0; i < static_cast<uint32_t>(stream.size()); ++i)
                {
                    uint32_t numBits = std::min<uint32_t>(8, remNumBits);
                    uint32_t newByte;

                    if (i < 2)
                    {
                        // first two bytes are headers
                        newByte = headerPtr[i];
                        assert(uint32_t(stream[i]) == 0xff);
                    }
                    else
                    {
                        // then body
                        newByte = stream[i];
                    }

                    for (uint32_t j = 0; j < numBits; ++j)
                    {
                        uint32_t bit = newByte & 1;
                        elemByte |= bit << bitPos;
                        newByte >>= 1;

                        bitPos = (bitPos + 1) & 7;

                        if (bitPos == 0)
                        {
                            tempStream.push_back(uint8_t(elemByte));
                            elemByte = 0;
                        }
                    }

                    remNumBits -= numBits;
                }

                assert(remNumBits == 0);

                if (bitPos != 0)
                {
                    tempStream.push_back(uint8_t(elemByte));
                }

                std::copy(tempStream.begin(), tempStream.end(), std::back_inserter(mergedGroup));
            }

            numBitsStream += streamSize[streamIdx];
        }
    }

    return result;
}

std::vector<uint8_t> WeightEncoder::InterleaveStreams(const std::vector<std::vector<uint8_t>>& streams,
                                                      uint32_t numBytesPerStream) const
{
    // Calculate how long the longest stream is, which determines how big our output will be.
    uint32_t maxLength = 0;
    for (const std::vector<uint8_t>& s : streams)
    {
        maxLength = std::max(maxLength, static_cast<uint32_t>(s.size()));
    }
    std::vector<uint8_t> result;
    result.reserve(maxLength * streams.size());

    // Keep adding data until we reach the end
    for (uint32_t streamOffset = 0; streamOffset < maxLength; streamOffset += numBytesPerStream)
    {
        // Go through each stream and add the requested number of bytes
        for (uint32_t streamIdx = 0; streamIdx < streams.size(); ++streamIdx)
        {
            const std::vector<uint8_t>& stream = streams[streamIdx];

            int32_t numBytesToCopy =
                std::max(0, std::min(static_cast<int32_t>(numBytesPerStream),
                                     static_cast<int32_t>(stream.size()) - static_cast<int32_t>(streamOffset)));
            if (numBytesToCopy > 0)
            {
                std::copy(stream.begin() + streamOffset, stream.begin() + streamOffset + numBytesToCopy,
                          std::back_inserter(result));
            }

            uint32_t numZeroesToAdd = numBytesPerStream - numBytesToCopy;
            if (numZeroesToAdd)
            {
                std::fill_n(std::back_inserter(result), numZeroesToAdd, 0);
            }
        }
    }

    return result;
}

}    // namespace support_library
}    // namespace ethosn
