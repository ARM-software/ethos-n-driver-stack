//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "WeightEncoder.hpp"

#include "Compiler.hpp"
#include "SubmapFilter.hpp"
#include "ThreadPool.hpp"
#include "Utils.hpp"

#include <ethosn_utils/Macros.hpp>

#include <algorithm>
#include <exception>
#include <future>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

namespace ethosn
{
namespace support_library
{

uint32_t g_NumWeightEncodingsStage1 = 0;
uint32_t g_NumWeightEncodingsStage2 = 0;

WeightEncodingRequest::WeightEncodingRequest(const HardwareCapabilities& capabilities)
    : m_Capabilities(capabilities)
{}

WeightEncodingRequest::WeightEncodingRequest(TensorInfo weightsTensorInfo,
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
                                             const WeightCompressionParams& testParams)
    : m_Capabilities(capabilities)
    , m_WeightsTensorInfo(std::move(weightsTensorInfo))
    , m_WeightsData(weightsData)
    , m_BiasTensorInfo(std::move(biasTensorInfo))
    , m_BiasData(std::move(biasData))
    , m_InputQuantizationInfo(inputQuantizationInfo)
    , m_OutputQuantizationInfo(outputQuantizationInfo)
    , m_StripeDepth(stripeDepth)
    , m_StrideY(strideY)
    , m_StrideX(strideX)
    , m_PaddingTop(paddingTop)
    , m_PaddingLeft(paddingLeft)
    , m_IterationSize(iterationSize)
    , m_Operation(operation)
    , m_Algorithm(algorithm)
    , m_Mode(mode)
    , m_TestParams(testParams)
{}

bool WeightEncodingRequest::operator==(const WeightEncodingRequest& r) const
{
    // Compare things in an order such that we avoid comparing the big data (weights and bias) until
    // we absolutely need to
    bool same = m_WeightsTensorInfo == r.m_WeightsTensorInfo && m_BiasTensorInfo == r.m_BiasTensorInfo &&
                m_InputQuantizationInfo == r.m_InputQuantizationInfo &&
                m_OutputQuantizationInfo == r.m_OutputQuantizationInfo && m_StripeDepth == r.m_StripeDepth &&
                m_StrideY == r.m_StrideY && m_StrideX == r.m_StrideX && m_PaddingTop == r.m_PaddingTop &&
                m_PaddingLeft == r.m_PaddingLeft && m_IterationSize == r.m_IterationSize &&
                m_Operation == r.m_Operation && m_Algorithm == r.m_Algorithm;
    if (!same)
    {
        return false;
    }

    // At this point just bias and weights need to be compared. Do bias first because it's smaller.
    if (m_BiasData != r.m_BiasData)
    {
        return false;
    }

    // Now just need to compare weights, but we can skip this if the shared_ptrs are the same
    if (m_WeightsData == r.m_WeightsData)
    {
        return true;
    }

    if (*m_WeightsData != *r.m_WeightsData)
    {
        return false;
    }

    return true;
}

namespace
{

typedef int16_t Weight;
typedef uint16_t WeightSymbol;

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

struct EncodedOfm
{
    std::vector<uint8_t> m_EncodedWeights;
    uint32_t m_NumOfBits;
};

/// Stores a list of bit arrays, which are conceptually joined together end-to-end.
/// This makes it very fast to join multiple bit arrays without having to do any copying.
class BitstreamRope
{
public:
    struct Element
    {
        std::vector<uint8_t> bytes;
        uint32_t numBits;
    };

    void ReserveNumElements(size_t numElements);
    void AddElement(Element&& e);
    void AddRope(BitstreamRope&& rope);

    uint32_t GetTotalBits() const;

    /// Joins the stored bits to a single array. This is an expensive operation so is best left
    /// until a linear array is really needed.
    std::vector<uint8_t> Resolve() const;

private:
    uint32_t m_TotalBits = 0;
    std::vector<Element> m_Elements;
};

class BitstreamWriter;

uint8_t WeightOffsetClamp(WeightSymbol offset)
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
Weight AbsWeight(Weight weight)
{
    return static_cast<Weight>(weight < 0 ? -weight : weight);
}

WeightSymbol WeightToSymbol(Weight weight)
{
    // See Ethos-N78 MCE specification, section 6.8.6.3.2
    return static_cast<WeightSymbol>((AbsWeight(weight) << 1u) - (weight < 0));
}

Weight SymbolToWeight(WeightSymbol weightSymbol)
{
    uint16_t sign = weightSymbol & 1;
    uint16_t mag  = static_cast<uint16_t>((weightSymbol + 1) >> 1);
    return static_cast<Weight>(sign ? -mag : mag);
}

/// Private implementation of results passed from stage 1 encoding to stage 2 encoding.
struct Stage1Results : public IStage1Results
{
    Stage1Results(WeightEncodingRequest&& r)
        : request(std::move(r))
    {}

    WeightEncodingRequest request;

    uint32_t numOfms          = 0;
    uint32_t numIterationsOfm = 0;
    uint32_t numOfmInParallel = 0;
    std::vector<SubmapFilter> wideSubfilters;
    std::vector<EncodedOfm> encodedStreams;
};

/// Private implementation of the pending calculation of stage 1 results.
struct Stage1ResultsFuture : public IStage1ResultsFuture
{
    std::unique_ptr<IStage1Results> Wait() override;

    /// Shared state between all worker threads for stage 1.
    /// Includes the shared input data as well as the shared results.
    struct SharedState
    {
        // Inputs not needed in stage 2
        std::vector<std::vector<uint32_t>> perOgOfms;
        std::vector<std::unique_ptr<WeightCompressionParams>> compressionParams;
        std::vector<SubmapFilter> subfilters;

        // Results
        std::unique_ptr<Stage1Results> results;
    };

    /// This shared state is jointly owned by all the worker threads, plus this future object.
    std::shared_ptr<SharedState> m_SharedState;
    /// The futures for each of the tasks enqueued onto the thread pool.
    /// These are needed so that we can wait on them to finish.
    std::vector<std::future<void>> m_WaitHandles;
};

namespace
{

template <typename T>
std::vector<Weight>
    ConvertToUncompressedWeights(const T* const weights, const size_t numWeights, const int32_t zeroPoint)
{
    std::vector<Weight> uncompressedWeights;
    uncompressedWeights.reserve(numWeights);

    const auto correctZeroPoint = [zeroPoint](const T w) { return static_cast<Weight>(w - zeroPoint); };

    std::transform(weights, &weights[numWeights], std::back_inserter(uncompressedWeights), correctZeroPoint);

    return uncompressedWeights;
}
}    // namespace

/// BitstreamWriter is a helper class that supports writing packed bitfields into a vector.
class BitstreamWriter
{
public:
    BitstreamWriter(uint32_t capacityBits);

    // Returns the current write position in the bitstream (in bits)
    size_t GetOffset();

    // Write an element to end of the stream.
    void Write(uint8_t elem, uint32_t numBits);

    // Write an element to the stream. Offset specifies where to start writing in the stream.
    template <class T>
    void Write(const T* elem, uint32_t numBits);

    // Returns the stream as a uint8_t vector
    const std::vector<uint8_t>& GetBitstream();

private:
    std::vector<uint8_t> m_Bitstream;
    size_t m_EndPos;
};

BitstreamWriter::BitstreamWriter(uint32_t capacityBits)
    : m_EndPos(0)
{
    m_Bitstream.reserve(utils::DivRoundUp(capacityBits, 8));
}

size_t BitstreamWriter::GetOffset()
{
    return m_EndPos;
}

void BitstreamWriter::Write(uint8_t elem, uint32_t numBits)
{
    if (numBits == 0)
    {
        return;
    }
    assert(numBits <= 8);

    // Make sure there is enough space in the vector for the new bits, so we can index into it later
    const size_t requiredSize = utils::DivRoundUp(static_cast<uint32_t>(m_EndPos) + numBits, 8);
    if (requiredSize > m_Bitstream.size())
    {
        m_Bitstream.push_back(0);
    }

    // The operation is split into two parts - "a" and "b". a is the part which is appended to the partially-complete
    // byte at the end of m_Bitstream, and b is the part which is appended as a new byte to m_Bitstream.
    // There is always an "a", but not always a "b" (if the number of bits we are appending doesn't overflow into the
    // next byte)

    const uint32_t destBitIdxA = m_EndPos % 8;
    const uint32_t numBitsA    = std::min<uint32_t>(8 - destBitIdxA, numBits);
    const uint8_t bitsA        = static_cast<uint8_t>(elem & ((1u << numBitsA) - 1u));
    uint8_t& destA             = m_Bitstream[m_EndPos / 8];
    destA                      = static_cast<uint8_t>(destA | (bitsA << destBitIdxA));

    const uint32_t numBitsB = numBits - numBitsA;
    if (numBitsB > 0)
    {
        const uint8_t bitsB = static_cast<uint8_t>((elem >> numBitsA) & ((1u << numBitsB) - 1u));
        uint8_t& destB      = m_Bitstream.back();
        destB               = bitsB;
    }

    m_EndPos += numBits;
}

template <class T>
void BitstreamWriter::Write(const T* elem, uint32_t numBits)
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(elem);

    while (numBits > 0)
    {
        Write(*p, std::min(numBits, 8U));

        if (numBits <= 8)
        {
            break;
        }
        numBits -= 8;
        ++p;
    }
}

const std::vector<uint8_t>& BitstreamWriter::GetBitstream()
{
    return m_Bitstream;
}

/// Generate vector of weight compression parameters
std::vector<std::unique_ptr<WeightCompressionParams>> GenerateCompressionParams(uint32_t numOfmInParallel)
{
    std::vector<std::unique_ptr<WeightCompressionParams>> params(numOfmInParallel);
    std::generate(params.begin(), params.end(), std::make_unique<WeightCompressionParams>);
    return params;
}

/// Number of Ofm processed in parallel which is the minimum number of
/// weights streams that need to be loaded at the same time for all the
/// mce interfaces to start producing an Ofm each.
uint32_t GetNumOfmInParallel(const uint32_t numOfm,
                             const uint32_t numSrams,
                             const uint32_t stripeDepth,
                             const DataFormat dataFormat)
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

static uint8_t CalcBitWidth(size_t value, uint8_t minWidth)
{
    uint8_t bitwidth = minWidth;
    while ((1ull << bitwidth) <= value)
    {
        ++bitwidth;
    }
    return bitwidth;
}

/// Create vector of weight symbol frequency pairs where the DIROFS, Palette size and Palette has
/// been applied.
WeightSymbolFreqInfo
    CreateUncompressedSymbolFreqs(const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs,
                                  const std::map<Weight, uint8_t>& inversePalette,
                                  size_t paletteSize,
                                  uint8_t weightOffset)
{
    WeightSymbolFreqInfo symbolFreqInfo{};
    symbolFreqInfo.m_SymbolFreqPairs.reserve(symbolFreqPairs.size());
    symbolFreqInfo.m_MaxSymbol = 0;
    symbolFreqInfo.m_MinSymbol = std::numeric_limits<WeightSymbol>::max();

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

        symbolFreqInfo.m_MinSymbol = std::min(uncompressedSymbol, symbolFreqInfo.m_MinSymbol);
        symbolFreqInfo.m_MaxSymbol = std::max(uncompressedSymbol, symbolFreqInfo.m_MaxSymbol);
        symbolFreqInfo.m_SymbolFreqPairs.emplace_back(std::make_pair(uncompressedSymbol, symbolFreqPair.second));
    }

    return symbolFreqInfo;
}

/// Find the optimal GRC parameters for the specified weight symbol frequency pairs.
uint32_t FindGRCParams(WeightCompressionParams& params,
                       const WeightSymbolFreqInfo& symbolFreqPairInfo,
                       const WeightSymbolFreqInfo& noPaletteSymbolFreqPairInfo,
                       const HardwareCapabilities& capabilities)
{
    const uint32_t ifmConsumedPerEnginex3d4 =
        (3 * capabilities.GetIgsPerEngine() * capabilities.GetNumberOfEngines()) / 4;

    constexpr uint8_t maxNumQuotientBits = 31;
    constexpr uint32_t wDiv0             = static_cast<uint32_t>(WeightCompressionParams::WDivisor::WDIV_0);
    constexpr uint32_t wDiv5             = static_cast<uint32_t>(WeightCompressionParams::WDivisor::WDIV_5);

    // If the no palette vector is not empty, it should be used for the uncompressed bitcost
    const auto& uncompressedSymbolFreqInfo =
        (noPaletteSymbolFreqPairInfo.m_SymbolFreqPairs.empty() ? symbolFreqPairInfo : noPaletteSymbolFreqPairInfo);

    // Calculate the bitcost to use uncompressed symbols
    uint8_t symbolBitWidth = CalcBitWidth(uncompressedSymbolFreqInfo.m_MaxSymbol, 2);

    uint32_t uncompressedBitcost = 0;
    for (const auto& symbolFreqPair : uncompressedSymbolFreqInfo.m_SymbolFreqPairs)
    {
        uncompressedBitcost += (symbolFreqPair.second * symbolBitWidth);
    }

    const uint32_t minWidth = CalcBitWidth(symbolFreqPairInfo.m_MinSymbol, 2);
    const uint32_t maxWidth = CalcBitWidth(symbolFreqPairInfo.m_MaxSymbol, 1);
    // If the largest symbol has a bit width larger than wDiv5, the start divisor must be adjusted to
    // not exceed maxNumQuotientBits.
    const uint32_t startDiv = std::max(maxWidth > wDiv5 ? maxWidth - wDiv5 : wDiv0, std::min(wDiv5, minWidth - 2));
    const uint32_t endDiv   = std::min(wDiv5, maxWidth - 1);

    // Calculate the bitcost for each WDiv to find the one with the lowest overall bitcost. Use the
    // uncompressed bitcost as the initial best choice to include it in the selection process.
    uint32_t bestBitcost                       = uncompressedBitcost;
    WeightCompressionParams::WDivisor bestWDiv = WeightCompressionParams::WDivisor::UNCOMPRESSED;
    bool truncated                             = false;
    for (uint32_t i = startDiv; i <= endDiv; ++i)
    {
        uint32_t sumQuots        = 0;
        uint32_t sumTruncQuots   = 0;
        uint32_t wUnary1Len      = 0;
        uint32_t wUnary1TruncLen = 0;
        uint32_t sumRemain       = 0;
        bool canTruncate         = (symbolFreqPairInfo.m_SymbolFreqPairs.size() <= 3);
        for (const auto& symbolFreqPair : symbolFreqPairInfo.m_SymbolFreqPairs)
        {
            const uint32_t numQuotientBits = (symbolFreqPair.first >> i);
            canTruncate                    = canTruncate && numQuotientBits < 3;

            if (numQuotientBits > maxNumQuotientBits)
            {
                // Too many quotient bits, skip to next WDiv
                sumQuots = UINT32_MAX;
                break;
            }

            sumQuots += (numQuotientBits + 1) * symbolFreqPair.second;
            wUnary1Len += ((numQuotientBits + 1) / 2) * symbolFreqPair.second;

            sumTruncQuots += (numQuotientBits > 0 ? 2 : 1) * symbolFreqPair.second;
            wUnary1TruncLen += (numQuotientBits > 0 ? 1 : 0) * symbolFreqPair.second;

            sumRemain += i * symbolFreqPair.second;
        }

        if (sumQuots == UINT32_MAX)
        {
            continue;
        }

        if (canTruncate)
        {
            sumQuots   = sumTruncQuots;
            wUnary1Len = wUnary1TruncLen;
        }

        // Calculate the total bitcost for the GRC chunk packing with padding
        // See Ethos-N78 MCE Specification, section 6.8.6.3.5
        uint32_t bitcost =
            utils::RoundUpToNearestMultiple(sumQuots - wUnary1Len, ifmConsumedPerEnginex3d4) + wUnary1Len + sumRemain;

        if (bitcost < bestBitcost)
        {
            bestBitcost = bitcost;
            bestWDiv    = static_cast<WeightCompressionParams::WDivisor>(i);
            truncated   = canTruncate;
        }
    }

    params.m_Wdiv = bestWDiv;
    // Ignore truncated if uncompressed is used
    params.m_TruncationEnabled = truncated && bestWDiv != WeightCompressionParams::WDivisor::UNCOMPRESSED;

    return bestBitcost;
}

/// Create a palette of the specified size
void CreatePalette(WeightCompressionParams& params,
                   const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs,
                   uint8_t paletteSize,
                   bool palettePadding)
{
    // See Ethos-N78 MCE Specification, section 6.8.6.3.4
    std::vector<uint16_t> palette(paletteSize);
    std::map<Weight, uint8_t> inversePalette;
    uint8_t noPaddingSize = static_cast<uint8_t>(paletteSize - palettePadding);

    assert(paletteSize > 0 && paletteSize <= 32);

    WeightSymbol maxSymbol = std::max_element(symbolFreqPairs.begin(), symbolFreqPairs.begin() + noPaddingSize)->first;
    const uint32_t maxWeightMag    = static_cast<uint32_t>(AbsWeight(SymbolToWeight(maxSymbol)));
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

/// Find Palette parameters for the specified weight symbol frequency pairs
bool FindPaletteParams(WeightCompressionParams& params,
                       const std::vector<std::pair<WeightSymbol, uint32_t>>& symbolFreqPairs)
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
            weightOffset   = params.m_Zdiv != WeightCompressionParams::ZDivisor::RLE_DISABLED;
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

/// Find the optimal RLE parameters for the specified weights
uint32_t FindRLEParams(WeightCompressionParams& params,
                       const ZeroGroupInfo& zeroGroupInfo,
                       const HardwareCapabilities& capabilities)
{
    const uint32_t ifmConsumedPerEnginex3d4 =
        (3 * capabilities.GetIgsPerEngine() * capabilities.GetNumberOfEngines()) / 4;
    const uint32_t ifmConsumedPerEngined2 = (capabilities.GetIgsPerEngine() * capabilities.GetNumberOfEngines()) / 2;

    constexpr uint32_t zDiv3 = static_cast<uint32_t>(WeightCompressionParams::ZDivisor::ZDIV_3);

    const uint32_t minWidth = CalcBitWidth(zeroGroupInfo.m_MinGroup, 2);
    const uint32_t maxWidth = CalcBitWidth(zeroGroupInfo.m_MaxGroup, 1);
    const uint32_t startDiv = std::min(zDiv3, minWidth - 2);
    const uint32_t endDiv   = std::min(zDiv3, maxWidth - 1);

    // Find the ZDiv with the lowest overall bitcost
    uint32_t bestBitcost                       = UINT32_MAX;
    WeightCompressionParams::ZDivisor bestZDiv = WeightCompressionParams::ZDivisor::ZDIV_0;
    for (uint32_t i = startDiv; i <= endDiv; ++i)
    {
        uint32_t sumQuots  = 0;
        uint32_t sumRemain = 0;
        for (uint32_t group : zeroGroupInfo.m_ZeroGroups)
        {
            sumQuots += (group >> i) + 1;
            sumRemain += i;
        }

        // Calculate the total bitcost for the RLE chunk packing with padding
        // See Ethos-N78 MCE Specification, section 6.8.6.3.5
        uint32_t packSize = i < zDiv3 ? ifmConsumedPerEnginex3d4 : ifmConsumedPerEngined2;
        uint32_t bitcost  = utils::RoundUpToNearestMultiple(sumQuots, packSize) + sumRemain;

        if (bitcost < bestBitcost)
        {
            bestBitcost = bitcost;
            bestZDiv    = static_cast<WeightCompressionParams::ZDivisor>(i);
        }
    }

    params.m_Zdiv = bestZDiv;

    return bestBitcost;
}

/// Find optimal compression parameter for the specified weights
void FindWeightCompressionParams(WeightCompressionParams& newParams,
                                 const WeightCompressionParams& prevParams,
                                 const std::vector<uint8_t>& weights,
                                 const TensorInfo& weightsTensorInfo,
                                 const HardwareCapabilities& capabilities)
{
    const int32_t zeroPoint           = weightsTensorInfo.m_QuantizationInfo.GetZeroPoint();
    const uint8_t rawZeroPoint        = static_cast<uint8_t>(zeroPoint);
    const uint32_t conversionOffset   = (weightsTensorInfo.m_DataType == DataType::INT8_QUANTIZED) ? 128U : 0;
    const int32_t conversionZeroPoint = static_cast<int32_t>(conversionOffset) + zeroPoint;

    // Make frequency table containing an entry for each different weight symbol
    std::array<std::pair<WeightSymbol, uint32_t>, 256> frequencyTable;

    // Initialize the table, filling in the weight symbol and zeroing the frequency
    uint32_t rawWeight = 0;
    for (auto& pair : frequencyTable)
    {
        const Weight weight =
            static_cast<Weight>(static_cast<uint8_t>(rawWeight + conversionOffset) - conversionZeroPoint);
        pair.first  = WeightToSymbol(weight);
        pair.second = 0;
        ++rawWeight;
    }

    ZeroGroupInfo zeroGroupInfo{};
    zeroGroupInfo.m_ZeroGroups.reserve(weights.size() + 1);
    zeroGroupInfo.m_MaxGroup = 0U;
    zeroGroupInfo.m_MinGroup = UINT32_MAX;
    auto lastNonZeroIt       = weights.begin();
    for (auto wIt = weights.begin(); wIt != weights.end(); ++wIt)
    {
        frequencyTable[*wIt].second++;

        if (*wIt != rawZeroPoint)
        {
            const uint32_t numZeroes = static_cast<uint32_t>(wIt - lastNonZeroIt);
            zeroGroupInfo.m_ZeroGroups.push_back(numZeroes);
            zeroGroupInfo.m_MinGroup = std::min(numZeroes, zeroGroupInfo.m_MinGroup);
            zeroGroupInfo.m_MaxGroup = std::max(numZeroes, zeroGroupInfo.m_MaxGroup);
            lastNonZeroIt            = wIt + 1;
        }
    }

    const uint32_t numZeroes = static_cast<uint32_t>(weights.end() - lastNonZeroIt);
    zeroGroupInfo.m_ZeroGroups.push_back(numZeroes);
    zeroGroupInfo.m_MinGroup = std::min(numZeroes, zeroGroupInfo.m_MinGroup);
    zeroGroupInfo.m_MaxGroup = std::max(numZeroes, zeroGroupInfo.m_MaxGroup);

    // Convert to vector and sort
    WeightSymbolFreqInfo sortedSymbolFreqInfo{};
    sortedSymbolFreqInfo.m_MaxSymbol = 0U;
    sortedSymbolFreqInfo.m_MinSymbol = std::numeric_limits<WeightSymbol>::max();
    WeightSymbol minNonZeroSymbol    = std::numeric_limits<WeightSymbol>::max();
    for (const std::pair<WeightSymbol, uint32_t>& p : frequencyTable)
    {
        if (p.second > 0)
        {
            sortedSymbolFreqInfo.m_MinSymbol = std::min(sortedSymbolFreqInfo.m_MinSymbol, p.first);
            minNonZeroSymbol                 = p.first > 0 ? std::min(minNonZeroSymbol, p.first) : minNonZeroSymbol;
            sortedSymbolFreqInfo.m_MaxSymbol = std::max(sortedSymbolFreqInfo.m_MaxSymbol, p.first);
            sortedSymbolFreqInfo.m_SymbolFreqPairs.push_back(p);
        }
    }

    std::sort(sortedSymbolFreqInfo.m_SymbolFreqPairs.begin(), sortedSymbolFreqInfo.m_SymbolFreqPairs.end(),
              [](const auto& a, const auto& b) {
                  // If two symbols have the same frequency, place the larger symbol first to give it a better
                  // chance to be placed in the palette.
                  return a.second > b.second || (a.second == b.second && a.first > b.first);
              });

    auto zeroIter =
        find_if(sortedSymbolFreqInfo.m_SymbolFreqPairs.begin(), sortedSymbolFreqInfo.m_SymbolFreqPairs.end(),
                [](const std::pair<WeightSymbol, uint32_t>& e) { return e.first == 0; });

    std::vector<std::pair<uint32_t, WeightCompressionParams>> passCostParamPairs;
    // If there are zero weights, run an extra pass with RLE enabled
    uint32_t numPasses = zeroGroupInfo.m_MaxGroup > 0 ? 2 : 1;
    for (uint32_t pass = 0; pass < numPasses; ++pass)
    {
        WeightCompressionParams params = newParams;
        uint32_t bitCost               = 0;

        // Only use RLE for the second pass
        if (pass > 0)
        {
            bitCost += FindRLEParams(params, zeroGroupInfo, capabilities);
            // If there are only zero weights, there is nothing more to do.
            if (sortedSymbolFreqInfo.m_SymbolFreqPairs.size() == 1)
            {
                // There are only zero weights so only the ZDivisor will be used. All other compression
                // parameters should stay the same as the previous OFM.
                WeightCompressionParams::ZDivisor zDiv            = params.m_Zdiv;
                WeightCompressionParams::EncodingParams encParams = params.m_EncodingParams;
                params                                            = prevParams;
                params.m_Zdiv                                     = zDiv;
                params.m_EncodingParams                           = encParams;
                params.m_InitialParameters                        = false;
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
            sortedSymbolFreqInfo.m_SymbolFreqPairs.erase(zeroIter);
            sortedSymbolFreqInfo.m_MinSymbol = minNonZeroSymbol;
        }

        // Attempt to find palette parameters that fit the weight symbols
        if (!FindPaletteParams(params, sortedSymbolFreqInfo.m_SymbolFreqPairs))
        {
            // No palette will be used so find the smallest symbol to use as weight offset
            params.m_WeightOffset = WeightOffsetClamp(sortedSymbolFreqInfo.m_MinSymbol);
            params.m_PaletteBits  = 0;
        }

        // To be able to find the best GRC params, we first need to create a vector with the final
        // symbols that should be compressed.
        WeightSymbolFreqInfo uncompressedSymbolFreqInfo =
            CreateUncompressedSymbolFreqs(sortedSymbolFreqInfo.m_SymbolFreqPairs, params.m_InversePalette,
                                          params.m_Palette.size(), params.m_WeightOffset);

        // If a palette is used and it does not contain all the values, the GRC param finder needs an
        // additional vector where the palette is not used to correctly evaluate the cost of using
        // uncompressed mode.
        WeightSymbolFreqInfo uncompressedNoPaletteSymbolFreqInfo = { {}, 0, 0 };
        uint8_t noPaletteOffset                                  = 0;
        // Inverse palette has the actual size without padding
        if (params.m_InversePalette.size() != sortedSymbolFreqInfo.m_SymbolFreqPairs.size())
        {
            noPaletteOffset = WeightOffsetClamp(sortedSymbolFreqInfo.m_MinSymbol);
            uncompressedNoPaletteSymbolFreqInfo =
                CreateUncompressedSymbolFreqs(sortedSymbolFreqInfo.m_SymbolFreqPairs, {}, 0, noPaletteOffset);
        }

        bitCost += FindGRCParams(params, uncompressedSymbolFreqInfo, uncompressedNoPaletteSymbolFreqInfo, capabilities);
        if (params.m_Wdiv == WeightCompressionParams::WDivisor::UNCOMPRESSED &&
            !uncompressedNoPaletteSymbolFreqInfo.m_SymbolFreqPairs.empty())
        {
            params.m_Palette.clear();
            params.m_InversePalette.clear();

            // Change to offset without the palette
            params.m_WeightOffset = noPaletteOffset;
            // Calculate the uncompressed bitwidth
            params.m_PaletteBits = CalcBitWidth(uncompressedNoPaletteSymbolFreqInfo.m_MaxSymbol, 2) - 2;
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

/// Select compression parameters based through analyzis of the weight stream.
const WeightCompressionParams
    SelectWeightCompressionParams(const WeightEncodingRequest& request,
                                  const std::vector<uint8_t>& weights,
                                  const TensorInfo& weightsTensorInfo,
                                  const WeightCompressionParams::EncodingParams& encodingParams,
                                  const WeightCompressionParams& prevCompParams)
{
    WeightCompressionParams params(encodingParams);

    switch (request.m_Mode)
    {
        case WeightCompMode::UNCOMPRESSED:
            assert(params.m_Wdiv == WeightCompressionParams::WDivisor::UNCOMPRESSED);
            assert(params.m_Zdiv == WeightCompressionParams::ZDivisor::RLE_DISABLED);
            assert(params.m_Palette.size() == 0);
            break;
        case WeightCompMode::DIRECT_RLE:
            params.m_Wdiv         = request.m_TestParams.m_Wdiv;
            params.m_Zdiv         = request.m_TestParams.m_Zdiv;
            params.m_WeightOffset = 1;
            break;
        case WeightCompMode::DIRECT_TRUNC:
            params.m_TruncationEnabled = true;
            params.m_Wdiv              = request.m_TestParams.m_Wdiv;
            break;
        case WeightCompMode::DIRECT:
            params.m_Wdiv = request.m_TestParams.m_Wdiv;
            assert(params.m_Zdiv == WeightCompressionParams::ZDivisor::RLE_DISABLED);
            break;
        case WeightCompMode::PALETTE_RLE:
            ETHOSN_FALLTHROUGH;    // intentional fallthrough
        case WeightCompMode::PALETTE_DIRECT_RLE:
            params.m_WeightOffset = 1;
            ETHOSN_FALLTHROUGH;    // intentional fallthrough
        case WeightCompMode::PALETTE:
            ETHOSN_FALLTHROUGH;    // intentional fallthrough
        case WeightCompMode::PALETTE_DIRECT:
            params.m_Wdiv = request.m_TestParams.m_Wdiv;
            // sanity check WDIV != 7 for palette direct modes
            assert(params.m_Wdiv != WeightCompressionParams::WDivisor::UNCOMPRESSED ||
                   ((request.m_Mode != WeightCompMode::PALETTE_DIRECT) &&
                    (request.m_Mode != WeightCompMode::PALETTE_DIRECT_RLE)));
            params.m_Zdiv              = request.m_TestParams.m_Zdiv;
            params.m_TruncationEnabled = false;
            params.m_Palette           = request.m_TestParams.m_Palette;
            params.m_InversePalette    = request.m_TestParams.m_InversePalette;
            params.m_PaletteBits       = request.m_TestParams.m_PaletteBits;
            break;
        case WeightCompMode::PALETTE_DIRECT_TRUNC_RLE:
            params.m_WeightOffset = 1;
            ETHOSN_FALLTHROUGH;    // intentional fallthrough
        case WeightCompMode::PALETTE_TRUNC_RLE:
            params.m_TruncationEnabled = true;
            ETHOSN_FALLTHROUGH;    // intentional fallthrough
        case WeightCompMode::PALETTE_TRUNC:
            ETHOSN_FALLTHROUGH;    // intentional fallthrough
        case WeightCompMode::PALETTE_DIRECT_TRUNC:
            params.m_Wdiv              = request.m_TestParams.m_Wdiv;
            params.m_Zdiv              = request.m_TestParams.m_Zdiv;
            params.m_TruncationEnabled = true;
            params.m_Palette           = request.m_TestParams.m_Palette;
            params.m_InversePalette    = request.m_TestParams.m_InversePalette;
            params.m_PaletteBits       = request.m_TestParams.m_PaletteBits;
            break;
        case WeightCompMode::AUTO:
            FindWeightCompressionParams(params, prevCompParams, weights, weightsTensorInfo, request.m_Capabilities);
            break;
        default:
            throw NotSupportedException("Unsupported weight compression mode");
            break;
    }

    return params;
}

/// Get the size in bytes of the OFM bias.
uint32_t GetOfmBiasSize(const TensorInfo& weightsTensorInfo)
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

/// Determine if OFM parameters need to be reloaded.
bool GetOfmReload(const WeightCompressionParams& compParams,
                  const WeightCompressionParams& prevCompParams,
                  const bool firstOfm)
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

/// Convert 8-bit weights to 9-bit weights including zero point.
std::vector<Weight> GetUncompressedWeights(const std::vector<uint8_t>& weights, const TensorInfo& weightsTensorInfo)
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

/// Convert 9-bit signed weight to 9-bit unsigned weight symbol.
WeightSymbol DirectEncode(const Weight weight, const WeightCompressionParams& compParams)
{
    WeightSymbol x = WeightToSymbol(weight);

    x = static_cast<WeightSymbol>(x + compParams.m_Palette.size());

    assert(compParams.m_WeightOffset >= 1 || compParams.m_Zdiv == WeightCompressionParams::ZDivisor::RLE_DISABLED);

    assert(x >= compParams.m_WeightOffset);
    x = static_cast<WeightSymbol>(x - compParams.m_WeightOffset);

    assert(x >= compParams.m_Palette.size());

    return x;
}

/// Palette or direct encode the uncompressed weight symbol stream.
void PaletteZrunEncode(const std::vector<Weight>& uncompressedWeights,
                       const WeightCompressionParams& compParams,
                       std::vector<WeightSymbol>& weightSymbols,
                       std::vector<WeightSymbol>& zeroSymbols)
{
    // Please refer to Ethos-N78 MCE specification, section 6.8.6.3.2
    const std::map<Weight, uint8_t>& invPalette = compParams.m_InversePalette;

    std::vector<Weight>::const_iterator wItr;
    uint32_t zeroCnt = 0;

    wItr = uncompressedWeights.begin();
    while (wItr != uncompressedWeights.end())
    {
        if (compParams.m_Zdiv != WeightCompressionParams::ZDivisor::RLE_DISABLED)
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

        if (compParams.m_Zdiv != WeightCompressionParams::ZDivisor::RLE_DISABLED)
        {
            // After encountering a non zero symbol, writes
            // accumulated RLE symbol then resets the RLE.
            zeroSymbols.push_back(static_cast<WeightSymbol>(zeroCnt));
            zeroCnt = 0;
        }

        // sanity check: non-zero weight if RLE
        assert(value != 0 || compParams.m_Zdiv == WeightCompressionParams::ZDivisor::RLE_DISABLED);

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

    if (compParams.m_Zdiv != WeightCompressionParams::ZDivisor::RLE_DISABLED)
    {
        zeroSymbols.push_back(static_cast<WeightSymbol>(zeroCnt));
    }

    assert((zeroSymbols.size() == (weightSymbols.size() + 1)) ||
           (compParams.m_Zdiv == WeightCompressionParams::ZDivisor::RLE_DISABLED));
}

/// Golomb Rice code the palette/direct weight symbol stream and pack the
/// symbols into chunks.
void GRCCompressPackChunk(const std::vector<WeightSymbol>& weightSymbols,
                          const std::vector<WeightSymbol>& zeroSymbols,
                          const WeightCompressionParams& compParams,
                          BitstreamWriter& writer,
                          const HardwareCapabilities& capabilities)
{
    const uint32_t ifmConsumedPerEnginex3d4 =
        (3 * capabilities.GetIgsPerEngine() * capabilities.GetNumberOfEngines()) / 4;
    const uint32_t ifmConsumedPerEngined2 = (capabilities.GetIgsPerEngine() * capabilities.GetNumberOfEngines()) / 2;

    bool unCompressed = compParams.m_Wdiv == WeightCompressionParams::WDivisor::UNCOMPRESSED;
    bool rleEnabled   = compParams.m_Zdiv != WeightCompressionParams::ZDivisor::RLE_DISABLED;

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

    int32_t nWeights = static_cast<int32_t>(weightSymbols.size());
    int32_t nZeros   = static_cast<int32_t>(zeroSymbols.size());

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
    int32_t zUnaryLen =
        (zDivisor < 3) ? static_cast<int32_t>(ifmConsumedPerEnginex3d4) : static_cast<int32_t>(ifmConsumedPerEngined2);

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
        bool wEnable    = (balance < static_cast<int32_t>(ifmConsumedPerEngined2)) && (wPos < nWeights);
        bool zEnable    = balance >= 0 && rleEnabled && zPos < nZeros;

        // maximum number of weight symbols
        int32_t maxNumWunary0Bits = (unCompressed && (wDivisor > 5)) ? static_cast<int32_t>(ifmConsumedPerEngined2)
                                                                     : static_cast<int32_t>(ifmConsumedPerEnginex3d4);

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

                        int32_t value = static_cast<int32_t>(weightSymbols[static_cast<size_t>(wPos)]);

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
                        int32_t value = static_cast<int32_t>(zeroSymbols[static_cast<size_t>(zPos)]);
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
            writer.Write(reinterpret_cast<const uint8_t*>(&wUnary0), static_cast<uint32_t>(maxNumWunary0Bits));
        }

        if (zEnable)
        {
            writer.Write(reinterpret_cast<const uint8_t*>(&zUnary), static_cast<uint32_t>(zUnaryLen));
        }

        if (wEnable && !unCompressed)
        {
            writer.Write(reinterpret_cast<const uint8_t*>(&wUnary1), static_cast<uint32_t>(wUnary1Len));
        }

        if (!wRemain[rmdPrevIdx].empty())
        {
            std::vector<int32_t>::iterator it;
            for (it = wRemain[rmdPrevIdx].begin(); it != wRemain[rmdPrevIdx].end(); ++it)
            {
                assert(*it <= 31 || unCompressed);
                int32_t value = *it;
                writer.Write(reinterpret_cast<const uint8_t*>(&value), static_cast<uint32_t>(wDivisor));
            }

            wRemain[rmdPrevIdx].clear();
        }

        if (!zRemain[rmdPrevIdx].empty())
        {
            std::vector<int32_t>::iterator it;
            for (it = zRemain[rmdPrevIdx].begin(); it != zRemain[rmdPrevIdx].end(); ++it)
            {
                assert(*it <= 7);
                writer.Write(static_cast<uint8_t>(*it), static_cast<uint32_t>(zDivisor));
            }

            zRemain[rmdPrevIdx].clear();
        }

        rmdIdx     = (rmdIdx + 1) % numRmdEntries;
        rmdPrevIdx = (rmdPrevIdx + 1) % numRmdEntries;

        prevWenable = wEnable;
        prevZenable = zEnable;

    } while (prevWenable || prevZenable);
}

/// Write the weight stream header. There is exactly one header per OFM.
void WriteWeightHeader(BitstreamWriter& writer,
                       const uint32_t streamLength,
                       const uint64_t ofmBias,
                       const size_t ofmBiasLength,
                       const bool ofmReload,
                       const uint32_t ofmScaling,
                       const uint32_t ofmShift,
                       const uint32_t ofmZeroPointCorrection)
{
    // See Ethos-N78 MCE Specification, section 6.8.6.2.2
    writer.Write(&streamLength, 16);
    writer.Write(&ofmBias, static_cast<uint32_t>(ofmBiasLength * 8U));
    writer.Write(&ofmReload, 1);

    if (ofmReload)
    {
        writer.Write(&ofmScaling, 16);
        writer.Write(&ofmShift, 6);
        writer.Write(&ofmZeroPointCorrection, 8);
    }
}

/// Write the weight payload header. There may be one or multiple payload headers in the weight stream.
void WritePayloadHeader(BitstreamWriter& writer, const size_t payloadLength, const WeightCompressionParams& compParams)
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
                Weight value = static_cast<Weight>(*itr);
                writer.Write(&value, compParams.m_PaletteBits + 2);
            }
        }
    }
}

// Calculates the exact offset and size in DRAM of each weight stripe
std::vector<WeightsMetadata> CalculateWeightsMetadata(const std::vector<BitstreamRope>& streamPerStripeOg,
                                                      uint32_t numOgPerStripe)
{
    std::vector<WeightsMetadata> metadata;
    uint32_t runningSize = 0;
    for (size_t i = 0; i < streamPerStripeOg.size(); i += numOgPerStripe)
    {
        uint32_t stripeSize = 0;
        for (size_t j = 0; j < numOgPerStripe; ++j)
        {
            stripeSize += static_cast<uint32_t>(utils::DivRoundUp(streamPerStripeOg[i + j].GetTotalBits(), 8));
        }
        metadata.push_back(WeightsMetadata{ runningSize, stripeSize });
        runningSize += stripeSize;
    }

    return metadata;
}

/// Gets the raw (unencoded) stream for all the weights required to calculate a single OFM.
std::vector<uint8_t> GetRawOfmStream(const uint8_t* weightData,
                                     uint32_t ofmIdx,
                                     uint32_t iteration,
                                     const TensorInfo& weightsTensorInfo,
                                     uint32_t strideY,
                                     uint32_t strideX,
                                     uint32_t iterationSize,
                                     ethosn::command_stream::MceOperation operation,
                                     CompilerMceAlgorithm algorithm,
                                     const std::vector<SubmapFilter>& subfilters,
                                     const std::vector<SubmapFilter>& wideSubfilters,
                                     const HardwareCapabilities& capabilities)
{
    assert(algorithm != CompilerMceAlgorithm::None);

    const uint32_t numUninterleavedIfmsPerIteration = iterationSize / (strideX * strideY);

    const uint32_t filterX = weightsTensorInfo.m_Dimensions[1];
    const uint32_t filterY = weightsTensorInfo.m_Dimensions[0];

    uint32_t numEngines      = capabilities.GetNumberOfEngines();
    uint32_t numIgsPerEngine = capabilities.GetIgsPerEngine();

    std::vector<uint8_t> result;
    result.reserve(filterX * filterY * iterationSize);

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
                    // We must tightly pack the final subfilter in the final slice
                    // (where each slice is the set of weights for as many IFMs as there are IGs).
                    const uint32_t numChannels =
                        (filterIdx == subfilters.size() - 1) ? channelsInThisSlice : numIfmsProcessedInParallel;

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
                                                   ? filter.GetWeightAt(weightData, y, x, i, ofmIdx)
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
                                isValidData ? filter.GetWeightAt(weightData, y, x, channel, ofmIdx)
                                            : static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.GetZeroPoint());
                            result.push_back(weight);
                            ++count;
                        }
                    }
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
        const uint32_t numSrams        = capabilities.GetNumberOfSrams();

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
                    weight = filter.GetWeightAt(weightData, 0, 0, rawIdx, ofmIdx);
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

        // Decompose the ofm index to find which ifm it corresponds to.
        const uint32_t channelMultiplierIdx = ofmIdx / numIfms;
        const uint32_t ifmIdx               = ofmIdx % numIfms;

        // Compared to 'regular' HWIO weights, we only need to specify the weights for as many IFMs as there are IGs
        // rather than all of the IFMs.
        // Only packs on set of weights and the HW will insert 0s accordingly after decoding.
        for (uint32_t filterIdx = 0; filterIdx < subfilters.size(); ++filterIdx)
        {
            const SubmapFilter& filter = subfilters[filterIdx];

            // Add weight data in row-major order, with the slice of as many IFMs as there are IGs, tightly packed
            // for each filter coordinate.
            for (uint32_t h = 0; h < filter.GetFilterY(); ++h)
            {
                for (uint32_t w = 0; w < filter.GetFilterX(); ++w)
                {
                    result.push_back(filter.GetWeightAt(weightData, h, w, ifmIdx, channelMultiplierIdx));
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

void BitstreamRope::ReserveNumElements(size_t numElements)
{
    m_Elements.reserve(numElements);
}

void BitstreamRope::AddElement(BitstreamRope::Element&& element)
{
    m_Elements.push_back(std::move(element));
    m_TotalBits += element.numBits;
}

void BitstreamRope::AddRope(BitstreamRope&& rope)
{
    for (Element& e : rope.m_Elements)
    {
        AddElement(std::move(e));
    }

    // Moved-from
    rope.m_Elements.clear();
    rope.m_TotalBits = 0;
}

uint32_t BitstreamRope::GetTotalBits() const
{
    return m_TotalBits;
}

std::vector<uint8_t> BitstreamRope::Resolve() const
{
    std::vector<uint8_t> result;
    result.reserve(utils::DivRoundUp(GetTotalBits(), 8));

    uint32_t numBitsResult = 0;
    for (uint32_t elementIdx = 0; elementIdx < m_Elements.size(); ++elementIdx)
    {
        const Element& element = m_Elements[elementIdx];

        // We need to append the new `stream` onto `mergedGroup` bit-by-bit.
        const uint32_t bitPos = numBitsResult % 8;
        if (bitPos == 0)
        {
            // Simple case where we are byte-aligned, and can do a regular copy
            std::copy(element.bytes.begin(), element.bytes.end(), std::back_inserter(result));
        }
        else
        {
            // Otherwise, we need to shuffle around the bits when appending.
            // Note that least-significant-bits are considered as "first" (little endian), which is left-most/position 0 in this diagram:
            //
            // This diagram shows the alignment between the two sets of bytes, for the case where `bitPos` = 6.
            //
            //  |---------------|
            //  |0 1 2 3 4 5 - -| <-     output bit stream (`mergedGroup`), before appending the new stream.
            //  |---------------|        Last two bits are not yet set.
            //
            //                   new byte 0        new byte 1       new byte 2
            //              |---------------|---------------|---------------|
            //              |0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7|    <- bit stream being appended (`stream`)
            //              |---------------|---------------|---------------|
            //
            //
            //  |---------------|---------------|---------------|
            //  |0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7|    <- output bit stream (`mergedGroup`) after new stream appended
            //  |---------------|---------------|---------------|
            //
            //  * The first bit of the new stream is placed at bit position 6 in the already-existing final byte of `mergedGroup`
            //  * the second bit of the new stream is placed at bit position 7 in the same byte
            //  * The third bit of the new stream is placed in a new byte in `mergedGroup`, at bit position 0

            const uint32_t invBitPos = 8 - bitPos;

            // Function to construct the left/least-significant-bits of an output byte
            // This comes from the right/most-significant-bits of the previous byte in the new stream
            auto firstPart = [&](uint8_t prevStreamByte) -> uint8_t {
                return static_cast<uint8_t>(prevStreamByte >> invBitPos);
            };
            // Function to construct the right/most-significant-bits of an output byte
            // This comes from the left/least-significant-bits of the next byte in the new stream
            auto secondPart = [&](uint8_t nextStreamByte) -> uint8_t {
                return static_cast<uint8_t>(((nextStreamByte & ((1 << invBitPos) - 1)) << (bitPos)));
            };

            // Get the last element of the merged stream, as we'll need to append some bits to this and update it
            uint8_t lastByte = static_cast<uint8_t>(result.back());
            // Merge with new stream byte
            lastByte      = static_cast<uint8_t>(lastByte | secondPart(element.bytes[0]));
            result.back() = lastByte;

            // Loop byte-by-byte appending each of the rest of the bytes of the new stream
            // This might be faster to do multiple bytes at a time (e.g. 64-bit words). Would need to be careful about alignment though.
            uint8_t prevStreamByte = element.bytes[0];
            for (size_t i = 1; i < element.bytes.size() - 0; ++i)
            {
                const uint8_t nextStreamByte = element.bytes[i];
                const uint8_t newByte        = firstPart(prevStreamByte) | secondPart(nextStreamByte);
                result.push_back(newByte);

                prevStreamByte = nextStreamByte;
            }

            // The second half of the final byte now needs adding, but this final byte
            // might not be a full byte, so check if there's actually anything to add
            uint32_t finalByteNumBits = element.numBits % 8;
            if (finalByteNumBits == 0)
            {
                finalByteNumBits = 8;
            }
            if (finalByteNumBits > invBitPos)
            {
                result.push_back(firstPart(prevStreamByte));
            }
        }

        numBitsResult += element.numBits;
    }

    return result;
}

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
std::vector<BitstreamRope> MergeStreams(std::vector<BitstreamRope>& streams,
                                        uint32_t numGroups,
                                        uint32_t numIterations,
                                        uint32_t numOfmPerSram)
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
    std::vector<BitstreamRope> result(numGroups);
    for (uint32_t groupIdx = 0; groupIdx < numGroups; ++groupIdx)
    {
        const std::vector<uint32_t>& group = groups[groupIdx];
        BitstreamRope& mergedGroup         = result[groupIdx];

        // Calculate size required and reserve space, to reduce overhead from reallocations
        mergedGroup.ReserveNumElements(group.size());

        for (uint32_t streamIdxWithinGroup = 0; streamIdxWithinGroup < group.size(); ++streamIdxWithinGroup)
        {
            uint32_t streamIdx    = group[streamIdxWithinGroup];
            BitstreamRope& stream = streams[streamIdx];

            mergedGroup.AddRope(std::move(stream));
        }
    }

    return result;
}

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
std::vector<BitstreamRope> MergeStreamsOgAndUpdateHeaders(std::vector<EncodedOfm>& streams,
                                                          uint32_t numGroups,
                                                          const uint32_t streamHeadersUpdateAlignment)
{
    // Assign each stream to a group (each group is stored as a vector of the stream indexes assigned to it).
    std::vector<std::vector<uint32_t>> groups(numGroups);
    for (uint32_t streamIdx = 0; streamIdx < streams.size(); ++streamIdx)
    {
        uint32_t groupIdx = streamIdx % numGroups;
        groups[groupIdx].push_back(streamIdx);
    }

    // For each group, merge all its streams together into one.
    std::vector<BitstreamRope> result(numGroups);
    for (uint32_t groupIdx = 0; groupIdx < numGroups; ++groupIdx)
    {
        const std::vector<uint32_t>& group = groups[groupIdx];
        BitstreamRope& mergedGroup         = result[groupIdx];

        // Pre-allocate a conservative estimate of capacity, to reduce number of reallocations as the vector grows.
        if (group.size() > 0)
        {
            mergedGroup.ReserveNumElements(group.size());
        }

        uint32_t numBitsStream = 0;

        for (uint32_t streamIdxWithinGroup = 0; streamIdxWithinGroup < group.size(); ++streamIdxWithinGroup)
        {
            uint32_t streamIdx           = group[streamIdxWithinGroup];
            std::vector<uint8_t>& stream = streams[streamIdx].m_EncodedWeights;

            // start position in byte
            uint32_t start = numBitsStream / 8;

            // start position in word (16 bytes)
            uint32_t startWord = start / streamHeadersUpdateAlignment;

            // end position in word
            // Note Ethos-N78: weight stream header starts at the SRAM bit position
            // following the last bit of the preceding weight stream.
            uint32_t endWord = numBitsStream + streams[streamIdx].m_NumOfBits;
            endWord          = (endWord + (streamHeadersUpdateAlignment * 8) - 1) / (streamHeadersUpdateAlignment * 8);
            uint16_t headerLength = static_cast<uint16_t>(endWord - startWord);
            uint8_t* headerPtr    = reinterpret_cast<uint8_t*>(&headerLength);

            // 2 x header bytes for new stream
            const uint8_t h1 = headerPtr[0];
            const uint8_t h2 = headerPtr[1];

            // Update header bytes
            stream[0] = h1;
            stream[1] = h2;

            mergedGroup.AddElement({ std::move(stream), streams[streamIdx].m_NumOfBits });

            numBitsStream += streams[streamIdx].m_NumOfBits;
        }
    }

    return result;
}

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
std::vector<uint8_t> InterleaveStreams(const std::vector<BitstreamRope>& streams, uint32_t numBytesPerStream)
{
    std::vector<std::vector<uint8_t>> resolvedStreams(streams.size());
    for (uint32_t i = 0; i < streams.size(); ++i)
    {
        resolvedStreams[i] = streams[i].Resolve();
    }

    // Calculate how long the longest stream is, which determines how big our output will be.
    uint32_t maxLength = 0;
    for (const std::vector<uint8_t>& s : resolvedStreams)
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
            const std::vector<uint8_t>& stream = resolvedStreams[streamIdx];

            int32_t numBytesToCopy =
                std::max(0, std::min(static_cast<int32_t>(numBytesPerStream),
                                     static_cast<int32_t>(stream.size()) - static_cast<int32_t>(streamOffset)));
            if (numBytesToCopy > 0)
            {
                std::copy(stream.begin() + streamOffset, stream.begin() + streamOffset + numBytesToCopy,
                          std::back_inserter(result));
            }

            uint32_t numZeroesToAdd = numBytesPerStream - static_cast<uint32_t>(numBytesToCopy);
            if (numZeroesToAdd)
            {
                std::fill_n(std::back_inserter(result), numZeroesToAdd, 0);
            }
        }
    }

    return result;
}

/// Encodes all the weights required to calculate a single OFM.
EncodedOfm EncodeOfm(const WeightEncodingRequest& request,
                     uint32_t ofmIdx,
                     uint32_t numOfmInParallel,
                     uint32_t numIterationsOfm,
                     uint32_t iteration,
                     const WeightCompressionParams::EncodingParams& params,
                     std::vector<std::unique_ptr<WeightCompressionParams>>& compressionParams,
                     const std::vector<SubmapFilter>& subfilters,
                     const std::vector<SubmapFilter>& wideSubfilters)
{
    uint32_t wdIdx = (ofmIdx % request.m_StripeDepth) % numOfmInParallel;

    // Grab a reference to previous compression parameters
    WeightCompressionParams& prevCompParams = static_cast<WeightCompressionParams&>(*compressionParams[wdIdx]);

    if (!prevCompParams.m_InitialParameters)
    {
        if (numIterationsOfm > 1)
        {
            prevCompParams.m_InitialParameters = iteration == 0;
        }

        uint32_t numOfmSetsPerStripe = utils::DivRoundUp(request.m_StripeDepth, numOfmInParallel);
        assert(numOfmSetsPerStripe >= 1);

        if ((ofmIdx % request.m_StripeDepth) == wdIdx && numOfmSetsPerStripe > 1)
        {
            prevCompParams.m_InitialParameters = true;
        }
    }

    std::vector<uint8_t> weights =
        GetRawOfmStream(request.m_WeightsData->data(), ofmIdx, iteration, request.m_WeightsTensorInfo,
                        request.m_StrideY, request.m_StrideX, request.m_IterationSize, request.m_Operation,
                        request.m_Algorithm, subfilters, wideSubfilters, request.m_Capabilities);

    const WeightCompressionParams compParams =
        SelectWeightCompressionParams(request, weights, request.m_WeightsTensorInfo, params, prevCompParams);

    const uint32_t ofmBiasSize = GetOfmBiasSize(request.m_WeightsTensorInfo);

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
    const auto isPerChannelQuantization = request.m_WeightsTensorInfo.m_QuantizationInfo.GetScales().size() > 1;
    const bool ofmReload =
        isPerChannelQuantization || GetOfmReload(compParams, prevCompParams, ofmIdx < numOfmInParallel);

    // Over-estimate of how many bits we need. This could be more accurate as we've already decided the best scheme.
    const uint32_t capacityBits = std::max(static_cast<uint32_t>(weights.size()) * 8 * 2, 1024u);
    BitstreamWriter writer(capacityBits);
    std::vector<WeightSymbol> weightSymbols, zeroSymbols;

    std::vector<Weight> uncompressedWeights = GetUncompressedWeights(weights, request.m_WeightsTensorInfo);
    PaletteZrunEncode(uncompressedWeights, compParams, weightSymbols, zeroSymbols);

    // Note the weight stream length will be filled later
    WriteWeightHeader(writer, 0xffff, static_cast<uint64_t>(params.m_OfmBias), ofmBiasSize, ofmReload,
                      params.m_OfmScaleFactor, params.m_OfmShift, params.m_OfmZeroPoint);

    uint32_t pldLen = static_cast<uint32_t>(weightSymbols.size());

    WritePayloadHeader(writer, pldLen, compParams);

    GRCCompressPackChunk(weightSymbols, zeroSymbols, compParams, writer, request.m_Capabilities);

    // Remember current compression parameters
    prevCompParams = compParams;

    return { std::move(writer.GetBitstream()), static_cast<uint32_t>(writer.GetOffset()) };
}

}    // namespace

uint64_t GetUncompressedWeightStripeSize(const WeightEncodingRequest& r)
{
    const uint64_t wh =
        static_cast<uint64_t>(r.m_WeightsTensorInfo.m_Dimensions[0]) * r.m_WeightsTensorInfo.m_Dimensions[1];
    if (r.m_Operation == command_stream::MceOperation::DEPTHWISE_CONVOLUTION)
    {
        return wh * r.m_StripeDepth;
    }
    else
    {
        return wh * r.m_IterationSize / (r.m_StrideX * r.m_StrideY) * r.m_StripeDepth;
    }
}

EncodedWeights EncodeWeights(WeightEncodingRequest&& request, ThreadPool& threadPool)
{
    std::unique_ptr<IStage1ResultsFuture> future  = EncodeWeightsStage1Async(std::move(request), threadPool);
    std::unique_ptr<IStage1Results> stage1Results = future->Wait();
    return EncodeWeightsStage2(std::move(stage1Results));
}

std::unique_ptr<IStage1ResultsFuture> EncodeWeightsStage1Async(WeightEncodingRequest&& requestIn,
                                                               ThreadPool& threadPool)
{
    g_NumWeightEncodingsStage1++;

    g_Logger.Verbose("Encode %lu weights, stripeDepth = %u, iterationSize = %u, algorithm = %s...",
                     requestIn.m_WeightsData->size(), requestIn.m_StripeDepth, requestIn.m_IterationSize,
                     ToString(requestIn.m_Algorithm).c_str());

    // State which will be shared between all the thread pool tasks
    auto sharedState = std::make_shared<Stage1ResultsFuture::SharedState>();

    // Move the request into the shared state, as it is needed by all the tasks.
    sharedState->results                 = std::make_unique<Stage1Results>(std::move(requestIn));
    Stage1Results& results               = *sharedState->results;
    const WeightEncodingRequest& request = sharedState->results->request;

    assert(request.m_StripeDepth > 0);
    assert(request.m_IterationSize > 0);

    const uint32_t filterX = request.m_WeightsTensorInfo.m_Dimensions[1];
    const uint32_t filterY = request.m_WeightsTensorInfo.m_Dimensions[0];

    results.numOfms = 0;
    if (request.m_WeightsTensorInfo.m_DataFormat == DataFormat::HWIO)
    {
        results.numOfms = request.m_WeightsTensorInfo.m_Dimensions[3];
    }
    else if (request.m_WeightsTensorInfo.m_DataFormat == DataFormat::HWIM)
    {
        results.numOfms = request.m_WeightsTensorInfo.m_Dimensions[2] * request.m_WeightsTensorInfo.m_Dimensions[3];
    }
    else
    {
        assert(false);
    }

    // Bias dimensions should be valid
    assert((request.m_BiasTensorInfo.m_Dimensions[0] * request.m_BiasTensorInfo.m_Dimensions[1] *
                request.m_BiasTensorInfo.m_Dimensions[2] ==
            1) &&
           request.m_BiasTensorInfo.m_Dimensions[3] == results.numOfms);

    // Zero point value should be within allowed range
    const utils::DataTypeRange zeroPointBounds = utils::GetRangeOfDataType(request.m_WeightsTensorInfo.m_DataType);
    ETHOSN_UNUSED(zeroPointBounds);
    assert(request.m_WeightsTensorInfo.m_QuantizationInfo.GetZeroPoint() <= zeroPointBounds.max &&
           request.m_WeightsTensorInfo.m_QuantizationInfo.GetZeroPoint() >= zeroPointBounds.min);

    uint32_t ifmChannels     = request.m_WeightsTensorInfo.m_Dimensions[2] * request.m_StrideX * request.m_StrideY;
    results.numIterationsOfm = request.m_WeightsTensorInfo.m_DataFormat == DataFormat::HWIM
                                   ? 1
                                   : utils::DivRoundUp(ifmChannels, request.m_IterationSize);

    // The number of OFMs that can be processed in parallel is limited to the stripe depth
    uint32_t numSrams        = request.m_Capabilities.GetNumberOfSrams();
    results.numOfmInParallel = GetNumOfmInParallel(request.m_Capabilities.GetNumberOfOgs(), numSrams,
                                                   request.m_StripeDepth, request.m_WeightsTensorInfo.m_DataFormat);

    sharedState->compressionParams = GenerateCompressionParams(results.numOfmInParallel);

    // Decide if wide filter is needed
    const uint32_t maxFilterSize = request.m_Algorithm == CompilerMceAlgorithm::Direct ? 7 : 1;
    sharedState->subfilters =
        GetSubmapFilters(filterX, filterY, request.m_StrideX, request.m_StrideY, request.m_PaddingLeft,
                         request.m_PaddingTop, request.m_WeightsTensorInfo.m_Dimensions);
    const uint32_t wideKernelSize = request.m_Capabilities.GetWideKernelSize();
    results.wideSubfilters =
        GetSubmapFilters(filterX, filterY, wideKernelSize, maxFilterSize, request.m_WeightsTensorInfo.m_Dimensions);

    // Encode each OFM stream independently
    // Split the work for each OG so that the OFMs for each OG can be encoded in parallel.
    // Assign each OFM to an OG
    sharedState->perOgOfms.resize(results.numOfmInParallel);
    for (uint32_t ofm = 0; ofm < (results.numOfms * results.numIterationsOfm); ++ofm)
    {
        const uint32_t ofmIdx = ofm / results.numIterationsOfm;
        const uint32_t ogIdx  = (ofmIdx % request.m_StripeDepth) % results.numOfmInParallel;
        sharedState->perOgOfms[ogIdx].push_back(ofm);
    }

    const size_t numWeightScales = request.m_WeightsTensorInfo.m_QuantizationInfo.GetScales().size();

    results.encodedStreams.resize(results.numOfms * results.numIterationsOfm);

    auto future           = std::make_unique<Stage1ResultsFuture>();
    future->m_SharedState = sharedState;
    future->m_WaitHandles.resize(results.numOfmInParallel);

    // Process each OG independently
    for (int og = 0; og < static_cast<int>(results.numOfmInParallel); ++og)
    {
        future->m_WaitHandles[og] = threadPool.AddToQueue(
            // Note that we copy the shared state (i.e. add reference to the shared_ptr) into the task
            [sharedStateCopy = sharedState, numWeightScales](int og) {
                Stage1Results& results               = *sharedStateCopy->results;
                const WeightEncodingRequest& request = sharedStateCopy->results->request;
                for (uint32_t ofm : sharedStateCopy->perOgOfms[static_cast<size_t>(og)])
                {
                    const uint32_t iteration = ofm % results.numIterationsOfm;
                    const uint32_t ofmIdx    = ofm / results.numIterationsOfm;

                    // Calculate encoding parameters from the various quantization infos
                    WeightCompressionParams::EncodingParams params;
                    double overallScale =
                        (request.m_InputQuantizationInfo.GetScale() *
                         request.m_WeightsTensorInfo.m_QuantizationInfo.GetScale(numWeightScales > 1 ? ofmIdx : 0)) /
                        request.m_OutputQuantizationInfo.GetScale();
                    utils::CalculateQuantizedMultiplierSmallerThanOne(overallScale, params.m_OfmScaleFactor,
                                                                      params.m_OfmShift);

                    params.m_OfmBias      = request.m_BiasData[ofmIdx];
                    params.m_OfmZeroPoint = static_cast<uint32_t>(request.m_OutputQuantizationInfo.GetZeroPoint());
                    params.m_FilterZeroPoint =
                        static_cast<uint32_t>(request.m_WeightsTensorInfo.m_QuantizationInfo.GetZeroPoint());

                    EncodedOfm encodedOfm = EncodeOfm(
                        request, ofmIdx, results.numOfmInParallel, results.numIterationsOfm, iteration, params,
                        sharedStateCopy->compressionParams, sharedStateCopy->subfilters, results.wideSubfilters);

                    results.encodedStreams[ofm] = std::move(encodedOfm);
                }
            },
            og);
    }

    return future;
}

std::unique_ptr<IStage1Results> Stage1ResultsFuture::Wait()
{
    // The results are produced by many tasks, so wait for them all to finish.
    for (auto&& h : m_WaitHandles)
    {
        h.wait();
    }

    return std::move(m_SharedState->results);
}

EncodedWeights EncodeWeightsStage2(std::unique_ptr<IStage1Results> stage1ResultsInterface)
{
    g_NumWeightEncodingsStage2++;

    Stage1Results& stage1Results         = static_cast<Stage1Results&>(*stage1ResultsInterface);
    const WeightEncodingRequest& request = stage1Results.request;

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
    std::vector<BitstreamRope> streamPerStripeOg;
    const uint32_t numStripes = utils::DivRoundUp(stage1Results.numOfms, stage1Results.request.m_StripeDepth);
    for (uint32_t stripeIdx = 0; stripeIdx < numStripes; ++stripeIdx)
    {
        const uint32_t firstOfmInStripe = request.m_StripeDepth * stripeIdx * stage1Results.numIterationsOfm;
        const uint32_t lastOfmInStripe =
            std::min<uint32_t>(stage1Results.numOfms, request.m_StripeDepth * (stripeIdx + 1)) *
            stage1Results.numIterationsOfm;
        std::vector<EncodedOfm> encodedOfmStreamsForThisStripe(
            std::begin(stage1Results.encodedStreams) + firstOfmInStripe,
            std::begin(stage1Results.encodedStreams) + lastOfmInStripe);
        std::vector<BitstreamRope> streamPerOgForThisStripe = MergeStreamsOgAndUpdateHeaders(
            encodedOfmStreamsForThisStripe, stage1Results.numOfmInParallel * stage1Results.numIterationsOfm,
            dmaEngineAlignment);
        streamPerStripeOg.insert(std::end(streamPerStripeOg), std::make_move_iterator(streamPerOgForThisStripe.begin()),
                                 std::make_move_iterator(streamPerOgForThisStripe.end()));
    }

    uint32_t maxLength = 0;
    for (const BitstreamRope& s : streamPerStripeOg)
    {
        maxLength = std::max(maxLength, utils::DivRoundUp(s.GetTotalBits(), 8));
    }

    // Ensure all streams are of equal size as SRAM offsets are same on all CEs
    // Because the weights will be DMA'd in stripes, there is an alignment requirement for the start of each stripe
    // (the DMA can only transfer blocks aligned to 16-bytes).
    // Therefore we pad each stream to 16 bytes.
    maxLength = utils::RoundUpToNearestMultiple(maxLength, dmaEngineAlignment);
    for (BitstreamRope& s : streamPerStripeOg)
    {
        uint32_t numPaddingBits = maxLength * 8 - s.GetTotalBits();
        if (numPaddingBits > 0)
        {
            BitstreamRope::Element paddingElement;
            paddingElement.bytes = std::vector<uint8_t>(utils::DivRoundUp(numPaddingBits, 8), static_cast<uint8_t>(0));
            paddingElement.numBits = numPaddingBits;
            s.AddElement(std::move(paddingElement));
        }
    }

    // Number of Ofm processed in parallel which is the minimum number of
    // weights streams that need to be loaded at the same time for all the
    // mce interfaces to start producing an Ofm each.
    uint32_t numSrams       = request.m_Capabilities.GetNumberOfSrams();
    uint32_t numOfmsPerSram = request.m_Capabilities.GetNumberOfOgs() / numSrams;
    assert(numOfmsPerSram >= 1);

    EncodedWeights encodedWeights;
    encodedWeights.m_Metadata = CalculateWeightsMetadata(streamPerStripeOg, stage1Results.numOfmInParallel);

    // Merge together all the stripes into groups based on the SRAM they will be loaded into.
    // Stream = group of stripes that are loaded into a particular SRAM
    std::vector<BitstreamRope> mergedStreams =
        MergeStreams(streamPerStripeOg, stage1Results.request.m_Capabilities.GetNumberOfSrams(),
                     stage1Results.numIterationsOfm, numOfmsPerSram);

    // Merge all the SRAM streams together by interleaving 16 bytes from each.
    // This is so the DMA will distribute the correct weight data to the correct SRAM.
    encodedWeights.m_Data         = InterleaveStreams(mergedStreams, dmaEngineAlignment);
    encodedWeights.m_IsWideFilter = stage1Results.wideSubfilters.size() > 1;

    encodedWeights.m_MaxSize = 0;

    for (uint32_t i = 0; i < encodedWeights.m_Metadata.size(); ++i)
    {
        encodedWeights.m_MaxSize = std::max(encodedWeights.m_Metadata[i].m_Size, encodedWeights.m_MaxSize);
    }

    return encodedWeights;
}

}    // namespace support_library
}    // namespace ethosn
