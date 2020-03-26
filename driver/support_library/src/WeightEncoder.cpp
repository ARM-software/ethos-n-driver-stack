//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "WeightEncoder.hpp"

#include "Compiler.hpp"
#include "GraphNodes.hpp"
#include "SubmapFilter.hpp"
#include "Utils.hpp"

#include <algorithm>
#include <exception>

namespace ethosn
{
namespace support_library
{

/**
 * BitstreamWriter is a helper class that supports writing packed bitfields into a vector.
 */
class BitstreamWriter
{
public:
    BitstreamWriter();

    /**
     * Returns the current write position in the bitstream (in bits)
     */
    size_t GetOffset();

    /**
     * Write an element to the stream. Offset specifies where to start writing in the stream.
     */
    void Write(uint8_t elem, int numBits, size_t offset);

    /**
     * Write an element to end of the stream.
     */
    void Write(uint8_t elem, int numBits);

    /**
     * Reserve space in the stream by writing 0 bits
     */
    void Reserve(size_t numBits);

    /**
     * Returns the stream as a uint8_t vector
     */
    const std::vector<uint8_t>& GetBitstream();

    /**
     * Clears the content of the stream and resets the write position
     */
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
            m_Bitstream.push_back((elem >> i) & 1);
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
                   const uint32_t zeroPoint,
                   int blockSize);

    virtual void CompressWeight(uint8_t weight);
    void Flush();

protected:
    const int m_BlockSize;

    uint16_t m_Mask;
    int m_NumWeights;
    size_t m_MaskOffset;
    uint32_t m_ZeroPoint;
};

ZeroCompressor::ZeroCompressor(std::vector<uint8_t>& result,
                               uint32_t indexSize,
                               const std::vector<uint8_t>& lut,
                               bool lutReload,
                               const uint32_t zeroPoint,
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
                                                                const uint32_t zeroPoint,
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

WeightEncoder::WeightEncoder(const HardwareCapabilities& capabilities)
    : m_Capabilities(capabilities)
{}

EncodedWeights WeightEncoder::Encode(const MceOperationNode& mceOperation,
                                     uint32_t stripeDepth,
                                     uint32_t stripeSize,
                                     const QuantizationInfo& outputQuantizationInfo) const
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
                                     const QuantizationInfo& outputQuantizationInfo) const
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
                                     CompilerMceAlgorithm algorithm) const
{
    uint32_t numOfms;
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
        // Weight tensor must be HWIO or HWIM
        assert(false);
    }

    // Bias dimensions should be valid
    assert((biasTensorInfo.m_Dimensions[0] * biasTensorInfo.m_Dimensions[1] * biasTensorInfo.m_Dimensions[2] == 1) ||
           biasTensorInfo.m_Dimensions[3] == numOfms);

    // Zero point value should be within allowed range
    assert(weightsTensorInfo.m_QuantizationInfo.m_ZeroPoint <= UINT8_MAX);
    assert(weightsTensorInfo.m_QuantizationInfo.m_ZeroPoint >= 0);

    uint32_t ifmChannels = weightsTensorInfo.m_Dimensions[2] * strideX * strideY;
    uint32_t numIterationsOfm =
        weightsTensorInfo.m_DataFormat == DataFormat::HWIM ? 1 : utils::DivRoundUp(ifmChannels, iterationSize);

    // Number of Ofm processed in parallel which is the minimum number of
    // weights streams that need to be loaded at the same time for all the
    // mce interfaces to start producing an Ofm each.
    uint32_t numOfmInParallel;
    uint32_t numSrams       = m_Capabilities.GetNumberOfSrams();
    uint32_t numOfmsPerSram = m_Capabilities.GetNumberOfOfm() / numSrams;

    if (weightsTensorInfo.m_DataFormat == DataFormat::HWIO)
    {
        numOfmInParallel = m_Capabilities.GetNumberOfOfm();
    }
    else if (weightsTensorInfo.m_DataFormat == DataFormat::HWIM)
    {
        numOfmInParallel = numSrams;
    }
    else
    {
        // Weight tensor must be HWIO or HWIM
        assert(false);
    }

    // Encode each OFM stream independently
    std::vector<std::vector<uint8_t>> encodedStreams;
    std::vector<WeightCompressionParams> compressionParams;
    encodedStreams.reserve(numOfms * numIterationsOfm);
    compressionParams.reserve(numOfms * numIterationsOfm);
    for (uint32_t ofm = 0; ofm < (numOfms * numIterationsOfm); ++ofm)
    {
        // numIterationsOfm >= 1, fully connected
        //                   = 1, otherwise
        uint32_t iteration = ofm % numIterationsOfm;
        uint32_t ofmIdx    = ofm / numIterationsOfm;

        // Calculate encoding parameters from the various quantization infos
        EncodingParams params;
        double overallScale = (inputQuantizationInfo.m_Scale * weightsTensorInfo.m_QuantizationInfo.m_Scale) /
                              outputQuantizationInfo.m_Scale;
        utils::CalculateQuantizedMultiplierSmallerThanOne(overallScale, params.m_OfmScaleFactor, params.m_OfmShift);
        params.m_OfmBias         = biasData[ofmIdx];
        params.m_OfmZeroPoint    = outputQuantizationInfo.m_ZeroPoint;
        params.m_FilterZeroPoint = weightsTensorInfo.m_QuantizationInfo.m_ZeroPoint;

        // Lookup the compression parameters for the previous OFM associated with the same CE. This is used
        // to modify the compression of this current OFM.
        const WeightCompressionParams* previousOfmSameCeCompressionParams =
            ofmIdx < numOfmInParallel ? nullptr : &compressionParams[ofm - numOfmInParallel];

        EncodedOfm encodedOfm =
            EncodeOfm(weightsData, ofmIdx, iteration, weightsTensorInfo, strideY, strideX, paddingTop, paddingLeft,
                      iterationSize, operation, algorithm, params, previousOfmSameCeCompressionParams);
        encodedStreams.push_back(std::move(encodedOfm.m_EncodedWeights));
        compressionParams.push_back(encodedOfm.m_CompressionParameters);
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
        std::vector<std::vector<uint8_t>> streamPerOgForThisStripe =
            MergeStreams(encodedOfmStreamsForThisStripe, numOfmInParallel * numIterationsOfm, 1, 1, dmaEngineAlignment);
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
    uint32_t numIfmPerEngine = m_Capabilities.GetIfmPerEngine();
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

        const uint32_t numIfmsProcessedInParallel = numIfmPerEngine * numEngines;

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
                                return static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.m_ZeroPoint);
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
                                                         weightsTensorInfo.m_QuantizationInfo.m_ZeroPoint);
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
                                            : static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.m_ZeroPoint);
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
                    result.push_back(static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.m_ZeroPoint));
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

        assert(numIfms % 1024 == 0);

        for (SubmapFilter filter : subfilters)
        {
            for (uint32_t encodedIdx = 0; encodedIdx < numUninterleavedIfmsPerIteration; ++encodedIdx)
            {
                uint32_t rawIdx;

                uint32_t brickIdx = encodedIdx / 1024;
                uint32_t idxBrick = encodedIdx % 1024;

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

                rawIdx =
                    iterationOffset + brickIdx * 1024 + qbrickIdx * qbrickSize + patchIdx * patchSize + patchOffset;

                if (rawIdx < numIfms)
                {
                    weight = filter.GetWeightAt(wd, 0, 0, rawIdx, ofmIdx);
                }
                else
                {
                    weight = static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.m_ZeroPoint);
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
        // rather than all of the IFMs. Mathematically we only need to supply 1 (as each OFM is dependent on only 1 IFM),
        // but the HW requires a full set of 16 weights so we just set the others to zero. Add weight data in row-major
        // order, with a slice of as many IFMs as there are IGs, tightly packed for each filter coordinate.
        for (uint32_t filterIdx = 0; filterIdx < subfilters.size(); ++filterIdx)
        {
            const SubmapFilter& filter = subfilters[filterIdx];

            // If there are multiple subfilters, the data in all except the last must be padded to the number of IGs.
            // The last one may be left without padding, if we are not using zero compression.
            const uint32_t numChannels = (filterIdx == subfilters.size() - 1 && tightlyPackLastSliceLastSubfilter)
                                             ? (ifmIdx % numIfmsProcessedInParallel) + 1
                                             : numIfmsProcessedInParallel;
            // Add weight data in row-major order, with the slice of as many IFMs as there are IGs, tightly packed
            // for each filter coordinate.
            for (uint32_t h = 0; h < filter.GetFilterY(); ++h)
            {
                for (uint32_t w = 0; w < filter.GetFilterX(); ++w)
                {
                    for (uint32_t i = 0; i < numChannels; ++i)
                    {
                        uint8_t weight;

                        if (i == ifmIdx % numIfmsProcessedInParallel)
                        {
                            weight = filter.GetWeightAt(wd, h, w, ifmIdx, channelMultiplierIdx);
                        }
                        else
                        {
                            weight = static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.m_ZeroPoint);
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

WeightEncoder::WeightCompressionParams
    WeightEncoder::ChooseCompressionParameters(const std::vector<uint8_t>& rawWeightsForZeroMaskCompression,
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
        scheme.numZeroPointElements = scheme.frequencies[weightsTensorInfo.m_QuantizationInfo.m_ZeroPoint];
        scheme.compressedSize       = scheme.compressedSizeCalculator(scheme);
    }

    const Scheme& bestScheme = *std::min_element(
        schemes.begin(), schemes.end(), [](auto a, auto b) -> bool { return a.compressedSize < b.compressedSize; });

    WeightCompressionParams params;
    params.m_LutReload  = bestScheme.m_Lut;
    params.m_MaskEnable = bestScheme.m_ZeroMask;
    params.m_IndexSize  = 0;    // 8-bit weights, Lut disabled

    if (params.m_LutReload)
    {
        // Enable Lut compression
        /* IndexSize:  Bits per index (number of weights):
            1           3 (0 - 8 weights)
            2           4 (9 - 16 weights)
            3           5 (17 - 32 weights)
        */
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
        // ZeroPoint must be greater than zero and less than 255
        uint8_t zeroPoint = static_cast<uint8_t>(weightsTensorInfo.m_QuantizationInfo.m_ZeroPoint);
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

WeightEncoder::EncodedOfm
    WeightEncoder::EncodeOfm(const uint8_t* weightData,
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
                             const EncodingParams& params,
                             const WeightCompressionParams* previousOfmSameCeCompressionParams) const
{
    // Get the raw (unencoded) weight stream. Note we must do this twice - once to get a stream suited
    // for zero mask compression and again to get one suited to no zero mask compression. Yuck!
    std::vector<uint8_t> rawWeightsForZeroMaskCompression =
        GetRawOfmStream(weightData, ofmIdx, iteration, weightsTensorInfo, strideY, strideX, paddingTop, paddingLeft,
                        iterationSize, operation, algorithm, true);
    std::vector<uint8_t> rawWeightsForNoZeroMaskCompression =
        GetRawOfmStream(weightData, ofmIdx, iteration, weightsTensorInfo, strideY, strideX, paddingTop, paddingLeft,
                        iterationSize, operation, algorithm, false);

    // Choose the best compression scheme
    WeightCompressionParams compressionParams = ChooseCompressionParameters(
        rawWeightsForZeroMaskCompression, rawWeightsForNoZeroMaskCompression, weightsTensorInfo);
    std::vector<uint8_t>& rawWeights =
        compressionParams.m_MaskEnable ? rawWeightsForZeroMaskCompression : rawWeightsForNoZeroMaskCompression;

    // If the Lut is the same as for previous OFM for the current CE then don't reload it
    const uint32_t numOfmsPerSram = m_Capabilities.GetNumberOfOfm() / m_Capabilities.GetNumberOfSrams();
    if (compressionParams.m_IndexSize != 0 && previousOfmSameCeCompressionParams &&
        previousOfmSameCeCompressionParams->m_Lut == compressionParams.m_Lut &&
        // Disable for configurations with more than one OFM per SRAM, since they use a different CE OFM
        // fetching strategy
        numOfmsPerSram == 1)
    {
        compressionParams.m_LutReload = false;
    }

    EncodedOfm result;
    result.m_CompressionParameters       = compressionParams;
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
    header.m_SignExtend      = 0;    // Only required for 16-bit weights (we only support 8-bit)
    header.m_Padding         = 0;    // Unused padding.

    // Compress each weight using the above chosen compression parameters
    std::shared_ptr<WeightCompressor> compressor = CreateWeightCompressor(
        encodedWeights, compressionParams.m_IndexSize, compressionParams.m_Lut, compressionParams.m_LutReload,
        compressionParams.m_MaskEnable, params.m_FilterZeroPoint, m_Capabilities.GetNumberOfSrams());

    for (size_t i = 0; i < rawWeights.size(); ++i)
    {
        compressor->CompressWeight(rawWeights[i]);
    }

    compressor->Flush();

    return result;
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

                uint32_t startWord = start / streamHeadersUpdateAlignment;
                uint32_t endWord =
                    utils::DivRoundUp(static_cast<uint32_t>(mergedGroup.size()), streamHeadersUpdateAlignment);
                header.m_StreamLength = static_cast<uint16_t>(endWord - startWord);
            }
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
