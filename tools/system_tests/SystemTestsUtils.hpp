//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "Tensor.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>

#include <ethosn_driver_library/Inference.hpp>
#include <ethosn_driver_library/Network.hpp>
#include <ethosn_support_library/Support.hpp>
#include <ethosn_utils/Log.hpp>
#include <ethosn_utils/Macros.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__unix__)
#include <unistd.h>
#elif defined(_MSC_VER)
#include <io.h>
#endif

namespace ethosn
{
namespace system_tests
{

extern utils::log::Logger<utils::log::Severity::Debug> g_Logger;

/*****************************************************************************
 * String helper functions
 *****************************************************************************/

static inline std::string Split(const std::string& s, const std::string& delim, std::size_t& pos)
{
    if (pos >= s.length())
    {
        return "";
    }

    std::size_t end = s.find(delim, pos);
    if (end == std::string::npos)
    {
        end = s.length();
    }

    std::string str = s.substr(pos, end - pos);
    pos             = end + 1;

    return str;
}

template <typename Vector>
void DebugVector(const char* const name, const Vector& data, const size_t max)
{
    std::stringstream ss;
    ss << name << ": size=" << data.size();

    const size_t end = std::min(data.size(), max);
    for (size_t i = 0; i < end; ++i)
    {
        if ((i & 0xf) == 0)
        {
            ss << std::dec << std::setfill(' ');
            ss << "\n[" << std::setw(4) << i << "] ";
        }
        ss << std::hex << std::setfill('0');
        ss << std::setw(2 * sizeof(data[i])) << +data[i] << " ";
    }
    g_Logger.Debug("%s", ss.str().c_str());
}

void DebugTensor(const char* const name, const BaseTensor& tensor, const size_t max);

template <typename Vector>
void PrintDifferencesNhwc(
    const Vector& data, const Vector& reference, const uint32_t height, const uint32_t width, const uint32_t depth)
{
    const uint32_t strideX = depth;
    const uint32_t strideY = depth * width;
    const uint32_t strideZ = 1;

    const char* const prefixGood = isatty(fileno(stderr)) ? " " : "  ";
    const char* const suffixGood = isatty(fileno(stderr)) ? "" : " ";

    const char* const prefixBad = isatty(fileno(stderr)) ? " \033[91m" : " *";
    const char* const suffixBad = isatty(fileno(stderr)) ? "\033[0m" : "*";

    std::stringstream ss;
    ss << std::setfill('0');

    for (uint32_t z = 0; z < depth; ++z)
    {
        ss << std::dec;
        ss << "z=" << z << std::endl;
        ss << std::hex;
        for (uint32_t y = 0; y < height; ++y)
        {
            for (uint32_t x = 0; x < width; ++x)
            {
                const auto element    = data[(x * strideX) + (y * strideY) + (z * strideZ)];
                const auto RefElement = reference[(x * strideX) + (y * strideY) + (z * strideZ)];

                const bool elementsMatch = element == RefElement;

                constexpr size_t numHexDigits = 2 * sizeof(element);

                ss << (elementsMatch ? prefixGood : prefixBad);
                ss << std::setw(numHexDigits) << +element;
                ss << "(" << std::setw(numHexDigits) << +RefElement << ")";
                ss << (elementsMatch ? suffixGood : suffixBad);
            }
            ss << std::endl;
        }
    }

    std::cerr << ss.str();
}

inline void WriteHex(std::ostream& os, const uint32_t startAddr, const uint8_t* const data, const uint32_t numBytes)
{
    const std::ios_base::fmtflags flags = os.flags();

    os << std::hex << std::setfill('0');

    // Loop over rows (16-bytes each)
    for (uint32_t i = 0; i < numBytes; i += 16)
    {
        const uint32_t addr = startAddr + i;
        os << std::setw(8) << addr << ':';

        // Loop over columns (4-bytes each)
        for (uint32_t j = 0; j < 16; j += 4)
        {
            os << ' ';
            // Loop over bytes within the column.
            // Hex files are little-endian so we loop over the bytes in reverse order
            for (int32_t k = 3; k >= 0; --k)
            {
                const uint32_t b     = i + j + static_cast<uint32_t>(k);
                const uint32_t value = (b < numBytes) ? data[b] : 0;
                os << std::setw(2) << value;
            }
        }

        os << std::endl;
    }

    os.flags(flags);
}

/// Returns the first argument rounded UP to the nearest multiple of the second argument
constexpr uint32_t RoundUpToNearestMultiple(uint32_t num, uint32_t nearestMultiple)
{
    uint32_t remainder = num % nearestMultiple;

    if (remainder == 0)
    {
        return num;
    }

    return num + nearestMultiple - remainder;
}

constexpr uint32_t g_BrickWidth          = 4;
constexpr uint32_t g_BrickHeight         = 4;
constexpr uint32_t g_BrickSlice          = g_BrickWidth * g_BrickHeight;
constexpr uint32_t g_BrickDepth          = 16;
constexpr uint32_t g_BrickSize           = g_BrickSlice * g_BrickDepth;
constexpr uint32_t g_BrickCountInGroup   = 4;
constexpr uint32_t g_BrickGroupSizeBytes = g_BrickSize * g_BrickCountInGroup;

/// Calculates the quotient of numerator and denominator as an integer where the result is rounded up to the nearest
/// integer. i.e. ceil(numerator/denominator).
template <typename T1, typename T2>
constexpr auto DivRoundUp(T1 numerator, T2 denominator)
{
    auto result = (numerator + denominator - 1) / denominator;
    static_assert(std::is_integral<decltype(result)>::value, "DivRoundUp only supports integer division");
    return result;
}

constexpr uint32_t GetTotalSizeNhwcb(uint32_t w, uint32_t h, uint32_t c)
{
    return DivRoundUp(w, 8) * DivRoundUp(h, 8) * DivRoundUp(c, 16) * g_BrickGroupSizeBytes;
}

// Helper class to read data from a tightly-packed multidimensional array
template <typename T, uint32_t D>
class MultiDimensionalArray
{
public:
    MultiDimensionalArray(T* data, const std::array<uint32_t, D>& dims)
        : m_Data(data)
        , m_Dims(dims)
    {}

    T GetElement(const std::array<uint32_t, D>& indexes) const
    {
        return m_Data[GetOffset(indexes)];
    }

    void SetElement(const std::array<uint32_t, D>& indexes, T value) const
    {
        m_Data[GetOffset(indexes)] = value;
    }

    uint32_t GetDimSize(uint32_t dim) const
    {
        return m_Dims[dim];
    }

    uint32_t GetSize() const
    {
        return std::accumulate(m_Dims.begin(), m_Dims.end(), 1, std::multiplies<uint32_t>());
    }

private:
    uint32_t GetOffset(const std::array<uint32_t, D>& indexes) const
    {
        uint32_t offset  = 0;
        uint32_t product = 1;
        for (int32_t d = static_cast<int32_t>(indexes.size()) - 1; d >= 0; d--)
        {
            assert(indexes[d] < m_Dims[d]);
            offset += product * indexes[d];
            product *= m_Dims[d];
        }

        return offset;
    }

    T* m_Data;
    std::array<uint32_t, D> m_Dims;
};

template <typename T>
void ConvertNhwcToNhwcb(
    const T* nhwcInPtr, T* nhwcbOutPtr, uint32_t tensorHeight, uint32_t tensorWidth, uint32_t tensorDepth)
{
    uint32_t newHeight = DivRoundUp(tensorHeight, 8);
    uint32_t newWidth  = DivRoundUp(tensorWidth, 8);
    uint32_t newDepth  = DivRoundUp(tensorDepth, 16);
    MultiDimensionalArray<T, 8> nhwcbOut(nhwcbOutPtr, { 1, newHeight, newWidth, newDepth, g_BrickCountInGroup,
                                                        g_BrickDepth, g_BrickHeight, g_BrickWidth });
    MultiDimensionalArray<const T, 4> nhwcIn(nhwcInPtr, { 1, tensorHeight, tensorWidth, tensorDepth });
    uint32_t brickHeightShift = 2;
    uint32_t brickWidthShift  = 2;
    uint32_t brickDepthShift  = 4;
    uint32_t brickHeightMask  = (1 << brickHeightShift) - 1;
    uint32_t brickWidthMask   = (1 << brickWidthShift) - 1;
    uint32_t brickDepthMask   = (1 << brickDepthShift) - 1;
    uint32_t hIdx, oH, hB;
    uint32_t wIdx, oW, wB;
    uint32_t oD;
    for (uint32_t height = 0; height < tensorHeight; height++)
    {
        hIdx = height >> 3;
        oH   = height & brickHeightMask;
        hB   = (height & 7) >> brickHeightShift;
        for (uint32_t width = 0; width < tensorWidth; width++)
        {
            wIdx = width >> 3;
            oW   = width & brickWidthMask;
            wB   = ((width & 7) >> brickWidthShift) * 2 + hB;
            for (uint32_t depth = 0; depth < tensorDepth; depth++)
            {
                oD = depth & brickDepthMask;
                nhwcbOut.SetElement({ 0, hIdx, wIdx, static_cast<uint32_t>(floor(depth / 16)), wB, oD, oH, oW },
                                    nhwcIn.GetElement({ 0, height, width, depth }));
            }
        }
    }
    return;
}

OwnedTensor
    ConvertNhwcToNhwcb(const BaseTensor& nhwcInPtr, uint32_t tensorHeight, uint32_t tensorWidth, uint32_t tensorDepth);

// NHWCB iteration: x,y,d,bnum,bdepth,bgx,bgy
// NHWC iteration: d,x,y
// Brick number (bnum) in brickgroup:
//
//       /   /   /
//      /---/---/
//     /   /   /|/
//    /---/---/ /
//    | 0 | 2 |/|/
//    +---+---/ /
//    | 1 | 3 |/
//    +---+---/
//
template <typename T>
void ConvertNhwcbToNhwc(
    const T* nhwcbInPtr, T* nhwcOutPtr, uint32_t tensorHeight, uint32_t tensorWidth, uint32_t tensorDepth)
{
    uint32_t srcWidthBGs    = DivRoundUp(tensorWidth, (g_BrickWidth * 2));
    uint32_t srcDepthBricks = DivRoundUp(tensorDepth, g_BrickDepth);

    const uint32_t nhwcSlice      = tensorWidth * tensorDepth;
    const uint32_t brickGroupSize = g_BrickSize * g_BrickCountInGroup;
    const uint32_t bgStick        = brickGroupSize * srcDepthBricks;
    const uint32_t bgStickRow     = bgStick * srcWidthBGs;

    // Iterate over destination NHWC, sampling from NHWCB source
    for (uint32_t y = 0; y < tensorHeight; y += 1)
    {
        uint32_t yBrick  = y / g_BrickHeight;
        uint32_t yOffset = y % g_BrickHeight;
        uint32_t yBG     = yBrick / 2;
        yBrick &= 1;

        for (uint32_t x = 0; x < tensorWidth; x += 1)
        {
            uint32_t xBrick  = x / g_BrickWidth;
            uint32_t xOffset = x % g_BrickHeight;
            uint32_t xBG     = xBrick / 2;
            xBrick &= 1;
            uint32_t brickNo = (xBrick * 2) + yBrick;

            for (uint32_t d = 0; d < tensorDepth; d += 1)
            {
                uint32_t dBrick  = d / g_BrickDepth;
                uint32_t dOffset = d % g_BrickDepth;

                T val = nhwcbInPtr[(bgStickRow * yBG) + (bgStick * xBG) +                             // Brick group
                                   (brickGroupSize * dBrick) + (g_BrickSize * brickNo) +              // Brick in group
                                   (dOffset * g_BrickSlice) + (yOffset * g_BrickWidth) + xOffset];    // Pixel in brick

                nhwcOutPtr[(y * nhwcSlice) + (tensorDepth * x) + d] = val;
            }
        }
    }
}

OwnedTensor
    ConvertNhwcbToNhwc(const BaseTensor& nhwcbInPtr, uint32_t tensorHeight, uint32_t tensorWidth, uint32_t tensorDepth);

constexpr uint32_t CalcConvOutSize(const uint32_t inSize,
                                   const uint32_t kSize,
                                   const uint32_t stride,
                                   const uint32_t pad,
                                   const bool isTranspose = false)
{
    if (isTranspose)
    {
        // This is the inverse calculation of a convolution.
        // The input size is what the output size would be in a convolution with given kSize, stride and pad:
        //
        //     outSize = ((inSize * stride) + kSize) - (stride + pad)

        // Separate positive contribution from negative contribution and use max to make sure we don't overflow
        const uint32_t positive = (inSize * stride) + kSize;
        const uint32_t negative = stride + pad;

        return std::max(positive, negative) - negative;
    }
    else
    {
        // Output size of a convolution:
        //
        //     outSize = (inSize + stride + pad - kSize) / stride

        // Separate positive contribution from negative contribution and use max to make sure we don't overflow
        const uint32_t positive = inSize + stride + pad;
        const uint32_t negative = kSize;

        return (std::max(positive, negative) - negative) / stride;
    }
}

constexpr uint32_t CalcConvOutSize(const uint32_t inSize,
                                   const uint32_t kSize,
                                   const uint32_t stride,
                                   const uint32_t padLeftOrTop,
                                   const uint32_t padRightOrBottom,
                                   const bool isTranspose = false)
{
    return CalcConvOutSize(inSize, kSize, stride, (padLeftOrTop + padRightOrBottom), isTranspose);
}

constexpr uint32_t CalcConvOutSize(const uint32_t inSize,
                                   const uint32_t kSize,
                                   const uint32_t stride,
                                   const bool padSame,
                                   const bool isTranspose = false)
{
    return CalcConvOutSize(inSize, kSize, stride, padSame ? kSize - 1U : 0U, isTranspose);
}

constexpr std::pair<uint32_t, uint32_t> CalcConvPadding(const uint32_t inSize,
                                                        const uint32_t outSize,
                                                        const uint32_t kSize,
                                                        const uint32_t stride,
                                                        const bool isTranspose = false)
{
    // The relationship between input size (i), output size (o), kernel size (k), stride (s) and pad size (p)
    // in a convolution is:
    //
    //     i + p = (o*s) + k - s
    //
    // And helper function CalcConvOutSize gives:
    //
    //     o = CalcConvOutSize(i, k, s, p, false)
    //     i = CalcConvOutSize(o, k, s, p, true)
    //     p = CalcConvOutSize(o, k, s, i, true)
    //
    const uint32_t padSize = isTranspose ? CalcConvOutSize(inSize, kSize, stride, outSize, true)
                                         : CalcConvOutSize(outSize, kSize, stride, inSize, true);

    const uint32_t padBefore = padSize / 2;
    const uint32_t padAfter  = padSize - padBefore;

    return { padBefore, padAfter };
}

template <typename PadType>
constexpr std::pair<uint32_t, std::pair<uint32_t, uint32_t>> CalcConvOutSizeAndPadding(const uint32_t inSize,
                                                                                       const uint32_t kSize,
                                                                                       const uint32_t stride,
                                                                                       const PadType pad,
                                                                                       const bool isTranspose = false)
{
    const uint32_t outSize                      = CalcConvOutSize(inSize, kSize, stride, pad, isTranspose);
    const std::pair<uint32_t, uint32_t> padding = CalcConvPadding(inSize, outSize, kSize, stride, isTranspose);
    return { outSize, padding };
}

// Permutes a weight tensor for normal convolution from Ethos-N to Arm NN
template <typename T>
std::vector<std::remove_cv_t<T>> ConvertConvolutionWeightData(const MultiDimensionalArray<T, 4>& ethosnInput)
{
    std::vector<std::remove_cv_t<T>> result;
    result.reserve(ethosnInput.GetSize());
    for (uint32_t outerDim = 0; outerDim < ethosnInput.GetDimSize(3); ++outerDim)
    {
        for (uint32_t h = 0; h < ethosnInput.GetDimSize(0); ++h)
        {
            for (uint32_t w = 0; w < ethosnInput.GetDimSize(1); ++w)
            {
                for (uint32_t i = 0; i < ethosnInput.GetDimSize(2); ++i)
                {
                    result.push_back(ethosnInput.GetElement({ h, w, i, outerDim }));
                }
            }
        }
    }
    return result;
}

// Permutes a weight tensor for depthwise from Ethos-N (HWIM) to Arm NN (1HW(I*M))
template <typename T>
std::vector<std::remove_cv_t<T>> ConvertDepthwiseConvolutionWeightData(const MultiDimensionalArray<T, 4>& ethosnInput)
{
    std::vector<std::remove_cv_t<T>> result;
    result.reserve(ethosnInput.GetSize());
    for (uint32_t h = 0; h < ethosnInput.GetDimSize(0); ++h)
    {
        for (uint32_t w = 0; w < ethosnInput.GetDimSize(1); ++w)
        {
            for (uint32_t i = 0; i < ethosnInput.GetDimSize(2); ++i)
            {
                for (uint32_t m = 0; m < ethosnInput.GetDimSize(3); ++m)
                {
                    result.push_back(ethosnInput.GetElement({ h, w, i, m }));
                }
            }
        }
    }
    return result;
}

// Generate random weight data for Arm NN and Ethos-N. The input 'dims' is in HWIO/HWIM format.
inline std::pair<std::vector<uint8_t>, std::vector<uint8_t>>
    GenerateWeightData(const std::array<uint32_t, 4>& dims, size_t max, bool depthwise = false)
{
    std::vector<uint8_t> ethosnWeightData(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>()), 0);
    generate(ethosnWeightData.begin(), ethosnWeightData.end(),
             [max]() -> uint8_t { return static_cast<uint8_t>(rand() % (max + 1)); });

    MultiDimensionalArray<uint8_t, 4> ethosnWeightTensor(ethosnWeightData.data(), dims);
    std::vector<uint8_t> armnnWeightData(ethosnWeightData.size(), 0);
    if (depthwise)
    {
        armnnWeightData = ConvertDepthwiseConvolutionWeightData(ethosnWeightTensor);
    }
    else
    {
        armnnWeightData = ConvertConvolutionWeightData(ethosnWeightTensor);
    }

    return std::pair<std::vector<uint8_t>, std::vector<uint8_t>>(std::move(ethosnWeightData),
                                                                 std::move(armnnWeightData));
}

inline uint32_t GetIfmGlobal(uint32_t numIfms, uint32_t numCe, uint32_t strideX, uint32_t strideY)
{
    uint32_t result;
    if (strideX == 1 && strideY == 1)
    {
        result = numIfms;
    }
    else
    {
        if (numIfms % numCe)
        {
            // Original number of IFMs is not a multiple of 16
            result = DivRoundUp(numIfms, numCe) * numCe * strideX * strideY - (numCe - (numIfms % numCe));
        }
        else
        {
            // Original number of IFMs is a multiple of 16
            result = numIfms * strideX * strideY;
        }
    }
    return result;
}

// Interleave the input data (in NHWC format)
template <typename T>
std::vector<T>
    InterleaveNhwcInputData(MultiDimensionalArray<T, 4> ethosnInput, uint32_t strideX, uint32_t strideY, int32_t ch)
{
    std::vector<T> result;
    result.reserve(ethosnInput.GetSize());
    for (uint32_t outerDim = 0; outerDim < ethosnInput.GetDimSize(0); ++outerDim)
    {
        for (uint32_t h = 0; h < ethosnInput.GetDimSize(1); h = h + strideY)
        {
            for (uint32_t w = 0; w < ethosnInput.GetDimSize(2); w = w + strideX)
            {
                // Number of input IFMs (original)
                uint32_t numIfms = ethosnInput.GetDimSize(3);
                // Number of submapped IFMs (interleave)
                uint32_t ifmGlobal = GetIfmGlobal(numIfms, ch, strideX, strideY);
                // Interleave ch limit
                uint32_t chLimit = (numIfms % ch) ? (numIfms / ch) + 1 : (numIfms / ch);
                for (uint32_t k = 0; k < chLimit; ++k)
                {
                    for (uint32_t y = 0; y < strideY; ++y)
                    {
                        for (uint32_t x = 0; x < strideX; ++x)
                        {
                            uint32_t limit = std::min(static_cast<int32_t>(ifmGlobal), ch);
                            for (uint32_t i = 0; i < limit; ++i)
                            {
                                if ((i + k) < numIfms)
                                {
                                    result.push_back(ethosnInput.GetElement({ outerDim, h + y, w + x, i + k * ch }));
                                }
                                else
                                {
                                    // Padding with zeros for the remaining channels in the group of ch
                                    result.push_back(0);
                                }
                            }
                            ifmGlobal -= ch;
                        }
                    }
                }
            }
        }
    }
    return result;
}

bool DumpData(const char* filename, const BaseTensor& t);

template <typename InputDataType>
inline bool DumpData(const char* filename, const std::vector<InputDataType>& inputData)
{
    std::ofstream stream(filename);
    WriteHex(stream, 0x0, reinterpret_cast<const uint8_t*>(inputData.data()), static_cast<uint32_t>(inputData.size()));
    return stream.good();
}

/// Gets the absolute differences between corresponding elements in two arrays.
/// If the arrays have different lengths, the extra elements in the larger array are ignored.
template <typename T>
std::vector<T> GetAbsoluteDifferences(const std::vector<T>& a, const std::vector<T>& b)
{
    size_t size = std::min(a.size(), b.size());
    std::vector<T> differences(size, 0);
    for (uint32_t i = 0; i < size; ++i)
    {
        const int64_t absDiff = static_cast<int64_t>(std::abs(static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i])));
        const T upperBound    = std::numeric_limits<T>::max();
        differences[i]        = absDiff > upperBound ? upperBound : static_cast<T>(absDiff);
    }
    return differences;
}

// float specialization of GetAbsoluteDifferences (see above).
template <>
std::vector<float> GetAbsoluteDifferences<float>(const std::vector<float>& a, const std::vector<float>& b);

template <typename T>
uint32_t GetMaxAbsDifference(const std::vector<T>& a, const std::vector<T>& b)
{
    std::vector<T> differences = GetAbsoluteDifferences(a, b);
    T maxDifference            = *std::max_element(differences.begin(), differences.end());
    return static_cast<uint32_t>(maxDifference);
}

/// Compares two arrays.
/// Returns true iff all elements in the arrays are within the given tolerance of each other and
/// the arrays are the same size.
template <typename T>
bool CompareArrays(const std::vector<T>& a, const std::vector<T>& b, float tolerance)
{
    if (a.size() != b.size())
    {
        return false;
    }
    const uint32_t UTolerance = static_cast<uint32_t>(tolerance);

    return GetMaxAbsDifference(a, b) <= UTolerance;
}

// float specialization of CompareArrays (see above).
template <>
bool CompareArrays<float>(const std::vector<float>& a, const std::vector<float>& b, float tolerance);

/// Compares two tensors, which must have the same datatype.
/// Returns true iff all elements in the tensors are within the given tolerance of each other and
/// the tensors are the same size.
bool CompareTensors(const BaseTensor& a, const BaseTensor& b, float tolerance);

std::string DumpOutputToFiles(const BaseTensor& output,
                              const BaseTensor& refOutput,
                              const std::string& filePrefix,
                              const std::string& outputName,
                              size_t outputNumber);

std::string DumpFiles(const BaseTensor& ethosn, const BaseTensor& cpu, std::string& outputNames, float tolerance);

/// Copies the contents of the given driver_library Buffers to the given destinations.
void CopyBuffers(const std::vector<ethosn::driver_library::Buffer*>& sourceBuffers,
                 const std::vector<uint8_t*>& destPointers);

struct Stats
{
    Stats()
        : m_Count(0)
        , m_Frequencies()
        , m_Mean(0)
        , m_Variance(0)
        , m_StandardDeviation(0)
        , m_Mode(0)
        , m_Median(0)
        , m_Max(0)
        , m_DataTypeMin(0)
        , m_DataTypeMax(0)
    {}
    template <typename T>
    Stats(const std::vector<T>& data);

    void PrintHistogram(std::ostream& stream);

    uint32_t m_Count;
    std::map<int64_t, size_t> m_Frequencies;
    float m_Mean;
    float m_Variance;
    float m_StandardDeviation;
    float m_Mode;
    float m_Median;
    int64_t m_Max;

    int64_t m_DataTypeMin;
    int64_t m_DataTypeMax;
};

using InferenceInputBuffers          = std::vector<std::shared_ptr<ethosn::driver_library::Buffer>>;
using InferenceOutputBuffers         = std::vector<std::shared_ptr<ethosn::driver_library::Buffer>>;
using MultipleInferenceOutputBuffers = std::vector<std::vector<std::shared_ptr<ethosn::driver_library::Buffer>>>;

using InferenceInputBuffersPtr          = std::vector<ethosn::driver_library::Buffer*>;
using InferenceOutputBuffersPtr         = std::vector<ethosn::driver_library::Buffer*>;
using MultipleInferenceOutputBuffersPtr = std::vector<std::vector<ethosn::driver_library::Buffer*>>;

using InferenceOutputsPtr         = std::vector<uint8_t*>;
using MultipleInferenceOutputsPtr = std::vector<InferenceOutputsPtr>;

// Helper class to manage a dma buf device file descriptor with C++ RIIA
// to enable RIAA deallocation
class DmaBufferDevice
{
public:
    DmaBufferDevice(DmaBufferDevice&& otherDmaBufferDevice);
    DmaBufferDevice(const char* dmaBufferDeviceFile);

    ~DmaBufferDevice();

    // The returned file descriptor will only be valid as long as this object is in scope
    // take care when using the fd
    int GetFd() const;

    explicit operator bool() const
    {
        return m_DevFd >= 0;
    }

private:
    int m_DevFd;
};

// Helper class to handle dma buf memory allocation file descriptors with C++ RIIA
// to enable RIAA deallocation
class DmaBuffer
{
public:
    DmaBuffer();
    DmaBuffer(DmaBuffer&& otherDmaBuffer);

    DmaBuffer(const std::unique_ptr<DmaBufferDevice>& dmaBufHeap, size_t len);

    ~DmaBuffer();

    DmaBuffer& operator=(DmaBuffer&& otherDmaBuffer);

    // The returned file descriptor will only be valid as long as this object is in scope
    // take care when using the fd
    int GetFd() const;

    size_t GetSize() const;

    void PopulateData(const uint8_t* inData, size_t len);

    void RetrieveData(uint8_t* outData, size_t len);

    explicit operator bool() const
    {
        return m_DmaBufFd >= 0;
    }

private:
    int m_DmaBufFd;
    size_t m_Size;
};

using InferenceDmaBuffers         = std::vector<std::shared_ptr<DmaBuffer>>;
using MultipleInferenceDmaBuffers = std::vector<std::vector<std::shared_ptr<DmaBuffer>>>;

using InferenceResult = std::vector<std::unique_ptr<ethosn::driver_library::Inference>>;

bool IsStatisticalOutputGood(const MultipleInferenceOutputs& output);

bool IsStatisticalOutputGood(const InferenceOutputs& output);

std::string GetCacheFilename(const std::string& sourceFilename, const std::string& cacheFolderOverride);
InferenceOutputs RunNetworkCached(const std::string& cacheFilename, std::function<InferenceOutputs()> runNetworkFunc);

bool IsDataTypeSigned(DataType dataType);

bool IsDataTypeSigned(ethosn::support_library::DataType dataType);

// Resize mode
enum class ResizeMode
{
    DROP,
    REPEAT,
};

struct ScaleParams
{
    uint32_t m_Size   = 0U;
    float m_Ratio     = 0.f;
    ResizeMode m_Mode = ResizeMode::REPEAT;
};

struct ResizeParams
{
    ethosn::support_library::ResizeAlgorithm m_Algo = ethosn::support_library::ResizeAlgorithm::NEAREST_NEIGHBOUR;
    ScaleParams m_Height;
    ScaleParams m_Width;
};

uint32_t CalcUpsampleOutputSize(const ScaleParams& params, const uint32_t inputSize);

std::vector<char> CreateCacheHeader(const InferenceOutputs& outputs);

InferenceOutputs GetOutputTensorsFromCache(std::vector<char>& cacheHeader);

std::vector<uint32_t> GetBinaryDataFromHexFile(std::istream& input, uint32_t startAddress, uint32_t lengthBytes);
OwnedTensor LoadTensorFromHexStream(std::istream& input, DataType dataType, size_t numElements);
OwnedTensor LoadTensorFromBinaryStream(std::istream& input, DataType dataType, size_t numElements);

float GetReferenceComparisonTolerance(const std::map<std::string, float>& referenceComparisonTolerances,
                                      const std::string& outputName);

}    // namespace system_tests
}    // namespace ethosn
