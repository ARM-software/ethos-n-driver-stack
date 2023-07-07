//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "Capabilities.hpp"

#include <ethosn_command_stream/BinaryTuple.hpp>
#include <ethosn_command_stream/CommandData.hpp>
#include <ethosn_command_stream/PleOperation.hpp>
#include <ethosn_command_stream/cascading/CommandStream.hpp>
#include <ethosn_utils/Log.hpp>
#include <ethosn_utils/Macros.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>

namespace ethosn
{
namespace support_library
{

#if defined(ETHOSN_LOGGING)
constexpr ethosn::utils::log::Severity g_LogCompileTimeMaxSeverity = ethosn::utils::log::Severity::Debug;
#else
constexpr ethosn::utils::log::Severity g_LogCompileTimeMaxSeverity = ethosn::utils::log::Severity::Info;
#endif
using LoggerType = ethosn::utils::log::Logger<g_LogCompileTimeMaxSeverity>;
extern LoggerType g_Logger;

class Node;
class FuseOnlyPleOperationNode;
class MceOperationNode;
enum class CascadingBufferFormat;

enum class CompilerDataFormat
{
    NONE,
    NHWC,
    NCHW,
    NHWCB,
    WEIGHT
};

enum class CompilerDataCompressedFormat
{
    NONE,
    FCAF_DEEP,
    FCAF_WIDE
};

bool IsCompressed(CompilerDataCompressedFormat compressedFormat);

inline CompilerDataFormat ConvertExternalToCompilerDataFormat(DataFormat dataFormat)
{
    assert(dataFormat == DataFormat::NHWC || dataFormat == DataFormat::NHWCB || dataFormat == DataFormat::HWIO ||
           dataFormat == DataFormat::HWIM);
    if (dataFormat == DataFormat::NHWC)
    {
        return CompilerDataFormat::NHWC;
    }
    else if (dataFormat == DataFormat::NHWCB)
    {
        return CompilerDataFormat::NHWCB;
    }
    else
    {
        return CompilerDataFormat::WEIGHT;
    }
}

/// The types of algorithm an MceOperation can use or None if it hasn't been decided yet
/// The decision of what algorithm to use is based on several factors including the AlgorithmHint.
enum class CompilerMceAlgorithm
{
    None,
    Winograd,
    Direct
};

struct WinogradOutputShape
{
    uint32_t m_Width;
    uint32_t m_Height;
};

class HardwareCapabilities
{
public:
    HardwareCapabilities(const FirmwareAndHardwareCapabilities& fwAndHwCapabilities);

    uint32_t GetTotalSramSize() const;
    uint32_t GetNumberOfEngines() const;
    uint32_t GetIgsPerEngine() const;
    uint32_t GetOgsPerEngine() const;
    uint32_t GetNumberOfOgs() const;
    uint32_t GetNumberOfSrams() const;
    uint32_t GetNumberofSramsPerEngine() const;
    uint32_t GetMaxPleSize() const;
    uint32_t GetBoundaryStripeHeight() const;
    uint32_t GetNumBoundarySlots() const;
    uint32_t GetNumCentralSlots() const;
    uint32_t GetTotalAccumulatorsPerOg() const;
    uint32_t GetMacUnitsPerOg() const;
    uint32_t GetNumberOfPleLanes() const;

    // It is always 16 MACs per wingorad output block either for 1D (1x3/3x1 filter) or 2D (3x3 filter).
    uint32_t GetMacsPerWinogradOutputBlock() const
    {
        return 16U;
    }

    WinogradOutputShape Get3x3WinogradOutputSize() const
    {
        return { 2, 2 };
    }

    WinogradOutputShape Get3x1WinogradOutputSize() const
    {
        return { 2, 4 };
    }

    WinogradOutputShape Get1x3WinogradOutputSize() const
    {
        return { 4, 2 };
    }

    uint32_t GetWideKernelSize() const
    {
        return 3U;
    }

    std::vector<char> GetData() const
    {
        auto const caps = reinterpret_cast<const char*>(&m_FirmwareAndHardwareCapabilities);
        std::vector<char> ret(caps, caps + sizeof(m_FirmwareAndHardwareCapabilities));
        return ret;
    }

private:
    FirmwareAndHardwareCapabilities m_FirmwareAndHardwareCapabilities;
};

constexpr TensorShape g_PatchShape      = TensorShape{ 1, 4, 4, 1 };
constexpr TensorShape g_BrickGroupShape = TensorShape{ 1, 8, 8, 16 };

constexpr TensorShape g_FcafDeepCellShape = TensorShape{ 1, 8, 8, 32 };
constexpr TensorShape g_FcafWideCellShape = TensorShape{ 1, 8, 16, 16 };
/// The number of bytes of a compressed FCAF cell in DRAM. This is the same for both FCAF_DEEP and FCAF_WIDE.
constexpr uint32_t g_FcafCellSizeBytes = 2112U;

namespace utils
{

template <typename T>
T Clamp(const T& value, const T& low, const T& high)
{
    assert(low <= high);
    if (value < low)
    {
        return low;
    }
    if (value > high)
    {
        return high;
    }
    return value;
}

template <typename Container, typename T>
constexpr std::pair<bool, size_t> FindIndex(const Container& container, T value)
{
    auto it = std::find(container.begin(), container.end(), value);
    if (it == container.end())
    {
        return { false, std::numeric_limits<size_t>::max() };
    }
    return { true, std::distance(container.begin(), it) };
}

template <typename Container, typename T>
constexpr std::pair<bool, T> Find(const Container& container, T value)
{
    auto it = std::find(container.begin(), container.end(), value);
    if (it == container.end())
    {
        return { false, T() };
    }
    return { true, *it };
}

/// Looks up a key in a std::map (or std::unordered_map), returning a given default value
/// if the key is not present in the map.
template <typename MapContainer, typename K, typename V>
constexpr V GetWithDefault(const MapContainer& container, const K& key, const V& def)
{
    auto it = container.find(key);
    if (it == container.end())
    {
        return def;
    }
    return it->second;
}

template <typename Container, typename Func>
constexpr std::pair<bool, size_t> FindIndexIf(const Container& container, Func func)
{
    auto it = std::find_if(container.begin(), container.end(), func);
    if (it == container.end())
    {
        return { false, std::numeric_limits<size_t>::max() };
    }
    return { true, std::distance(container.begin(), it) };
}

template <typename Container, typename Func>
Container FilterNot(Container container, Func func)
{
    // Copy the original container so we don't modify the input.
    Container containerCopy = container;
    auto removeIterator     = std::remove_if(containerCopy.begin(), containerCopy.end(), func);
    containerCopy.erase(removeIterator, std::end(containerCopy));
    return containerCopy;
}

template <typename Container, typename Func>
Container Filter(Container col, Func func)
{
    auto res = FilterNot(col, [func](typename Container::value_type i) { return !func(i); });
    return res;
}

/// Projects each element of the given container using the given map function, and returns the maximum
/// of the projected elements (determined by operator<).
/// Example: Max({1,3,2}, [](auto x){ return x * 2; }) == 6
template <typename TElement, typename Container, typename MapFunc>
TElement Max(Container container, MapFunc mapFunc)
{
    auto it = std::max_element(container.begin(), container.end(),
                               [&](const auto& a, const auto& b) { return mapFunc(a) < mapFunc(b); });
    return mapFunc(*it);
}

/// Calculates the quotient of numerator and denominator as an integer where the result is rounded up to the nearest
/// integer. i.e. ceil(numerator/denominator).
constexpr uint32_t DivRoundUp(uint32_t numerator, uint32_t denominator)
{
    return (numerator + denominator - 1) / denominator;
}

uint32_t RoundDownToPow2(uint32_t x);

/// Returns the first argument rounded UP to the nearest multiple of the second argument
template <typename T, typename S>
constexpr T RoundUpToNearestMultiple(T num, S nearestMultiple)
{
    T remainder = num % nearestMultiple;

    if (remainder == 0)
    {
        return num;
    }

    return num + nearestMultiple - remainder;
}

/// Splits the given multiplier (which must be between 0 and 1) into integer scale and shift amounts,
/// for use in a quantized multiplication.
inline void CalculateQuantizedMultiplierSmallerThanOne(double multiplier, uint16_t& outScale, uint32_t& outShift)
{
    assert(multiplier >= 0.0f);
    if (multiplier == 0.0f)
    {
        outScale = 0;
        outShift = 0;
    }
    else
    {
        const int exp = std::ilogb(multiplier);
        outShift      = static_cast<uint32_t>(-exp - 1);
        outShift += 16;
        outShift             = outShift > 47 ? 47 : outShift;
        uint32_t outScaleU32 = static_cast<uint32_t>(std::lround(std::scalbn(multiplier, static_cast<int>(outShift))));
        assert(outScaleU32 <= (1U << 16));
        if (outScaleU32 == (1U << 16))
        {
            if (outShift > 0)
            {
                outScaleU32 /= 2U;
                --outShift;
            }
            else
            {
                --outScaleU32;
            }
        }
        outScale = static_cast<uint16_t>(outScaleU32);
    }
}

inline void CalculateRescaleMultiplierAndShift(const double rescaleFactor, uint16_t& mult, uint16_t& shift)
{
    int exp;
    const double fr = std::frexp(rescaleFactor, &exp);

    if (exp < -16)
    {
        mult  = 0;
        shift = 0;
    }
    else
    {
        const int precision = std::max(0, 32 - std::max(16, exp));
        mult                = static_cast<uint16_t>(fr * (1U << precision));
        shift               = static_cast<uint16_t>(std::max(precision, exp) - exp);
    }
}

constexpr uint32_t GetElementSizeBytes(ethosn::support_library::DataType type)
{
    switch (type)
    {
        case ethosn::support_library::DataType::UINT8_QUANTIZED:
        case ethosn::support_library::DataType::INT8_QUANTIZED:
            return 1;
        case ethosn::support_library::DataType::INT32_QUANTIZED:
            return 4;
        default:
            return 0;
    }
}

template <typename T>
constexpr uint32_t GetHeight(const T& tensorShape)
{
    return tensorShape[1];
}

template <typename T>
constexpr uint32_t GetWidth(const T& tensorShape)
{
    return tensorShape[2];
}

template <typename T>
constexpr uint32_t GetChannels(const T& tensorShape)
{
    return tensorShape[3];
}

// Get the estimated size of the weights in bytes. This includes the size of the header as well as the data.
uint32_t EstimateWeightSizeBytes(const ethosn::support_library::TensorShape& shape,
                                 const HardwareCapabilities& capabilities,
                                 bool isHwim);

constexpr uint32_t GetNumElements(const ethosn::support_library::TensorShape& shape)
{
    return shape[0] * shape[1] * shape[2] * shape[3];
}

constexpr uint32_t GetNumStripesH(const ethosn::support_library::TensorShape& shape,
                                  const ethosn::support_library::TensorShape& stripeShape)
{
    return DivRoundUp(shape[1], stripeShape[1]);
}

constexpr uint32_t GetNumStripesW(const ethosn::support_library::TensorShape& shape,
                                  const ethosn::support_library::TensorShape& stripeShape)
{
    return DivRoundUp(shape[2], stripeShape[2]);
}

constexpr uint32_t GetNumStripesC(const ethosn::support_library::TensorShape& shape,
                                  const ethosn::support_library::TensorShape& stripeShape)
{
    return DivRoundUp(shape[3], stripeShape[3]);
}

constexpr uint32_t GetNumStripesTotal(const ethosn::support_library::TensorShape& shape,
                                      const ethosn::support_library::TensorShape& stripeShape)
{
    return GetNumStripesH(shape, stripeShape) * GetNumStripesW(shape, stripeShape) * GetNumStripesC(shape, stripeShape);
}

// Get the total size of a tensor in bytes.
constexpr uint32_t TotalSizeBytes(const ethosn::support_library::TensorInfo& info)
{
    return GetElementSizeBytes(info.m_DataType) * GetNumElements(info.m_Dimensions);
}

inline ethosn::support_library::TensorShape
    RoundUpHeightAndWidthToBrickGroup(const ethosn::support_library::TensorShape& shape)
{
    ethosn::support_library::TensorShape roundUp{ shape[0], utils::RoundUpToNearestMultiple(shape[1], 8U),
                                                  utils::RoundUpToNearestMultiple(shape[2], 8U), shape[3] };
    return roundUp;
}

inline uint32_t TotalSizeBytesNHWCB(const ethosn::support_library::TensorInfo& info)
{
    return GetElementSizeBytes(info.m_DataType) * info.m_Dimensions[0] *
           utils::RoundUpToNearestMultiple(info.m_Dimensions[1], 8U) *
           utils::RoundUpToNearestMultiple(info.m_Dimensions[2], 8U) *
           utils::RoundUpToNearestMultiple(info.m_Dimensions[3], 16U);
}

inline uint32_t TotalSizeBytesFCAF(const ethosn::support_library::TensorShape& tensorShape,
                                   const ethosn::support_library::TensorShape& cellShape)
{
    return g_FcafCellSizeBytes * DivRoundUp(GetHeight(tensorShape), GetHeight(cellShape)) *
           DivRoundUp(GetWidth(tensorShape), GetWidth(cellShape)) *
           DivRoundUp(GetChannels(tensorShape), GetChannels(cellShape));
}

inline uint32_t TotalSizeBytesFCAFDeep(const ethosn::support_library::TensorInfo& tensorInfo)
{
    return TotalSizeBytesFCAF(tensorInfo.m_Dimensions, g_FcafDeepCellShape);
}

inline uint32_t TotalSizeBytesFCAFWide(const ethosn::support_library::TensorInfo& tensorInfo)
{
    return TotalSizeBytesFCAF(tensorInfo.m_Dimensions, g_FcafWideCellShape);
}

uint32_t CalculateBufferSize(const TensorShape& shape, CascadingBufferFormat dataFormat);

uint32_t GetNumOrigChannels(uint32_t nChannels,
                            uint32_t strideX,
                            uint32_t strideY,
                            const HardwareCapabilities& capabilities);

uint32_t GetNumSubmapChannels(uint32_t nChannels,
                              uint32_t strideX,
                              uint32_t strideY,
                              const HardwareCapabilities& capabilities);

uint32_t CalculateDramOffset(const CascadingBufferFormat dataFormat,
                             const TensorShape& tensorSize,
                             const TensorShape& offset);

uint32_t CalculateDramOffsetNHWCB(const TensorShape& tensorShape, uint32_t offsetY, uint32_t offsetX, uint32_t offsetC);

uint32_t CalculateDramOffsetNHWC(const TensorShape& tensorShape, uint32_t offsetY, uint32_t offsetX, uint32_t offsetC);

uint32_t
    CalculateDramOffsetFcafDeep(const TensorShape& tensorShape, uint32_t offsetY, uint32_t offsetX, uint32_t offsetC);
uint32_t
    CalculateDramOffsetFcafWide(const TensorShape& tensorShape, uint32_t offsetY, uint32_t offsetX, uint32_t offsetC);

inline uint32_t
    GetTensorIndex(const TensorShape& tensorShape, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3)
{
    assert(dim0 < tensorShape[0] && dim1 < tensorShape[1] && dim2 < tensorShape[2] && dim3 < tensorShape[3]);
    // clang-format off
    uint32_t index = dim0 * tensorShape[1] * tensorShape[2] * tensorShape[3] +
                     dim1 * tensorShape[2] * tensorShape[3] +
                     dim2 * tensorShape[3] +
                     dim3;
    // clang-format on
    return index;
}

// Helper class to read data from a tightly-packed multidimensional array (e.g. weights data)
class ConstTensorData
{
public:
    ConstTensorData(const uint8_t* data, const TensorShape& tensorShape)
        : m_Data(data)
        , m_TensorShape(tensorShape)
    {}

    const uint8_t& GetElementRef(uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3) const
    {
        return m_Data[GetTensorIndex(m_TensorShape, dim0, dim1, dim2, dim3)];
    }

    uint8_t GetElement(uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3) const
    {
        return GetElementRef(dim0, dim1, dim2, dim3);
    }

    const TensorShape& GetShape() const
    {
        return m_TensorShape;
    }

private:
    const uint8_t* m_Data;
    TensorShape m_TensorShape;
};

// Helper class to read data from a tightly-packed multidimensional array (e.g. weights data)
class TensorData
{
public:
    TensorData(uint8_t* data, const TensorShape& tensorShape)
        : m_Data(data)
        , m_TensorShape(tensorShape)
    {}

    uint8_t& GetElementRef(uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3) const
    {
        return m_Data[GetTensorIndex(m_TensorShape, dim0, dim1, dim2, dim3)];
    }

    void SetElement(uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3, uint8_t value) const
    {
        GetElementRef(dim0, dim1, dim2, dim3) = value;
    }

    const TensorShape& GetShape() const
    {
        return m_TensorShape;
    }

private:
    uint8_t* m_Data;
    TensorShape m_TensorShape;
};

namespace
{

enum class NodeState
{
    Visiting,
    Visited,
};

template <typename TNodeId>
bool Visit(TNodeId current,
           std::function<std::vector<TNodeId>(TNodeId)> getIncomingEdges,
           std::vector<TNodeId>& outSorted,
           std::map<TNodeId, NodeState>& nodeStates)
{
    auto currentStateIt = nodeStates.find(current);
    if (currentStateIt != nodeStates.end())
    {
        if (currentStateIt->second == NodeState::Visited)
        {
            return true;
        }
        if (currentStateIt->second == NodeState::Visiting)
        {
            return false;
        }
        else
        {
            assert(false);
        }
    }

    nodeStates[current] = NodeState::Visiting;

    for (TNodeId inputNode : getIncomingEdges(current))
    {
        Visit(inputNode, getIncomingEdges, outSorted, nodeStates);
    }

    nodeStates[current] = NodeState::Visited;

    outSorted.push_back(current);
    return true;
}

}    // namespace

// Sorts an directed acyclic graph (DAG) into a flat list such that all inputs to a node are before the node itself.
// Returns true if successful or false if there is an error in the graph structure (e.g. it contains a cycle).
// The graph is defined entirely by the "getIncomingEdges" function which the user provides. For a given node,
// it must return the list of nodes which are required to come before it.
// "targetNodes" is the list of nodes where the search begins - i.e. the nodes that you want to evaluate.
// The implementation is based on https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
template <typename TNodeId, typename TTargetNodes>
bool GraphTopologicalSort(const TTargetNodes& targetNodes,
                          std::function<std::vector<TNodeId>(TNodeId)> getIncomingEdges,
                          std::vector<TNodeId>& outSorted)
{
    outSorted.clear();
    std::map<TNodeId, NodeState> nodeStates;

    for (TNodeId targetNode : targetNodes)
    {
        if (!Visit(targetNode, getIncomingEdges, outSorted, nodeStates))
        {
            return false;
        }
    }

    return true;
}

/// e.g. Map({1, 2, 3}, [](int x) { return 2*x; }) == {2, 4, 6}
template <typename Out, typename In, typename Func>
constexpr std::vector<Out> Map(const std::vector<In>& container, Func func)
{
    std::vector<Out> result;
    std::transform(container.begin(), container.end(), std::back_inserter(result), func);
    return result;
}

struct Fraction
{
    uint32_t m_Numerator;
    uint32_t m_Denominator;

    constexpr Fraction(uint32_t value)
        : m_Numerator(value)
        , m_Denominator(1)
    {}
    constexpr Fraction(uint32_t numerator, uint32_t denominator)
        : m_Numerator(numerator)
        , m_Denominator(denominator)
    {}

    Fraction operator*(const Fraction& rhs) const
    {
        return { m_Numerator * rhs.m_Numerator, m_Denominator * rhs.m_Denominator };
    }
};

inline uint32_t operator*(const Fraction& f, uint32_t i)
{
    return (i * f.m_Numerator) / f.m_Denominator;
}
inline uint32_t operator*(uint32_t i, const Fraction& f)
{
    return f * i;
}
inline uint32_t operator/(uint32_t i, const Fraction& f)
{
    return (i * f.m_Denominator) / f.m_Numerator;
}

/// Represents a scaling of a 3D shape by a different amount in each dimension.
/// Each amount is represented by a fraction, to allow both magnifications (e.g. upscaling)
/// and reductions (e.g. pooling) by exact amounts.
struct ShapeMultiplier
{
    Fraction m_H;
    Fraction m_W;
    Fraction m_C;

    ShapeMultiplier operator*(const ShapeMultiplier& rhs) const
    {
        return { m_H * rhs.m_H, m_W * rhs.m_W, m_C * rhs.m_C };
    }

    static const ShapeMultiplier Identity;
};

inline TensorShape operator*(const TensorShape& lhs, const ShapeMultiplier& rhs)
{
    return { lhs[0], lhs[1] * rhs.m_H, lhs[2] * rhs.m_W, lhs[3] * rhs.m_C };
}

constexpr ShapeMultiplier g_IdentityShapeMultiplier = { Fraction{ 1, 1 }, Fraction{ 1, 1 }, Fraction{ 1, 1 } };

command_stream::DataType GetCommandDataType(const DataType supportLibraryDataType);

struct DataTypeRange
{
    int32_t min;
    int32_t max;
};

DataTypeRange GetRangeOfDataType(const DataType type);

template <typename T>
constexpr DataTypeRange GetTypeLimits()
{
    return { std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max() };
}

bool IsCompressionFormatCompatibleWithStripeShape(CompilerDataCompressedFormat compressionFormat,
                                                  const TensorShape& stripeShape,
                                                  const TensorShape& dramTensorShape);

struct NeedBoundary
{
    bool m_Before;
    bool m_After;
};

inline NeedBoundary GetBoundaryRequirements(const uint32_t padBefore,
                                            const uint32_t ifmStripeSize,
                                            const uint32_t ofmStripeSize,
                                            const uint32_t weightSize,
                                            bool isUpscale)
{
    return NeedBoundary{ padBefore > 0, ((ofmStripeSize + weightSize - padBefore - 1U) > ifmStripeSize) || isUpscale };
}

std::vector<command_stream::BlockConfig>
    FilterPleBlockConfigs(const command_stream::PleOperation pleOp,
                          const std::vector<command_stream::BlockConfig>& allowedBlockConfigs);
bool PleBlockConfigAllowed(const command_stream::PleOperation pleOp,
                           const command_stream::BlockConfig allowedBlockConfig);

constexpr int32_t g_IdentityWeightValue = 128;
constexpr float g_IdentityWeightScale   = 1.f / static_cast<float>(g_IdentityWeightValue);

/// Gets the internal data, reinterpreted as an array of the given type.
/// Note this incurs a full copy of the data.
template <typename T, typename S>
inline std::vector<T> GetDataVectorAs(const std::vector<S>& data)
{
    assert(data.size() % sizeof(T) == 0);    // Otherwise won't fit exactly in result type.
    size_t numElements = data.size() / sizeof(T);
    std::vector<T> result(numElements);
    std::memcpy(result.data(), data.data(), data.size());
    return result;
}

unsigned CalculateSpaceToDepthSramUsage(uint32_t blockSize, uint32_t s1, uint32_t s2);

std::pair<uint32_t, uint32_t>
    CalculateSpaceToDepthBlockSizes(const TensorShape tensor, uint32_t usedSrams, uint32_t blockSize);

std::tuple<bool, bool, bool> IsSplitting(const TensorShape& tensorShape, const TensorShape& stripeShape);

bool IsFullTensor(const TensorShape& tensorShape, const TensorShape& stripeShape);

ethosn::command_stream::DumpDram GetDumpDramCommand(
    const TensorShape& shape, uint32_t bufferId, DataType dataType, int32_t zeroPoint, const char* format);

/// Checks if two ranges overlap.
bool CheckOverlap(uint32_t startA, uint32_t sizeA, uint32_t startB, uint32_t sizeB);

}    // namespace utils

}    // namespace support_library

}    // namespace ethosn

inline ethosn::support_library::TensorShape& operator+=(ethosn::support_library::TensorShape& l,
                                                        const ethosn::support_library::TensorShape& r)
{
    l[0] += r[0];
    l[1] += r[1];
    l[2] += r[2];
    l[3] += r[3];
    return l;
}
