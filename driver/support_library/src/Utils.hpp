//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "Capabilities.hpp"

#include <ethosn_command_stream/BinaryTuple.hpp>
#include <ethosn_command_stream/PleOperation.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>

// To prevent the warning that we arent using a variable, use this macro.
#define UNUSED(x) (void)(x)

namespace ethosn
{
namespace support_library
{

class HardwareCapabilities
{
public:
    HardwareCapabilities(const FirmwareAndHardwareCapabilities& fwAndHwCapabilities);

    uint32_t GetTotalSramSize() const;
    uint32_t GetNumberOfEngines() const;
    uint32_t GetIfmPerEngine() const;
    uint32_t GetOfmPerEngine() const;
    uint32_t GetNumberOfOfm() const;
    uint32_t GetNumberOfSrams() const;
    uint32_t GetNumberofSramsPerEngine() const;
    uint32_t GetMaxPleSize() const;
    uint32_t GetBoundaryStripeHeight() const;
    uint32_t GetNumBoundarySlots() const;
    uint32_t GetNumCentralSlots() const;
    const TensorShape& GetBrickGroupShape() const;
    const TensorShape& GetPatchShape() const;
    uint32_t GetTotalAccumulatorsPerEngine() const;
    uint32_t GetMacUnitsPerEngine() const;

    uint32_t GetMacsPerWinograd2D() const
    {
        return 16U;
    }

    uint32_t GetOutputSizePerWinograd2D() const
    {
        return 2U;
    }

    uint32_t GetMacsPerWinograd1D() const
    {
        return 4U;
    }

    uint32_t GetOutputSizePerWinograd1D() const
    {
        return 1U;
    }

    uint32_t GetWideKernelSize() const
    {
        return 3U;
    }

private:
    FirmwareAndHardwareCapabilities m_FirmwareAndHardwareCapabilities;
};

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

/// Calculates the quotient of numerator and denominator as an integer where the result is rounded up to the nearest
/// integer. i.e. ceil(numerator/denominator).
constexpr uint32_t DivRoundUp(uint32_t numerator, uint32_t denominator)
{
    return (numerator + denominator - 1) / denominator;
}

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
    assert(multiplier >= 0.0f && multiplier < 1.0f);
    if (multiplier == 0.0f)
    {
        outScale = 0;
        outShift = 0;
    }
    else
    {
        const int exp = std::ilogb(multiplier);
        outShift      = static_cast<uint32_t>(-exp - 1);
        assert(outShift < 32);
        uint32_t outScaleU32 = static_cast<uint32_t>(std::lround(std::scalbn(multiplier, 16 + outShift)));
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
            return 1;
        case ethosn::support_library::DataType::INT32_QUANTIZED:
            return 4;
        default:
            return 0;
    }
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
    ethosn::support_library::TensorShape roundUp{ shape[0], utils::RoundUpToNearestMultiple(shape[1], 8),
                                                  utils::RoundUpToNearestMultiple(shape[2], 8), shape[3] };
    return roundUp;
}

inline uint32_t TotalSizeBytesNHWCB(const ethosn::support_library::TensorInfo& info)
{
    return GetElementSizeBytes(info.m_DataType) * info.m_Dimensions[0] *
           utils::RoundUpToNearestMultiple(info.m_Dimensions[1], 8) *
           utils::RoundUpToNearestMultiple(info.m_Dimensions[2], 8) *
           utils::RoundUpToNearestMultiple(info.m_Dimensions[3], 16);
}

inline uint32_t TotalSizeBytesNHWCBCompressed(const ethosn::support_library::TensorInfo& info)
{
    assert(info.m_DataType == DataType::UINT8_QUANTIZED);
    auto shape                            = info.m_Dimensions;
    constexpr uint32_t brickGroupWidth    = 8;
    constexpr uint32_t brickGroupHeight   = 8;
    constexpr uint32_t brickGroupChannels = 16;

    const uint32_t numBrickGroupRows  = DivRoundUp(shape[1], brickGroupHeight);
    const uint32_t numBrickGroupCols  = DivRoundUp(shape[2], brickGroupWidth);
    const uint32_t numBrickGroupDepth = DivRoundUp(shape[3], brickGroupChannels);

    // The compressed row sizes are stored contiguously as 16 bit values at the beginning of the buffer and
    // is aligned to 64 bytes.
    // See the Activation Compression Specification
    // The row sizes are defined to be 16-bits.
    const uint32_t rowSizeMetaData = static_cast<uint32_t>(sizeof(uint16_t)) * numBrickGroupRows;

    // Compressed NHWCB needs 1 byte of metadata per patch group.
    const uint32_t sizeOfBrickGroupRow =
        numBrickGroupDepth * numBrickGroupCols * (brickGroupWidth * brickGroupHeight + 1) * brickGroupChannels;

    return utils::RoundUpToNearestMultiple(rowSizeMetaData, 64) +
           utils::RoundUpToNearestMultiple(sizeOfBrickGroupRow, 64) * numBrickGroupRows;
}

uint32_t GetNumOrigChannels(uint32_t nChannels,
                            uint32_t strideX,
                            uint32_t strideY,
                            const HardwareCapabilities& capabilities);

uint32_t GetNumSubmapChannels(uint32_t nChannels,
                              uint32_t strideX,
                              uint32_t strideY,
                              const HardwareCapabilities& capabilities);

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

template <typename TElement>
std::vector<TElement*> GetRawPointers(const std::vector<std::unique_ptr<TElement>>& container)
{
    std::vector<TElement*> result;
    std::transform(container.begin(), container.end(), std::back_inserter(result),
                   [](const std::unique_ptr<TElement>& x) -> TElement* { return x.get(); });
    return result;
}

std::string ReplaceAll(std::string str, const std::string& from, const std::string& to);

/// e.g. Map({1, 2, 3}, [](int x) { return 2*x; }) == {2, 4, 6}
template <typename Out, typename In, typename Func>
constexpr std::vector<Out> Map(const std::vector<In>& container, Func func)
{
    std::vector<Out> result;
    std::transform(container.begin(), container.end(), std::back_inserter(result), func);
    return result;
}

template <typename T>
T NextEnumValue(T current)
{
    return static_cast<T>(static_cast<uint32_t>(current) + 1);
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

constexpr ShapeMultiplier g_IdentityShapeMultiplier = { Fraction{ 1, 1 }, Fraction{ 1, 1 }, Fraction{ 1, 1 } };

}    // namespace utils

}    // namespace support_library

}    // namespace ethosn
