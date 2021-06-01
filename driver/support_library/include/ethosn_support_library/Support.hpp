//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Optional.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <valarray>
#include <vector>

// Version information
#define ETHOSN_SUPPORT_LIBRARY_VERSION_MAJOR 1
#define ETHOSN_SUPPORT_LIBRARY_VERSION_MINOR 1
#define ETHOSN_SUPPORT_LIBRARY_VERSION_PATCH 0

namespace ethosn
{
namespace support_library
{
class Network;
class Input;
class Output;
class Operand;
class Constant;
class SupportQueries;

struct Version
{
    Version();
    Version(const uint32_t Major, const uint32_t Minor, const uint32_t Patch);
    Version(const char* version);

    const std::string ToString() const;

    uint32_t Major;
    uint32_t Minor;
    uint32_t Patch;
};

#define COMPILER_ALGORITHM_MODE                                                                                        \
    X(Auto)                                                                                                            \
    X(CascadingOnly)                                                                                                   \
    X(NonCascadingOnly)

#define X(value) value,
enum class CompilerAlgorithm
{
    COMPILER_ALGORITHM_MODE
};
#undef X

const char* EthosNCompilerAlgorithmAsString(CompilerAlgorithm mode);
CompilerAlgorithm EthosNCompilerAlgorithmFromString(const char* mode);

struct CompilationOptions
{
    enum class DebugLevel
    {
        None,
        Medium,
        High,
    };

    struct DebugInfo
    {
        DebugLevel m_DumpDebugFiles = DebugLevel::None;
        std::string m_DebugDir      = ".";
        bool m_DumpRam              = false;
        bool m_InitialSramDump      = false;
    };

    bool m_Strategy0                     = true;
    bool m_Strategy1                     = true;
    bool m_Strategy3                     = true;
    bool m_Strategy4                     = true;
    bool m_Strategy6                     = true;
    bool m_Strategy7                     = true;
    bool m_BlockConfig16x16              = true;
    bool m_BlockConfig32x8               = true;
    bool m_BlockConfig8x32               = true;
    bool m_BlockConfig16x8               = true;
    bool m_BlockConfig8x16               = true;
    bool m_BlockConfig8x8                = true;
    bool m_EnableIntermediateCompression = true;
    bool m_DisableWinograd               = false;

    bool m_StrictPrecision =
        false;    // Set this to true to create a more precise but slower compiled network. At the moment this will disable the concat optimization.

    /// If enabled, files containing details of the compilation process will be dumped to m_DebugDir.
    /// These can be helpful for debugging compilation issues.
    DebugInfo m_DebugInfo;
    /// The m_CompilerAlgorithm can be used to force one approach over another as cascaded vs non cascaded.
    /// "CascadingOnly" means that the cascaded approach will be used
    /// "NonCascadingOnly" means that the non cascaded approach will be used
    /// "Auto" means the compiler decides to do what is best which is
    /// - for compilation: using non cascaded approach
    /// - for estimation: executing cascaded and non cascaded approach and returning
    ///                   the one which is the more performant
    CompilerAlgorithm m_CompilerAlgorithm = CompilerAlgorithm::NonCascadingOnly;
};

/// Contains options for performance estimation
struct EstimationOptions
{
    /// The proportion of space saved with activation compression, where it can be used. (Default 0.0f indicates no compression)
    /// Appropriate values for this parameter are determined by network topology, weights and input data. Please contact Arm for more details.
    float m_ActivationCompressionSaving = 0.0f;
    /// Switch to override the weight compression with the space saving proportion below
    bool m_UseWeightCompressionOverride = false;
    /// The proportion of space saved with weight compression if m_UseWeightCompressionOverride is set to true (0.0f indicates no compression)
    /// Appropriate values for this parameter are determined weights and the compression method used. Please contact Arm for more details.
    float m_WeightCompressionSaving = 0.0f;
    /// Switch to use "current" numbers which estimates the performance as measured with todays software.
    /// Default is to be using "future" estimates, i.e. possible future performance of the stack.
    bool m_Current = false;
};

struct MceStats
{
    MceStats()
        : m_Operations(0)
        , m_CycleCount(0)
    {}

    MceStats operator+(const MceStats& rhs) const
    {
        return (MceStats(*this) += rhs);
    }

    MceStats operator+=(const MceStats& rhs)
    {
        m_Operations += rhs.m_Operations;
        m_CycleCount += rhs.m_CycleCount;
        return *this;
    }

    // Number of MAC operations (multiplications and additions)
    uint64_t m_Operations;
    // Number of cycles to complete all MAC operations, expressed in cycles
    uint64_t m_CycleCount;
};

struct PleStats
{
    PleStats()
        : m_Operation(0)
        , m_NumOfPatches(0)
    {}

    uint32_t m_Operation;
    uint32_t m_NumOfPatches;
};

struct MemoryStats
{
    MemoryStats()
        : m_DramParallel(0)
        , m_DramNonParallel(0)
        , m_Sram(0)
    {}

    MemoryStats operator+(const MemoryStats& rhs) const
    {
        return (MemoryStats(*this) += rhs);
    }

    MemoryStats operator+=(const MemoryStats& rhs)
    {
        m_DramParallel += rhs.m_DramParallel;
        m_DramNonParallel += rhs.m_DramNonParallel;
        m_Sram += rhs.m_Sram;
        return *this;
    }

    // Data that can be transferred (from/to Dram) in parallel with other operations, expressed in bytes
    uint32_t m_DramParallel;
    // Data that can NOT be transferred (from/to Dram) in parallel with other operations, expressed in bytes
    uint32_t m_DramNonParallel;
    // Data located in internal memory, expressed in bytes
    uint32_t m_Sram;
};

struct StripesStats
{
    StripesStats()
        : m_NumCentralStripes(0)
        , m_NumBoundaryStripes(0)
        , m_NumReloads(0)
    {}

    StripesStats operator+(const StripesStats& rhs) const
    {
        return (StripesStats(*this) += rhs);
    }

    StripesStats operator+=(const StripesStats& rhs)
    {
        m_NumCentralStripes += rhs.m_NumCentralStripes;
        m_NumBoundaryStripes += rhs.m_NumBoundaryStripes;
        m_NumReloads += rhs.m_NumReloads;
        return *this;
    }

    uint32_t m_NumCentralStripes;
    uint32_t m_NumBoundaryStripes;
    // Number of data reloads (depends on the streaming strategy selected)
    uint32_t m_NumReloads;
};

struct InputStats
{
    InputStats() noexcept
        : m_MemoryStats()
        , m_StripesStats()
    {}

    InputStats operator+(const InputStats& rhs) const
    {
        return (InputStats(*this) += rhs);
    }

    InputStats operator+=(const InputStats& rhs)
    {
        m_MemoryStats += rhs.m_MemoryStats;
        m_StripesStats += rhs.m_StripesStats;
        return *this;
    }

    MemoryStats m_MemoryStats;
    StripesStats m_StripesStats;
};

using OutputStats = InputStats;

struct WeightsStats : InputStats
{
    float m_WeightCompressionSavings;
};

/// The performance stats for a single pass.
struct PassStats
{
    PassStats()
        : m_Input()
        , m_Output()
        , m_Weights()
        , m_Mce()
        , m_Ple()
    {}
    InputStats m_Input;
    OutputStats m_Output;
    WeightsStats m_Weights;
    MceStats m_Mce;
    PleStats m_Ple;
};

/// Performance data for a single pass pairs performance stats with network topology meta-data.
struct PassPerformanceData
{
    PassPerformanceData()
        : m_OperationIds()
        , m_ParentIds()
        , m_Stats()
    {}

    /// The set of operations from the input Network that are associated with this pass.
    /// Note that one pass may be associated with multiple operations (e.g. if operations were fused)
    /// and one operation may be associated with multiple passes (e.g. if an operation was split).
    std::set<uint32_t> m_OperationIds;
    /// List of ID's of parent passes grouped in concatenation groups.
    /// An integer ID represents the position in m_Stream inside the container NetworkPerformanceData.
    /// A parent pass is any pass that produces data that this pass consumes.
    /// The result of multiple passes may be concatenated together before this pass consumes the
    /// concatenated tensor.
    /// The string is formatted as a Json array with each element representing an input to the pass.
    /// Each element in turn can be a parent pass ID or an array representing a concatenation, and the
    /// same applies recursively to the elements of that array.
    std::string m_ParentIds;
    PassStats m_Stats;
};

struct NetworkPerformanceData
{
    /// The performance figures grouped into passes. Each pass will be associated with one or more operations
    /// from the source Network. Note that the original operations may have been merged and/or reordered in this view,
    /// according to how the Ethos-N optimizes the network.
    std::vector<PassPerformanceData> m_Stream;
    std::map<uint32_t, std::string> m_OperationIdFailureReasons;
};

/// Prints the given NetworkPerformanceData struct in a JSON format to the given stream.
void PrintNetworkPerformanceDataJson(std::ostream& os, uint32_t indentNumTabs, const NetworkPerformanceData& perfData);

// Data types for tensors
enum class DataType
{
    UINT8_QUANTIZED,    ///< Contiguously packed 8-bit unsigned integers, interpreted as based on the QuantizationInfo.
    INT8_QUANTIZED,     ///< Contiguously packed 8-bit signed integers, interpreted as based on the QuantizationInfo.
    INT32_QUANTIZED     ///< Contiguously packed 32-bit signed integers, interpreted as based on the QuantizationInfo.
};

// Gives names to dimensions in a tensor
// (N = batch, H = height, W = width, C = channel, I = input_channel, O = output_channel, M = channel_multiplier, B = Brick)
enum class DataFormat
{
    NHWC,
    NCHW,
    HWIO,
    NHWCB,
    HWIM,
};

// Pooling function types
enum class PoolingType
{
    MAX,
    AVG,
};

// Resize algorithm
enum class ResizeAlgorithm
{
    NEAREST_NEIGHBOUR,
    BILINEAR,
};

// QuantizationScales vector supporting per element and scalar multiplication
struct QuantizationScales : public std::valarray<float>
{
public:
    // Inherit valarray constructors
    using std::valarray<float>::valarray;

    QuantizationScales(std::valarray<float>&& rhs)
        : std::valarray<float>(std::move(rhs))
    {}

    explicit QuantizationScales(const std::vector<float>& vect)
        : std::valarray<float>(vect.data(), vect.size())
    {}
};

// QuantizationScales operator overloads to support boolean comparison
bool operator==(const QuantizationScales& lhs, const QuantizationScales& rhs);
bool operator!=(const QuantizationScales& lhs, const QuantizationScales& rhs);

// QuantizationScales operator overloads to support scalar scales operations
QuantizationScales operator/(const QuantizationScales& lhs, const QuantizationScales& rhs);
QuantizationScales operator*(const QuantizationScales& lhs, const QuantizationScales& rhs);

// Scale and zero point for quantized asymmetric values
struct QuantizationInfo
{
    using QuantizationDim = utils::Optional<uint32_t>;

    QuantizationInfo()
        : QuantizationInfo(0, QuantizationScales{ 1.0f })
    {}

    QuantizationInfo(const int32_t zeroPoint, const float scale)
        : QuantizationInfo(zeroPoint, QuantizationScales{ scale })
    {}

    QuantizationInfo(const int32_t zeroPoint,
                     QuantizationScales scales,
                     const QuantizationDim dim = utils::EmptyOptional{})
        : m_ZeroPoint(zeroPoint)
        , m_Scales(std::move(scales))
        , m_QuantizationDim(dim)
    {}

    bool operator==(const QuantizationInfo& quantizationInfo) const
    {
        return (m_ZeroPoint == quantizationInfo.m_ZeroPoint) && (m_Scales == quantizationInfo.m_Scales) &&
               (m_QuantizationDim == quantizationInfo.m_QuantizationDim);
    }

    bool operator!=(const QuantizationInfo& rhs) const
    {
        return !(*this == rhs);
    }

    int32_t GetZeroPoint() const
    {
        return m_ZeroPoint;
    }

    void SetZeroPoint(int32_t zeroPoint)
    {
        m_ZeroPoint = zeroPoint;
    }

    float GetScale() const
    {
        assert(m_Scales.size() == 1);
        return m_Scales[0];
    }

    float GetScale(std::size_t index) const
    {
        return m_Scales[index];
    }

    void SetScale(float scale)
    {
        assert(m_Scales.size() == 1);
        m_Scales[0] = scale;
    }

    const QuantizationScales& GetScales() const
    {
        return m_Scales;
    }

    void SetScales(const QuantizationScales& scales)
    {
        m_Scales = scales;
    }

    void SetScales(const std::vector<float>& scales)
    {
        m_Scales = QuantizationScales(scales.data(), scales.size());
    }

    QuantizationDim GetQuantizationDim() const
    {
        return m_QuantizationDim;
    }

    void SetQuantizationDim(unsigned int quantizationDim)
    {
        m_QuantizationDim = quantizationDim;
    }

private:
    int32_t m_ZeroPoint;
    QuantizationScales m_Scales;
    QuantizationDim m_QuantizationDim;
};

using TensorShape = std::array<uint32_t, 4>;

// Tensor dimensions, data format and quantization info
struct TensorInfo
{
    TensorInfo(const TensorShape& dims       = {},
               DataType dataType             = DataType::UINT8_QUANTIZED,
               DataFormat dataFormat         = DataFormat::NHWC,
               const QuantizationInfo& qInfo = {})
        : m_Dimensions(dims)
        , m_DataType(dataType)
        , m_DataFormat(dataFormat)
        , m_QuantizationInfo(qInfo)
    {}

    bool operator==(const TensorInfo& rhs) const
    {
        return m_Dimensions == rhs.m_Dimensions && m_DataType == rhs.m_DataType && m_DataFormat == rhs.m_DataFormat &&
               m_QuantizationInfo == rhs.m_QuantizationInfo;
    }

    bool operator!=(const TensorInfo& rhs) const
    {
        return !(*this == rhs);
    }

    TensorShape m_Dimensions;
    DataType m_DataType;
    DataFormat m_DataFormat;
    QuantizationInfo m_QuantizationInfo;
};

// Padding specification for convolution/pooling/... operations
struct Padding
{
    constexpr Padding(const uint32_t top    = 0,
                      const uint32_t bottom = 0,
                      const uint32_t left   = 0,
                      const uint32_t right  = 0)
        : m_Top(top)
        , m_Bottom(bottom)
        , m_Left(left)
        , m_Right(right)
    {}

    bool operator==(const Padding& rhs) const
    {
        return m_Top == rhs.m_Top && m_Bottom == rhs.m_Bottom && m_Left == rhs.m_Left && m_Right == rhs.m_Right;
    }

    bool operator!=(const Padding& rhs) const
    {
        return !(*this == rhs);
    }

    uint32_t m_Top;
    uint32_t m_Bottom;
    uint32_t m_Left;
    uint32_t m_Right;
};

// Stride specification for convolution/pooling/... operations
struct Stride
{
    constexpr Stride(const uint32_t x = 1, const uint32_t y = 1)
        : m_X(x)
        , m_Y(y)
    {}

    bool operator==(const Stride& rhs) const
    {
        return m_X == rhs.m_X && m_Y == rhs.m_Y;
    }

    uint32_t m_X;
    uint32_t m_Y;
};

// Parameters that specify a convolution operation
struct ConvolutionInfo
{
    ConvolutionInfo(const Padding& padding = {}, const Stride& stride = {}, const QuantizationInfo& qInfo = {})
        : m_Padding(padding)
        , m_Stride(stride)
        , m_OutputQuantizationInfo(qInfo)
    {}

    bool operator==(const ConvolutionInfo& rhs) const
    {
        return m_Padding == rhs.m_Padding && m_Stride == rhs.m_Stride &&
               m_OutputQuantizationInfo == rhs.m_OutputQuantizationInfo;
    }

    bool operator!=(const ConvolutionInfo& rhs) const
    {
        return !(*this == rhs);
    }

    Padding m_Padding;
    Stride m_Stride;
    QuantizationInfo m_OutputQuantizationInfo;
};

// Parameters that specify a fully connected operation
struct FullyConnectedInfo
{
    FullyConnectedInfo(const QuantizationInfo& qInfo = {})
        : m_OutputQuantizationInfo(qInfo)
    {}

    QuantizationInfo m_OutputQuantizationInfo;
};

// Parameters that specify a reinterpret quantization operation
struct ReinterpretQuantizationInfo
{
    ReinterpretQuantizationInfo(const QuantizationInfo& qInfo = {})
        : m_OutputQuantizationInfo(qInfo)
    {}

    QuantizationInfo m_OutputQuantizationInfo;
};

// Parameters that specify a relu operation
struct ReluInfo
{
    ReluInfo()
        : m_LowerBound(0)
        , m_UpperBound(255)
    {}

    constexpr ReluInfo(int16_t lowerBound, int16_t upperBound)
        : m_LowerBound(lowerBound)
        , m_UpperBound(upperBound)
    {}

    bool operator==(const ReluInfo& rhs) const
    {
        return m_LowerBound == rhs.m_LowerBound && m_UpperBound == rhs.m_UpperBound;
    }

    bool operator!=(const ReluInfo& rhs) const
    {
        return !(*this == rhs);
    }

    /// The bounds of the relu, specified in the quantised space of the input to the Relu operation.
    /// @{
    int16_t m_LowerBound;
    int16_t m_UpperBound;
    /// @}
};

// Parameters that specify a LeakyRelu operation
struct LeakyReluInfo
{
    LeakyReluInfo()
        : m_Alpha(0.01f)
        , m_OutputQuantizationInfo()
    {}

    LeakyReluInfo(const float alpha, const QuantizationInfo& qInfo)
        : m_Alpha(alpha)
        , m_OutputQuantizationInfo(qInfo)
    {}

    bool operator==(const LeakyReluInfo& rhs) const
    {
        return ((m_Alpha == rhs.m_Alpha) && (m_OutputQuantizationInfo == rhs.m_OutputQuantizationInfo));
    }

    bool operator!=(const LeakyReluInfo& rhs) const
    {
        return !(*this == rhs);
    }

    float m_Alpha;
    QuantizationInfo m_OutputQuantizationInfo;
};

// Parameters that specify a Requantize operation
struct RequantizeInfo
{
    RequantizeInfo()
        : m_OutputQuantizationInfo()
    {}

    RequantizeInfo(const QuantizationInfo& qInfo)
        : m_OutputQuantizationInfo(qInfo)
    {}

    bool operator==(const RequantizeInfo& rhs) const
    {
        return (m_OutputQuantizationInfo == rhs.m_OutputQuantizationInfo);
    }

    bool operator!=(const RequantizeInfo& rhs) const
    {
        return !(*this == rhs);
    }

    QuantizationInfo m_OutputQuantizationInfo;
};

// Parameters that specify a pooling operation
struct PoolingInfo
{
    constexpr PoolingInfo(uint32_t poolingSizeX,
                          uint32_t poolingSizeY,
                          uint32_t poolingStrideX,
                          uint32_t poolingStrideY,
                          Padding padding,
                          PoolingType type)
        : m_PoolingSizeX(poolingSizeX)
        , m_PoolingSizeY(poolingSizeY)
        , m_PoolingStrideX(poolingStrideX)
        , m_PoolingStrideY(poolingStrideY)
        , m_Padding(padding)
        , m_PoolingType(type)
    {}

    constexpr bool operator==(const PoolingInfo& rhs) const
    {
        return m_PoolingSizeX == rhs.m_PoolingSizeX && m_PoolingSizeY == rhs.m_PoolingSizeY &&
               m_Padding == rhs.m_Padding && m_PoolingStrideX == rhs.m_PoolingStrideX &&
               m_PoolingStrideY == rhs.m_PoolingStrideY && m_PoolingType == rhs.m_PoolingType;
    }

    constexpr bool operator!=(const PoolingInfo& rhs) const
    {
        return !(*this == rhs);
    }

    uint32_t m_PoolingSizeX;
    uint32_t m_PoolingSizeY;
    uint32_t m_PoolingStrideX;
    uint32_t m_PoolingStrideY;
    Padding m_Padding;
    PoolingType m_PoolingType;
};

struct ConcatenationInfo
{
    ConcatenationInfo(uint32_t axis, const QuantizationInfo& qInfo)
        : m_Axis(axis)
        , m_OutputQuantizationInfo(qInfo)
    {}

    bool operator==(const ConcatenationInfo& rhs) const
    {
        return m_Axis == rhs.m_Axis && m_OutputQuantizationInfo == rhs.m_OutputQuantizationInfo;
    }

    bool operator!=(const ConcatenationInfo& rhs) const
    {
        return !(*this == rhs);
    }

    uint32_t m_Axis;
    QuantizationInfo m_OutputQuantizationInfo;
};

struct SplitInfo
{
    SplitInfo(uint32_t axis, const std::vector<uint32_t>& sizes)
        : m_Axis(axis)
        , m_Sizes(sizes)
    {}

    bool operator==(const SplitInfo& rhs) const
    {
        return m_Axis == rhs.m_Axis && m_Sizes == rhs.m_Sizes;
    }

    bool operator!=(const SplitInfo& rhs) const
    {
        return !(*this == rhs);
    }

    uint32_t m_Axis;
    std::vector<uint32_t> m_Sizes;
};

struct DepthToSpaceInfo
{
    DepthToSpaceInfo(uint32_t blockSize)
        : m_BlockSize(blockSize)
    {}

    bool operator==(const DepthToSpaceInfo& rhs) const
    {
        return m_BlockSize == rhs.m_BlockSize;
    }

    bool operator!=(const DepthToSpaceInfo& rhs) const
    {
        return !(*this == rhs);
    }

    uint32_t m_BlockSize;
};

using SpaceToDepthInfo = DepthToSpaceInfo;

struct TransposeInfo
{
    TransposeInfo(const std::array<uint32_t, 4>& permutation)
        : m_Permutation(permutation)
    {}

    bool operator==(const TransposeInfo& rhs) const
    {
        return (m_Permutation == rhs.m_Permutation);
    }

    bool operator!=(const TransposeInfo& rhs) const
    {
        return !(*this == rhs);
    }

    std::array<uint32_t, 4> m_Permutation;
};

struct ResizeInfo
{
    ResizeInfo()
        : m_Algo{}
        , m_NewHeight{}
        , m_NewWidth{}
        , m_OutputQuantizationInfo{}
    {}

    ResizeInfo(const ResizeAlgorithm algo,
               const uint32_t newHeight,
               const uint32_t newWidth,
               const QuantizationInfo& qInfo = {})
        : m_Algo(algo)
        , m_NewHeight(newHeight)
        , m_NewWidth(newWidth)
        , m_OutputQuantizationInfo(qInfo)
    {}

    bool operator==(const ResizeInfo& rhs) const
    {
        return (m_Algo == rhs.m_Algo && m_NewHeight == rhs.m_NewHeight && m_NewWidth == rhs.m_NewWidth &&
                m_OutputQuantizationInfo == rhs.m_OutputQuantizationInfo);
    }

    bool operator!=(const ResizeInfo& rhs) const
    {
        return !(*this == rhs);
    }

    ResizeAlgorithm m_Algo;
    uint32_t m_NewHeight;
    uint32_t m_NewWidth;
    QuantizationInfo m_OutputQuantizationInfo;
};

struct EstimateOnlyInfo
{
    EstimateOnlyInfo(const std::vector<TensorInfo>& outputInfos)
        : m_OutputInfos(outputInfos)
    {}

    bool operator==(const EstimateOnlyInfo& rhs) const
    {
        return m_OutputInfos == rhs.m_OutputInfos;
    }

    bool operator!=(const EstimateOnlyInfo& rhs) const
    {
        return !(*this == rhs);
    }

    std::vector<TensorInfo> m_OutputInfos;
};

struct BufferInfo
{
public:
    BufferInfo()
        : m_Size(0)
    {}

    constexpr BufferInfo(uint32_t size)
        : m_Size(size)
    {}

    bool operator==(const BufferInfo& rhs) const
    {
        return m_Size == rhs.m_Size;
    }

    /// Size (in bytes) of this buffer.
    uint32_t m_Size;
};

// Please see the example network below:
// (X), (Y) and (Z) are unsupported operations
// (Add) is an Addition operation and (O) is the output node.
//
//      (X)   (Y)
//      /\     |
//   0 /  \ 1  | 0  <- Indices of the output slots
//    /    \   |
//   /   ---------
//  (Z)  | (Add) |
//       |   |   |  <- The part of the network that we support
//       |  (O)  |
//       ---------
//
// In this example, the Addition operation takes input buffers from the two unsupported (X) and (Y) operations.
// Therefore the call to GetInputBufferInfos() will return two InputBufferInfos:
// - The first one is the second output of operation (X) and will have m_SourceOperationOutputIndex = 1
// - The second one is the first (and only) output of operation(Y) and will have m_SourceOperationOutputIndex = 0
struct InputBufferInfo : BufferInfo
{
    InputBufferInfo()
        : m_SourceOperationId(0)
        , m_SourceOperationOutputIndex(0)
    {}

    constexpr InputBufferInfo(uint32_t size, uint32_t operationId, uint32_t sourceOperationOutputIndex)
        : BufferInfo(size)
        , m_SourceOperationId(operationId)
        , m_SourceOperationOutputIndex(sourceOperationOutputIndex)
    {}

    bool operator==(const InputBufferInfo& rhs) const
    {
        return BufferInfo::operator==(rhs) && m_SourceOperationId == rhs.m_SourceOperationId &&
               m_SourceOperationOutputIndex == rhs.m_SourceOperationOutputIndex;
    }

    uint32_t m_SourceOperationId;             ///< Identifies which operation produced this buffer.
    uint32_t m_SourceOperationOutputIndex;    ///< Identifies which output of the source operation produced this buffer
};

// Please see the example network above.
struct OutputBufferInfo : BufferInfo
{
    OutputBufferInfo()
        : m_SourceOperationId(0)
        , m_SourceOperationOutputIndex(0)
    {}

    constexpr OutputBufferInfo(uint32_t size, uint32_t operationId, uint32_t sourceOperationOutputIndex)
        : BufferInfo(size)
        , m_SourceOperationId(operationId)
        , m_SourceOperationOutputIndex(sourceOperationOutputIndex)
    {}

    bool operator==(const OutputBufferInfo& rhs) const
    {
        return BufferInfo::operator==(rhs) && m_SourceOperationId == rhs.m_SourceOperationId &&
               m_SourceOperationOutputIndex == rhs.m_SourceOperationOutputIndex;
    }

    uint32_t m_SourceOperationId;             ///< Identifies which operation produced this buffer.
    uint32_t m_SourceOperationOutputIndex;    ///< Identifies which output of the source operation produced this buffer
};

inline std::ostream& operator<<(std::ostream& os, const BufferInfo& value)
{
    std::ios::fmtflags f(os.flags());
    os << std::hex << "{ Size = 0x" << value.m_Size << " }";
    os.flags(f);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const InputBufferInfo& value)
{
    std::ios::fmtflags f(os.flags());
    os << std::hex << "{ Size = 0x" << value.m_Size << ", OpId = " << value.m_SourceOperationId
       << ", Index = " << value.m_SourceOperationOutputIndex << " }";
    os.flags(f);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const OutputBufferInfo& value)
{
    std::ios::fmtflags f(os.flags());
    os << std::hex << "{ Size = 0x" << value.m_Size << ", OpId = " << value.m_SourceOperationId
       << ", Index = " << value.m_SourceOperationOutputIndex << " }";
    os.flags(f);
    return os;
}

/// The result of compiling a network using Compile(...).
class CompiledNetwork
{
public:
    virtual ~CompiledNetwork()
    {}

    virtual const std::set<uint32_t>& GetOperationIds() const = 0;

    /// Details of each input buffer.
    /// The array is in the same order as the user provided inputs via AddInput()
    virtual const std::vector<InputBufferInfo>& GetInputBufferInfos() const = 0;
    /// Details of each output buffer.
    /// The array is in the same order as the user provided outputs via AddOutput()
    virtual const std::vector<OutputBufferInfo>& GetOutputBufferInfos() const = 0;

    /// Serializes this object to a binary data stream, for consumption by the Driver Library
    /// (see ethosn::driver_library::Network constructor).
    /// If writing to the given stream fails, no additional error reporting is performed. The caller must check the
    /// state of the stream themselves after this method returns.
    virtual void Serialize(std::ostream&) const = 0;
};

/// Exception type thrown for unexpected internal errors.
class InternalErrorException : public std::exception
{
public:
    InternalErrorException(const char* reason)
        : m_Reason(reason)
    {}

    virtual const char* what() const noexcept override
    {
        return m_Reason.c_str();
    }

private:
    std::string m_Reason;
};

/// Exception type thrown when an operation is added to a Network which is not supported.
class NotSupportedException : public std::exception
{
public:
    NotSupportedException(const char* reason)
        : m_Reason(reason)
    {}

    virtual const char* what() const noexcept override
    {
        return m_Reason.c_str();
    }

private:
    std::string m_Reason;
};

/// Exception type thrown when data passed to the Support Library is of the wrong version.
class VersionMismatchException : public std::exception
{
public:
    VersionMismatchException(const char* reason)
        : m_Reason(reason)
    {}

    VersionMismatchException(std::string reason)
        : m_Reason(reason)
    {}

    virtual const char* what() const noexcept override
    {
        return m_Reason.c_str();
    }

private:
    std::string m_Reason;
};

/// The return value of adding a new operation to the Network, for operations which have a single output.
template <typename TensorType>
struct TensorAndId
{
    // The tensor representing the single output of the new operation.
    std::shared_ptr<TensorType> tensor;
    // The unique ID for the new operation.
    uint32_t operationId;
};

/// The return value of adding a new operation to the Network, for operations which have multiple outputs.
struct TensorsAndId
{
    // The tensors representing the outputs of the new operation.
    std::vector<std::shared_ptr<Operand>> tensors;
    // The unique ID for the new operation.
    uint32_t operationId;
};

// Get a version string for this library
const Version GetLibraryVersion();

/// Call the Compiler to process the network and get the outputs that need to be passed through Arm NN to the driver lib
///
/// @throws InternalErrorException for unexpected internal compilation errors.
std::vector<std::unique_ptr<CompiledNetwork>> Compile(const Network& network, const CompilationOptions& options);

// Call the Compiler to estimate the performance of the network
NetworkPerformanceData EstimatePerformance(const Network& network,
                                           const CompilationOptions& compilationOptions,
                                           const EstimationOptions& estimationOptions = {});

enum class EthosNVariant
{
    ETHOS_N77,    ///< Not supported and will error at runtime if used. Kept for backwards-compatibility.
    ETHOS_N57,    ///< Not supported and will error at runtime if used. Kept for backwards-compatibility.
    ETHOS_N37,    ///< Not supported and will error at runtime if used. Kept for backwards-compatibility.
    ETHOS_N78_1TOPS_2PLE_RATIO,
    ETHOS_N78_1TOPS_4PLE_RATIO,
    ETHOS_N78_2TOPS_2PLE_RATIO,
    ETHOS_N78_2TOPS_4PLE_RATIO,
    ETHOS_N78_4TOPS_2PLE_RATIO,
    ETHOS_N78_4TOPS_4PLE_RATIO,
    ETHOS_N78_8TOPS_2PLE_RATIO
};

const char* EthosNVariantAsString(EthosNVariant ethosnType);
EthosNVariant EthosNVariantFromString(const char* ethosnType);

/// Gets firmware and hardware capabilities from this library in the absence of the real device.
/// Optionally SRAM size can be overridden.
/// Will throw a NotSupportedException if an unsupported variant is requested.
std::vector<char> GetFwAndHwCapabilities(EthosNVariant variant, uint32_t sramSizeBytes = 0);

/// Creates a new Network
///
/// @param caps: An opaque block of data containing the capabilities of the hardware and the firmware.
///        This should be obtained from the Driver Library's GetFirmwareAndHardwareCapabilities() API.
///        It can also be obtained from the Support Library's GetFwAndHwCapabilities() API for offline compilation.
///        Note that the size and format of this data may vary depending on the version of the hardware/firmware.
std::shared_ptr<Network> CreateNetwork(const std::vector<char>& caps);

// Create a new Network for performance estimation
std::shared_ptr<Network> CreateEstimationNetwork(const std::vector<char>& caps);

// Add Input to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand> AddInput(const std::shared_ptr<Network>& network, const TensorInfo& info);

// Add Output to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Output> AddOutput(const std::shared_ptr<Network>& network,
                              Operand& operand,
                              const DataFormat outputFormat = DataFormat::NHWC);

// Add Constant to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Constant> AddConstant(const std::shared_ptr<Network>& network, const TensorInfo& info, const void* data);

// Get the Operand produced by a Constant
std::shared_ptr<Operand> GetOperand(const std::shared_ptr<Constant>& constant);

/// Add Convolution to a Network. The returned shared_ptr ref-counts the network.
/// The bias parameter should be a 1D tensor of shape [ 1, 1, 1, O ], where O is the number of output channels for the
/// convolution. It should have a data type of INT32_QUANTIZED and have quantisation parameters exactly equal to:
///    - Scale = input scale * weight scale
///    - Zero point = 0
/// Although it would be possible for the support library to perform any required requantisation to meet these
/// parameters, requiring the user to do it reduces the risk of precision issues leading to a slightly different
/// result than expected.
TensorAndId<Operand> AddConvolution(const std::shared_ptr<Network>& network,
                                    Operand& input,
                                    Constant& bias,
                                    Constant& weights,
                                    const ConvolutionInfo& convInfo);

// Add Depthwise Convolution to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand> AddDepthwiseConvolution(const std::shared_ptr<Network>& network,
                                             Operand& input,
                                             Constant& bias,
                                             Constant& weights,
                                             const ConvolutionInfo& convInfo);

// Add Transpose Convolution to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand> AddTransposeConvolution(const std::shared_ptr<Network>& network,
                                             Operand& input,
                                             Constant& bias,
                                             Constant& weights,
                                             const ConvolutionInfo& convInfo);

// Add Concatenation to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand> AddConcatenation(const std::shared_ptr<Network>& network,
                                      const std::vector<Operand*>& layers,
                                      const ConcatenationInfo& concatInfo);

TensorsAndId AddSplit(const std::shared_ptr<Network>& network, Operand& input, const SplitInfo& splitInfo);

// Add Addition to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand> AddAddition(const std::shared_ptr<Network>& network,
                                 Operand& layer1,
                                 Operand& layer2,
                                 const QuantizationInfo& outputQuantizationInfo);

// Add FullyConnected to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand> AddFullyConnected(const std::shared_ptr<Network>& network,
                                       Operand& input,
                                       Constant& bias,
                                       Constant& weights,
                                       FullyConnectedInfo fullyConnectedInfo);

// Add a ReinterpretQuantization to a Network.
// This operation doesn't correspond to an actual network's operation but it useful
// when the user needs to change the quantization parameters at a late stage of
// a network's construction.
TensorAndId<Operand> AddReinterpretQuantization(const std::shared_ptr<Network>& network,
                                                Operand& input,
                                                const ReinterpretQuantizationInfo& reinterpretQuantizationInfo);

// Add Relu to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand> AddRelu(const std::shared_ptr<Network>& network, Operand& input, const ReluInfo& reluInfo);

// Add LeakyRelu to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand>
    AddLeakyRelu(const std::shared_ptr<Network>& network, Operand& input, const LeakyReluInfo& leakyReluInfo);

// Add Requantize to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand>
    AddRequantize(const std::shared_ptr<Network>& network, Operand& input, const RequantizeInfo& requantizeInfo);

// Add Softmax to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand> AddSoftmax(const std::shared_ptr<Network>& network, Operand& input);

// Add Sigmoid to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand> AddSigmoid(const std::shared_ptr<Network>& network, Operand& input);

// Add Mean to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand> AddMeanXy(const std::shared_ptr<Network>& network, Operand& input);

// Add Pooling to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand>
    AddPooling(const std::shared_ptr<Network>& network, Operand& input, const PoolingInfo& poolingInfo);

// Add Reshape to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand>
    AddReshape(const std::shared_ptr<Network>& network, Operand& input, const TensorShape& newDimensions);

// Add DepthToSpace to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand>
    AddDepthToSpace(const std::shared_ptr<Network>& network, Operand& input, const DepthToSpaceInfo& depthToSpaceInfo);

// Add SpaceToDepth to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand>
    AddSpaceToDepth(const std::shared_ptr<Network>& network, Operand& input, const SpaceToDepthInfo& spaceToDepthInfo);

// Add Transpose to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand>
    AddTranspose(const std::shared_ptr<Network>& network, Operand& input, const TransposeInfo& transposeInfo);

// Add Resize to a Network. The returned shared_ptr ref-counts the network.
TensorAndId<Operand> AddResize(const std::shared_ptr<Network>& network, Operand& input, const ResizeInfo& resizeInfo);

// Add EstimateOnly to a Network. The returned shared_ptr ref-counts the network.
TensorsAndId AddEstimateOnly(const std::shared_ptr<Network>& network,
                             const std::vector<Operand*>& inputs,
                             const EstimateOnlyInfo& estimateOnlyInfo);

/// Gets the TensorInfo of the given Operand.
TensorInfo GetTensorInfo(const std::shared_ptr<Operand>& operand);

}    // namespace support_library
}    // namespace ethosn
