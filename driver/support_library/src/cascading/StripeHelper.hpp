//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "Plan.hpp"
#include "Utils.hpp"

#include <vector>

namespace ethosn
{
namespace support_library
{
namespace impl
{

/// Settings to specify which stripe splitting strategies and block sizes can be used.
struct StripeConfig
{
    struct
    {
        bool beginning = true;
        bool middle    = true;
        bool end       = true;
        bool lonely    = true;
    } planTypes;

    /// Set of flags to specify which dimensions can be split.
    /// Any dimensions not mentioned in a name are implicitly not split.
    struct
    {
        bool mceAndPleOutputHeight            = true;
        bool mceOutputHeightOnly              = true;
        bool widthOnly                        = true;
        bool widthHeight                      = true;
        bool widthHeightOutputDepth           = true;
        bool widthHeightOutputDepthInputDepth = true;
        bool outputDepthInputDepth            = true;
        bool mceAndPleOutputDepth             = true;
        bool mceOutputDepthOnly               = true;
        bool inputDepthOnly                   = true;
        bool none                             = true;
    } splits;

    struct
    {
        uint32_t min = 1;
        uint32_t max = std::numeric_limits<uint32_t>::max();
    } blockWidthMultiplier;
    struct
    {
        uint32_t min = 1;
        uint32_t max = std::numeric_limits<uint32_t>::max();
    } blockHeightMultiplier;
    struct
    {
        uint32_t min = 1;
        uint32_t max = std::numeric_limits<uint32_t>::max();
    } ifmDepthMultiplier;
    struct
    {
        uint32_t min = 1;
        uint32_t max = std::numeric_limits<uint32_t>::max();
    } ofmDepthMultiplier;

    std::vector<ethosn::command_stream::BlockConfig> blockConfigs = { { 16u, 16u },
                                                                      { 16u, 8u },
                                                                      { 8u, 16u },
                                                                      { 8u, 8u },
                                                                      {
                                                                          32u,
                                                                          8u,
                                                                      },
                                                                      { 8u, 32u } };

    /// Disables all splitting strategies and block configs.
    /// After calling this you will most likely want to re-enable some, otherwise no
    /// plans will be generated!
    void DisableAll()
    {
        DisableAllSplits();
        blockConfigs.clear();
    }

    /// Disables all splitting strategies.
    /// After calling this you will most likely want to re-enable some, otherwise no
    /// plans will be generated!
    void DisableAllSplits()
    {
        splits.mceAndPleOutputHeight            = false;
        splits.mceOutputHeightOnly              = false;
        splits.widthOnly                        = false;
        splits.widthHeight                      = false;
        splits.widthHeightOutputDepth           = false;
        splits.widthHeightOutputDepthInputDepth = false;
        splits.outputDepthInputDepth            = false;
        splits.mceAndPleOutputDepth             = false;
        splits.mceOutputDepthOnly               = false;
        splits.inputDepthOnly                   = false;
        splits.none                             = false;
    }

    /// Helper functions to disable all splitting strategies which split tensors
    /// in certain dimensions.
    /// @{
    void DisableSplitHeight()
    {
        splits.mceAndPleOutputHeight            = false;
        splits.mceOutputHeightOnly              = false;
        splits.widthHeight                      = false;
        splits.widthHeightOutputDepth           = false;
        splits.widthHeightOutputDepthInputDepth = false;
    }
    void DisableSplitWidth()
    {
        splits.widthOnly                        = false;
        splits.widthHeight                      = false;
        splits.widthHeightOutputDepth           = false;
        splits.widthHeightOutputDepthInputDepth = false;
    }
    void DisableSplitInputDepth()
    {
        splits.widthHeightOutputDepthInputDepth = false;
        splits.outputDepthInputDepth            = false;
        splits.inputDepthOnly                   = false;
    }
    void DisableSplitOutputDepth()
    {
        splits.widthHeightOutputDepth           = false;
        splits.widthHeightOutputDepthInputDepth = false;
        splits.outputDepthInputDepth            = false;
        splits.mceAndPleOutputDepth             = false;
        splits.mceOutputDepthOnly               = false;
    }
    /// @}
};

/// Gets a StripeConfig with everything enabled, unless there is a debug config file provided
/// which overrides this for the identifier given.
StripeConfig GetDefaultStripeConfig(const CompilationOptions& compilationOptions, const char* identifier);

using NumStripesType = uint32_t;
struct NumStripes
{
    NumStripesType m_Min;
    NumStripesType m_Max;
    bool operator<(const NumStripes& rhs) const;
};

struct MceStripesInfo
{
    TensorShape m_Input;
    TensorShape m_Output;
    TensorShape m_Weight;
    command_stream::BlockConfig m_BlockConfig = { 8U, 8U };

    bool operator<(const MceStripesInfo& rhs) const;
};

struct PleStripesInfo
{
    TensorShape m_Input;
    TensorShape m_Output;
    command_stream::BlockConfig m_BlockConfig = { 8U, 8U };
    bool operator<(const PleStripesInfo& rhs) const;
};

struct MemoryStripeInfo
{
    NumStripes m_Range;
    TensorShape m_Shape;
    bool operator<(const MemoryStripeInfo& rhs) const;
};

struct InputMemoryStripeInfo : public MemoryStripeInfo
{
    InputMemoryStripeInfo() = default;
    InputMemoryStripeInfo(const MemoryStripeInfo& m, const PackedBoundaryThickness& t, uint32_t l)
        : MemoryStripeInfo(m)
        , m_PackedBoundaryThickness(t)
        , m_NumLoads(l)
    {}

    PackedBoundaryThickness m_PackedBoundaryThickness;
    uint32_t m_NumLoads;
    bool operator<(const InputMemoryStripeInfo& rhs) const;
};

struct WeightMemoryStripeInfo : public MemoryStripeInfo
{
    WeightMemoryStripeInfo() = default;
    WeightMemoryStripeInfo(const MemoryStripeInfo& m, uint32_t l)
        : MemoryStripeInfo(m)
        , m_NumLoads(l)
    {}

    uint32_t m_NumLoads;
    bool operator<(const WeightMemoryStripeInfo& rhs) const;
};

struct MemoryStripesInfo
{
    InputMemoryStripeInfo m_Input;
    MemoryStripeInfo m_Output;
    WeightMemoryStripeInfo m_Weight;
    MemoryStripeInfo m_PleInput;
    bool operator<(const MemoryStripesInfo& rhs) const;
};

struct NumMemoryStripes
{
    NumStripesType m_Input;
    NumStripesType m_Output;
    NumStripesType m_Weight;
    NumStripesType m_PleInput;
    bool operator<(const NumMemoryStripes& rhs) const;
};

// The following structs are intermediate representations of plans
// describing the size of compute stripes and the size and number of memory stripes

// A representation of plans with both mce and ple operations
// this is to enable plans which need identity mce or identity ple operations
struct MceAndPleInfo
{
    MceStripesInfo m_MceCompute;
    PleStripesInfo m_PleCompute;
    MemoryStripesInfo m_Memory;

    bool operator<(const MceAndPleInfo& rhs) const;
};

// A representation of plans without an identity PLE operation
// this is to enable fusing with subsequent ple operations
struct MceOnlyInfo
{
    MceStripesInfo m_MceCompute;
    MemoryStripesInfo m_Memory;

    bool operator<(const MceOnlyInfo& rhs) const;
};

// A representation of plans without an identity MCE operation
// this is to enable fusing with preceding mce operations
struct PleOnlyInfo
{
    PleStripesInfo m_PleCompute;
    MemoryStripesInfo m_Memory;

    bool operator<(const PleOnlyInfo& rhs) const;
};

// A representation of plans that only use DMA and thus only
// have information about memory
struct DmaOnlyInfo
{
    MemoryStripeInfo m_Input;
    MemoryStripeInfo m_Output;

    bool operator<(const DmaOnlyInfo& rhs) const;
};

struct StripeInfos
{
    std::set<MceAndPleInfo> m_MceAndPleInfos;
    std::set<MceOnlyInfo> m_MceOnlyInfos;
    std::set<PleOnlyInfo> m_PleOnlyInfos;
    std::set<DmaOnlyInfo> m_DmaOnlyInfos;
};

uint32_t GetWeightStripeDepth(const TensorInfo& weightInfo, const TensorShape& weightStripeShape, const Stride& stride);

struct ConvData
{
    TensorInfo weightInfo;
    std::shared_ptr<const std::vector<uint8_t>> weightData;
    TensorInfo biasInfo;
    std::vector<int32_t> biasData;
};

Buffer* AddPleInputSramBuffer(OwnedOpGraph& opGraph,
                              NumStripesType numPleInputMemoryStripes,
                              const TensorShape& tensorShape,
                              const TensorShape& pleInputMemoryShape,
                              const QuantizationInfo& quantInfo,
                              DataType dataType);

std::pair<SramBuffer*, Op*> AddPleToOpGraph(OwnedOpGraph& opGraph,
                                            const TensorShape& memoryOutputShape,
                                            impl::NumMemoryStripes& numMemoryStripes,
                                            std::unique_ptr<Op> pleOp,
                                            const TensorShape& outputShape,
                                            const QuantizationInfo& outputQuantInfo,
                                            DataType outputDataType,
                                            const std::set<uint32_t>& sourceOperationIds);

/// Generates a stripe shape given an encoding and an input tensor
/// Tries to create a stripe with the stripe shape in the encoding, if the dimension is 0 then it uses the full length of that dimension.
TensorShape CreateStripe(TensorShape input, TensorShape inputEncoding, uint32_t channelRounding);

/// Allows easy looping over a set of possible stripe shapes based on a tensor shape, with a few customisable options.
/// Supports iterating so can be used in a range-based for loop, e.g. for (int x : StripeShapeLoop(...)) { ... }
/// The stripe shapes returned are logarithmically spaced, to avoid producing too many options (e.g. 1, 2, 4, 8, ...).
/// See StripeHelperTests.cpp for some examples.
struct StripeShapeLoop
{
    /// Creates a StripeShapeLoop that includes a final stripe shape which is >= the tensor size.
    static StripeShapeLoop Inclusive(uint32_t tensorSize,
                                     uint32_t baseSize,
                                     uint32_t minMultiplier = 1,
                                     uint32_t maxMultiplier = std::numeric_limits<uint32_t>::max())
    {
        maxMultiplier = std::min(maxMultiplier, utils::DivRoundUp(tensorSize, baseSize));
        return StripeShapeLoop(baseSize, minMultiplier, maxMultiplier);
    }

    /// Creates a StripeShapeLoop which yields stripe shapes which are always < the tensor size.
    /// Note that this may result in an empty range (no valid stripe shapes)
    static StripeShapeLoop Exclusive(uint32_t tensorSize,
                                     uint32_t baseSize,
                                     uint32_t minMultiplier = 1,
                                     uint32_t maxMultiplier = std::numeric_limits<uint32_t>::max())
    {
        maxMultiplier = std::min(maxMultiplier, utils::DivRoundUp(tensorSize, baseSize));
        // Reduce maxMultiplier so that it is the largest power of 2 that doesn't include the full stripe
        maxMultiplier = utils::RoundDownToPow2(maxMultiplier);
        if (maxMultiplier * baseSize >= tensorSize)
        {
            maxMultiplier /= 2U;
        }
        return StripeShapeLoop(baseSize, minMultiplier, maxMultiplier);
    }

    struct Iterator
    {
        Iterator(uint32_t multiplierValue, const StripeShapeLoop& parent)
            : m_MultiplierValue(multiplierValue)
            , m_Parent(parent)
        {}

        void operator++()
        {
            if (m_MultiplierValue == m_Parent.m_UpperMultiplier)
            {
                // This was the last value, so incrementing takes us to the end iterator (one-past-the-end)
                ++m_MultiplierValue;
            }
            else
            {
                // Iterate with *= 2 to reduce the number of stripe shapes produced (for compiler performance)
                // Note that the m_UpperMultiplier may not be a power of two. There is no point having a stripe
                // shape far larger than the tensor.
                m_MultiplierValue = std::min(m_MultiplierValue * 2, m_Parent.m_UpperMultiplier);
            }
        }

        bool operator!=(const Iterator& rhs) const
        {
            return m_MultiplierValue != rhs.m_MultiplierValue;
        }

        /// Gets the value of the iterator (i.e. the current stripe shape).
        uint32_t operator*() const
        {
            return m_MultiplierValue * m_Parent.m_BaseSize;
        }

        uint32_t m_MultiplierValue;
        const StripeShapeLoop& m_Parent;
    };

    Iterator begin() const
    {
        return Iterator(m_LowerMultiplier, *this);
    }
    Iterator end() const
    {
        return Iterator(m_UpperMultiplier + 1, *this);    // Plus 1 because m_UpperMultiplier is inclusive.
    }

private:
    /// Note that the lower and upper multipliers here are inclusive.
    StripeShapeLoop(uint32_t baseSize, uint32_t lowerMultiplier, uint32_t upperMultiplier)
        : m_BaseSize(baseSize)
        , m_LowerMultiplier(lowerMultiplier)
        , m_UpperMultiplier(upperMultiplier)
    {
        if (m_LowerMultiplier > m_UpperMultiplier)
        {
            // This is an empty-range, so we need to make sure begin() == end().
            // Because of the way we handle the end iterator with the +1, we override the values to some which meet
            // this criteria.
            m_LowerMultiplier = 1;
            m_UpperMultiplier = 0;
        }
    }

    uint32_t m_BaseSize;
    uint32_t m_LowerMultiplier;
    uint32_t m_UpperMultiplier;
};

enum class PlanPriority
{
    Low,
    High,
};

// Class used to generate stripes for the start of cascades. i.e. beginning and lonely cascades
// Middle and end cascades don't need this as they their plan generation is limited by the inputs.
class StripeGenerator
{
public:
    StripeGenerator(const TensorShape& mceInput,
                    const TensorShape& mceOutput,
                    const TensorShape& pleOutput,
                    uint32_t kernelHeight,
                    uint32_t kernelWidth,
                    uint32_t padTop,
                    uint32_t padLeft,
                    uint32_t upscaleFactor,
                    command_stream::MceOperation op,
                    command_stream::PleOperation pleOp,
                    const utils::ShapeMultiplier& mceShapeMult,
                    const utils::ShapeMultiplier& pleShapeMult,
                    const HardwareCapabilities& caps,
                    const StripeConfig& stripeConfig);

    // This method is intended to be called first with PlanPriority::High and after and only if needed
    // with PlanPriority::Low.
    StripeInfos GenerateStripes(CascadeType cascadeType, utils::Optional<PlanPriority> priorityFilter) const;

    void CreateNumStripes(CascadeType cascadeType,
                          bool requiresBoundaryData,
                          NumStripes& numStripesInput,
                          NumStripes& numStripesOutput,
                          NumStripes& numStripesWeights,
                          NumStripes& numStripesPleInput) const;

    StripeConfig ApplyPleKernelSplitRestrictions(CascadeType cascadeType) const;

    TensorShape m_MceInputTensorShape;
    TensorShape m_MceOutputTensorShape;
    TensorShape m_PleOutputTensorShape;
    uint32_t m_KernelHeight;
    uint32_t m_KernelWidth;
    uint32_t m_PadTop;
    uint32_t m_PadLeft;
    uint32_t m_UpscaleFactor;
    command_stream::MceOperation m_Operation;
    command_stream::PleOperation m_KernelOperation;
    utils::ShapeMultiplier m_MceShapeMultiplier;
    utils::ShapeMultiplier m_PleShapeMultiplier;

    const HardwareCapabilities& m_Capabilities;

    StripeConfig m_StripeConfig;

private:
    void GenerateStripes(const ethosn::command_stream::BlockConfig blockConfig,
                         CascadeType cascadeType,
                         utils::Optional<PlanPriority> priorityFilter,
                         StripeInfos& outStripeInfos) const;
};

/// Checks if a given SRAM buffer could be DMA'd to or from a DRAM buffer of the given format and shape,
/// at the given offset.
/// For example, this checks that an SRAM buffer with a stripe shape that splits depth cannot be DMA'd
/// to an NHWC DRAM buffer (as the firmware does not support this).
/// This accounts for possible reshaping and subtensors/offsets.
///@{
bool IsSramBufferCompatibleWithDramBuffer(const SramBuffer& sramBuffer,
                                          const DramBuffer& dramBuffer,
                                          const TensorShape& dramOffset);
bool IsSramBufferCompatibleWithDramBuffer(const SramBuffer& sramBuffer,
                                          CascadingBufferFormat dramFormat,
                                          const TensorShape& dramTensorShape,
                                          const TensorShape& dramOffset);
bool IsSramBufferCompatibleWithDramBuffer(const TensorShape& sramTensorShape,
                                          const TensorShape& stripeShape,
                                          bool forbidFcafWide,
                                          const PackedBoundaryThickness& packedBoundaryThickness,
                                          CascadingBufferFormat dramFormat,
                                          const TensorShape& dramTensorShape,
                                          const TensorShape& dramOffset);
/// @}

/// Returns the most efficient DRAM buffer format to use, that is compatible with being copied to/from
/// the given set of SRAM buffers. Assumes that the full tensor is going to be copied (i.e. no subtensors)
/// and no reshaping.
CascadingBufferFormat GetBestDramBufferFormat(const std::vector<const SramBuffer*>& sramBuffers,
                                              const CompilationOptions& compilationOptions);

/// Creates an SRAM buffer for use in a glue (or similar) which DMAs stuff into and out of SRAM.
/// The stripe shape is chosen (somewhat) optimally.
/// The stripe shape is chosen so that it is compatible with the given set of DRAM buffer formats,
/// so that it can be DMA'd into and out of SRAM to those formats. For example, if you request that
/// the buffer is compatible with FCAF, the stripe shape will be a multiple of the FCAF cell size.
std::unique_ptr<SramBuffer>
    MakeGlueIntermediateSramBuffer(const TensorShape& shape,
                                   const QuantizationInfo& quantInfo,
                                   DataType dataType,
                                   const std::vector<CascadingBufferFormat>& compatibleDramBufferFormats,
                                   const HardwareCapabilities& caps,
                                   uint32_t minWidthMultiplier  = 1,
                                   uint32_t maxWidthMultiplier  = std::numeric_limits<uint32_t>::max(),
                                   uint32_t minHeightMultiplier = 1,
                                   uint32_t maxHeightMultiplier = std::numeric_limits<uint32_t>::max(),
                                   uint32_t minDepthMultiplier  = 1,
                                   uint32_t maxDepthMultiplier  = std::numeric_limits<uint32_t>::max());

};    // namespace impl
}    // namespace support_library
}    // namespace ethosn
