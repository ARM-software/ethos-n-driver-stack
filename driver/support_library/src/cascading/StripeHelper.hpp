//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "Plan.hpp"
#include "Utils.hpp"

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
    InputMemoryStripeInfo(const MemoryStripeInfo& m,
                          const command_stream::cascading::PackedBoundaryThickness& t,
                          uint32_t l)
        : MemoryStripeInfo(m)
        , m_PackedBoundaryThickness(t)
        , m_NumLoads(l)
    {}

    command_stream::cascading::PackedBoundaryThickness m_PackedBoundaryThickness;
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

Buffer* AddPleInBuffer(OwnedOpGraph& opGraph,
                       NumStripesType numPleInputMemoryStripes,
                       const TensorShape& tensorShape,
                       const TensorShape& pleInputMemoryShape,
                       const QuantizationInfo& quantInfo,
                       DataType dataType,
                       Location location);

std::pair<Buffer*, Op*> AddPleToOpGraph(OwnedOpGraph& opGraph,
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
                    const Stride& stride,
                    uint32_t upscaleFactor,
                    command_stream::MceOperation op,
                    command_stream::PleOperation pleOp,
                    utils::ShapeMultiplier mceShapeMult,
                    utils::ShapeMultiplier pleShapeMult,
                    const HardwareCapabilities& caps,
                    StripeConfig stripeConfig);

    StripeInfos GenerateStripes(CascadeType cascadeType) const;

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
    Stride m_Stride;
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
                         StripeInfos& outStripeInfos) const;
};

};    // namespace impl
}    // namespace support_library
}    // namespace ethosn
