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

struct MemoryStripesInfo
{
    MemoryStripeInfo m_Input;
    MemoryStripeInfo m_Output;
    MemoryStripeInfo m_Weight;
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
    Lifetime m_Lifetime = Lifetime::Cascade;

    bool operator<(const MceAndPleInfo& rhs) const;
};

// A representation of plans without an identity PLE operation
// this is to enable fusing with subsequent ple operations
struct MceOnlyInfo
{
    MceStripesInfo m_MceCompute;
    MemoryStripesInfo m_Memory;
    Lifetime m_Lifetime = Lifetime::Cascade;

    bool operator<(const MceOnlyInfo& rhs) const;
};

// A representation of plans without an identity MCE operation
// this is to enable fusing with preceding mce operations
struct PleOnlyInfo
{
    PleStripesInfo m_PleCompute;
    MemoryStripesInfo m_Memory;
    Lifetime m_Lifetime = Lifetime::Cascade;

    bool operator<(const PleOnlyInfo& rhs) const;
};

// A representation of plans that only use DMA and thus only
// have information about memory
struct DmaOnlyInfo
{
    MemoryStripeInfo m_Input;
    MemoryStripeInfo m_Output;
    Lifetime m_Lifetime = Lifetime::Cascade;

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
                       NumStripesType& numPleInputMemoryStripes,
                       const TensorShape& tensorShape,
                       const TensorShape& pleInputMemoryShape,
                       const QuantizationInfo& quantInfo,
                       Lifetime lifetime,
                       TraversalOrder order,
                       Location location);

std::pair<Buffer*, Op*> AddPleToOpGraph(OwnedOpGraph& opGraph,
                                        Lifetime lifetime,
                                        TraversalOrder order,
                                        const TensorShape& memoryOutputShape,
                                        impl::NumMemoryStripes& numMemoryStripes,
                                        std::unique_ptr<Op> pleOp,
                                        const TensorShape& outputShape,
                                        const QuantizationInfo& outputQuantInfo,
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
                    const Stride& stride,
                    uint32_t upscaleFactor,
                    command_stream::MceOperation op,
                    utils::ShapeMultiplier shapeMult,
                    const HardwareCapabilities& caps);

    void GenerateStripes(const ethosn::command_stream::BlockConfig blockConfig,
                         CascadeType cascadeType,
                         StripeInfos* outStripeInfos) const;

    void CreateNumStripes(CascadeType cascadeType,
                          uint32_t kernelHeight,
                          NumStripes& numStripesInput,
                          NumStripes& numStripesOutput,
                          NumStripes& numStripesWeights,
                          NumStripes& numStripesPleInput) const;

    TensorShape m_MceInputTensorShape;
    TensorShape m_MceOutputTensorShape;
    TensorShape m_PleOutputTensorShape;
    uint32_t m_KernelHeight;
    uint32_t m_KernelWidth;
    Stride m_Stride;
    uint32_t m_UpscaleFactor;
    command_stream::MceOperation m_Operation;
    utils::ShapeMultiplier m_ShapeMultiplier;

    const HardwareCapabilities& m_Capabilities;
};

};    // namespace impl
}    // namespace support_library
}    // namespace ethosn
