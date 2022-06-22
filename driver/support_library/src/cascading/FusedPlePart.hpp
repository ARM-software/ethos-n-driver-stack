//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "StripeHelper.hpp"
#include "WeightEncoderCache.hpp"

namespace ethosn
{
namespace support_library
{

class FusedPlePart : public BasePart
{
public:
    FusedPlePart(PartId id,
                 const TensorShape& inputTensorShape,
                 const TensorShape& outputTensorShape,
                 const QuantizationInfo& inputQuantizationInfo,
                 const QuantizationInfo& outputQuantizationInfo,
                 command_stream::PleOperation op,
                 utils::ShapeMultiplier shapeMultiplier,
                 const EstimationOptions& estOpt,
                 const CompilationOptions& compOpt,
                 const HardwareCapabilities& capabilities,
                 std::set<uint32_t> correspondingOperationIds,
                 command_stream::DataType m_InputDataType,
                 command_stream::DataType m_OutputDataType);

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;

    utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const override;
    bool CanDoubleBufferWeights() const override;

    DotAttributes GetDotAttributes(DetailLevel detail) const override;

private:
    Plans GenerateContinueSectionPlans(ethosn::command_stream::BlockConfig blockConfig,
                                       Buffer* prevBuffer,
                                       uint32_t numWeightStripes,
                                       CascadeType cascadeType) const;

    Plans GetLonelyPlans(uint32_t numWeightStripes) const;

    Plans GetBeginningPlans(uint32_t numWeightStripes) const;

    Buffer* AddIdentityWeights(OwnedOpGraph& opGraph,
                               const impl::MceStripesInfo& mceComputeInfo,
                               const impl::NumStripesType& numMemoryWeightStripes,
                               const TensorShape& memoryWeightStripe,
                               const impl::ConvData& convData,
                               WeightEncoderCache& weightEncoderCache) const;

    void CreateFuseOnlyPlans(const impl::PleOnlyInfo& info, Plans& plans) const;

    void CreateIdentityMceAndFusedPlePlans(const impl::MceAndPleInfo& info,
                                           WeightEncoderCache& weightEncoderCache,
                                           Plans& plans,
                                           uint32_t numWeightStripes) const;

    std::pair<Buffer*, Buffer*> AddIdentityMceOpForSubGraph(OwnedOpGraph& opGraph,
                                                            const impl::MceStripesInfo& mceComputeInfo,
                                                            const impl::NumMemoryStripes& numMemoryStripes,
                                                            const impl::MemoryStripesInfo& memoryStripes,
                                                            const TensorShape& inpShape,
                                                            const QuantizationInfo& inpQuantInfo,
                                                            WeightEncoderCache& weightEncoderCache) const;

    TensorShape m_InputTensorShape;
    TensorShape m_OutputTensorShape;
    QuantizationInfo m_InputQuantizationInfo;
    QuantizationInfo m_OutputQuantizationInfo;
    command_stream::PleOperation m_KernelOperation;
    utils::ShapeMultiplier m_ShapeMultiplier;

    impl::StripeConfig m_StripeConfig;
    impl::StripeGenerator m_StripeGenerator;

    mutable WeightEncoderCache m_WeightEncoderCache;

    command_stream::DataType m_InputDataType;
    command_stream::DataType m_OutputDataType;
};
}    // namespace support_library
}    // namespace ethosn
