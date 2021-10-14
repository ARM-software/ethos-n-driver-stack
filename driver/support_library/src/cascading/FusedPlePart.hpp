//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "StripeHelper.hpp"

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
                 std::set<uint32_t> correspondingOperationIds);

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;

    utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const override;

private:
    Plans GenerateContinueSectionPlans(ethosn::command_stream::BlockConfig blockConfig,
                                       Buffer* prevBuffer,
                                       uint32_t numWeightStripes,
                                       CascadeType cascadeType) const;

    Plans GetLonelyPlans() const;

    Plans GetBeginningPlans() const;

    Buffer* AddIdentityWeights(OwnedOpGraph& opGraph,
                               Lifetime lifetime,
                               const impl::MceStripesInfo& mceComputeInfo,
                               const impl::NumStripesType& numMemoryWeightStripes,
                               const TensorShape& memoryWeightStripe,
                               TraversalOrder order,
                               const impl::ConvData& convData,
                               WeightEncoderCache& weightEncoderCache) const;

    void CreateFuseOnlyPlans(const impl::PleOnlyInfo& info, TraversalOrder order, Plans& plans) const;

    void CreateIdentityMceAndFusedPlePlans(const impl::MceAndPleInfo& info,
                                           TraversalOrder order,
                                           WeightEncoderCache& weightEncoderCache,
                                           Plans& plans) const;

    std::pair<Buffer*, Buffer*> AddIdentityMceOpForSubGraph(OwnedOpGraph& opGraph,
                                                            Lifetime lifetime,
                                                            const impl::MceStripesInfo& mceComputeInfo,
                                                            const impl::NumMemoryStripes& numMemoryStripes,
                                                            const impl::MemoryStripesInfo& memoryStripes,
                                                            const TensorShape& inpShape,
                                                            const QuantizationInfo& inpQuantInfo,
                                                            TraversalOrder order,
                                                            WeightEncoderCache& weightEncoderCache) const;

    TensorShape m_InputTensorShape;
    TensorShape m_OutputTensorShape;
    QuantizationInfo m_InputQuantizationInfo;
    QuantizationInfo m_OutputQuantizationInfo;
    command_stream::PleOperation m_KernelOperation;
    utils::ShapeMultiplier m_ShapeMultiplier;

    impl::StripeGenerator m_StripeGenerator;
};
}    // namespace support_library
}    // namespace ethosn
