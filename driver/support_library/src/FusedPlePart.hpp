//
// Copyright © 2021-2023 Arm Limited.
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

using namespace impl;
using namespace utils;

class FusedPlePart : public BasePart
{
public:
    FusedPlePart(PartId id,
                 const TensorShape& inputTensorShape,
                 const TensorShape& outputTensorShape,
                 const QuantizationInfo& inputQuantizationInfo,
                 const QuantizationInfo& outputQuantizationInfo,
                 PleOperation op,
                 const utils::ShapeMultiplier& shapeMultiplier,
                 const EstimationOptions& estOpt,
                 const CompilationOptions& compOpt,
                 const HardwareCapabilities& capabilities,
                 std::set<uint32_t> correspondingOperationIds,
                 DataType inputDataType,
                 DataType outputDataType,
                 DebuggingContext&,
                 ThreadPool& threadPool,
                 std::map<std::string, std::string> selectionStringParams,
                 std::map<std::string, int> selectionIntParams,
                 std::map<std::string, int> runtimeParams);
    FusedPlePart(FusedPlePart&&) = default;

    Plans GetPlans(CascadeType cascadeType,
                   BlockConfig blockConfig,
                   const std::vector<Buffer*>& sramBufferInputs,
                   uint32_t numWeightStripes) const override;

    utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const override;
    bool CanDoubleBufferWeights() const override;

    std::vector<BoundaryRequirements> GetInputBoundaryRequirements() const override;
    std::vector<bool> CanInputsTakePleInputSram() const override;

    DotAttributes GetDotAttributes(DetailLevel detail) const override;

    void PreprocessWeightsAsync() const override;

private:
    Plans GenerateContinueSectionPlans(BlockConfig blockConfig,
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
    PleOperation m_KernelOperation;
    utils::ShapeMultiplier m_ShapeMultiplier;

    impl::StripeConfig m_StripeConfig;
    impl::StripeGenerator m_StripeGenerator;

    mutable WeightEncoderCache m_WeightEncoderCache;

    DataType m_InputDataType;
    DataType m_OutputDataType;

    /// The set of parameters used to select which PLE kernel to use.
    /// @{
    std::map<std::string, std::string> m_SelectionStringParams;
    std::map<std::string, int> m_SelectionIntParams;
    /// @}
    /// The set of parameters passed to the selected PLE kernel at runtime.
    std::map<std::string, int> m_RuntimeParams;
};

}    // namespace support_library
}    // namespace ethosn
