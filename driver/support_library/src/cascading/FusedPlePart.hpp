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
    template <typename Ids>
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
                 Ids&& correspondingOperationIds,
                 DataType m_InputDataType,
                 DataType m_OutputDataType,
                 float alpha,
                 DebuggingContext&,
                 ThreadPool& threadPool)
        : BasePart(id, "FusedPlePart", std::forward<Ids>(correspondingOperationIds), estOpt, compOpt, capabilities)
        , m_InputTensorShape(inputTensorShape)
        , m_OutputTensorShape(outputTensorShape)
        , m_InputQuantizationInfo(inputQuantizationInfo)
        , m_OutputQuantizationInfo(outputQuantizationInfo)
        , m_KernelOperation(op)
        , m_ShapeMultiplier(shapeMultiplier)
        , m_StripeConfig(GetDefaultStripeConfig(compOpt, m_DebugTag.c_str()))
        , m_StripeGenerator(m_InputTensorShape,
                            m_InputTensorShape,
                            m_OutputTensorShape,
                            1,
                            1,
                            0,
                            0,
                            1,
                            command_stream::MceOperation::DEPTHWISE_CONVOLUTION,
                            op,
                            ShapeMultiplier::Identity,
                            shapeMultiplier,
                            capabilities,
                            m_StripeConfig)
        , m_WeightEncoderCache(capabilities, threadPool)
        , m_InputDataType(m_InputDataType)
        , m_OutputDataType(m_OutputDataType)
        , m_Input0Multiplier(0)
        , m_Input0Shift(0)
        , m_Input1Multiplier(0)
        , m_Input1Shift(0)
    {
        m_StripeGenerator.m_StripeConfig.blockConfigs =
            FilterPleBlockConfigs(m_KernelOperation, m_StripeGenerator.m_StripeConfig.blockConfigs);

        if (op == PleOperation::SIGMOID)
        {
            constexpr double log2e = 1.4426950408889634;

            const double inputScale = inputQuantizationInfo.GetScale();

            const double rescaleFactor = inputScale * (log2e * 256.);

            // Note that tanh shares the same PLE kernel with sigmoid
            // by applying different scaling factor to input and output
            // The output tensor scaling factor is 1/256 for sigmoid
            // and 1/128 for tanh.
            assert(outputQuantizationInfo.GetScale() == (1.f / 128) ||
                   outputQuantizationInfo.GetScale() == (1.f / 256));
            const double tanhFactor = (outputQuantizationInfo.GetScale() == (1.f / 128)) ? 2.0f : 1.0f;

            utils::CalculateRescaleMultiplierAndShift(rescaleFactor * tanhFactor, m_Input0Multiplier, m_Input0Shift);

            int absMax = static_cast<int>(std::ceil(std::ldexp(1., 15U + m_Input0Shift) / m_Input0Multiplier)) - 1;

            if (absMax == 0)
            {
                absMax = 1;

                m_Input0Multiplier = INT16_MAX;
                m_Input0Shift      = 0;
            }
        }
        else if (op == PleOperation::LEAKY_RELU)
        {
            const double alphaRescaleFactor =
                alpha * (inputQuantizationInfo.GetScale() / outputQuantizationInfo.GetScale());
            uint16_t alphaMult;
            uint16_t alphaShift;
            CalculateRescaleMultiplierAndShift(alphaRescaleFactor, alphaMult, alphaShift);

            const double inputToOutputRescaleFactor =
                (inputQuantizationInfo.GetScale() / outputQuantizationInfo.GetScale());
            uint16_t inputToOutputMult;
            uint16_t inputToOutputShift;
            CalculateRescaleMultiplierAndShift(inputToOutputRescaleFactor, inputToOutputMult, inputToOutputShift);

            m_Input0Multiplier = inputToOutputMult;
            m_Input0Shift      = inputToOutputShift;

            m_Input1Multiplier = alphaMult;
            m_Input1Shift      = alphaShift;
        }
    }
    FusedPlePart(FusedPlePart&&) = default;

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   const std::vector<Buffer*>& sramBufferInputs,
                   uint32_t numWeightStripes) const override;

    utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const override;
    bool CanDoubleBufferWeights() const override;

    std::vector<BoundaryRequirements> GetInputBoundaryRequirements() const override;
    std::vector<bool> CanInputsTakePleInputSram() const override;

    DotAttributes GetDotAttributes(DetailLevel detail) const override;

    void PreprocessWeightsAsync() const override;

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
    PleOperation m_KernelOperation;
    utils::ShapeMultiplier m_ShapeMultiplier;

    impl::StripeConfig m_StripeConfig;
    impl::StripeGenerator m_StripeGenerator;

    mutable WeightEncoderCache m_WeightEncoderCache;

    DataType m_InputDataType;
    DataType m_OutputDataType;

    uint16_t m_Input0Multiplier;
    uint16_t m_Input0Shift;
    uint16_t m_Input1Multiplier;
    uint16_t m_Input1Shift;
};
}    // namespace support_library
}    // namespace ethosn
