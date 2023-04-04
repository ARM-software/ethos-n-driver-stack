//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "StripeHelper.hpp"

namespace ethosn
{
namespace support_library
{

class StandalonePlePart : public BasePart
{
public:
    template <typename Ids>
    StandalonePlePart(PartId id,
                      const std::vector<TensorShape>& inputTensorShapes,
                      const TensorShape& outputTensorShape,
                      const std::vector<QuantizationInfo>& inputQuantizationInfos,
                      const QuantizationInfo& outputQuantizationInfo,
                      command_stream::PleOperation op,
                      const EstimationOptions& estOpt,
                      const CompilationOptions& compOpt,
                      const HardwareCapabilities& capabilities,
                      Ids&& correspondingOperationIds,
                      DataType dataType)
        : BasePart(id, "StandalonePlePart", std::forward<Ids>(correspondingOperationIds), estOpt, compOpt, capabilities)
        , m_InputTensorShapes(inputTensorShapes)
        , m_OutputTensorShape(outputTensorShape)
        , m_InputQuantizationInfos(inputQuantizationInfos)
        , m_OutputQuantizationInfo(outputQuantizationInfo)
        , m_KernelOperation(op)
        , m_DataType(dataType)
        , m_StripeConfig(impl::GetDefaultStripeConfig(compOpt, m_DebugTag.c_str()))
    {
        assert(m_InputQuantizationInfos.size() == m_InputTensorShapes.size());

        const double outputScale = outputQuantizationInfo.GetScale();
        const double inputScale0 = inputQuantizationInfos[0].GetScale();
        utils::CalculateRescaleMultiplierAndShift(inputScale0 / outputScale, m_Input0Multiplier, m_Input0Shift);

        if (inputTensorShapes.size() == 2)
        {
            const double inputScale1 = inputQuantizationInfos[1].GetScale();
            utils::CalculateRescaleMultiplierAndShift(inputScale1 / outputScale, m_Input1Multiplier, m_Input1Shift);
        }
    }

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* prevBuffer,
                   uint32_t numWeightStripes) const override;

    DotAttributes GetDotAttributes(DetailLevel detail) const override;

private:
    std::vector<TensorShape> m_InputTensorShapes;
    TensorShape m_OutputTensorShape;
    std::vector<QuantizationInfo> m_InputQuantizationInfos;
    QuantizationInfo m_OutputQuantizationInfo;
    command_stream::PleOperation m_KernelOperation;
    DataType m_DataType;
    impl::StripeConfig m_StripeConfig;
    uint16_t m_Input0Multiplier;
    uint16_t m_Input0Shift;
    uint16_t m_Input1Multiplier;
    uint16_t m_Input1Shift;
};
}    // namespace support_library
}    // namespace ethosn
