//
// Copyright © 2022 Arm Limited.
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
    StandalonePlePart(PartId id,
                      const std::vector<TensorShape>& inputTensorShapes,
                      const TensorShape& outputTensorShape,
                      const std::vector<QuantizationInfo>& inputQuantizationInfos,
                      const QuantizationInfo& outputQuantizationInfo,
                      command_stream::PleOperation op,
                      const EstimationOptions& estOpt,
                      const CompilationOptions& compOpt,
                      const HardwareCapabilities& capabilities,
                      std::set<uint32_t> correspondingOperationIds,
                      DataType dataType);

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
};
}    // namespace support_library
}    // namespace ethosn
