//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "StripeHelper.hpp"

namespace ethosn
{
namespace support_library
{

class ReshapePart : public BasePart
{
public:
    template <typename Ids>
    ReshapePart(PartId id,
                const TensorShape& inputTensorShape,
                const TensorShape& outputTensorShape,
                const QuantizationInfo& quantizationInfo,
                DataType dataType,
                Ids&& correspondingOperationIds,
                const EstimationOptions& estOpt,
                const CompilationOptions& compOpt,
                const HardwareCapabilities& capabilities)
        : BasePart(id, "ReshapePart", std::forward<Ids>(correspondingOperationIds), estOpt, compOpt, capabilities)
        , m_InputTensorShape{ inputTensorShape }
        , m_OutputTensorShape{ outputTensorShape }
        , m_OutputQuantizationInfo(quantizationInfo)
        , m_DataType(dataType)
        , m_StripeConfig(impl::GetDefaultStripeConfig(compOpt, m_DebugTag.c_str()))
    {}

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;
    bool IsOutputGuaranteedNhwc() const override;
    DotAttributes GetDotAttributes(DetailLevel detail) const override;
    virtual ~ReshapePart();

    std::vector<BoundaryRequirements> GetInputBoundaryRequirements() const override;
    std::vector<bool> CanInputsTakePleInputSram() const override;

private:
    const TensorShape m_InputTensorShape;
    const TensorShape m_OutputTensorShape;
    QuantizationInfo m_OutputQuantizationInfo;
    DataType m_DataType;
    impl::StripeConfig m_StripeConfig;
};

}    // namespace support_library
}    // namespace ethosn
