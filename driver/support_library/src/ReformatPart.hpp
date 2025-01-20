//
// Copyright © 2025 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "StripeHelper.hpp"

namespace ethosn
{
namespace support_library
{

class ReformatPart : public BasePart
{
public:
    template <typename Ids>
    ReformatPart(PartId id,
                 const TensorShape& inputTensorShape,
                 BufferFormat inputBufferFormat,
                 BufferFormat inputTransferFormat,
                 const TensorShape& outputTensorShape,
                 BufferFormat outputBufferFormat,
                 BufferFormat outputTransferFormat,
                 const QuantizationInfo& quantizationInfo,
                 DataType dataType,
                 Ids&& correspondingOperationIds,
                 const EstimationOptions& estOpt,
                 const CompilationOptions& compOpt,
                 const HardwareCapabilities& capabilities)
        : BasePart(id, "ReformatPart", std::forward<Ids>(correspondingOperationIds), estOpt, compOpt, capabilities)
        , m_InputTensorShape{ inputTensorShape }
        , m_InputBufferFormat{ inputBufferFormat }
        , m_InputTransferFormat{ inputTransferFormat }
        , m_OutputTensorShape{ outputTensorShape }
        , m_OutputBufferFormat{ outputBufferFormat }
        , m_OutputTransferFormat{ outputTransferFormat }
        , m_OutputQuantizationInfo(quantizationInfo)
        , m_DataType(dataType)
        , m_StripeConfig(impl::GetDefaultStripeConfig(compOpt, m_DebugTag.c_str()))
    {}

    Plans GetPlans(CascadeType cascadeType,
                   BlockConfig blockConfig,
                   const std::vector<Buffer*>& sramBufferInputs,
                   uint32_t numWeightStripes) const override;
    bool IsOutputGuaranteedNhwc() const override;
    DotAttributes GetDotAttributes(DetailLevel detail) const override;
    virtual ~ReformatPart();

    std::vector<BoundaryRequirements> GetInputBoundaryRequirements() const override;
    std::vector<bool> CanInputsTakePleInputSram() const override;

private:
    const TensorShape m_InputTensorShape;
    const BufferFormat m_InputBufferFormat;
    const BufferFormat m_InputTransferFormat;

    const TensorShape m_OutputTensorShape;
    const BufferFormat m_OutputBufferFormat;
    const BufferFormat m_OutputTransferFormat;

    QuantizationInfo m_OutputQuantizationInfo;
    DataType m_DataType;
    impl::StripeConfig m_StripeConfig;
};

}    // namespace support_library
}    // namespace ethosn
