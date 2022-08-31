//
// Copyright © 2021-2022 Arm Limited.
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
    ReshapePart(PartId id,
                const TensorShape& inputTensorShape,
                const TensorShape& outputTensorShape,
                const CompilerDataFormat& compilerDataFormat,
                const QuantizationInfo& quantizationInfo,
                DataType dataType,
                const std::set<uint32_t>& correspondingOperationIds,
                const EstimationOptions& estOpt,
                const CompilationOptions& compOpt,
                const HardwareCapabilities& capabilities);
    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;
    DotAttributes GetDotAttributes(DetailLevel detail) const override;
    virtual ~ReshapePart();

private:
    const TensorShape m_InputTensorShape;
    const TensorShape m_OutputTensorShape;
    QuantizationInfo m_OutputQuantizationInfo;
    DataType m_DataType;
    impl::StripeConfig m_StripeConfig;
};

}    // namespace support_library
}    // namespace ethosn
