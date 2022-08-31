//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"

namespace ethosn
{
namespace support_library
{

class ConstantPart : public BasePart
{
public:
    ConstantPart(PartId id,
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
    virtual ~ConstantPart();

private:
    const TensorShape m_OutputTensorShape;
    QuantizationInfo m_OutputQuantizationInfo;
    DataType m_OutputDataType;

    void CreatePlanForConstantPart(TraversalOrder order, Plans& plans) const;
};

}    // namespace support_library
}    // namespace ethosn
