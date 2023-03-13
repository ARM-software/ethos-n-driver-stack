//
// Copyright © 2021-2023 Arm Limited.
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
                 const HardwareCapabilities& capabilities,
                 const std::vector<uint8_t>& constantData);
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
    CompilerDataFormat m_CompilerDataFormat;
    std::shared_ptr<std::vector<uint8_t>> m_ConstantData;    // Shared ptr to avoid copying for every plan we make

    void CreatePlanForConstantPart(Plans& plans) const;
};

}    // namespace support_library
}    // namespace ethosn
