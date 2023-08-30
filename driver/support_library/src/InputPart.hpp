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

class InputPart : public BasePart
{
public:
    template <typename Ids>
    InputPart(PartId id,
              const TensorShape& outputTensorShape,
              const CompilerDataFormat& compilerDataFormat,
              const QuantizationInfo& quantizationInfo,
              DataType dataType,
              Ids&& correspondingOperationIds,
              const EstimationOptions& estOpt,
              const CompilationOptions& compOpt,
              const HardwareCapabilities& capabilities)
        : BasePart(id, "InputPart", std::forward<Ids>(correspondingOperationIds), estOpt, compOpt, capabilities)
        , m_OutputTensorShape{ outputTensorShape }
        , m_OutputQuantizationInfo(quantizationInfo)
        , m_OutputDataType(dataType)
        , m_CompilerDataFormat(compilerDataFormat)
    {}

    Plans GetPlans(CascadeType cascadeType,
                   BlockConfig blockConfig,
                   const std::vector<Buffer*>& sramBufferInputs,
                   uint32_t numWeightStripes) const override;
    DotAttributes GetDotAttributes(DetailLevel detail) const override;
    virtual ~InputPart();

    std::vector<BoundaryRequirements> GetInputBoundaryRequirements() const override;
    std::vector<bool> CanInputsTakePleInputSram() const override;

private:
    const TensorShape m_OutputTensorShape;
    QuantizationInfo m_OutputQuantizationInfo;
    DataType m_OutputDataType;
    CompilerDataFormat m_CompilerDataFormat;

    void CreatePlanForInputPart(Plans& plans) const;
};

}    // namespace support_library
}    // namespace ethosn
