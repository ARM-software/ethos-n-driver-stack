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
    template <typename Ids>
    ConstantPart(PartId id,
                 const TensorShape& outputTensorShape,
                 const CompilerDataFormat& compilerDataFormat,
                 const QuantizationInfo& quantizationInfo,
                 DataType dataType,
                 Ids&& correspondingOperationIds,
                 const EstimationOptions& estOpt,
                 const CompilationOptions& compOpt,
                 const HardwareCapabilities& capabilities,
                 const std::vector<uint8_t>& constantData)
        : BasePart(id, "ConstantPart", std::forward<Ids>(correspondingOperationIds), estOpt, compOpt, capabilities)
        , m_OutputTensorShape{ outputTensorShape }
        , m_OutputQuantizationInfo(quantizationInfo)
        , m_OutputDataType(dataType)
        , m_CompilerDataFormat(compilerDataFormat)
        , m_ConstantData(std::make_shared<std::vector<uint8_t>>(constantData))
    {}

    Plans GetPlans(CascadeType cascadeType,
                   BlockConfig blockConfig,
                   const std::vector<Buffer*>& sramBufferInputs,
                   uint32_t numWeightStripes) const override;
    DotAttributes GetDotAttributes(DetailLevel detail) const override;
    virtual ~ConstantPart();

    std::vector<BoundaryRequirements> GetInputBoundaryRequirements() const override;
    std::vector<bool> CanInputsTakePleInputSram() const override;

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
