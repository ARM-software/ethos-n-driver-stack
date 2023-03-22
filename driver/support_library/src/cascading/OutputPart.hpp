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

class OutputPart : public BasePart
{
public:
    template <typename Ids>
    OutputPart(PartId id,
               const TensorShape& inputTensorShape,
               const CompilerDataFormat& compilerDataFormat,
               const QuantizationInfo& quantizationInfo,
               DataType dataType,
               Ids&& correspondingOperationIds,
               const uint32_t producerOutputIndx,
               const EstimationOptions& estOpt,
               const CompilationOptions& compOpt,
               const HardwareCapabilities& capabilities)
        : BasePart(id, "OutputPart", std::forward<Ids>(correspondingOperationIds), estOpt, compOpt, capabilities)
        , m_InputTensorShape{ inputTensorShape }
        , m_InputQuantizationInfo(quantizationInfo)
        , m_InputDataType(dataType)
        , m_ProducerOutputIndx{ producerOutputIndx }
        , m_CompilerDataFormat(compilerDataFormat)
    {}

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;
    DotAttributes GetDotAttributes(DetailLevel detail) const override;
    virtual ~OutputPart();

    std::vector<BoundaryRequirements> GetInputBoundaryRequirements() const override;
    std::vector<bool> CanInputsTakePleInputSram() const override;

private:
    const TensorShape m_InputTensorShape;
    QuantizationInfo m_InputQuantizationInfo;
    DataType m_InputDataType;
    const uint32_t m_ProducerOutputIndx;
    CompilerDataFormat m_CompilerDataFormat;

    void CreatePlanForOutputPart(Plans& plans) const;
};

}    // namespace support_library
}    // namespace ethosn
