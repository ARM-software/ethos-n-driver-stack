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

class OutputPart : public BasePart
{
public:
    OutputPart(PartId id,
               const TensorShape& inputTensorShape,
               const CompilerDataFormat& compilerDataFormat,
               const QuantizationInfo& quantizationInfo,
               const std::set<uint32_t>& correspondingOperationIds,
               const uint32_t producerOutputIndx,
               const EstimationOptions& estOpt,
               const CompilationOptions& compOpt,
               const HardwareCapabilities& capabilities);
    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;
    DotAttributes GetDotAttributes(DetailLevel detail) const override;
    virtual ~OutputPart();

private:
    const TensorShape m_InputTensorShape;
    QuantizationInfo m_InputQuantizationInfo;
    const uint32_t m_ProducerOutputIndx;

    void CreatePlanForOutputPart(TraversalOrder order, Plans& plans) const;
};

}    // namespace support_library
}    // namespace ethosn
