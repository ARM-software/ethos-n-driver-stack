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

class EstimateOnlyPart : public BasePart
{
public:
    EstimateOnlyPart(PartId id,
                     const std::string& reasonForEstimateOnly,
                     const std::vector<TensorInfo>& inputTensorsInfo,
                     const std::vector<TensorInfo>& outputTensorsInfo,
                     const CompilerDataFormat& compilerDataFormat,
                     const std::set<uint32_t>& correspondingOperationIds,
                     const EstimationOptions& estOpt,
                     const CompilationOptions& compOpt,
                     const HardwareCapabilities& capabilities);
    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;
    DotAttributes GetDotAttributes(DetailLevel detail) const override;
    virtual ~EstimateOnlyPart();

private:
    const std::vector<TensorInfo> m_InputTensorsInfo;
    const std::vector<TensorInfo> m_OutputTensorsInfo;
    const std::string m_ReasonForEstimateOnly;
    CompilerDataFormat m_CompilerDataFormat;

    void CreatePlanForEstimateOnlyPart(TraversalOrder order, Plans& plans) const;
};

}    // namespace support_library
}    // namespace ethosn
