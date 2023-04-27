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

class EstimateOnlyPart : public BasePart
{
public:
    template <typename Ids>
    EstimateOnlyPart(PartId id,
                     const std::string& reasonForEstimateOnly,
                     const std::vector<TensorInfo>& inputTensorsInfo,
                     const std::vector<TensorInfo>& outputTensorsInfo,
                     const CompilerDataFormat& compilerDataFormat,
                     Ids&& correspondingOperationIds,
                     const EstimationOptions& estOpt,
                     const CompilationOptions& compOpt,
                     const HardwareCapabilities& capabilities)
        : BasePart(id, "EstimateOnlyPart", std::forward<Ids>(correspondingOperationIds), estOpt, compOpt, capabilities)
        , m_InputTensorsInfo{ inputTensorsInfo }
        , m_OutputTensorsInfo{ outputTensorsInfo }
        , m_ReasonForEstimateOnly{ reasonForEstimateOnly }
        , m_CompilerDataFormat(compilerDataFormat)
    {}

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;
    DotAttributes GetDotAttributes(DetailLevel detail) const override;
    virtual ~EstimateOnlyPart();

    std::vector<BoundaryRequirements> GetInputBoundaryRequirements() const override;

private:
    const std::vector<TensorInfo> m_InputTensorsInfo;
    const std::vector<TensorInfo> m_OutputTensorsInfo;
    const std::string m_ReasonForEstimateOnly;
    CompilerDataFormat m_CompilerDataFormat;

    void CreatePlanForEstimateOnlyPart(Plans& plans) const;
};

}    // namespace support_library
}    // namespace ethosn
