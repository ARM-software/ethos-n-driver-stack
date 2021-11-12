//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"

namespace ethosn
{
namespace support_library
{

class ConcatPart : public BasePart
{
public:
    ConcatPart(PartId id,
               const std::vector<TensorInfo>& inputTensorsInfo,
               const ConcatenationInfo& concatInfo,
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
    virtual ~ConcatPart();

private:
    const std::vector<TensorInfo> m_InputTensorsInfo;
    const ConcatenationInfo m_ConcatInfo;

    void CreateConcatDramPlan(Plans& plans) const;
};

}    // namespace support_library
}    // namespace ethosn
