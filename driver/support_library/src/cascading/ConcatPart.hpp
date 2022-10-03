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

class ConcatPart : public BasePart
{
public:
    ConcatPart(PartId id,
               const std::vector<TensorInfo>& inputTensorsInfo,
               const TensorInfo& outputTensorInfo,
               uint32_t axis,
               const std::vector<uint32_t>& offsets,
               bool preferNhwc,
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

    const TensorShape& GetOutputTensorShape() const;
    const std::vector<uint32_t>& GetOffsets() const;

private:
    const std::vector<TensorInfo> m_InputTensorsInfo;
    TensorInfo m_OutputTensorInfo;
    uint32_t m_Axis;
    std::vector<uint32_t> m_Offsets;
    impl::StripeConfig m_StripeConfig;
    bool m_PreferNhwc;

    void CreateConcatDramPlans(Plans& plans) const;
};

}    // namespace support_library
}    // namespace ethosn
