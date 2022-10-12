//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "StripeHelper.hpp"

namespace ethosn
{
namespace support_library
{

class SplitPart : public BasePart
{
public:
    SplitPart(PartId id,
              const TensorInfo& inputTensorInfo,
              const std::vector<TensorInfo>& outputTensorInfos,
              uint32_t axis,
              const std::vector<uint32_t>& offsets,
              const std::set<uint32_t>& correspondingOperationIds,
              const EstimationOptions& estOpt,
              const CompilationOptions& compOpt,
              const HardwareCapabilities& capabilities);
    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;
    DotAttributes GetDotAttributes(DetailLevel detail) const override;
    virtual ~SplitPart();

    const TensorShape& GetInputTensorShape() const;
    const std::vector<uint32_t>& GetOffsets() const;

private:
    TensorInfo m_InputTensorInfo;
    std::vector<TensorInfo> m_OutputTensorInfos;
    uint32_t m_Axis;
    std::vector<uint32_t> m_Offsets;

    void CreateSplitDramPlans(Plans& plans) const;

    impl::StripeConfig m_StripeConfig;
};

}    // namespace support_library
}    // namespace ethosn
