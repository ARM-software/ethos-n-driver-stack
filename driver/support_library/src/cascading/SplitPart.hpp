//
// Copyright © 2022-2023 Arm Limited.
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
    template <typename Ids>
    SplitPart(PartId id,
              const TensorInfo& inputTensorInfo,
              const std::vector<TensorInfo>& outputTensorInfos,
              uint32_t axis,
              const std::vector<uint32_t>& offsets,
              Ids&& correspondingOperationIds,
              const EstimationOptions& estOpt,
              const CompilationOptions& compOpt,
              const HardwareCapabilities& capabilities)
        : BasePart(id, "SplitPart", std::forward<Ids>(correspondingOperationIds), estOpt, compOpt, capabilities)
        , m_InputTensorInfo{ inputTensorInfo }
        , m_OutputTensorInfos(outputTensorInfos)
        , m_Axis(axis)
        , m_Offsets(offsets)
        , m_StripeConfig(impl::GetDefaultStripeConfig(compOpt, m_DebugTag.c_str()))
    {}

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;
    DotAttributes GetDotAttributes(DetailLevel detail) const override;
    virtual ~SplitPart();

    const TensorShape& GetInputTensorShape() const;
    const std::vector<uint32_t>& GetOffsets() const;

    std::vector<BoundaryRequirements> GetInputBoundaryRequirements() const override;
    std::vector<bool> CanInputsTakePleInputSram() const override;

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
