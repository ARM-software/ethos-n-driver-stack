//
// Copyright © 2021-2023 Arm Limited.
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
    template <typename Ids>
    ConcatPart(PartId id,
               const std::vector<TensorInfo>& inputTensorsInfo,
               const TensorInfo& outputTensorInfo,
               uint32_t axis,
               const std::vector<uint32_t>& offsets,
               bool preferNhwc,
               Ids&& correspondingOperationIds,
               const EstimationOptions& estOpt,
               const CompilationOptions& compOpt,
               const HardwareCapabilities& capabilities)
        : BasePart(id, "ConcatPart", std::forward<Ids>(correspondingOperationIds), estOpt, compOpt, capabilities)
        , m_InputTensorsInfo{ inputTensorsInfo }
        , m_OutputTensorInfo{ outputTensorInfo }
        , m_Axis(axis)
        , m_Offsets(offsets)
        , m_StripeConfig(impl::GetDefaultStripeConfig(compOpt, m_DebugTag.c_str()))
        , m_PreferNhwc(preferNhwc)
    {}

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
