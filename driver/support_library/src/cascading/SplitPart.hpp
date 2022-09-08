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
              const SplitInfo& splitInfo,
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
    virtual ~SplitPart();

private:
    const TensorInfo& m_InputTensorInfo;
    const SplitInfo m_SplitInfo;

    void CreateSplitDramPlans(Plans& plans) const;

    impl::StripeConfig m_StripeConfig;
};

}    // namespace support_library
}    // namespace ethosn
