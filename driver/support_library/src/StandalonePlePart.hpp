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

class StandalonePlePart : public BasePart
{
public:
    StandalonePlePart(PartId id,
                      const std::vector<TensorShape>& inputTensorShapes,
                      const TensorShape& outputTensorShape,
                      const std::vector<QuantizationInfo>& inputQuantizationInfos,
                      const QuantizationInfo& outputQuantizationInfo,
                      PleOperation op,
                      const EstimationOptions& estOpt,
                      const CompilationOptions& compOpt,
                      const HardwareCapabilities& capabilities,
                      std::set<uint32_t> correspondingOperationIds,
                      DataType dataType,
                      std::map<std::string, std::string> selectionStringParams,
                      std::map<std::string, int> selectionIntParams,
                      std::map<std::string, int> runtimeParams);

    Plans GetPlans(CascadeType cascadeType,
                   BlockConfig blockConfig,
                   const std::vector<Buffer*>& sramBufferInputs,
                   uint32_t numWeightStripes) const override;

    DotAttributes GetDotAttributes(DetailLevel detail) const override;

    std::vector<BoundaryRequirements> GetInputBoundaryRequirements() const override;
    std::vector<bool> CanInputsTakePleInputSram() const override;

private:
    std::vector<TensorShape> m_InputTensorShapes;
    TensorShape m_OutputTensorShape;
    std::vector<QuantizationInfo> m_InputQuantizationInfos;
    QuantizationInfo m_OutputQuantizationInfo;
    PleOperation m_KernelOperation;
    DataType m_DataType;
    impl::StripeConfig m_StripeConfig;
    /// The set of parameters used to select which PLE kernel to use.
    /// @{
    std::map<std::string, std::string> m_SelectionStringParams;
    std::map<std::string, int> m_SelectionIntParams;
    /// @}
    /// The set of parameters passed to the selected PLE kernel at runtime.
    std::map<std::string, int> m_RuntimeParams;
};

}    // namespace support_library
}    // namespace ethosn
