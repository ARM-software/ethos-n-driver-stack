//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "McePart.hpp"
#include "StripeHelper.hpp"
#include "WeightEncoderCache.hpp"

namespace ethosn
{
namespace support_library
{

class FullyConnectedPart : public McePart
{
public:
    FullyConnectedPart(PartId id,
                       const TensorShape& inputTensorShape,
                       const TensorShape& reinterpretedInputShape,
                       const TensorShape& outputTensorShape,
                       const QuantizationInfo& inputQuantizationInfo,
                       const QuantizationInfo& outputQuantizationInfo,
                       const TensorInfo& weightsInfo,
                       std::vector<uint8_t> weightsData,
                       const TensorInfo& biasInfo,
                       std::vector<int32_t> biasData,
                       const EstimationOptions& estOpt,
                       const CompilationOptions& compOpt,
                       const HardwareCapabilities& capabilities,
                       std::set<uint32_t> operationIds,
                       command_stream::DataType inputDataType,
                       command_stream::DataType outputDataType);

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;

    utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const override;

private:
    Plans GetLonelyPlans(uint32_t numWeightStripes) const;

    TensorShape m_OriginalInputShape;
};
}    // namespace support_library
}    // namespace ethosn
