//
// Copyright © 2021-2023 Arm Limited.
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
    template <typename Ids, typename Weights, typename Biases>
    FullyConnectedPart(PartId id,
                       const TensorShape& inputTensorShape,
                       const TensorShape& reinterpretedInputShape,
                       const TensorShape& outputTensorShape,
                       const QuantizationInfo& inputQuantizationInfo,
                       const QuantizationInfo& outputQuantizationInfo,
                       const TensorInfo& weightsInfo,
                       Weights&& weightsData,
                       const TensorInfo& biasInfo,
                       Biases&& biasData,
                       const EstimationOptions& estOpt,
                       const CompilationOptions& compOpt,
                       const HardwareCapabilities& capabilities,
                       Ids&& operationIds,
                       DataType inputDataType,
                       DataType outputDataType,
                       DebuggingContext& debuggingContext)
        : McePart(id,
                  reinterpretedInputShape,
                  outputTensorShape,
                  inputQuantizationInfo,
                  outputQuantizationInfo,
                  weightsInfo,
                  std::forward<Weights>(weightsData),
                  biasInfo,
                  std::forward<Biases>(biasData),
                  Stride{},
                  0,
                  0,
                  command_stream::MceOperation::FULLY_CONNECTED,
                  estOpt,
                  compOpt,
                  capabilities,
                  std::forward<Ids>(operationIds),
                  inputDataType,
                  outputDataType,
                  debuggingContext)
        , m_OriginalInputShape(inputTensorShape)
    {}

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   const std::vector<Buffer*>& sramBufferInputs,
                   uint32_t numWeightStripes) const override;

    utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const override;

private:
    Plans GetLonelyPlans(uint32_t numWeightStripes) const;

    TensorShape m_OriginalInputShape;
};
}    // namespace support_library
}    // namespace ethosn
