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
    struct ConstructionParams
    {
        ConstructionParams(const EstimationOptions& estOpt,
                           const CompilationOptions& compOpt,
                           const HardwareCapabilities& capabilities,
                           DebuggingContext& debuggingContext,
                           ThreadPool& threadPool)
            : m_EstOpt{ estOpt }
            , m_CompOpt{ compOpt }
            , m_Capabilities{ capabilities }
            , m_DebuggingContext(debuggingContext)
            , m_ThreadPool(threadPool)
        {}

        PartId m_Id                                 = 0xFFFFFFFF;
        TensorShape m_InputTensorShape              = {};
        TensorShape m_ReinterpretedInputTensorShape = {};
        TensorShape m_OutputTensorShape             = {};
        QuantizationInfo m_InputQuantizationInfo;
        QuantizationInfo m_OutputQuantizationInfo;
        TensorInfo m_WeightsInfo;
        std::vector<uint8_t> m_WeightsData;
        TensorInfo m_BiasInfo;
        std::vector<int32_t> m_BiasData;
        const EstimationOptions& m_EstOpt;
        const CompilationOptions& m_CompOpt;
        const HardwareCapabilities& m_Capabilities;
        std::set<uint32_t> m_OperationIds;
        DataType m_InputDataType  = DataType::UINT8_QUANTIZED;
        DataType m_OutputDataType = DataType::UINT8_QUANTIZED;
        DebuggingContext& m_DebuggingContext;
        ThreadPool& m_ThreadPool;
    };

    FullyConnectedPart(ConstructionParams&& params);

    Plans GetPlans(CascadeType cascadeType,
                   BlockConfig blockConfig,
                   const std::vector<Buffer*>& sramBufferInputs,
                   uint32_t numWeightStripes) const override;

    utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const override;

    void PreprocessWeightsAsync() const override;

private:
    Plans GetLonelyPlans(uint32_t numWeightStripes) const;

    impl::StripeInfos GenerateStripeInfos() const;

    TensorShape m_OriginalInputShape;
};
}    // namespace support_library
}    // namespace ethosn
