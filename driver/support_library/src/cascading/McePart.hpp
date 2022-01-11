//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "StripeHelper.hpp"
#include "WeightEncoderCache.hpp"

namespace ethosn
{
namespace support_library
{

class McePart : public BasePart
{
public:
    struct ConstructionParams
    {
        ConstructionParams(const EstimationOptions& estOpt,
                           const CompilationOptions& compOpt,
                           const HardwareCapabilities& capabilities)
            : m_EstOpt{ estOpt }
            , m_CompOpt{ compOpt }
            , m_Capabilities{ capabilities }
        {}

        PartId m_Id                     = 0xFFFFFFFF;
        TensorShape m_InputTensorShape  = {};
        TensorShape m_OutputTensorShape = {};
        QuantizationInfo m_InputQuantizationInfo;
        QuantizationInfo m_OutputQuantizationInfo;
        TensorInfo m_WeightsInfo;
        std::vector<uint8_t> m_WeightsData;
        TensorInfo m_BiasInfo;
        std::vector<int32_t> m_BiasData;
        Stride m_Stride;
        uint32_t m_PadTop                 = 0;
        uint32_t m_PadLeft                = 0;
        command_stream::MceOperation m_Op = command_stream::MceOperation::CONVOLUTION;
        const EstimationOptions& m_EstOpt;
        const CompilationOptions& m_CompOpt;
        const HardwareCapabilities& m_Capabilities;
        std::set<uint32_t> m_OperationIds;
        command_stream::DataType m_DataType         = command_stream::DataType::U8;
        uint32_t m_UpscaleFactor                    = 1;
        command_stream::UpsampleType m_UpsampleType = command_stream::UpsampleType::OFF;
        int16_t m_LowerBound                        = 0;
        int16_t m_UpperBound                        = 255;
    };

    McePart(PartId id,
            const TensorShape& inputTensorShape,
            const TensorShape& outputTensorShape,
            const QuantizationInfo& inputQuantizationInfo,
            const QuantizationInfo& outputQuantizationInfo,
            const TensorInfo& weightsInfo,
            std::vector<uint8_t> weightsData,
            const TensorInfo& biasInfo,
            std::vector<int32_t> biasData,
            Stride stride,
            uint32_t padTop,
            uint32_t padLeft,
            command_stream::MceOperation op,
            const EstimationOptions& estOpt,
            const CompilationOptions& compOpt,
            const HardwareCapabilities& capabilities,
            std::set<uint32_t> operationIds,
            command_stream::DataType dataType);
    McePart(ConstructionParams&& params);

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;

    utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const override;

    bool HasActivationBounds() const override;
    void ModifyActivationBounds(int16_t lowerBound, int16_t upperBound) override;

    DotAttributes GetDotAttributes(DetailLevel detail) const override;

protected:
    void CreateMceAndIdentityPlePlans(const impl::MceAndPleInfo& info,
                                      TraversalOrder order,
                                      WeightEncoderCache& weightEncoderCache,
                                      Plans& plans,
                                      uint32_t numWeightStripes) const;

    TensorShape m_InputTensorShape;
    TensorShape m_OutputTensorShape;
    mutable WeightEncoderCache m_WeightEncoderCache;

private:
    Plans GetLonelyPlans(uint32_t numWeightStripes) const;
    Plans GetBeginningPlans(uint32_t numWeightStripes) const;

    Plans GetMiddlePlans(ethosn::command_stream::BlockConfig blockConfig,
                         Buffer* sramBuffer,
                         uint32_t numWeightStripes) const;

    Plans GetEndPlans(ethosn::command_stream::BlockConfig blockConfig,
                      Buffer* sramBuffer,
                      uint32_t numWeightStripes) const;

    uint32_t CalculateTileSize(const HardwareCapabilities& caps,
                               const TensorShape& inputTensorShape,
                               const TensorShape& inputStripeShape,
                               const TensorShape& outputStripeShape,
                               uint32_t numStripes) const;

    std::pair<Buffer*, Op*> AddMceToOpGraph(OwnedOpGraph& opGraph,
                                            Lifetime lifetime,
                                            TraversalOrder order,
                                            const impl::MceStripesInfo& mceStripeInfo,
                                            const impl::MemoryStripesInfo& memoryStripesInfo,
                                            impl::NumMemoryStripes& numMemoryStripes,
                                            const TensorShape& inputShape,
                                            const QuantizationInfo& inputQuantInfo,
                                            impl::ConvData& convData,
                                            WeightEncoderCache& weightEncoderCache) const;

    void CreateMceOnlyPlans(const impl::MceOnlyInfo& info,
                            TraversalOrder order,
                            WeightEncoderCache& weightEncoderCache,
                            Plans& plans,
                            uint32_t numWeightStripes) const;

    Buffer* AddWeightBuffersAndDmaOpToMceOp(OwnedOpGraph& opGraph,
                                            Lifetime lifetime,
                                            const impl::MceStripesInfo& mceComputeInfo,
                                            const impl::NumStripesType& numMemoryWeightStripes,
                                            const TensorShape& memoryWeightStripe,
                                            TraversalOrder order,
                                            const impl::ConvData& convData,
                                            WeightEncoderCache& weightEncoderCache,
                                            CompilerMceAlgorithm mceOpAlgo) const;

    QuantizationInfo m_InputQuantizationInfo;
    QuantizationInfo m_OutputQuantizationInfo;
    TensorInfo m_WeightsInfo;
    std::shared_ptr<const std::vector<uint8_t>> m_WeightsData;
    TensorInfo m_BiasInfo;
    std::vector<int32_t> m_BiasData;
    Stride m_Stride;
    uint32_t m_UpscaleFactor;
    ethosn::command_stream::UpsampleType m_UpsampleType;
    uint32_t m_PadTop;
    uint32_t m_PadLeft;
    command_stream::MceOperation m_Operation;
    impl::StripeGenerator m_StripeGenerator;
    command_stream::DataType m_DataType;
    int16_t m_LowerBound;
    int16_t m_UpperBound;
};
}    // namespace support_library
}    // namespace ethosn
