//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../Utils.hpp"
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
                           const HardwareCapabilities& capabilities,
                           DebuggingContext& debuggingContext)
            : m_EstOpt{ estOpt }
            , m_CompOpt{ compOpt }
            , m_Capabilities{ capabilities }
            , m_DebuggingContext(debuggingContext)
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
        DataType m_InputDataType       = DataType::UINT8_QUANTIZED;
        DataType m_OutputDataType      = DataType::UINT8_QUANTIZED;
        uint32_t m_UpscaleFactor       = 1;
        MceUpsampleType m_UpsampleType = MceUpsampleType::OFF;
        int16_t m_LowerBound           = 0;
        int16_t m_UpperBound           = 255;
        bool m_IsChannelSelector       = false;
        DebuggingContext& m_DebuggingContext;
    };

    template <typename Ids, typename Weights, typename Biases>
    McePart(PartId id,
            const TensorShape& inputTensorShape,
            const TensorShape& outputTensorShape,
            const QuantizationInfo& inputQuantizationInfo,
            const QuantizationInfo& outputQuantizationInfo,
            const TensorInfo& weightsInfo,
            Weights&& weightsData,
            const TensorInfo& biasInfo,
            Biases&& biasData,
            const Stride& stride,
            uint32_t padTop,
            uint32_t padLeft,
            command_stream::MceOperation op,
            const EstimationOptions& estOpt,
            const CompilationOptions& compOpt,
            const HardwareCapabilities& capabilities,
            Ids&& operationIds,
            DataType inputDataType,
            DataType outputDataType,
            DebuggingContext& debuggingContext)
        : BasePart(id, "McePart", std::forward<Ids>(operationIds), estOpt, compOpt, capabilities)
        , m_InputTensorShape(inputTensorShape)
        , m_OutputTensorShape(outputTensorShape)
        , m_WeightEncoderCache{ capabilities, debuggingContext, m_DebugTag.c_str() }
        , m_InputQuantizationInfo(inputQuantizationInfo)
        , m_OutputQuantizationInfo(outputQuantizationInfo)
        , m_WeightsInfo(weightsInfo)
        , m_WeightsData(std::make_shared<std::vector<uint8_t>>(std::forward<Weights>(weightsData)))
        , m_BiasInfo(biasInfo)
        , m_BiasData(std::forward<Biases>(biasData))
        , m_Stride(stride)
        , m_UpscaleFactor(1U)
        , m_UpsampleType(MceUpsampleType::OFF)
        , m_PadTop(padTop)
        , m_PadLeft(padLeft)
        , m_Operation(op)
        , m_StripeConfig(impl::GetDefaultStripeConfig(compOpt, m_DebugTag.c_str()))
        , m_StripeGenerator(m_InputTensorShape,
                            m_OutputTensorShape,
                            m_OutputTensorShape,
                            m_WeightsInfo.m_Dimensions[0],
                            m_WeightsInfo.m_Dimensions[1],
                            m_PadTop,
                            m_PadLeft,
                            m_UpscaleFactor,
                            op,
                            command_stream::PleOperation::PASSTHROUGH,
                            utils::ShapeMultiplier{ 1, 1, utils::Fraction(1, stride.m_X * stride.m_Y) },
                            utils::ShapeMultiplier::Identity,
                            capabilities,
                            m_StripeConfig)
        , m_InputDataType(inputDataType)
        , m_OutputDataType(outputDataType)
        , m_LowerBound(outputDataType == DataType::UINT8_QUANTIZED ? 0 : -128)
        , m_UpperBound(outputDataType == DataType::UINT8_QUANTIZED ? 255 : 127)
        , m_IsChannelSelector(false)
    {}

    McePart(ConstructionParams&& params);

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig blockConfig,
                   Buffer* sramBuffer,
                   uint32_t numWeightStripes) const override;

    utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const override;

    bool HasActivationBounds() const override;
    void ModifyActivationBounds(int16_t lowerBound, int16_t upperBound) override;
    bool CanDoubleBufferWeights() const override;

    DotAttributes GetDotAttributes(DetailLevel detail) const override;

    void setUninterleavedInputShape(TensorShape uninterleavedInputShape);

    const std::vector<uint8_t>& GetWeightsData() const;
    const TensorInfo& GetWeightsInfo() const;
    const std::vector<int32_t>& GetBiasData() const;
    const TensorInfo& GetBiasInfo() const;
    const TensorShape& GetInputTensorShape() const;
    const TensorShape& GetOutputTensorShape() const;

    utils::Optional<utils::ConstTensorData> GetChannelSelectorWeights() const override;
    bool MergeWithChannelSelectorBefore(const utils::ConstTensorData& channelSelectorWeights) override;
    bool MergeWithChannelSelectorAfter(const utils::ConstTensorData& channelSelectorWeights) override;

protected:
    void CreateMceAndIdentityPlePlans(const impl::MceAndPleInfo& info,
                                      WeightEncoderCache& weightEncoderCache,
                                      Plans& plans,
                                      uint32_t numWeightStripes,
                                      bool couldSourceBeFcaf) const;

    utils::Optional<TensorShape> m_UninterleavedInputShape;
    TensorShape m_InputTensorShape;
    TensorShape m_OutputTensorShape;
    mutable WeightEncoderCache m_WeightEncoderCache;

    Plans GetLonelyPlans(uint32_t numWeightStripes) const;
    Plans GetBeginningPlans(uint32_t numWeightStripes) const;

    Plans GetMiddlePlans(ethosn::command_stream::BlockConfig blockConfig,
                         const SramBuffer* sramBuffer,
                         uint32_t numWeightStripes) const;

    Plans GetEndPlans(ethosn::command_stream::BlockConfig blockConfig,
                      const SramBuffer* sramBuffer,
                      uint32_t numWeightStripes) const;

    std::pair<Buffer*, Op*> AddMceToOpGraph(OwnedOpGraph& opGraph,
                                            const impl::MceStripesInfo& mceStripeInfo,
                                            const impl::MemoryStripesInfo& memoryStripesInfo,
                                            impl::NumMemoryStripes& numMemoryStripes,
                                            const TensorShape& inputShape,
                                            const QuantizationInfo& inputQuantInfo,
                                            impl::ConvData& convData,
                                            WeightEncoderCache& weightEncoderCache,
                                            bool couldSourceBeFcaf) const;

    void CreateMceOnlyPlans(const impl::MceOnlyInfo& info,
                            WeightEncoderCache& weightEncoderCache,
                            Plans& plans,
                            uint32_t numWeightStripes,
                            bool couldSourceBeFcaf) const;

    Buffer* AddWeightBuffersAndDmaOpToMceOp(OwnedOpGraph& opGraph,
                                            const impl::MceStripesInfo& mceComputeInfo,
                                            const impl::NumStripesType& numMemoryWeightStripes,
                                            const TensorShape& memoryWeightStripe,
                                            uint32_t numLoads,
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
    MceUpsampleType m_UpsampleType;
    uint32_t m_PadTop;
    uint32_t m_PadLeft;
    command_stream::MceOperation m_Operation;
    impl::StripeConfig m_StripeConfig;
    impl::StripeGenerator m_StripeGenerator;
    DataType m_InputDataType;
    DataType m_OutputDataType;
    int16_t m_LowerBound;
    int16_t m_UpperBound;
    bool m_IsChannelSelector;
};
}    // namespace support_library
}    // namespace ethosn
