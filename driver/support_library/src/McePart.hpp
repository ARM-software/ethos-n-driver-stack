//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "StripeHelper.hpp"
#include "Utils.hpp"
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
                           DebuggingContext& debuggingContext,
                           ThreadPool& threadPool)
            : m_EstOpt{ estOpt }
            , m_CompOpt{ compOpt }
            , m_Capabilities{ capabilities }
            , m_DebuggingContext(debuggingContext)
            , m_ThreadPool(threadPool)
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
        ThreadPool& m_ThreadPool;
    };

    McePart(ConstructionParams&& params);
    McePart(McePart&&) = default;

    Plans GetPlans(CascadeType cascadeType,
                   BlockConfig blockConfig,
                   const std::vector<Buffer*>& sramBufferInputs,
                   uint32_t numWeightStripes) const override;

    utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const override;

    bool HasActivationBounds() const override;
    void ApplyActivationBounds(int16_t lowerBound, int16_t upperBound) override;
    bool CanDoubleBufferWeights() const override;

    std::vector<BoundaryRequirements> GetInputBoundaryRequirements() const override;
    std::vector<bool> CanInputsTakePleInputSram() const override;

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

    void PreprocessWeightsAsync() const override;

protected:
    CompilerMceAlgorithm ResolveMceAlgorithm(const BlockConfig& blockConfig, uint32_t inputStripeChannels) const;

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

    Plans GetMiddlePlans(BlockConfig blockConfig, const SramBuffer* sramBuffer, uint32_t numWeightStripes) const;

    Plans GetEndPlans(BlockConfig blockConfig, const SramBuffer* sramBuffer, uint32_t numWeightStripes) const;

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
