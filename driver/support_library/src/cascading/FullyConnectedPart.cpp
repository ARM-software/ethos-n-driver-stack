//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "FullyConnectedPart.hpp"

using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{
using namespace impl;
using namespace utils;

FullyConnectedPart::FullyConnectedPart(PartId id,
                                       const TensorShape& inputTensorShape,
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
                                       command_stream::DataType dataType)
    : McePart(id,
              inputTensorShape,
              outputTensorShape,
              inputQuantizationInfo,
              outputQuantizationInfo,
              weightsInfo,
              weightsData,
              biasInfo,
              biasData,
              Stride{},
              0,
              0,
              command_stream::MceOperation::FULLY_CONNECTED,
              estOpt,
              compOpt,
              capabilities,
              operationIds,
              dataType)
{}

Plans FullyConnectedPart::GetPlans(CascadeType cascadeType,
                                   BlockConfig blockConfig,
                                   Buffer* sramBuffer,
                                   uint32_t numWeightStripes) const
{
    ETHOSN_UNUSED(blockConfig);
    ETHOSN_UNUSED(sramBuffer);
    // Only Lonely plans are supported at the moment as fully connected layers
    // are rare and usually very large. This means the likelihood they can be
    // cascaded is reduced and their impact on performance is small.
    if (cascadeType == CascadeType::Lonely)
    {
        return GetLonelyPlans(numWeightStripes);
    }
    else
    {
        return Plans{};
    }
}

utils::Optional<MceOperation> FullyConnectedPart::GetMceOperation() const
{
    return ethosn::command_stream::MceOperation::FULLY_CONNECTED;
}

Plans FullyConnectedPart::GetLonelyPlans(uint32_t numWeightStripes) const
{
    Plans ret;

    // Fully connected only supports 8x8 block configs
    const BlockConfig blockConfig = { 8u, 8u };

    StripeInfos stripeInfos;
    // Full IFM and Full OFM
    {
        TensorShape mceInputEncoding = { 0, 0, 0, 0 };
        TensorShape mceInputStripe =
            CreateStripe(m_InputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

        TensorShape mceOutputEncoding = { 0, 0, 0, 0 };
        TensorShape mceOutputStripe =
            CreateStripe(m_OutputTensorShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

        TensorShape weightStripe = { 1, 1, GetNumElements(mceInputStripe), GetChannels(mceOutputStripe) };

        NumStripes numStripesInput   = { 1, 1 };
        NumStripes numStripesWeights = { 1, 1 };
        NumStripes numStripesOutput  = { 1, 1 };

        MceAndPleInfo mceAndPleInfo;

        mceAndPleInfo.m_MceCompute.m_Input       = mceInputStripe;
        mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
        mceAndPleInfo.m_MceCompute.m_Weight      = weightStripe;
        mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
        mceAndPleInfo.m_PleCompute.m_Input       = mceOutputStripe;
        mceAndPleInfo.m_PleCompute.m_Output      = mceOutputStripe;
        mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

        mceAndPleInfo.m_Memory.m_Input    = { numStripesInput, mceInputStripe };
        mceAndPleInfo.m_Memory.m_Output   = { numStripesOutput, mceOutputStripe };
        mceAndPleInfo.m_Memory.m_Weight   = { numStripesWeights, weightStripe };
        mceAndPleInfo.m_Memory.m_PleInput = { { 0, 0 }, mceOutputStripe };
        stripeInfos.m_MceAndPleInfos.emplace(mceAndPleInfo);
    }
    // Full IFM and partial OFM
    {
        TensorShape mceInputEncoding = { 0, 0, 0, 0 };
        TensorShape mceInputStripe =
            CreateStripe(m_InputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

        TensorShape mceOutputEncoding = { 0, 0, 0, m_Capabilities.GetNumberOfOgs() };
        TensorShape mceOutputStripe =
            CreateStripe(m_OutputTensorShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

        TensorShape weightStripe = { 1, 1, GetNumElements(mceInputStripe), GetChannels(mceOutputStripe) };

        NumStripes numStripesInput   = { 1, 1 };
        NumStripes numStripesWeights = { 1, 2 };

        uint32_t maxNumStripesOutput = GetChannels(m_OutputTensorShape) > GetChannels(mceOutputStripe) ? 2 : 1;
        NumStripes numStripesOutput  = { 1, maxNumStripesOutput };

        MceAndPleInfo mceAndPleInfo;

        mceAndPleInfo.m_MceCompute.m_Input       = mceInputStripe;
        mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
        mceAndPleInfo.m_MceCompute.m_Weight      = weightStripe;
        mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
        mceAndPleInfo.m_PleCompute.m_Input       = mceOutputStripe;
        mceAndPleInfo.m_PleCompute.m_Output      = mceOutputStripe;
        mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

        mceAndPleInfo.m_Memory.m_Input    = { numStripesInput, mceInputStripe };
        mceAndPleInfo.m_Memory.m_Output   = { numStripesOutput, mceOutputStripe };
        mceAndPleInfo.m_Memory.m_Weight   = { numStripesWeights, weightStripe };
        mceAndPleInfo.m_Memory.m_PleInput = { { 0, 0 }, mceOutputStripe };
        stripeInfos.m_MceAndPleInfos.emplace(mceAndPleInfo);
    }

    // Partial IFM and partial OFM
    {
        TensorShape mceInputEncoding = { 0, 0, 0,
                                         m_Capabilities.GetIgsPerEngine() * m_Capabilities.GetNumberOfEngines() };
        TensorShape mceInputStripe =
            CreateStripe(m_InputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

        TensorShape mceOutputEncoding = { 0, 0, 0, m_Capabilities.GetNumberOfOgs() };
        TensorShape mceOutputStripe =
            CreateStripe(m_OutputTensorShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

        TensorShape weightStripe = { 1, 1, GetNumElements(mceInputStripe), GetChannels(mceOutputStripe) };

        uint32_t maxNumInputStripes = GetChannels(m_InputTensorShape) > GetChannels(mceInputStripe) ? 2 : 1;
        NumStripes numStripesInput  = { 1, maxNumInputStripes };

        uint32_t maxNumStripesOutput = GetChannels(m_OutputTensorShape) > GetChannels(mceOutputStripe) ? 2 : 1;
        NumStripes numStripesOutput  = { 1, maxNumStripesOutput };

        NumStripes numStripesWeights = { 1, 1 };

        MceAndPleInfo mceAndPleInfo;

        mceAndPleInfo.m_MceCompute.m_Input       = mceInputStripe;
        mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
        mceAndPleInfo.m_MceCompute.m_Weight      = weightStripe;
        mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
        mceAndPleInfo.m_PleCompute.m_Input       = mceOutputStripe;
        mceAndPleInfo.m_PleCompute.m_Output      = mceOutputStripe;
        mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

        mceAndPleInfo.m_Memory.m_Input    = { numStripesInput, mceInputStripe };
        mceAndPleInfo.m_Memory.m_Output   = { numStripesOutput, mceOutputStripe };
        mceAndPleInfo.m_Memory.m_Weight   = { numStripesWeights, weightStripe };
        mceAndPleInfo.m_Memory.m_PleInput = { { 0, 0 }, mceOutputStripe };
        stripeInfos.m_MceAndPleInfos.emplace(mceAndPleInfo);
    }

    for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        CreateMceAndIdentityPlePlans(i, TraversalOrder::Xyz, m_WeightEncoderCache, ret, numWeightStripes);
    }
    return ret;
}

}    // namespace support_library
}    // namespace ethosn
