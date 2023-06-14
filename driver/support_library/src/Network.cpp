//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Network.hpp"

#include "ConcreteOperations.hpp"
#include "Utils.hpp"

#include <algorithm>
#include <cstring>

namespace ethosn
{
namespace support_library
{

Constant& Network::AddConstant(const TensorInfo& info, const void* data)
{
    char reason[1024];
    SupportedLevel supportedLevel = m_Queries.IsConstantSupported(info, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }
    return AddOperation<Constant>(info, data);
}

Convolution& Network::AddConvolution(Operand& input, Constant& bias, Constant& weights, const ConvolutionInfo& convInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsConvolutionSupported(bias.GetTensorInfo(), weights.GetTensorInfo(), convInfo, input.GetTensorInfo(),
                                         nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Convolution>(input, bias, weights, convInfo);
}

DepthwiseConvolution&
    Network::AddDepthwiseConvolution(Operand& input, Constant& bias, Constant& weights, const ConvolutionInfo& convInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsDepthwiseConvolutionSupported(bias.GetTensorInfo(), weights.GetTensorInfo(), convInfo,
                                                  input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<DepthwiseConvolution>(input, bias, weights, convInfo);
}

TransposeConvolution&
    Network::AddTransposeConvolution(Operand& input, Constant& bias, Constant& weights, const ConvolutionInfo& convInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsTransposeConvolutionSupported(bias.GetTensorInfo(), weights.GetTensorInfo(), convInfo,
                                                  input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<TransposeConvolution>(input, bias, weights, convInfo);
}

Concatenation& Network::AddConcatenation(const std::vector<Operand*>& inputs, const ConcatenationInfo& concatInfo)
{
    char reason[1024];

    SupportedLevel supportedLevel = m_Queries.IsConcatenationSupported(
        utils::Map<TensorInfo>(inputs, [](Operand* x) { return x->GetTensorInfo(); }), concatInfo, nullptr, reason,
        sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Concatenation>(inputs, concatInfo);
}

Split& Network::AddSplit(Operand& input, const SplitInfo& splitInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsSplitSupported(input.GetTensorInfo(), splitInfo, nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Split>(input, splitInfo);
}

Addition& Network::AddAddition(Operand& layer1, Operand& layer2, const QuantizationInfo& outputQuantizationInfo)
{
    char reason[1024];

    SupportedLevel supportedLevel = m_Queries.IsAdditionSupported(
        layer1.GetTensorInfo(), layer2.GetTensorInfo(), outputQuantizationInfo, nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Addition>(layer1, layer2, outputQuantizationInfo);
}

FullyConnected& Network::AddFullyConnected(Operand& input,
                                           Constant& bias,
                                           Constant& weights,
                                           const FullyConnectedInfo fullyConnectedInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsFullyConnectedSupported(bias.GetTensorInfo(), weights.GetTensorInfo(), fullyConnectedInfo,
                                            input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<FullyConnected>(input, bias, weights, fullyConnectedInfo);
}

ReinterpretQuantization&
    Network::AddReinterpretQuantization(Operand& input, const ReinterpretQuantizationInfo& reinterpretQuantizationInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel = m_Queries.IsReinterpretQuantizationSupported(
        reinterpretQuantizationInfo, input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<ReinterpretQuantization>(input, reinterpretQuantizationInfo);
}

Relu& Network::AddRelu(Operand& input, const ReluInfo& reluInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsReluSupported(reluInfo, input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Relu>(input, reluInfo);
}

LeakyRelu& Network::AddLeakyRelu(Operand& input, const LeakyReluInfo& leakyReluInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsLeakyReluSupported(leakyReluInfo, input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<LeakyRelu>(input, leakyReluInfo);
}

Requantize& Network::AddRequantize(Operand& input, const RequantizeInfo& requantizeInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsRequantizeSupported(requantizeInfo, input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Requantize>(input, requantizeInfo);
}

Sigmoid& Network::AddSigmoid(Operand& input)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsSigmoidSupported(input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Sigmoid>(input);
}

Tanh& Network::AddTanh(Operand& input)
{
    char reason[1024];
    SupportedLevel supportedLevel = m_Queries.IsTanhSupported(input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Tanh>(input);
}

MeanXy& Network::AddMeanXy(Operand& input)
{
    char reason[1024];
    SupportedLevel supportedLevel = m_Queries.IsMeanXySupported(input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<MeanXy>(input);
}

Pooling& Network::AddPooling(Operand& input, const PoolingInfo& poolingInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsPoolingSupported(poolingInfo, input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Pooling>(input, poolingInfo);
}

Reshape& Network::AddReshape(Operand& input, const TensorShape& newDimensions)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsReshapeSupported(newDimensions, input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Reshape>(input, newDimensions);
}

DepthToSpace& Network::AddDepthToSpace(Operand& input, const DepthToSpaceInfo& depthToSpaceInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsDepthToSpaceSupported(input.GetTensorInfo(), depthToSpaceInfo, nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<DepthToSpace>(input, depthToSpaceInfo);
}

SpaceToDepth& Network::AddSpaceToDepth(Operand& input, const SpaceToDepthInfo& spaceToDepthInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsSpaceToDepthSupported(input.GetTensorInfo(), spaceToDepthInfo, nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<SpaceToDepth>(input, spaceToDepthInfo);
}

Transpose& Network::AddTranspose(Operand& input, const TransposeInfo& transposeInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsTransposeSupported(transposeInfo, input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Transpose>(input, transposeInfo);
}

Resize& Network::AddResize(Operand& input, const ResizeInfo& resizeInfo)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsResizeSupported(resizeInfo, input.GetTensorInfo(), nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Resize>(input, resizeInfo);
}

EstimateOnly& Network::AddEstimateOnly(const std::vector<Operand*>& inputs, const EstimateOnlyInfo& estimateOnly)
{
    char reason[1024];

    std::vector<TensorInfo> inputTensorInfos;
    inputTensorInfos.reserve(inputs.size());
    for (auto input : inputs)
    {
        inputTensorInfos.push_back(input->GetTensorInfo());
    }
    SupportedLevel supportedLevel =
        m_Queries.IsEstimateOnlySupported(inputTensorInfos, estimateOnly, nullptr, reason, sizeof(reason));

    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<EstimateOnly>(inputs, estimateOnly);
}

Input& Network::AddInput(const TensorInfo& info)
{
    char reason[1024];
    SupportedLevel supportedLevel = m_Queries.IsInputSupported(info, nullptr, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperationWithId<Input>(info);
}

Output& Network::AddOutput(Operand& operand, const DataFormat format)
{
    char reason[1024];
    SupportedLevel supportedLevel =
        m_Queries.IsOutputSupported(operand.GetTensorInfo(), format, reason, sizeof(reason));
    if (!CheckSupportedLevel(supportedLevel))
    {
        throw NotSupportedException(reason);
    }

    return AddOperation<Output>(operand, format);
}

}    // namespace support_library
}    // namespace ethosn
