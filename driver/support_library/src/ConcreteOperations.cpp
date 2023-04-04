//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ConcreteOperations.hpp"

#include "Network.hpp"
#include "Utils.hpp"

#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace ethosn
{
namespace support_library
{

namespace
{

template <bool IsTranspose>
uint32_t
    CalcConvolutionOutputSize(const uint32_t inSize, const uint32_t kSize, const uint32_t stride, const uint32_t pad)
{
    if (IsTranspose)
    {
        // This is the inverse calculation of a convolution
        // The input size is what the output size would be in a convolution

        // ((inSize * stride) + kSize) - (stride + pad)
        // Separate positive contribution from negative contribution and use max to make sure we don't overflow
        const uint32_t positive = (inSize * stride) + kSize;
        const uint32_t negative = stride + pad;

        return std::max(positive, negative) - negative;
    }
    else
    {
        // (inSize + stride + pad - kSize) / stride
        // Separate positive contribution from negative contribution and use max to make sure we don't overflow
        const uint32_t positive = inSize + stride + pad;
        const uint32_t negative = kSize;

        return (std::max(positive, negative) - negative) / stride;
    }
}

template <bool IsTranspose>
TensorInfo
    CalcOutputTensorInfo(const TensorInfo& inputInfo, const TensorInfo& weightsInfo, const ConvolutionInfo& convInfo)
{
    const TensorShape& inputShape   = inputInfo.m_Dimensions;
    const TensorShape& weightsShape = weightsInfo.m_Dimensions;

    const uint32_t padY = convInfo.m_Padding.m_Top + convInfo.m_Padding.m_Bottom;
    const uint32_t padX = convInfo.m_Padding.m_Left + convInfo.m_Padding.m_Right;

    TensorShape outputShape = { {
        inputShape[0],
        CalcConvolutionOutputSize<IsTranspose>(inputShape[1], weightsShape[0], convInfo.m_Stride.m_Y, padY),
        CalcConvolutionOutputSize<IsTranspose>(inputShape[2], weightsShape[1], convInfo.m_Stride.m_X, padX),
        weightsShape[3],
    } };

    if (weightsInfo.m_DataFormat == DataFormat::HWIM)
    {
        outputShape[3] *= inputShape[3];
    }

    return TensorInfo(outputShape, inputInfo.m_DataType, inputInfo.m_DataFormat, convInfo.m_OutputQuantizationInfo);
}

}    // namespace

Input::Input(const detail::PosInNetwork pos, uint32_t id, const TensorInfo& info)
    : VisitableOperation<Input>(pos, id, {}, { info })
    , m_Info(info)
{}

Output::Output(const detail::PosInNetwork pos, uint32_t id, Operand& operand, const DataFormat format)
    : VisitableOperation<Output>(pos, id, { &operand }, {})
    , m_OutputFormat(format)
{}

support_library::TensorInfo Output::GetTensorInfo() const
{
    TensorInfo info   = GetInput(0).GetTensorInfo();
    info.m_DataFormat = m_OutputFormat;
    return info;
}

Constant::Constant(const detail::PosInNetwork pos, uint32_t id, const TensorInfo& info, const void* data)
    : VisitableOperation<Constant>(pos, id, {}, { info })
{
    const uint8_t* begin = static_cast<const uint8_t*>(data);
    m_Data.assign(begin, begin + utils::TotalSizeBytes(info));
}

const support_library::TensorInfo& Constant::GetTensorInfo() const
{
    return GetOutput(0).GetTensorInfo();
}

const std::vector<uint8_t>& Constant::GetDataVector() const
{
    return m_Data;
}

Convolution::Convolution(const detail::PosInNetwork pos,
                         uint32_t id,
                         Operand& input,
                         Constant& bias,
                         Constant& weights,
                         const ConvolutionInfo& convInfo)
    : VisitableOperation<Convolution>(
          pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo(), weights.GetTensorInfo(), convInfo) })
    , m_Bias(bias)
    , m_Weights(weights)
    , m_ConvInfo(convInfo)
{}

support_library::TensorInfo Convolution::CalculateOutputTensorInfo(const TensorInfo& inputInfo,
                                                                   const TensorInfo& weightsInfo,
                                                                   const ConvolutionInfo& convInfo)
{
    return CalcOutputTensorInfo<false>(inputInfo, weightsInfo, convInfo);
}

DepthwiseConvolution::DepthwiseConvolution(const detail::PosInNetwork pos,
                                           uint32_t id,
                                           Operand& input,
                                           Constant& bias,
                                           Constant& weights,
                                           const ConvolutionInfo& convInfo)
    : VisitableOperation<DepthwiseConvolution>(
          pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo(), weights.GetTensorInfo(), convInfo) })
    , m_Bias(bias)
    , m_Weights(weights)
    , m_ConvInfo(convInfo)
{}

TensorInfo DepthwiseConvolution::CalculateOutputTensorInfo(const TensorInfo& inputInfo,
                                                           const TensorInfo& weightsInfo,
                                                           const ConvolutionInfo& convInfo)
{
    return CalcOutputTensorInfo<false>(inputInfo, weightsInfo, convInfo);
}

TransposeConvolution::TransposeConvolution(const detail::PosInNetwork pos,
                                           uint32_t id,
                                           Operand& input,
                                           Constant& bias,
                                           Constant& weights,
                                           const ConvolutionInfo& convInfo)
    : VisitableOperation<TransposeConvolution>(
          pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo(), weights.GetTensorInfo(), convInfo) })
    , m_Bias(bias)
    , m_Weights(weights)
    , m_ConvInfo(convInfo)
{}

support_library::TensorInfo TransposeConvolution::CalculateOutputTensorInfo(const TensorInfo& inputInfo,
                                                                            const TensorInfo& weightsInfo,
                                                                            const ConvolutionInfo& convInfo)
{
    return CalcOutputTensorInfo<true>(inputInfo, weightsInfo, convInfo);
}

Addition::Addition(const detail::PosInNetwork pos,
                   uint32_t id,
                   Operand& layer1,
                   Operand& layer2,
                   const QuantizationInfo& outputQuantizationInfo)
    : VisitableOperation<Addition>(
          pos,
          id,
          { &layer1, &layer2 },
          { CalculateOutputTensorInfo(layer1.GetTensorInfo(), layer2.GetTensorInfo(), outputQuantizationInfo) })
{}

TensorInfo Addition::CalculateOutputTensorInfo(const TensorInfo& inputInfo0,
                                               const TensorInfo& inputInfo1,
                                               const QuantizationInfo& outputQuantizationInfo)
{
    TensorShape outputShape;
    for (uint32_t i = 0; i < outputShape.size(); ++i)
    {
        outputShape[i] = std::max(inputInfo0.m_Dimensions[i], inputInfo1.m_Dimensions[i]);
    }
    assert(inputInfo0.m_DataType == inputInfo1.m_DataType);    // Checked by IsAdditionSupported

    TensorInfo outputInfo(outputShape, inputInfo0.m_DataType, DataFormat::NHWC, outputQuantizationInfo);
    return outputInfo;
}

FullyConnected::FullyConnected(const detail::PosInNetwork pos,
                               uint32_t id,
                               Operand& input,
                               Constant& bias,
                               Constant& weights,
                               const FullyConnectedInfo& fullyConnectedInfo)
    : VisitableOperation<FullyConnected>(
          pos,
          id,
          { &input },
          { CalculateOutputTensorInfo(input.GetTensorInfo(), weights.GetTensorInfo(), fullyConnectedInfo) })
    , m_Bias(bias)
    , m_Weights(weights)
    , m_FullyConnectedInfo(fullyConnectedInfo)
{}

TensorInfo FullyConnected::CalculateOutputTensorInfo(const TensorInfo& inputInfo,
                                                     const TensorInfo& weightsInfo,
                                                     const FullyConnectedInfo& fullyConnectedInfo)
{
    return TensorInfo({ inputInfo.m_Dimensions[0], 1, 1, weightsInfo.m_Dimensions[3] }, inputInfo.m_DataType,
                      inputInfo.m_DataFormat, fullyConnectedInfo.m_OutputQuantizationInfo);
}

Relu::Relu(const detail::PosInNetwork pos, uint32_t id, Operand& input, const ReluInfo& reluInfo)
    : VisitableOperation<Relu>(pos, id, { &input }, { input.GetTensorInfo() })
    , m_ReluInfo(reluInfo)
{}

ReinterpretQuantization::ReinterpretQuantization(const detail::PosInNetwork pos,
                                                 uint32_t id,
                                                 Operand& input,
                                                 const ReinterpretQuantizationInfo& reinterpretQuantizationInfo)
    : VisitableOperation<ReinterpretQuantization>(
          pos,
          id,
          { &input },
          { CalculateOutputTensorInfo(input.GetTensorInfo(), reinterpretQuantizationInfo.m_OutputQuantizationInfo) })
{}

TensorInfo
    ReinterpretQuantization::CalculateOutputTensorInfo(const TensorInfo& inputTensorInfo,
                                                       const ReinterpretQuantizationInfo& reinterpretQuantizationInfo)
{
    TensorInfo outputTensorInfo(inputTensorInfo);
    outputTensorInfo.m_QuantizationInfo = reinterpretQuantizationInfo.m_OutputQuantizationInfo;
    return outputTensorInfo;
}

LeakyRelu::LeakyRelu(const detail::PosInNetwork pos, uint32_t id, Operand& input, const LeakyReluInfo& leakyReluInfo)
    : VisitableOperation<LeakyRelu>(
          pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo(), leakyReluInfo) })
    , m_LeakyReluInfo(leakyReluInfo)
{}

TensorInfo LeakyRelu::CalculateOutputTensorInfo(const TensorInfo& inputInfo, const LeakyReluInfo& leakyReluInfo)
{
    TensorInfo outputInfo         = inputInfo;
    outputInfo.m_QuantizationInfo = leakyReluInfo.m_OutputQuantizationInfo;
    return outputInfo;
}

Requantize::Requantize(const detail::PosInNetwork pos,
                       uint32_t id,
                       Operand& input,
                       const RequantizeInfo& requantizeInfo)
    : VisitableOperation<Requantize>(
          pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo(), requantizeInfo) })
    , m_RequantizeInfo(requantizeInfo)
{}

TensorInfo Requantize::CalculateOutputTensorInfo(const TensorInfo& inputInfo, const RequantizeInfo& requantizeInfo)
{
    TensorInfo outputInfo         = inputInfo;
    outputInfo.m_QuantizationInfo = requantizeInfo.m_OutputQuantizationInfo;
    if (requantizeInfo.m_OutputDataType.has_value())
    {
        outputInfo.m_DataType = requantizeInfo.m_OutputDataType.value();
    }
    return outputInfo;
}

Sigmoid::Sigmoid(const detail::PosInNetwork pos, uint32_t id, Operand& input)
    : VisitableOperation<Sigmoid>(pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo()) })
{}

TensorInfo Sigmoid::CalculateOutputTensorInfo(const TensorInfo& inputInfo)
{
    const int32_t zeroPoint = (inputInfo.m_DataType == DataType::INT8_QUANTIZED) ? -128 : 0;

    TensorInfo outInfo         = inputInfo;
    outInfo.m_QuantizationInfo = QuantizationInfo(zeroPoint, 1.f / 256);

    return outInfo;
}

Tanh::Tanh(const detail::PosInNetwork pos, uint32_t id, Operand& input)
    : VisitableOperation<Tanh>(pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo()) })
{}

TensorInfo Tanh::CalculateOutputTensorInfo(const TensorInfo& inputInfo)
{
    const int32_t zeroPoint = (inputInfo.m_DataType == DataType::INT8_QUANTIZED) ? 0 : 128;

    TensorInfo outInfo         = inputInfo;
    outInfo.m_QuantizationInfo = QuantizationInfo(zeroPoint, 1.f / 128);

    return outInfo;
}

Pooling::Pooling(const detail::PosInNetwork pos, uint32_t id, Operand& input, const PoolingInfo& poolingInfo)
    : VisitableOperation<Pooling>(
          pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo(), poolingInfo) })
    , m_PoolingInfo(poolingInfo)
{}

MeanXy::MeanXy(const detail::PosInNetwork pos, uint32_t id, Operand& input)
    : VisitableOperation<MeanXy>(pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo()) })
{}

TensorInfo MeanXy::CalculateOutputTensorInfo(const TensorInfo& inputInfo)
{
    return TensorInfo({ inputInfo.m_Dimensions[0], 1, 1, inputInfo.m_Dimensions[3] }, inputInfo.m_DataType,
                      inputInfo.m_DataFormat, inputInfo.m_QuantizationInfo);
}

TensorInfo Pooling::CalculateOutputTensorInfo(const TensorInfo& inputInfo, const PoolingInfo& poolingInfo)
{
    // clang-format off
    uint32_t h = ((inputInfo.m_Dimensions[1] + poolingInfo.m_Padding.m_Top + poolingInfo.m_Padding.m_Bottom -
                  poolingInfo.m_PoolingSizeY) / poolingInfo.m_PoolingStrideY) + 1;

    uint32_t w = ((inputInfo.m_Dimensions[2] + poolingInfo.m_Padding.m_Left + poolingInfo.m_Padding.m_Right -
                  poolingInfo.m_PoolingSizeX) / poolingInfo.m_PoolingStrideX) + 1;
    // clang-format on
    return TensorInfo({ inputInfo.m_Dimensions[0], h, w, inputInfo.m_Dimensions[3] }, inputInfo.m_DataType,
                      inputInfo.m_DataFormat, inputInfo.m_QuantizationInfo);
}

Reshape::Reshape(const detail::PosInNetwork pos, uint32_t id, Operand& input, const TensorShape& newDimensions)
    : VisitableOperation<Reshape>(
          pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo(), newDimensions) })
    , m_NewDimensions(newDimensions)
{}

TensorInfo Reshape::CalculateOutputTensorInfo(const TensorInfo& inputInfo, const TensorShape& newDimensions)
{
    return TensorInfo(newDimensions, inputInfo.m_DataType, inputInfo.m_DataFormat, inputInfo.m_QuantizationInfo);
}

Concatenation::Concatenation(const detail::PosInNetwork pos,
                             uint32_t id,
                             const std::vector<Operand*>& inputs,
                             const ConcatenationInfo& concatInfo)
    : VisitableOperation<Concatenation>(
          pos,
          id,
          inputs,
          { CalculateOutputTensorInfo(utils::Map<TensorInfo>(inputs, [](Operand* x) { return x->GetTensorInfo(); }),
                                      concatInfo) })
    , m_ConcatInfo(concatInfo)
{}

TensorInfo Concatenation::CalculateOutputTensorInfo(const std::vector<TensorInfo>& inputInfos,
                                                    const ConcatenationInfo& concatInfo)
{
    size_t numInputs = inputInfos.size();
    assert(numInputs > 0);
    TensorInfo outputInfo                      = inputInfos[0];
    outputInfo.m_Dimensions[concatInfo.m_Axis] = 0;

    for (uint32_t i = 0; i < numInputs; ++i)
    {
        outputInfo.m_Dimensions[concatInfo.m_Axis] += inputInfos[i].m_Dimensions[concatInfo.m_Axis];
    }

    outputInfo.m_QuantizationInfo = concatInfo.m_OutputQuantizationInfo;
    return outputInfo;
}

Split::Split(const detail::PosInNetwork pos, uint32_t id, Operand& input, const SplitInfo& splitInfo)
    : VisitableOperation<Split>(pos, id, { &input }, CalculateOutputTensorInfos(input.GetTensorInfo(), splitInfo))
    , m_SplitInfo(splitInfo)
{}

std::vector<TensorInfo> Split::CalculateOutputTensorInfos(const TensorInfo& inputInfo, const SplitInfo& splitInfo)
{
    std::vector<TensorInfo> result;
    for (uint32_t i = 0; i < splitInfo.m_Sizes.size(); ++i)
    {
        TensorInfo outputInfo                     = inputInfo;
        outputInfo.m_Dimensions[splitInfo.m_Axis] = splitInfo.m_Sizes[i];
        result.push_back(outputInfo);
    }
    return result;
}

DepthToSpace::DepthToSpace(const detail::PosInNetwork pos,
                           uint32_t id,
                           Operand& input,
                           const DepthToSpaceInfo& depthToSpaceInfo)
    : VisitableOperation<DepthToSpace>(
          pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo(), depthToSpaceInfo) })
    , m_DepthToSpaceInfo(depthToSpaceInfo)
{}

TensorInfo DepthToSpace::CalculateOutputTensorInfo(const TensorInfo& inputInfo,
                                                   const DepthToSpaceInfo& depthToSpaceInfo)
{
    TensorInfo result      = inputInfo;
    uint32_t blockSize     = depthToSpaceInfo.m_BlockSize;
    result.m_Dimensions[1] = inputInfo.m_Dimensions[1] * blockSize;
    result.m_Dimensions[2] = inputInfo.m_Dimensions[2] * blockSize;
    assert(inputInfo.m_Dimensions[3] % (blockSize * blockSize) == 0);    // Checked by IsDepthToSpaceSupported
    result.m_Dimensions[3] = inputInfo.m_Dimensions[3] / (blockSize * blockSize);
    return result;
}

SpaceToDepth::SpaceToDepth(const detail::PosInNetwork pos,
                           uint32_t id,
                           Operand& input,
                           const SpaceToDepthInfo& spaceToDepthInfo)
    : VisitableOperation<SpaceToDepth>(
          pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo(), spaceToDepthInfo) })
    , m_SpaceToDepthInfo(spaceToDepthInfo)
{}

TensorInfo SpaceToDepth::CalculateOutputTensorInfo(const TensorInfo& inputInfo,
                                                   const SpaceToDepthInfo& spaceToDepthInfo)
{
    TensorInfo result  = inputInfo;
    uint32_t blockSize = spaceToDepthInfo.m_BlockSize;
    assert(inputInfo.m_Dimensions[1] % blockSize == 0 && inputInfo.m_Dimensions[2] % blockSize == 0);
    result.m_Dimensions[1] = inputInfo.m_Dimensions[1] / blockSize;
    result.m_Dimensions[2] = inputInfo.m_Dimensions[2] / blockSize;
    result.m_Dimensions[3] = inputInfo.m_Dimensions[3] * blockSize * blockSize;
    return result;
}

Transpose::Transpose(const detail::PosInNetwork pos, uint32_t id, Operand& input, const TransposeInfo& transposeInfo)
    : VisitableOperation<Transpose>(
          pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo(), transposeInfo) })
    , m_TransposeInfo(transposeInfo)
{}

TensorInfo Transpose::CalculateOutputTensorInfo(const TensorInfo& inputInfo, const TransposeInfo& transposeInfo)
{
    TensorInfo result      = inputInfo;
    auto& permutation      = transposeInfo.m_Permutation;
    result.m_Dimensions[1] = inputInfo.m_Dimensions[permutation[1]];
    result.m_Dimensions[2] = inputInfo.m_Dimensions[permutation[2]];
    result.m_Dimensions[3] = inputInfo.m_Dimensions[permutation[3]];
    return result;
}

Resize::Resize(const detail::PosInNetwork pos, uint32_t id, Operand& input, const ResizeInfo& resizeInfo)
    : VisitableOperation<Resize>(pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo(), resizeInfo) })
    , m_ResizeInfo(resizeInfo)
{}

TensorInfo Resize::CalculateOutputTensorInfo(const TensorInfo& inputInfo, const ResizeInfo& resizeInfo)
{
    TensorInfo outputInfo         = inputInfo;
    outputInfo.m_Dimensions[1]    = resizeInfo.m_NewHeight;
    outputInfo.m_Dimensions[2]    = resizeInfo.m_NewWidth;
    outputInfo.m_QuantizationInfo = resizeInfo.m_OutputQuantizationInfo;
    return outputInfo;
}

EstimateOnly::EstimateOnly(const detail::PosInNetwork pos,
                           uint32_t id,
                           const std::vector<Operand*>& inputs,
                           const EstimateOnlyInfo& info)
    : VisitableOperation<EstimateOnly>(pos, id, inputs, info.m_OutputInfos)
    , m_EstimateOnlyInfo(info)
{}

}    // namespace support_library

}    // namespace ethosn
