//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/SupportQueries.hpp"

#include "Network.hpp"
#include "Utils.hpp"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdarg.h>
#include <string>
#include <tuple>
#include <unordered_set>

namespace ethosn
{
namespace support_library
{

namespace
{

static const std::unordered_set<uint32_t> g_ConvolutionKernelSizes = { 1, 2, 3, 5, 7, 9 };

void SetReason(const char* reasonFull, char* reasonTruncated, size_t maxLength, ...)
{
    if (reasonTruncated)
    {
        va_list args;
        va_start(args, maxLength);
        vsnprintf(reasonTruncated, maxLength, reasonFull, args);
        va_end(args);
    }
}

constexpr std::pair<uint32_t, uint32_t>
    CalcSamePadding(const uint32_t inputSize, const uint32_t kernelSize, const uint32_t stride, const bool preferBefore)
{
    const uint32_t paddedSize = ((utils::DivRoundUp(inputSize, stride) - 1U) * stride) + kernelSize;
    const uint32_t padSize    = (paddedSize > inputSize) ? paddedSize - inputSize : 0U;

    const uint32_t pad0 = utils::DivRoundUp(padSize, 2U);
    const uint32_t pad1 = padSize - pad0;

    const uint32_t padBefore = preferBefore ? pad0 : pad1;
    const uint32_t padAfter  = preferBefore ? pad1 : pad0;

    return { padBefore, padAfter };
}

constexpr Padding CalcSamePadding(const TensorShape& inputShape,
                                  const TensorShape& weightsShape,
                                  const Stride& stride,
                                  const bool preferBefore)
{
    const std::pair<uint32_t, uint32_t> padY =
        CalcSamePadding(inputShape[1], weightsShape[0], stride.m_Y, preferBefore);
    const std::pair<uint32_t, uint32_t> padX =
        CalcSamePadding(inputShape[2], weightsShape[1], stride.m_X, preferBefore);

    Padding pad;
    pad.m_Top    = padY.first;
    pad.m_Bottom = padY.second;
    pad.m_Left   = padX.first;
    pad.m_Right  = padX.second;

    return pad;
}

constexpr Padding CalcSamePadding(const TensorInfo& inputInfo,
                                  const TensorInfo& weightsInfo,
                                  const Stride& stride,
                                  const bool preferBefore)
{
    return CalcSamePadding(inputInfo.m_Dimensions, weightsInfo.m_Dimensions, stride, preferBefore);
}

}    // namespace

const SupportedLevel SupportedLevel::Unsupported  = SupportedLevel(InternalSupportedLevel::Unsupported);
const SupportedLevel SupportedLevel::EstimateOnly = SupportedLevel(InternalSupportedLevel::EstimateOnly);
const SupportedLevel SupportedLevel::Supported    = SupportedLevel(InternalSupportedLevel::Supported);

SupportedLevel
    IsInputSupported(const TensorInfo& inputInfo, TensorInfo* outputInfo, char* reason, size_t reasonMaxLength)
{
    if (inputInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Input layer must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input layer must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        TensorInfo expectedOutputInfo = inputInfo;
        if (utils::TotalSizeBytes(*outputInfo) != 0 && *outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    return SupportedLevel::Supported;
}

SupportedLevel
    IsOutputSupported(const TensorInfo& inputInfo, const DataFormat format, char* reason, size_t reasonMaxLength)
{
    if (inputInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("An Output layer's input must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("An Output layer's input must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (format != DataFormat::NHWC && format != DataFormat::NHWCB)
    {
        SetReason("An Output layer's format must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsConstantSupported(const TensorInfo&, char*, size_t)
{
    return SupportedLevel::Supported;
}

SupportedLevel IsConvolutionSupported(const TensorInfo& biasInfo,
                                      const TensorInfo& weightsInfo,
                                      const ConvolutionInfo& convInfo,
                                      const TensorInfo& inputInfo,
                                      TensorInfo* outputInfo,
                                      char* reason,
                                      size_t reasonMaxLength)
{
    if (inputInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Input to conv must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input to conv must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Weights for conv must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_DataFormat != DataFormat::HWIO)
    {
        SetReason("Weights for conv must be HWIO", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (biasInfo.m_DataType != DataType::INT32_QUANTIZED)
    {
        SetReason("Bias for conv must be INT32_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (biasInfo.m_DataFormat != DataFormat::NHWC)
    {
        SetReason("Bias for conv must be NHWC", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if ((biasInfo.m_Dimensions[0] * biasInfo.m_Dimensions[1] * biasInfo.m_Dimensions[2] != 1) ||
        biasInfo.m_Dimensions[3] != weightsInfo.m_Dimensions[3])
    {
        SetReason("Invalid bias tensor dimensions", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_Dimensions[2] != inputInfo.m_Dimensions[3])
    {
        SetReason("Weights input channels dimension (I) must match Input channels dimension (C)", reason,
                  reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    const uint32_t kernelHeight = weightsInfo.m_Dimensions[0];
    const uint32_t kernelWidth  = weightsInfo.m_Dimensions[1];

    if (kernelHeight == 0 || kernelWidth == 0 || convInfo.m_Stride.m_X == 0 || convInfo.m_Stride.m_Y == 0)
    {
        SetReason("Invalid kernel/stride parameters", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    TensorInfo expectedOutputInfo = Convolution::CalculateOutputTensorInfo(inputInfo, weightsInfo, convInfo);
    if (utils::GetNumElements(expectedOutputInfo.m_Dimensions) == 0)
    {
        SetReason("Output tensor would be empty", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        if ((utils::TotalSizeBytes(*outputInfo) != 0) && (*outputInfo != expectedOutputInfo))
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    if (weightsInfo.m_QuantizationInfo.m_ZeroPoint > UINT8_MAX)
    {
        SetReason("Zero point value above allowed range", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if (weightsInfo.m_QuantizationInfo.m_ZeroPoint < 0)
    {
        SetReason("Zero point value below allowed range", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if (biasInfo.m_QuantizationInfo.m_ZeroPoint != 0 ||
        biasInfo.m_QuantizationInfo.m_Scale !=
            inputInfo.m_QuantizationInfo.m_Scale * weightsInfo.m_QuantizationInfo.m_Scale)
    {
        SetReason("Bias for conv must have quantization parameters with zero point of 0 and scale of input scale x "
                  "weight scale",
                  reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    static const std::unordered_set<uint32_t> validStrides = { 1, 2 };

    if ((g_ConvolutionKernelSizes.count(kernelHeight) == 0U) || (g_ConvolutionKernelSizes.count(kernelWidth) == 0U))
    {
        SetReason("Unsupported kernel size. Width/height must be in { 1, 2, 3, 5, 7, 9 }", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if ((convInfo.m_Stride.m_X != convInfo.m_Stride.m_Y) || (validStrides.count(convInfo.m_Stride.m_X) == 0U))
    {
        SetReason("Unsupported stride. Stride X and Y must be equal and in { 1, 2 }", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if ((convInfo.m_Stride.m_X > 1U && (kernelHeight > 7U || kernelWidth > 7U)))
    {
        SetReason("Unsupported stride for kernel width/height > 7. Stride X and Y must be 1", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if ((convInfo.m_Padding != Padding(0, 0, 0, 0)) &&
        (convInfo.m_Padding != CalcSamePadding(inputInfo, weightsInfo, convInfo.m_Stride, false)) &&
        (convInfo.m_Padding != CalcSamePadding(inputInfo, weightsInfo, convInfo.m_Stride, true)))
    {
        SetReason("Unsupported padding.", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    double overallScale = inputInfo.m_QuantizationInfo.m_Scale * weightsInfo.m_QuantizationInfo.m_Scale /
                          convInfo.m_OutputQuantizationInfo.m_Scale;
    if (overallScale < 0.0f || overallScale >= 1.0f)
    {
        SetReason("Overall scale (of the input * weights / output) should be in the range [0, 1)", reason,
                  reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsDepthwiseConvolutionSupported(const TensorInfo& biasInfo,
                                               const TensorInfo& weightsInfo,
                                               const ConvolutionInfo& convInfo,
                                               const TensorInfo& inputInfo,
                                               TensorInfo* outputInfo,
                                               char* reason,
                                               size_t reasonMaxLength)
{
    if (inputInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Input to depthwise conv must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input to depthwise conv must be NHWC OR NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Weights for depthwise conv must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_DataFormat != DataFormat::HWIM)
    {
        SetReason("Weights for depthwise conv must be HWIM", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (biasInfo.m_DataType != DataType::INT32_QUANTIZED)
    {
        SetReason("Bias for depthwise conv must be INT32_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (biasInfo.m_DataFormat != DataFormat::NHWC)
    {
        SetReason("Bias for depthwise conv must be NHWC", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if ((biasInfo.m_Dimensions[0] * biasInfo.m_Dimensions[1] * biasInfo.m_Dimensions[2] != 1) ||
        biasInfo.m_Dimensions[3] != weightsInfo.m_Dimensions[2] * weightsInfo.m_Dimensions[3])
    {
        SetReason("Invalid bias tensor dimensions", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_Dimensions[2] != inputInfo.m_Dimensions[3])
    {
        SetReason("Weights input channels dimension (I) must match Input channels dimension (C)", reason,
                  reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    const uint32_t kernelHeight = weightsInfo.m_Dimensions[0];
    const uint32_t kernelWidth  = weightsInfo.m_Dimensions[1];

    if (kernelHeight == 0 || kernelWidth == 0 || convInfo.m_Stride.m_X == 0 || convInfo.m_Stride.m_Y == 0)
    {
        SetReason("Invalid kernel/stride parameters", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    TensorInfo expectedOutputInfo = DepthwiseConvolution::CalculateOutputTensorInfo(inputInfo, weightsInfo, convInfo);
    if (utils::GetNumElements(expectedOutputInfo.m_Dimensions) == 0)
    {
        SetReason("Output tensor would be empty", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        if (utils::TotalSizeBytes(*outputInfo) != 0 && *outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    // We support channel multiplier > 1, if there is only 1 input channel as this can be converted to a normal convolution
    if (weightsInfo.m_Dimensions[3] != 1 && weightsInfo.m_Dimensions[2] != 1)
    {
        SetReason("If channel multiplier > 1 the weights input channels dimension must be 1", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if (weightsInfo.m_QuantizationInfo.m_ZeroPoint > UINT8_MAX)
    {
        SetReason("Zero point value above allowed range", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if (weightsInfo.m_QuantizationInfo.m_ZeroPoint < 0)
    {
        SetReason("Zero point value below allowed range", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if (biasInfo.m_QuantizationInfo.m_ZeroPoint != 0 ||
        biasInfo.m_QuantizationInfo.m_Scale !=
            inputInfo.m_QuantizationInfo.m_Scale * weightsInfo.m_QuantizationInfo.m_Scale)
    {
        SetReason("Bias for depthwise conv must have quantization parameters with zero point of 0 and scale of "
                  "input scale x weight scale",
                  reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    static const std::unordered_set<uint32_t> validStrides = { 1, 2 };

    if ((kernelHeight != kernelWidth) || (g_ConvolutionKernelSizes.count(kernelHeight) == 0U))
    {
        SetReason("Unsupported kernel size. Width/height must be in { 1, 2, 3, 5, 7, 9 }", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if ((convInfo.m_Stride.m_X != convInfo.m_Stride.m_Y) || (validStrides.count(convInfo.m_Stride.m_X) == 0U))
    {
        SetReason("Unsupported stride. Stride X and Y must be equal and in { 1, 2 }", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if ((convInfo.m_Stride.m_X != 1) && (kernelHeight == 1U) && (kernelWidth == 1U))
    {
        SetReason("Unsupported stride >1 with kernel size 1x1.", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if ((convInfo.m_Padding != Padding{}) &&
        (convInfo.m_Padding != CalcSamePadding(inputInfo, weightsInfo, convInfo.m_Stride, false)) &&
        (convInfo.m_Padding != CalcSamePadding(inputInfo, weightsInfo, convInfo.m_Stride, true)))
    {
        SetReason("Unsupported padding.", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    double overallScale = inputInfo.m_QuantizationInfo.m_Scale * weightsInfo.m_QuantizationInfo.m_Scale /
                          convInfo.m_OutputQuantizationInfo.m_Scale;
    if (overallScale < 0.0f || overallScale >= 1.0f)
    {
        SetReason("Overall scale (of the input * weights / output) should be in the range [0, 1)", reason,
                  reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsTransposeConvolutionSupported(const TensorInfo& biasInfo,
                                               const TensorInfo& weightsInfo,
                                               const ConvolutionInfo& convInfo,
                                               const TensorInfo& inputInfo,
                                               TensorInfo* outputInfo,
                                               char* reason,
                                               size_t reasonMaxLength)
{
    if (inputInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Input to transpose conv must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input to transpose conv must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Weights for transpose conv must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_DataFormat != DataFormat::HWIO)
    {
        SetReason("Weights for transpose conv must be HWIO", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (biasInfo.m_DataType != DataType::INT32_QUANTIZED)
    {
        SetReason("Bias for transpose conv must be INT32_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (biasInfo.m_DataFormat != DataFormat::NHWC)
    {
        SetReason("Bias for transpose conv must be NHWC", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if ((biasInfo.m_Dimensions[0] * biasInfo.m_Dimensions[1] * biasInfo.m_Dimensions[2] != 1) ||
        biasInfo.m_Dimensions[3] != weightsInfo.m_Dimensions[3])
    {
        SetReason("Invalid bias tensor dimensions", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_Dimensions[2] != inputInfo.m_Dimensions[3])
    {
        SetReason("Weights input channels dimension (I) must match Input channels dimension (C)", reason,
                  reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    const uint32_t kernelHeight = weightsInfo.m_Dimensions[0];
    const uint32_t kernelWidth  = weightsInfo.m_Dimensions[1];

    if (kernelHeight == 0 || kernelWidth == 0 || convInfo.m_Stride.m_X == 0 || convInfo.m_Stride.m_Y == 0)
    {
        SetReason("Invalid kernel/stride parameters", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    TensorInfo expectedOutputInfo = TransposeConvolution::CalculateOutputTensorInfo(inputInfo, weightsInfo, convInfo);
    if (utils::GetNumElements(expectedOutputInfo.m_Dimensions) == 0)
    {
        SetReason("Output tensor would be empty", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        if ((utils::TotalSizeBytes(*outputInfo) != 0) && (*outputInfo != expectedOutputInfo))
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    if (weightsInfo.m_QuantizationInfo.m_ZeroPoint > UINT8_MAX || weightsInfo.m_QuantizationInfo.m_ZeroPoint < 0)
    {
        SetReason("Zero point value outside allowed range (0-255)", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if (biasInfo.m_QuantizationInfo.m_ZeroPoint != 0 ||
        biasInfo.m_QuantizationInfo.m_Scale !=
            inputInfo.m_QuantizationInfo.m_Scale * weightsInfo.m_QuantizationInfo.m_Scale)
    {
        SetReason("Bias for transpose conv must have quantization parameters with zero point of 0 and "
                  "scale of input scale x weight scale",
                  reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    static const std::unordered_set<uint32_t> validStrides = { 2 };

    if ((g_ConvolutionKernelSizes.count(kernelHeight) == 0U) || (g_ConvolutionKernelSizes.count(kernelWidth) == 0U))
    {
        SetReason("Unsupported kernel size. Width/height must be in { 1, 2, 3, 5, 7, 9 }", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if ((convInfo.m_Stride.m_X != convInfo.m_Stride.m_Y) || (validStrides.count(convInfo.m_Stride.m_X) == 0U))
    {
        SetReason("Unsupported stride. Stride X and Y must be equal to 2", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    // Check that padding is either SAME or VALID. To calculate what SAME padding means, we first calculate the output
    // size and then use that to calculate what SAME padding would be for a regular convolution.
    TensorShape outputShape = expectedOutputInfo.m_Dimensions;
    if ((convInfo.m_Padding != Padding{}) &&
        (convInfo.m_Padding != CalcSamePadding(outputShape, weightsInfo, convInfo.m_Stride, false)) &&
        (convInfo.m_Padding != CalcSamePadding(outputShape, weightsInfo, convInfo.m_Stride, true)))
    {
        SetReason("Unsupported padding.", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    // Padding must be SAME when the kernel is > 7x7
    if ((convInfo.m_Padding == Padding{}) && (kernelHeight > 7 || kernelWidth > 7))
    {
        SetReason("Padding must be SAME for kernel > 7x7.", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    double overallScale = inputInfo.m_QuantizationInfo.m_Scale * weightsInfo.m_QuantizationInfo.m_Scale /
                          convInfo.m_OutputQuantizationInfo.m_Scale;
    if (overallScale < 0.0f || overallScale >= 1.0f)
    {
        SetReason("Overall scale (of the input * weights / output) should be in the range [0, 1)", reason,
                  reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsConcatenationSupported(const std::vector<TensorInfo>& inputInfos,
                                        const ConcatenationInfo& concatInfo,
                                        TensorInfo* outputInfo,
                                        char* reason,
                                        size_t reasonMaxLength)
{
    size_t numInputs = inputInfos.size();
    if (numInputs < 1)
    {
        SetReason("Must have at least one input", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    for (uint32_t i = 0; i < numInputs; ++i)
    {
        if (inputInfos[i].m_DataType != DataType::UINT8_QUANTIZED)
        {
            SetReason("Input tensors must have data type UINT8_QUANTIZED", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        if (inputInfos[i].m_DataFormat != DataFormat::NHWC && inputInfos[i].m_DataFormat != DataFormat::NHWCB)
        {
            SetReason("Input to concatenation must be NHWC or NHWCB", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
    }

    if (concatInfo.m_Axis >= 4)
    {
        SetReason("Concatenation axis must refer to a valid dimension (0-3)", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    // All inputs must have the same dimensions in all except the dimension that we are concatenating along
    for (uint32_t i = 0; i < numInputs; ++i)
    {
        for (uint32_t dim = 0; dim < 4; ++dim)
        {
            if (dim == concatInfo.m_Axis)
            {
                continue;
            }
            if (inputInfos[i].m_Dimensions[dim] != inputInfos[0].m_Dimensions[dim])
            {
                SetReason(
                    "Input tensors must have the same size along all dimensions except the concatenation dimension",
                    reason, reasonMaxLength);
                return SupportedLevel::Unsupported;
            }
        }
    }

    if (outputInfo != nullptr)
    {
        TensorInfo expectedOutputInfo = Concatenation::CalculateOutputTensorInfo(inputInfos, concatInfo);
        if (utils::TotalSizeBytes(*outputInfo) != 0 && *outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    switch (concatInfo.m_Axis)
    {
        case 0:
            SetReason("Concatenation cannot be performed along batch axis (axis 0)", reason, reasonMaxLength);
            return SupportedLevel::EstimateOnly;
        case 1:
            // Deliberate fallthrough
        case 2:
            // Concat along width and height can always be performed by building up the tensor in DRAM
            // using NHWC.
            break;
        case 3:
            // Concatenation along channels can only be performed by building up the tensor in DRAM
            // using NHWCB and therefore the channels dimensions of the input tensors must be suitable for DMAing.
            // A conservative test is multiple of 16, although we could probably support other cases too.
            for (uint32_t i = 0; i < numInputs; ++i)
            {
                if (inputInfos[i].m_Dimensions[3] % 16 != 0)
                {
                    SetReason("Concatenation along the channels dimension (axis 3) requires input tensors with a "
                              "multiple of 16 channels",
                              reason, reasonMaxLength);
                    return SupportedLevel::EstimateOnly;
                }
            }
            break;
        default:
            assert(false);
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsSplitSupported(const TensorInfo& inputInfo,
                                const SplitInfo& splitInfo,
                                std::vector<TensorInfo>* outputInfos,
                                char* reason,
                                size_t reasonMaxLength)
{
    size_t numOutputs = splitInfo.m_Sizes.size();
    if (numOutputs < 1)
    {
        SetReason("Must have at least 1 output", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Input tensor must have data type UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }
    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input tensor must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (splitInfo.m_Axis >= 4)
    {
        SetReason("Axis must refer to a valid dimension (0-3)", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    // Provided sizes must sum to the total along the axis.
    if (std::accumulate(splitInfo.m_Sizes.begin(), splitInfo.m_Sizes.end(), 0u) !=
        inputInfo.m_Dimensions[splitInfo.m_Axis])
    {
        SetReason("Sizes must sum to the total size of the input tensor along the split axis", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfos != nullptr)
    {
        if (outputInfos->size() != numOutputs)
        {
            SetReason("Provided outputInfos array has incorrect size", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }

        std::vector<TensorInfo> expectedOutputInfos = Split::CalculateOutputTensorInfos(inputInfo, splitInfo);
        for (uint32_t i = 0; i < numOutputs; ++i)
        {
            if (utils::TotalSizeBytes((*outputInfos)[i]) != 0 && (*outputInfos)[i] != expectedOutputInfos[i])
            {
                SetReason("Provided outputInfo at index %u is incorrect", reason, reasonMaxLength, i);
                return SupportedLevel::Unsupported;
            }
            (*outputInfos)[i] = expectedOutputInfos[i];
        }
    }

    switch (splitInfo.m_Axis)
    {
        case 0:
            SetReason("Split cannot be performed along batch axis (axis 0)", reason, reasonMaxLength);
            return SupportedLevel::EstimateOnly;
        case 1:
            // Deliberate fallthrough
        case 2:
            // Split along width and height can always be performed by extracting subtensors from DRAM using NHWC.
            break;
        case 3:
            // Split along channels can only be performed by extracting subtensors from a tensor in DRAM using NHWCB
            // and therefore the channels dimensions of the output tensors must be suitable for DMAing.
            // A conservative test is multiple of 16, although we could probably support other cases too.
            for (uint32_t i = 0; i < numOutputs; ++i)
            {
                if (splitInfo.m_Sizes[i] % 16 != 0)
                {
                    SetReason("Split along the channels dimension (axis 3) requires all output sizes (specified in "
                              "splitInfo.m_Sizes) to be multiples of 16",
                              reason, reasonMaxLength);
                    return SupportedLevel::EstimateOnly;
                }
            }
            break;
        default:
            assert(false);
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsAdditionSupported(const TensorInfo& inputInfo0,
                                   const TensorInfo& inputInfo1,
                                   const QuantizationInfo& outputQuantizationInfo,
                                   TensorInfo* outputInfo,
                                   char* reason,
                                   size_t reasonMaxLength)
{
    const TensorShape& shape0 = inputInfo0.m_Dimensions;
    const TensorShape& shape1 = inputInfo1.m_Dimensions;
    const bool isDim1Equal    = shape0[1] == shape1[1];
    const bool isDim2Equal    = shape0[2] == shape1[2];
    const bool isDim3Equal    = shape0[3] == shape1[3];

    // To be able to stretch along a dimension. The dimension size in one of the tensors must be 1.
    const bool canStretchDim1 = shape0[1] == 1 || shape1[1] == 1;
    const bool canStretchDim2 = shape0[2] == 1 || shape1[2] == 1;
    const bool canStretchDim3 = shape0[3] == 1 || shape1[3] == 1;

    const bool isDim1Compatible = isDim1Equal || canStretchDim1;
    const bool isDim2Compatible = isDim2Equal || canStretchDim2;
    const bool isDim3Compatible = isDim3Equal || canStretchDim3;

    // From the AndroidNN spec:
    // Two dimensions are compatible when:
    //  they are equal, or
    //  one of them is 1
    if (!isDim1Compatible)
    {
        SetReason("Height must be either equal or one of the tensor's height must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }
    if (!isDim2Compatible)
    {
        SetReason("Width must be either equal or one of the tensor's height must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }
    if (!isDim3Compatible)
    {
        SetReason("Channels must be either equal or one of the tensor's height must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (((inputInfo0.m_DataFormat != DataFormat::NHWC) && (inputInfo0.m_DataFormat != DataFormat::NHWCB)) ||
        ((inputInfo1.m_DataFormat != DataFormat::NHWC) && (inputInfo1.m_DataFormat != DataFormat::NHWCB)))
    {
        SetReason("Input to addition must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        TensorInfo expectedOutputInfo =
            Addition::CalculateOutputTensorInfo(inputInfo0, inputInfo1, outputQuantizationInfo);
        if (utils::TotalSizeBytes(*outputInfo) != 0 && *outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    // We only support no stretching dimensions or stretching both height and width.
    using DimFlags                                       = std::array<bool, 3>;
    DimFlags stretchDimensions                           = { !isDim1Equal, !isDim2Equal, !isDim3Equal };
    std::array<DimFlags, 2> supportedStretchedDimensions = { DimFlags{ false, false, false },
                                                             DimFlags{ true, true, false } };

    if (!utils::Find(supportedStretchedDimensions, stretchDimensions).first)
    {
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsFullyConnectedSupported(const TensorInfo& biasInfo,
                                         const TensorInfo& weightsInfo,
                                         const FullyConnectedInfo& fullyConnectedInfo,
                                         const TensorInfo& inputInfo,
                                         TensorInfo* outputInfo,
                                         char* reason,
                                         size_t reasonMaxLength)
{
    if (inputInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Input to fully connected must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWCB && inputInfo.m_DataFormat != DataFormat::NHWC)
    {
        SetReason("Invalid data format. Only NHWC and NHWCB are supported for fully connected", reason,
                  reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Weights for fully connected must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }
    if (weightsInfo.m_DataFormat != DataFormat::HWIO)
    {
        SetReason("Weights for fully connected must be HWIO", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }
    if (weightsInfo.m_Dimensions[0] != 1 || weightsInfo.m_Dimensions[1] != 1)
    {
        SetReason("Weights tensor must have H and W set to 1 as these dimensions are not needed.", reason,
                  reasonMaxLength);
        return SupportedLevel::Unsupported;
    }
    uint32_t reshapedInputChannels = inputInfo.m_Dimensions[1] * inputInfo.m_Dimensions[2] * inputInfo.m_Dimensions[3];
    if (weightsInfo.m_Dimensions[2] != reshapedInputChannels)
    {
        SetReason("Weights tensor must have I dimension equal to the number of channels of the input tensor.", reason,
                  reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (biasInfo.m_DataType != DataType::INT32_QUANTIZED)
    {
        SetReason("Bias for fully connected must be INT32_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (biasInfo.m_DataFormat != DataFormat::NHWC)
    {
        SetReason("Bias for fully connected must be NHWC", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if ((biasInfo.m_Dimensions[0] * biasInfo.m_Dimensions[1] * biasInfo.m_Dimensions[2] != 1) ||
        biasInfo.m_Dimensions[3] != weightsInfo.m_Dimensions[3])
    {
        SetReason("Invalid bias tensor dimensions", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        TensorInfo expectedOutputInfo =
            FullyConnected::CalculateOutputTensorInfo(inputInfo, weightsInfo, fullyConnectedInfo);
        if (utils::TotalSizeBytes(*outputInfo) != 0 && *outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    if (inputInfo.m_Dimensions[0] != 1 || inputInfo.m_Dimensions[1] != 1 || inputInfo.m_Dimensions[2] != 1)
    {
        SetReason("Input to fully connected is expected to be one dimensional using the channels dimension.", reason,
                  reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if (weightsInfo.m_QuantizationInfo.m_ZeroPoint > UINT8_MAX)
    {
        SetReason("Zero point value above allowed range", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if (weightsInfo.m_QuantizationInfo.m_ZeroPoint < 0)
    {
        SetReason("Zero point value below allowed range", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if (biasInfo.m_QuantizationInfo.m_ZeroPoint != 0 ||
        biasInfo.m_QuantizationInfo.m_Scale !=
            inputInfo.m_QuantizationInfo.m_Scale * weightsInfo.m_QuantizationInfo.m_Scale)
    {
        SetReason("Bias for fully connected must have quantization parameters with zero point of 0 and scale of "
                  "input scale x weight scale",
                  reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    double overallScale = inputInfo.m_QuantizationInfo.m_Scale * weightsInfo.m_QuantizationInfo.m_Scale /
                          fullyConnectedInfo.m_OutputQuantizationInfo.m_Scale;
    if (overallScale < 0.0f || overallScale >= 1.0f)
    {
        SetReason("Overall scale (of the input * weights / output) should be in the range [0, 1)", reason,
                  reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsReluSupported(
    const ReluInfo& reluInfo, const TensorInfo& inputInfo, TensorInfo* outputInfo, char* reason, size_t reasonMaxLength)
{
    if (reluInfo.m_LowerBound > reluInfo.m_UpperBound)
    {
        SetReason("Relu has lower bound > upper bound", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Input to relu must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input to relu must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        TensorInfo expectedOutputInfo = inputInfo;
        if (utils::TotalSizeBytes(*outputInfo) != 0 && *outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsSoftmaxSupported(const TensorInfo&, TensorInfo*, char* reason, size_t reasonMaxLength)
{
    SetReason("Softmax operation is not supported", reason, reasonMaxLength);
    return SupportedLevel::EstimateOnly;
}

SupportedLevel
    IsSigmoidSupported(const TensorInfo& inputInfo, TensorInfo* outputInfo, char* reason, size_t reasonMaxLength)
{
    if (inputInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Input to sigmoid layer must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        const TensorInfo expectedOutputInfo = Sigmoid::CalculateOutputTensorInfo(inputInfo);

        if (utils::TotalSizeBytes(*outputInfo) == 0)
        {
            *outputInfo = expectedOutputInfo;
        }
        else if (*outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsPoolingSupported(const PoolingInfo& poolingInfo,
                                  const TensorInfo& inputInfo,
                                  TensorInfo* outputInfo,
                                  char* reason,
                                  size_t reasonMaxLength)
{
    const uint32_t inputHeight = inputInfo.m_Dimensions[1];
    const uint32_t inputWidth  = inputInfo.m_Dimensions[2];

    if (inputInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Input to pooling layer must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (poolingInfo.m_PoolingSizeX == 0 || poolingInfo.m_PoolingSizeY == 0 || poolingInfo.m_PoolingStrideX == 0 ||
        poolingInfo.m_PoolingStrideY == 0)
    {
        SetReason("Invalid pooling size/stride", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        TensorInfo expectedOutputInfo = Pooling::CalculateOutputTensorInfo(inputInfo, poolingInfo);
        if (utils::TotalSizeBytes(*outputInfo) != 0 && *outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    if (poolingInfo.m_PoolingType == PoolingType::AVG)
    {
        const bool isMean = (poolingInfo.m_Padding == Padding{ 0, 0, 0, 0 }) &&
                            (poolingInfo.m_PoolingSizeX == inputWidth) && (poolingInfo.m_PoolingSizeY == inputHeight);

        if (isMean)
        {
            if ((poolingInfo.m_PoolingSizeX != poolingInfo.m_PoolingSizeY) ||
                (poolingInfo.m_PoolingSizeX != 7U && poolingInfo.m_PoolingSizeX != 8U))
            {
                SetReason("Unsupported configuration in AVG pooling", reason, reasonMaxLength);
                return SupportedLevel::EstimateOnly;
            }
        }
        else if (poolingInfo.m_PoolingSizeX == 3)
        {
            if (poolingInfo != PoolingInfo{ 3, 3, 1, 1, { 1, 1, 1, 1 }, PoolingType::AVG })
            {
                SetReason("Unsupported configuration in AVG pooling", reason, reasonMaxLength);
                return SupportedLevel::EstimateOnly;
            }

            // Maximum width x height is implementation dependent
            constexpr uint32_t maxXySize = (60U << 10U);

            if ((inputWidth * inputHeight) > maxXySize)
            {
                SetReason("AVG pooling 3x3_1_1: maximum input width x height (60K) exceeded", reason, reasonMaxLength);
                return SupportedLevel::EstimateOnly;
            }
        }
        else
        {
            SetReason("Unsupported configuration in AVG pooling", reason, reasonMaxLength);
            return SupportedLevel::EstimateOnly;
        }
    }
    else if (poolingInfo.m_PoolingType == PoolingType::MAX)
    {
        const Padding noPad    = { 0, 0, 0, 0 };
        const Padding padAfter = { 0, 1, 0, 1 };

        const PoolingInfo supportedConfigs[] = {
            { 2, 2, 2, 2, noPad, PoolingType::MAX },
            { 2, 2, 2, 2, padAfter, PoolingType::MAX },
            { 3, 3, 2, 2, noPad, PoolingType::MAX },
            { 3, 3, 2, 2, padAfter, PoolingType::MAX },
        };

        const auto it = std::find(std::begin(supportedConfigs), std::end(supportedConfigs), poolingInfo);

        if (it == std::end(supportedConfigs))
        {
            SetReason("Unsupported configuration in Max pooling", reason, reasonMaxLength);
            return SupportedLevel::EstimateOnly;
        }

        if (poolingInfo.m_PoolingSizeX == 2)
        {
            if ((poolingInfo.m_Padding == noPad) && (((inputWidth % 2U) != 0) || ((inputHeight % 2U) != 0)))
            {
                SetReason("Max pooling 2x2_2_2 with no padding: input sizes must be even", reason, reasonMaxLength);
                return SupportedLevel::EstimateOnly;
            }

            if ((poolingInfo.m_Padding == padAfter) && (((inputWidth % 2U) == 0) || ((inputHeight % 2U) == 0)))
            {
                SetReason("Max pooling 2x2_2_2 with padding: input sizes must be odd", reason, reasonMaxLength);
                return SupportedLevel::EstimateOnly;
            }
        }

        if (poolingInfo.m_PoolingSizeX == 3)
        {
            // Maximum width is implementation dependent
            constexpr uint32_t maxWidth = 481;

            if (inputWidth > maxWidth)
            {
                SetReason("Max pooling 3x3_2_2: maximum input width (481) exceeded", reason, reasonMaxLength);
                return SupportedLevel::EstimateOnly;
            }

            if ((poolingInfo.m_Padding == noPad) && (((inputWidth % 2U) == 0) || ((inputHeight % 2U) == 0)))
            {
                SetReason("Max pooling 3x3_2_2 with no padding: input sizes must be odd", reason, reasonMaxLength);
                return SupportedLevel::EstimateOnly;
            }

            if ((poolingInfo.m_Padding == padAfter) && (((inputWidth % 2U) != 0) || ((inputHeight % 2U) != 0)))
            {
                SetReason("Max pooling 3x3_2_2 with padding: input sizes must be even", reason, reasonMaxLength);
                return SupportedLevel::EstimateOnly;
            }
        }

        if ((inputWidth < poolingInfo.m_PoolingSizeX) || (inputHeight < poolingInfo.m_PoolingSizeY))
        {
            SetReason("Input size must not be smaller than the pooling size", reason, reasonMaxLength);
            return SupportedLevel::EstimateOnly;
        }
    }
    else
    {
        SetReason("Unsupported pooling algorithm", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsReshapeSupported(const TensorShape& newDimensions,
                                  const TensorInfo& inputInfo,
                                  TensorInfo* outputInfo,
                                  char* reason,
                                  size_t reasonMaxLength)
{
    if (utils::TotalSizeBytes(inputInfo) != utils::TotalSizeBytes(newDimensions))
    {
        SetReason("Total elements in the input doesn't match new dimensions", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        TensorInfo expectedOutputInfo = Reshape::CalculateOutputTensorInfo(inputInfo, newDimensions);

        if (utils::TotalSizeBytes(*outputInfo) != 0 && *outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsDepthToSpaceSupported(const TensorInfo& inputInfo,
                                       const DepthToSpaceInfo& depthToSpaceInfo,
                                       TensorInfo* outputInfo,
                                       char* reason,
                                       size_t reasonMaxLength)
{
    if (inputInfo.m_DataType != DataType::UINT8_QUANTIZED)
    {
        SetReason("Input must be UINT8_QUANTIZED", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_Dimensions[3] % (depthToSpaceInfo.m_BlockSize * depthToSpaceInfo.m_BlockSize) != 0)
    {
        SetReason("Number of channels of input must be an exact multiple of the square of the block size", reason,
                  reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        TensorInfo expectedOutputInfo = DepthToSpace::CalculateOutputTensorInfo(inputInfo, depthToSpaceInfo);
        if (utils::TotalSizeBytes(*outputInfo) != 0 && *outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    if (depthToSpaceInfo.m_BlockSize != 2)
    {
        SetReason("Only block size of 2 is supported", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel IsEstimateOnlySupported(const std::vector<TensorInfo>&,
                                       const EstimateOnlyInfo& info,
                                       std::vector<TensorInfo>* outputInfos,
                                       char* reason,
                                       size_t reasonMaxLength)
{
    if (outputInfos != nullptr)
    {
        for (uint32_t i = 0; i < outputInfos->size(); ++i)
        {
            TensorInfo outputInfo         = (*outputInfos)[i];
            TensorInfo expectedOutputInfo = info.m_OutputInfos[i];
            if (utils::TotalSizeBytes(outputInfo) != 0 && outputInfo != expectedOutputInfo)
            {
                SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
                return SupportedLevel::Unsupported;
            }
            (*outputInfos)[i] = expectedOutputInfo;
        }
    }
    return SupportedLevel::EstimateOnly;
}

}    // namespace support_library
}    // namespace ethosn
