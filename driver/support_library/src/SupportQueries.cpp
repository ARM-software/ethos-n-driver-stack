//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/SupportQueries.hpp"

#include "CapabilitiesInternal.hpp"
#include "Network.hpp"
#include "Utils.hpp"

#include <cmath>
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

using namespace utils;

namespace
{

enum class PadMode
{
    PreferBefore,
    PreferAfter,
    Symmetric
};

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
    CalcSamePadding(const uint32_t inputSize, const uint32_t kernelSize, const uint32_t stride, const PadMode mode)
{
    const uint32_t paddedSize = ((utils::DivRoundUp(inputSize, stride) - 1U) * stride) + kernelSize;
    const uint32_t padSize    = (paddedSize > inputSize) ? paddedSize - inputSize : 0U;

    const uint32_t pad0 = utils::DivRoundUp(padSize, 2U);
    const uint32_t pad1 = padSize - pad0;

    if (mode == PadMode::Symmetric)
    {
        return { pad0, pad0 };
    }

    const uint32_t padBefore = (mode == PadMode::PreferBefore) ? pad0 : pad1;
    const uint32_t padAfter  = (mode == PadMode::PreferAfter) ? pad0 : pad1;

    return { padBefore, padAfter };
}

constexpr Padding CalcSamePadding(const TensorShape& inputShape,
                                  const TensorShape& weightsShape,
                                  const Stride& stride,
                                  const PadMode mode)
{
    const std::pair<uint32_t, uint32_t> padY = CalcSamePadding(inputShape[1], weightsShape[0], stride.m_Y, mode);
    const std::pair<uint32_t, uint32_t> padX = CalcSamePadding(inputShape[2], weightsShape[1], stride.m_X, mode);

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
                                  const PadMode mode)
{
    return CalcSamePadding(inputInfo.m_Dimensions, weightsInfo.m_Dimensions, stride, mode);
}

bool IsPaddingSupported(const TensorInfo& inputInfo,
                        const TensorInfo& weightsInfo,
                        const Stride& stride,
                        const Padding& padInfo)
{

    return ((padInfo == Padding{ 0, 0, 0, 0 }) ||
            (padInfo == CalcSamePadding(inputInfo, weightsInfo, stride, PadMode::Symmetric)) ||
            (padInfo == CalcSamePadding(inputInfo, weightsInfo, stride, PadMode::PreferBefore)) ||
            (padInfo == CalcSamePadding(inputInfo, weightsInfo, stride, PadMode::PreferAfter)));
}

using PossibleTypeList = std::initializer_list<DataType>;

bool IsDataTypeIn(const TensorInfo& info, const PossibleTypeList& possibleTypes)
{
    return std::find(possibleTypes.begin(), possibleTypes.end(), info.m_DataType) != possibleTypes.end();
}

bool IsInputDataTypeSupported(const TensorInfo& info, const char* what, char* reason, size_t reasonMaxLength)
{
    bool isSupported = IsDataTypeIn(info, { DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED });
    if (!isSupported)
    {
        SetReason("%s must be UINT8_QUANTIZED or INT8_QUANTIZED", reason, reasonMaxLength, what);
    }

    return isSupported;
}

bool IsWeightsDataTypeSupported(const TensorInfo& info, const char* what, char* reason, size_t reasonMaxLength)
{
    bool isSupported = IsDataTypeIn(info, { DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED });
    if (!isSupported)
    {
        SetReason("%s must be UINT8_QUANTIZED or INT8_QUANTIZED", reason, reasonMaxLength, what);
    }

    return isSupported;
}

bool IsBiasDataTypeSupported(const TensorInfo& info, const char* what, char* reason, size_t reasonMaxLength)
{
    bool isSupported = IsDataTypeIn(info, { DataType::INT32_QUANTIZED });
    if (!isSupported)
    {
        SetReason("%s must be INT32_QUANTIZED", reason, reasonMaxLength, what);
    }

    return isSupported;
}

bool IsQuantisationZeroPointInRange(const TensorInfo& tensor)
{
    const utils::DataTypeRange dataTypeRange = utils::GetRangeOfDataType(tensor.m_DataType);
    const int32_t minAllowed                 = dataTypeRange.min;
    const int32_t maxAllowed                 = dataTypeRange.max;

    const int32_t zeroPoint = tensor.m_QuantizationInfo.GetZeroPoint();
    return (zeroPoint >= minAllowed) && (zeroPoint <= maxAllowed);
}

bool HasQuantizationDim(const TensorInfo& info)
{
    return info.m_QuantizationInfo.GetQuantizationDim().has_value();
}

bool IsQuantizationDimSupported(const TensorInfo& info,
                                uint32_t expectedDim,
                                const char* name,
                                const char* what,
                                char* reason,
                                size_t reasonMaxLength)
{
    if (HasQuantizationDim(info))
    {
        uint32_t dim = info.m_QuantizationInfo.GetQuantizationDim().value();

        if (dim != expectedDim)
        {
            SetReason("%s: Per channel quantization axis must be %d for %s", reason, reasonMaxLength, what, expectedDim,
                      name);
            return false;
        }

        if (info.m_QuantizationInfo.GetScales().size() != info.m_Dimensions[dim])
        {
            SetReason("%s must have quantization parameters with same number of elements as the quantisation dim",
                      reason, reasonMaxLength, what);
            return false;
        }
    }

    return true;
}

bool IsQuantizationDimSupported(const TensorInfo* biasInfo,
                                const TensorInfo* weightsInfo,
                                const TensorInfo* inputInfo,
                                const QuantizationInfo* outputQuantInfo,
                                const char* what,
                                char* reason,
                                size_t reasonMaxLength)
{
    if (biasInfo != nullptr)
    {
        if (!IsQuantizationDimSupported(*biasInfo, 3U, "Biases", what, reason, reasonMaxLength))
        {
            return false;
        }
    }

    if (weightsInfo != nullptr)
    {
        if (!IsQuantizationDimSupported(*weightsInfo, 3U, "Weights", what, reason, reasonMaxLength))
        {
            return false;
        }
    }

    if (inputInfo != nullptr)
    {
        if (HasQuantizationDim(*inputInfo))
        {
            SetReason("%s: Quantization Dim should not be used on Input", reason, reasonMaxLength, what);
            return false;
        }
        if (inputInfo->m_QuantizationInfo.GetScales().size() != 1)
        {
            SetReason("%s: Input quantization scales must have a size of 1", reason, reasonMaxLength, what);
            return false;
        }
    }

    if (outputQuantInfo != nullptr)
    {
        if (outputQuantInfo->GetQuantizationDim().has_value())
        {
            SetReason("%s: Quantization Dim should not be used on Output", reason, reasonMaxLength, what);
            return false;
        }
        if (outputQuantInfo->GetScales().size() != 1)
        {
            SetReason("%s: Output quantization scales must have a size of 1", reason, reasonMaxLength, what);
            return false;
        }
    }

    return true;
}

size_t GetTotalSramSize(const std::vector<char>& caps)
{
    return static_cast<size_t>(GetValidCapabilities(caps).m_TotalSramSize);
}

bool IsTensorDepthSupported(
    const std::vector<char>& caps, const TensorInfo& info, const char* what, char* reason, size_t reasonMaxLength)
{
    if (info.m_Dimensions[2] == 1)
    {
        // Note:
        //   This is a relax check if Width size is 1 because the DMA is capable of splitting in channels
        //   (we can 'pretend' that the channels is actually the width)
        return true;
    }

    // Assume the worse case situation which is that we have to convert between NHWC and NHWCB.
    // Due to hardware DMA limitations, we can't split NHWC in depth, so the minimum chunk of data we can
    // convert is 8 x 8 x C, and therefore we need to be able to fit this in SRAM.
    size_t maxChunkSize = 8 * 8 * info.m_Dimensions[3];
    size_t sramSize     = GetTotalSramSize(caps);

    if (maxChunkSize > sramSize)
    {
        SetReason("%s: Tensor max depth cannot fit in SRAM (%d / %d)", reason, reasonMaxLength, what, maxChunkSize,
                  sramSize);
        return false;
    }

    return true;
}

constexpr bool IsQuantizationScaleSupported(float input, float output)
{
    // We implement requantize with identity convolutions so the same quantization restrictions apply
    float multiplier = input * utils::g_IdentityWeightScale / output;
    return (multiplier >= 0.f && multiplier < 1.f);
}

bool IsQuantizationScaleSupported(const QuantizationScales& overallScale,
                                  const char* what,
                                  char* reason,
                                  size_t reasonMaxLength)
{
    // The shift is encoded with 5 bits
    constexpr uint32_t maxShift = (1 << 5);
    const float minScale        = std::exp2f(-(static_cast<float>(maxShift)));
    if (overallScale.min() < minScale || overallScale.max() >= 1.0f)
    {
        SetReason("%s: Overall scale (of the input * weights / output) should be in the range [%e, 1)", reason,
                  reasonMaxLength, what, minScale);
        return false;
    }
    return true;
}

}    // namespace

const SupportedLevel SupportedLevel::Unsupported  = SupportedLevel(InternalSupportedLevel::Unsupported);
const SupportedLevel SupportedLevel::EstimateOnly = SupportedLevel(InternalSupportedLevel::EstimateOnly);
const SupportedLevel SupportedLevel::Supported    = SupportedLevel(InternalSupportedLevel::Supported);

SupportQueries::SupportQueries(const std::vector<char>& caps)
    : m_Capabilities(caps)
{
    ValidateCapabilities(m_Capabilities);
}

SupportedLevel SupportQueries::IsInputSupported(const TensorInfo& inputInfo,
                                                TensorInfo* outputInfo,
                                                char* reason,
                                                size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input layer must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, nullptr, "Input layer", reason, reasonMaxLength))
    {
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

SupportedLevel SupportQueries::IsOutputSupported(const TensorInfo& inputInfo,
                                                 const DataFormat format,
                                                 char* reason,
                                                 size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Output layer's input", reason, reasonMaxLength))
    {
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

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, nullptr, "Output layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    return SupportedLevel::Supported;
}

SupportedLevel
    SupportQueries::IsConstantSupported(const TensorInfo& constantInfo, char* reason, size_t reasonMaxLength) const
{
    if (!IsTensorDepthSupported(m_Capabilities, constantInfo, "Constant layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    return SupportedLevel::Supported;
}

SupportedLevel SupportQueries::IsConvolutionSupported(const TensorInfo& biasInfo,
                                                      const TensorInfo& weightsInfo,
                                                      const ConvolutionInfo& convInfo,
                                                      const TensorInfo& inputInfo,
                                                      TensorInfo* outputInfo,
                                                      char* reason,
                                                      size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to conv", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to conv", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input to conv must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsWeightsDataTypeSupported(weightsInfo, "Weight for conv", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_DataFormat != DataFormat::HWIO)
    {
        SetReason("Weights for conv must be HWIO", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsBiasDataTypeSupported(biasInfo, "Bias for conv", reason, reasonMaxLength))
    {
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

    if (!IsTensorDepthSupported(m_Capabilities, expectedOutputInfo, "Output of conv", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(&biasInfo, &weightsInfo, &inputInfo, &convInfo.m_OutputQuantizationInfo,
                                    "Convolution", reason, reasonMaxLength))
    {
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

    if (!(IsQuantisationZeroPointInRange(weightsInfo)))
    {
        SetReason("Zero point value of weight is not in range", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if (biasInfo.m_QuantizationInfo.GetZeroPoint() != 0)
    {
        SetReason("Bias for conv must have quantization parameters with zero point of 0", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    QuantizationScales intermediateScales =
        inputInfo.m_QuantizationInfo.GetScales() * weightsInfo.m_QuantizationInfo.GetScales();
    if (biasInfo.m_QuantizationInfo.GetScales() != intermediateScales)
    {
        SetReason("Bias for conv must have quantization parameters with scale of input scale x weight scale", reason,
                  reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    static const std::unordered_set<uint32_t> validStrides = { 1, 2 };

    if ((g_ConvolutionKernelSizes.count(kernelHeight) == 0U) || (g_ConvolutionKernelSizes.count(kernelWidth) == 0U))
    {
        SetReason("Unsupported kernel size. Width(%d)/height(%d) must be in { 1, 2, 3, 5, 7, 9 }", reason,
                  reasonMaxLength, kernelWidth, kernelHeight);
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

    if (!IsPaddingSupported(inputInfo, weightsInfo, convInfo.m_Stride, convInfo.m_Padding))
    {
        SetReason("Unsupported padding.", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    QuantizationScales overallScale = intermediateScales / convInfo.m_OutputQuantizationInfo.GetScales();
    if (!IsQuantizationScaleSupported(overallScale, "Convolution", reason, reasonMaxLength))
    {
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel SupportQueries::IsDepthwiseConvolutionSupported(const TensorInfo& biasInfo,
                                                               const TensorInfo& weightsInfo,
                                                               const ConvolutionInfo& convInfo,
                                                               const TensorInfo& inputInfo,
                                                               TensorInfo* outputInfo,
                                                               char* reason,
                                                               size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to depthwise conv", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to depthwise conv", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input to depthwise conv must be NHWC OR NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsWeightsDataTypeSupported(weightsInfo, "Weight for conv", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_DataFormat != DataFormat::HWIM)
    {
        SetReason("Weights for depthwise conv must be HWIM", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsBiasDataTypeSupported(biasInfo, "Bias for depthwise conv", reason, reasonMaxLength))
    {
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

    if (!IsTensorDepthSupported(m_Capabilities, expectedOutputInfo, "Output of depthwise conv", reason,
                                reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(&biasInfo, &weightsInfo, &inputInfo, &convInfo.m_OutputQuantizationInfo,
                                    "Depthwise Convolution", reason, reasonMaxLength))
    {
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

    if (!(IsQuantisationZeroPointInRange(weightsInfo)))
    {
        SetReason("Zero point value of weight is not in range", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    QuantizationScales intermediateScales =
        inputInfo.m_QuantizationInfo.GetScales() * weightsInfo.m_QuantizationInfo.GetScales();
    if (biasInfo.m_QuantizationInfo.GetScales() != intermediateScales)
    {
        SetReason("Bias for depthwise conv must have quantization parameters with zero point of 0 and scale of "
                  "input scale x weight scale",
                  reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    static const std::unordered_set<uint32_t> validStrides = { 1, 2 };

    if ((kernelHeight != kernelWidth) || (g_ConvolutionKernelSizes.count(kernelHeight) == 0U))
    {
        SetReason("Unsupported kernel size. Width(%d)/height(%d) must be in { 1, 2, 3, 5, 7, 9 }", reason,
                  reasonMaxLength, kernelWidth, kernelHeight);
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

    if (!IsPaddingSupported(inputInfo, weightsInfo, convInfo.m_Stride, convInfo.m_Padding))
    {
        SetReason("Unsupported padding.", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    QuantizationScales overallScale = intermediateScales / convInfo.m_OutputQuantizationInfo.GetScales();
    if (!IsQuantizationScaleSupported(overallScale, "Depthwise Convolution", reason, reasonMaxLength))
    {
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel SupportQueries::IsTransposeConvolutionSupported(const TensorInfo& biasInfo,
                                                               const TensorInfo& weightsInfo,
                                                               const ConvolutionInfo& convInfo,
                                                               const TensorInfo& inputInfo,
                                                               TensorInfo* outputInfo,
                                                               char* reason,
                                                               size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to transpose conv", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to transpose conv", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input to transpose conv must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsWeightsDataTypeSupported(weightsInfo, "Weights for transpose conv", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (weightsInfo.m_DataFormat != DataFormat::HWIO)
    {
        SetReason("Weights for transpose conv must be HWIO", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsBiasDataTypeSupported(biasInfo, "Bias for transpose conv", reason, reasonMaxLength))
    {
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

    if (!IsTensorDepthSupported(m_Capabilities, expectedOutputInfo, "Output of transpose conv", reason,
                                reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(&biasInfo, &weightsInfo, &inputInfo, &convInfo.m_OutputQuantizationInfo,
                                    "Transpose Convolution", reason, reasonMaxLength))
    {
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

    if (!(IsQuantisationZeroPointInRange(weightsInfo)))
    {
        SetReason("Zero point value of weight is not in range", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    QuantizationScales intermediateScales =
        inputInfo.m_QuantizationInfo.GetScales() * weightsInfo.m_QuantizationInfo.GetScales();
    if (biasInfo.m_QuantizationInfo.GetZeroPoint() != 0 ||
        biasInfo.m_QuantizationInfo.GetScales() != intermediateScales)
    {
        SetReason("Bias for transpose conv must have quantization parameters with zero point of 0 and "
                  "scale of input scale x weight scale",
                  reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    static const std::unordered_set<uint32_t> validStrides = { 2 };

    if ((g_ConvolutionKernelSizes.count(kernelHeight) == 0U) || (g_ConvolutionKernelSizes.count(kernelWidth) == 0U))
    {
        SetReason("Unsupported kernel size. Width(%d)/height(%d) must be in { 1, 2, 3, 5, 7, 9 }", reason,
                  reasonMaxLength, kernelWidth, kernelHeight);
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
    if (!IsPaddingSupported(outputShape, weightsInfo, convInfo.m_Stride, convInfo.m_Padding))
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

    QuantizationScales overallScale = intermediateScales / convInfo.m_OutputQuantizationInfo.GetScales();
    if (!IsQuantizationScaleSupported(overallScale, "Tranpose Convolution", reason, reasonMaxLength))
    {
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel SupportQueries::IsConcatenationSupported(const std::vector<TensorInfo>& inputInfos,
                                                        const ConcatenationInfo& concatInfo,
                                                        TensorInfo* outputInfo,
                                                        char* reason,
                                                        size_t reasonMaxLength) const
{
    size_t numInputs = inputInfos.size();
    if (numInputs < 1)
    {
        SetReason("Must have at least one input", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    for (uint32_t i = 0; i < numInputs; ++i)
    {
        if (inputInfos[i].m_Dimensions[0] != 1)
        {
            SetReason("Batch size must be 1", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }

        if (!IsTensorDepthSupported(m_Capabilities, inputInfos[i], "Input tensors", reason, reasonMaxLength))
        {
            return SupportedLevel::Unsupported;
        }

        if (!IsInputDataTypeSupported(inputInfos[i], "Input tensors", reason, reasonMaxLength))
        {
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

    if (std::any_of(inputInfos.begin(), inputInfos.end(), HasQuantizationDim))
    {
        SetReason("Quantization Dim should not be used on any Inputs of Concat", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(nullptr, nullptr, nullptr, &concatInfo.m_OutputQuantizationInfo, "Concatenation",
                                    reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    // We implement requantize with identity convolutions so the same quantization restrictions apply
    float outputScale = concatInfo.m_OutputQuantizationInfo.GetScale();
    for (const auto& info : inputInfos)
    {

        if (!IsQuantizationScaleSupported(info.m_QuantizationInfo.GetScale(), outputScale))
        {
            // We might be able to support this in the future if we add a generic requantize in the PLE
            SetReason("Output scales must be bigger than input scale / 128", reason, reasonMaxLength);
            return SupportedLevel::EstimateOnly;
        }
    }

    TensorInfo expectedOutputInfo = Concatenation::CalculateOutputTensorInfo(inputInfos, concatInfo);

    if (!IsTensorDepthSupported(m_Capabilities, expectedOutputInfo, "Output of concatenation", reason, reasonMaxLength))
    {
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

SupportedLevel SupportQueries::IsSplitSupported(const TensorInfo& inputInfo,
                                                const SplitInfo& splitInfo,
                                                std::vector<TensorInfo>* outputInfos,
                                                char* reason,
                                                size_t reasonMaxLength) const
{
    size_t numOutputs = splitInfo.m_Sizes.size();

    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input tensor", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (numOutputs < 1)
    {
        SetReason("Must have at least 1 output", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input tensor", reason, reasonMaxLength))
    {
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

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, nullptr, "Split", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (outputInfos != nullptr)
    {
        if (outputInfos->size() != numOutputs)
        {
            SetReason("Provided outputInfos array has incorrect size", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
    }

    std::vector<TensorInfo> expectedOutputInfos = Split::CalculateOutputTensorInfos(inputInfo, splitInfo);
    for (uint32_t i = 0; i < numOutputs; ++i)
    {
        if (!IsTensorDepthSupported(m_Capabilities, expectedOutputInfos[i], "Output of split", reason, reasonMaxLength))
        {
            return SupportedLevel::Unsupported;
        }

        if (outputInfos != nullptr)
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

SupportedLevel SupportQueries::IsAdditionSupported(const TensorInfo& inputInfo0,
                                                   const TensorInfo& inputInfo1,
                                                   const QuantizationInfo& outputQuantizationInfo,
                                                   TensorInfo* outputInfo,
                                                   char* reason,
                                                   size_t reasonMaxLength) const
{
    const TensorShape& shape0 = inputInfo0.m_Dimensions;
    const TensorShape& shape1 = inputInfo1.m_Dimensions;
    const bool isDim1Equal    = shape0[1] == shape1[1];
    const bool isDim2Equal    = shape0[2] == shape1[2];
    const bool isDim3Equal    = shape0[3] == shape1[3];

    // To be able to stretch along a dimension the dimension size in one of the tensors must be 1.
    // Note that we no longer support any form of stretching, but the logic in this function has been
    // preserved to make it easy to add back later if necessary.
    const bool canStretchDim1 = shape0[1] == 1 || shape1[1] == 1;
    const bool canStretchDim2 = shape0[2] == 1 || shape1[2] == 1;
    const bool canStretchDim3 = shape0[3] == 1 || shape1[3] == 1;

    const bool isDim1Compatible = isDim1Equal || canStretchDim1;
    const bool isDim2Compatible = isDim2Equal || canStretchDim2;
    const bool isDim3Compatible = isDim3Equal || canStretchDim3;

    if ((inputInfo0.m_Dimensions[0] != 1) || (inputInfo1.m_Dimensions[0] != 1))
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo0, "Input0 to addition", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo1, "Input1 to addition", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

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

    if (!IsInputDataTypeSupported(inputInfo0, "Input to addition", reason, reasonMaxLength) ||
        !IsInputDataTypeSupported(inputInfo1, "Input to addition", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (inputInfo0.m_DataType != inputInfo1.m_DataType)
    {
        SetReason("Inputs to addition must have the same data type", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (((inputInfo0.m_DataFormat != DataFormat::NHWC) && (inputInfo0.m_DataFormat != DataFormat::NHWCB)) ||
        ((inputInfo1.m_DataFormat != DataFormat::NHWC) && (inputInfo1.m_DataFormat != DataFormat::NHWCB)))
    {
        SetReason("Input to addition must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (HasQuantizationDim(inputInfo0) || HasQuantizationDim(inputInfo1))
    {
        SetReason("Quantization Dim should not be used on any Inputs of Addition", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        // Note:
        //   Here we don't need to check IsTensorDepthSupported since addition of two layers will result in an output
        //   tensor info that depth is the max of any of the input.
        //   This would lead to dead code as if we don't fail with input0 or input1 we will not fail on output.
        TensorInfo expectedOutputInfo =
            Addition::CalculateOutputTensorInfo(inputInfo0, inputInfo1, outputQuantizationInfo);
        if (utils::TotalSizeBytes(*outputInfo) != 0 && *outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    // We only support no stretching dimensions
    using DimFlags                                       = std::array<bool, 3>;
    DimFlags stretchDimensions                           = { !isDim1Equal, !isDim2Equal, !isDim3Equal };
    std::array<DimFlags, 1> supportedStretchedDimensions = { DimFlags{ false, false, false } };
    if (!utils::Find(supportedStretchedDimensions, stretchDimensions).first)
    {
        SetReason("Cannot stretch along the requested dimensions.", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel SupportQueries::IsFullyConnectedSupported(const TensorInfo& biasInfo,
                                                         const TensorInfo& weightsInfo,
                                                         const FullyConnectedInfo& fullyConnectedInfo,
                                                         const TensorInfo& inputInfo,
                                                         TensorInfo* outputInfo,
                                                         char* reason,
                                                         size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to fully connected", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to fully connected", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWCB && inputInfo.m_DataFormat != DataFormat::NHWC)
    {
        SetReason("Invalid data format. Only NHWC and NHWCB are supported for fully connected", reason,
                  reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsWeightsDataTypeSupported(weightsInfo, "Weights for fully connected", reason, reasonMaxLength))
    {
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

    if (!IsBiasDataTypeSupported(biasInfo, "Bias for fully connected", reason, reasonMaxLength))
    {
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

    if (!IsQuantizationDimSupported(&biasInfo, &weightsInfo, &inputInfo, &fullyConnectedInfo.m_OutputQuantizationInfo,
                                    "Fully Connected", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    TensorInfo expectedOutputInfo =
        FullyConnected::CalculateOutputTensorInfo(inputInfo, weightsInfo, fullyConnectedInfo);

    if (!IsTensorDepthSupported(m_Capabilities, expectedOutputInfo, "Output of fully connected", reason,
                                reasonMaxLength))
    {
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

    if (inputInfo.m_Dimensions[0] != 1 || inputInfo.m_Dimensions[1] != 1 || inputInfo.m_Dimensions[2] != 1)
    {
        SetReason("Input to fully connected is expected to be one dimensional using the channels dimension.", reason,
                  reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    if (!(IsQuantisationZeroPointInRange(weightsInfo)))
    {
        SetReason("Zero point value of weight is not in range", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    QuantizationScales intermediateScales =
        inputInfo.m_QuantizationInfo.GetScales() * weightsInfo.m_QuantizationInfo.GetScales();
    if (biasInfo.m_QuantizationInfo.GetZeroPoint() != 0 ||
        biasInfo.m_QuantizationInfo.GetScales() != intermediateScales)
    {
        SetReason("Bias for fully connected must have quantization parameters with zero point of 0 and scale of "
                  "input scale x weight scale",
                  reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    QuantizationScales overallScale = intermediateScales / fullyConnectedInfo.m_OutputQuantizationInfo.GetScales();
    if (!IsQuantizationScaleSupported(overallScale, "Fully Connected", reason, reasonMaxLength))
    {
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel SupportQueries::IsReluSupported(const ReluInfo& reluInfo,
                                               const TensorInfo& inputInfo,
                                               TensorInfo* outputInfo,
                                               char* reason,
                                               size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to relu", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (reluInfo.m_LowerBound > reluInfo.m_UpperBound)
    {
        SetReason("Relu has lower bound > upper bound", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to relu", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input to relu must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, nullptr, "Relu", reason, reasonMaxLength))
    {
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

SupportedLevel SupportQueries::IsLeakyReluSupported(const LeakyReluInfo& leakyReluInfo,
                                                    const TensorInfo& inputInfo,
                                                    TensorInfo* outputInfo,
                                                    char* reason,
                                                    size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to leaky relu", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to leaky relu", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input to leaky relu must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, &leakyReluInfo.m_OutputQuantizationInfo, "Leaky Relu",
                                    reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        TensorInfo expectedOutputInfo = LeakyRelu::CalculateOutputTensorInfo(inputInfo, leakyReluInfo);
        if (utils::TotalSizeBytes(*outputInfo) != 0 && *outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    if (leakyReluInfo.m_Alpha >= 1.0f || leakyReluInfo.m_Alpha <= 0.0f)
    {
        SetReason("Leaky relu alpha must be less than 1 and greater than 0", reason, reasonMaxLength);
        return SupportedLevel::EstimateOnly;
    }

    return SupportedLevel::Supported;
}

SupportedLevel SupportQueries::IsRequantizeSupported(const RequantizeInfo& requantizeInfo,
                                                     const TensorInfo& inputInfo,
                                                     TensorInfo* outputInfo,
                                                     char* reason,
                                                     size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to requantize", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to requantize", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input to requantize must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, &requantizeInfo.m_OutputQuantizationInfo,
                                    "Requantize", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationScaleSupported(inputInfo.m_QuantizationInfo.GetScale(),
                                      requantizeInfo.m_OutputQuantizationInfo.GetScale()))
    {
        SetReason("Output scale must be bigger than input scale / 128", reason, reasonMaxLength);
        // We might be able to support this in the future if we add a generic requantize in the PLE
        return SupportedLevel::EstimateOnly;
    }

    TensorInfo expectedOutputInfo = Requantize::CalculateOutputTensorInfo(inputInfo, requantizeInfo);

    if (!(IsQuantisationZeroPointInRange(expectedOutputInfo)))
    {
        SetReason("Zero point out of range", reason, reasonMaxLength);
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

    return SupportedLevel::Supported;
}

SupportedLevel
    SupportQueries::IsReinterpretQuantizationSupported(const ReinterpretQuantizationInfo& reinterpretQuantizationInfo,
                                                       const TensorInfo& inputInfo,
                                                       TensorInfo* outputInfo,
                                                       char* reason,
                                                       size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to reinterpret quantization", reason,
                                reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to reinterpret quantization", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input to reinterpret quantization must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!(IsQuantisationZeroPointInRange(inputInfo)))
    {
        SetReason("Zero point out of range for input info", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    TensorInfo expectedOutputInfo =
        ReinterpretQuantization::CalculateOutputTensorInfo(inputInfo, reinterpretQuantizationInfo);

    if (!(IsQuantisationZeroPointInRange(expectedOutputInfo)))
    {
        SetReason("Zero point out of range for expected output info", reason, reasonMaxLength);
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

    return SupportedLevel::Supported;
}

SupportedLevel
    SupportQueries::IsSoftmaxSupported(const TensorInfo&, TensorInfo*, char* reason, size_t reasonMaxLength) const
{
    SetReason("Softmax operation is not supported", reason, reasonMaxLength);
    return SupportedLevel::EstimateOnly;
}

SupportedLevel SupportQueries::IsMeanXySupported(const TensorInfo& inputInfo,
                                                 TensorInfo* outputInfo,
                                                 char* reason,
                                                 size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to MeanXy layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to MeanXy layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, nullptr, "Mean", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!(((inputInfo.m_Dimensions[1] == 8) && (inputInfo.m_Dimensions[2] == 8)) ||
          ((inputInfo.m_Dimensions[1] == 7) && (inputInfo.m_Dimensions[2] == 7))))
    {
        SetReason("MeanXy is supported for 7x7 and 8x8 as HeightxWidth only", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        const TensorInfo expectedOutputInfo = MeanXy::CalculateOutputTensorInfo(inputInfo);

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

SupportedLevel SupportQueries::IsSigmoidSupported(const TensorInfo& inputInfo,
                                                  TensorInfo* outputInfo,
                                                  char* reason,
                                                  size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to sigmoid layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to sigmoid layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, nullptr, "Sigmoid", reason, reasonMaxLength))
    {
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

SupportedLevel SupportQueries::IsTanhSupported(const TensorInfo& inputInfo,
                                               TensorInfo* outputInfo,
                                               char* reason,
                                               size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to tanh layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to tanh layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, nullptr, "tanh", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        const TensorInfo expectedOutputInfo = Tanh::CalculateOutputTensorInfo(inputInfo);

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

bool IsAvgPool_3x3_1_1_FitSram(const uint32_t height,
                               const uint32_t width,
                               const uint32_t depth,
                               const std::vector<char>& caps)
{
    const HardwareCapabilities hwCapabilities(GetValidCapabilities(caps));

    const uint32_t inputXySizeNhwcb = utils::RoundUpToNearestMultiple(height, hwCapabilities.GetBrickGroupShape()[1]) *
                                      utils::RoundUpToNearestMultiple(width, hwCapabilities.GetBrickGroupShape()[2]);

    const uint32_t outputXySizeNhwcb = inputXySizeNhwcb;

    const uint32_t totalSramSize = static_cast<uint32_t>(GetTotalSramSize(caps));

    const uint32_t numSrams = hwCapabilities.GetNumberOfSrams();

    const uint32_t sramSizePerEmc = utils::DivRoundUp(totalSramSize, numSrams);

    // max kernel size per EMC
    const uint32_t maxPleKernelSize = hwCapabilities.GetMaxPleSize();

    // Note average pooling only allows split in Z axis and split in X or Y is NOT allowed.
    // If input depth is smaller than number of SRAMs, then no split is allowed and number of strips in tile = 1
    // Otherwise, the IFM and OFM can be split in depth and the number of stripes in tile = 2
    // Refer to PlePass.cpp for more detail.
    uint32_t numStripesTile = (depth > numSrams) ? 2 : 1;

    return (numStripesTile * (inputXySizeNhwcb + outputXySizeNhwcb) + maxPleKernelSize) <= sramSizePerEmc;
}

SupportedLevel SupportQueries::IsPoolingSupported(const PoolingInfo& poolingInfo,
                                                  const TensorInfo& inputInfo,
                                                  TensorInfo* outputInfo,
                                                  char* reason,
                                                  size_t reasonMaxLength) const
{
    const uint32_t inputHeight = inputInfo.m_Dimensions[1];
    const uint32_t inputWidth  = inputInfo.m_Dimensions[2];

    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to pooling layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to pooling layer", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (poolingInfo.m_PoolingSizeX == 0 || poolingInfo.m_PoolingSizeY == 0 || poolingInfo.m_PoolingStrideX == 0 ||
        poolingInfo.m_PoolingStrideY == 0)
    {
        SetReason("Invalid pooling size/stride", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, nullptr, "Pooling", reason, reasonMaxLength))
    {
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

            const uint32_t inputDepth = inputInfo.m_Dimensions[3];

            if (!IsAvgPool_3x3_1_1_FitSram(inputHeight, inputWidth, inputDepth, m_Capabilities))
            {
                SetReason("AVG pooling 3x3_1_1: maximum input width x height cannot fit into SRAM", reason,
                          reasonMaxLength);
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
            { 1, 1, 2, 2, noPad, PoolingType::MAX },    { 2, 2, 2, 2, noPad, PoolingType::MAX },
            { 2, 2, 2, 2, padAfter, PoolingType::MAX }, { 3, 3, 2, 2, noPad, PoolingType::MAX },
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
            constexpr unsigned maxWidth = 417;

            if (inputWidth > maxWidth)
            {
                SetReason("Max pooling 3x3_2_2: maximum input width (%u) exceeded", reason, reasonMaxLength, maxWidth);
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

SupportedLevel SupportQueries::IsReshapeSupported(const TensorShape& newDimensions,
                                                  const TensorInfo& inputInfo,
                                                  TensorInfo* outputInfo,
                                                  char* reason,
                                                  size_t reasonMaxLength) const
{
    if ((inputInfo.m_Dimensions[0] != 1) || (newDimensions[0] != 1))
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to reshape", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (utils::TotalSizeBytes(inputInfo) != utils::TotalSizeBytes(newDimensions))
    {
        SetReason("Total elements in the input doesn't match new dimensions", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, nullptr, "Reshape", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    TensorInfo expectedOutputInfo = Reshape::CalculateOutputTensorInfo(inputInfo, newDimensions);

    if (!IsTensorDepthSupported(m_Capabilities, expectedOutputInfo, "Output of reshape", reason, reasonMaxLength))
    {
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

    return SupportedLevel::Supported;
}

SupportedLevel SupportQueries::IsDepthToSpaceSupported(const TensorInfo& inputInfo,
                                                       const DepthToSpaceInfo& depthToSpaceInfo,
                                                       TensorInfo* outputInfo,
                                                       char* reason,
                                                       size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to depth to space", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to depth to space", reason, reasonMaxLength))
    {
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

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, nullptr, "Depth to Space", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        // Note:
        //   Here we don't need to check IsTensorDepthSupported since calculated output of DepthToSpace results in an
        //   output tensor info that depth is the depth of input tensor info divided by the square of block size.
        //   This would lead to dead code as if we don't fail with input we will not fail on output.
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

bool IsSpaceToDepthFitInSram(const TensorInfo& inputInfo,
                             const SpaceToDepthInfo& spaceToDepthInfo,
                             const std::vector<char>& caps)
{
    const uint32_t numSrams = GetValidCapabilities(caps).m_NumberOfEngines * GetValidCapabilities(caps).m_EmcPerEngine;
    const uint32_t sramSize = static_cast<uint32_t>(GetTotalSramSize(caps));

    const uint32_t blockSize = spaceToDepthInfo.m_BlockSize;

    const uint32_t ifmHeight   = GetHeight(inputInfo.m_Dimensions);
    const uint32_t ifmChannels = GetChannels(inputInfo.m_Dimensions);

    // usedEmcs must evenly divide ifmChannels * blockSize
    uint32_t usedEmcs = std::min(numSrams, blockSize * ifmChannels);
    while ((blockSize * ifmChannels) % usedEmcs != 0)
    {
        --usedEmcs;
    }

    // SpaceToDepth only splits the input tensor in Y-axis.
    // Support query only needs to find if the smallest possible
    // subdivision can be fit into SRAM.
    TensorShape outIfmStripeShape = inputInfo.m_Dimensions;
    for (uint32_t divisor = ifmHeight; divisor >= 1; --divisor)
    {
        if ((ifmHeight % divisor) == 0 && ((ifmHeight / divisor) % blockSize) == 0)
        {
            outIfmStripeShape[1] = ifmHeight / divisor;
            break;
        }
    }

    uint32_t s1, s2;

    std::tie(s1, s2)         = CalculateSpaceToDepthBlockSizes(outIfmStripeShape, usedEmcs, blockSize);
    const uint32_t sramUsage = CalculateSpaceToDepthSramUsage(blockSize, s1, s2) * numSrams;

    return (sramUsage <= sramSize);
}

SupportedLevel SupportQueries::IsSpaceToDepthSupported(const TensorInfo& inputInfo,
                                                       const SpaceToDepthInfo& spaceToDepthInfo,
                                                       TensorInfo* outputInfo,
                                                       char* reason,
                                                       size_t reasonMaxLength) const
{
    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to space to depth", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to space to depth", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (spaceToDepthInfo.m_BlockSize <= 1)
    {
        SetReason("Block size must be larger than 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_Dimensions[1] % spaceToDepthInfo.m_BlockSize != 0 ||
        inputInfo.m_Dimensions[2] % spaceToDepthInfo.m_BlockSize != 0)
    {
        SetReason("Input width and height must be a multiple of the block size", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, nullptr, "Space to Depth", reason, reasonMaxLength))
    {
        return SupportedLevel::EstimateOnly;
    }

    TensorInfo expectedOutputInfo = SpaceToDepth::CalculateOutputTensorInfo(inputInfo, spaceToDepthInfo);

    if (!IsTensorDepthSupported(m_Capabilities, expectedOutputInfo, "Output of space to depth", reason,
                                reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        if (*outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
    }

    if (!IsSpaceToDepthFitInSram(inputInfo, spaceToDepthInfo, m_Capabilities))
    {
        SetReason("The tensor is too large", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    return SupportedLevel::Supported;
}

SupportedLevel SupportQueries::IsEstimateOnlySupported(const std::vector<TensorInfo>&,
                                                       const EstimateOnlyInfo& info,
                                                       std::vector<TensorInfo>* outputInfos,
                                                       char* reason,
                                                       size_t reasonMaxLength) const
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

uint32_t DepthwiseConvSramEstimate(const TensorInfo& inputInfo, const std::vector<char>& caps)
{
    // Weight tensor shape of a 1x1 depthwise convolution
    const TensorShape weightStripe = { 1, 1, inputInfo.m_Dimensions[3], 1 };
    const uint32_t weightSize      = EstimateWeightSizeBytes(weightStripe, GetValidCapabilities(caps), true);

    const uint32_t inputSize = TotalSizeBytesNHWCB(inputInfo);

    // OFM size is the same as IFM's because we only swap H and W in SRAM,
    // and NHWCB block width = height = 8.
    const uint32_t outputSize = inputSize;
    const uint32_t numSrams = GetValidCapabilities(caps).m_NumberOfEngines * GetValidCapabilities(caps).m_EmcPerEngine;
    const uint32_t maxPleKernelSize = GetValidCapabilities(caps).m_MaxPleSize * numSrams;

    // The total size consumed by a single stripe depthwise conv = IFM + OFM + WEIGHT + PLE
    const uint32_t totalSize = inputSize + outputSize + weightSize + maxPleKernelSize;

    return totalSize;
}

// Transpose only supports strategy 3. Hence any transpose request that needs IFM/OFM streaming is rejected
bool IsTransposeTensorSupported(const std::vector<char>& caps,
                                const TensorInfo& inputInfo,
                                const TransposeInfo& transposeInfo)
{
    const uint32_t sramSize = static_cast<uint32_t>(GetTotalSramSize(caps));
    uint32_t totalSize;
    assert(GetElementSizeBytes(inputInfo.m_DataType) == 1 && "transpose only support 8-bit");

    if (transposeInfo.m_Permutation[1] == 2 && transposeInfo.m_Permutation[2] == 3 &&
        transposeInfo.m_Permutation[3] == 1)
    {
        // (0, 2, 3, 1)
        // Loads to SRAM pretending it is NCHW->0231, then saves to DRAM as NHWC.
        // The compiler will insert a format conversion node that convert from
        // NCHW to NHWC.
        // The compiler will usually assign a format conversion pass to the conversion
        // node if it cannot assign it to a McePle plass due to the multiple stripes
        // restriction.
        totalSize = TotalSizeBytesNHWCB(inputInfo);
    }
    else if (transposeInfo.m_Permutation[1] == 1 && transposeInfo.m_Permutation[2] == 2 &&
             transposeInfo.m_Permutation[3] == 3)
    {
        // (0, 1, 2, 3) should always be supported
        totalSize = sramSize;
    }
    else
    {
        // The compiler will insert an identity mce op node before the format conversion or PLE node (created for transpose)
        // that it has failed to assign a pass to (because only strategy 3 is allowed). In this way the compiler will be able
        // to assign a pass to identity MCE op + format conversion/PLE as long as they can be fit into the SRAM. And the input
        // tensor to transpose operation can be directly used to perform a "conservative" SRAM usage estimate.

        TensorInfo tensorInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);

        if ((transposeInfo.m_Permutation[1] == 3 && transposeInfo.m_Permutation[2] == 1 &&
             transposeInfo.m_Permutation[3] == 2) ||
            (transposeInfo.m_Permutation[1] == 2 && transposeInfo.m_Permutation[2] == 1 &&
             transposeInfo.m_Permutation[3] == 3))
        {
            // (0, 3, 1, 2)
            // (0, 2, 1, 3)
            // Both load data to SRAM as NHWC. Therefore the input to transpose is used to
            // estimate the SRAM usage for the identity MCE operation
            tensorInfo.m_Dimensions = inputInfo.m_Dimensions;
        }
        else if ((transposeInfo.m_Permutation[1] == 1 && transposeInfo.m_Permutation[2] == 3 &&
                  transposeInfo.m_Permutation[3] == 2) ||
                 (transposeInfo.m_Permutation[1] == 3 && transposeInfo.m_Permutation[2] == 2 &&
                  transposeInfo.m_Permutation[3] == 1))

        {
            // (0, 1, 3, 2)
            // (0, 3, 2, 1)
            // Both load data to SRAM as NCHW. Therefore the input to transpose is permuted to
            // estimate the SRAM usage for the identity MCE operation
            tensorInfo.m_Dimensions = { 1, inputInfo.m_Dimensions[2], inputInfo.m_Dimensions[3],
                                        inputInfo.m_Dimensions[1] };
        }

        totalSize = DepthwiseConvSramEstimate(tensorInfo, caps);
    }

    return (totalSize <= sramSize);
}

SupportedLevel SupportQueries::IsTransposeSupported(const TransposeInfo& transposeInfo,
                                                    const TensorInfo& inputInfo,
                                                    TensorInfo* outputInfo,
                                                    char* reason,
                                                    size_t reasonMaxLength) const
{

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to transpose", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to transpose", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (transposeInfo.m_Permutation[0] != 0)
    {
        SetReason("Transpose of batch(0) axis is not allowed", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if ((transposeInfo.m_Permutation[1] > 3) || (transposeInfo.m_Permutation[2] > 3) ||
        (transposeInfo.m_Permutation[3] > 3))
    {
        SetReason("Transpose of dimensions > 3 are not allowed", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if ((transposeInfo.m_Permutation[0] == transposeInfo.m_Permutation[1]) ||
        (transposeInfo.m_Permutation[0] == transposeInfo.m_Permutation[2]) ||
        (transposeInfo.m_Permutation[0] == transposeInfo.m_Permutation[3]) ||
        (transposeInfo.m_Permutation[1] == transposeInfo.m_Permutation[2]) ||
        (transposeInfo.m_Permutation[1] == transposeInfo.m_Permutation[3]) ||
        (transposeInfo.m_Permutation[2] == transposeInfo.m_Permutation[3]))
    {
        SetReason("Transpose axes must be unique", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsQuantizationDimSupported(nullptr, nullptr, &inputInfo, nullptr, "Transpose", reason, reasonMaxLength))
    {
        return SupportedLevel::EstimateOnly;
    }

    TensorInfo expectedOutputInfo = Transpose::CalculateOutputTensorInfo(inputInfo, transposeInfo);

    if (!IsTensorDepthSupported(m_Capabilities, expectedOutputInfo, "Output of transpose", reason, reasonMaxLength))
    {
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

    if (!IsTransposeTensorSupported(m_Capabilities, inputInfo, transposeInfo))
    {
        SetReason("The tensors are too large", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    return SupportedLevel::Supported;
}

SupportedLevel SupportQueries::IsResizeSupported(const ResizeInfo& resizeInfo,
                                                 const TensorInfo& inputInfo,
                                                 TensorInfo* outputInfo,
                                                 char* reason,
                                                 size_t reasonMaxLength) const
{
    if (inputInfo.m_Dimensions[0] != 1)
    {
        SetReason("Batch size must be 1", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (!IsTensorDepthSupported(m_Capabilities, inputInfo, "Input to resize", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (!IsInputDataTypeSupported(inputInfo, "Input to resize", reason, reasonMaxLength))
    {
        return SupportedLevel::Unsupported;
    }

    if (inputInfo.m_DataFormat != DataFormat::NHWC && inputInfo.m_DataFormat != DataFormat::NHWCB)
    {
        SetReason("Input must be NHWC or NHWCB", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    constexpr uint32_t upscaleFactor = 2U;
    const uint32_t maxUpscaledHeight = upscaleFactor * inputInfo.m_Dimensions[1];
    const uint32_t maxUpscaledWidth  = upscaleFactor * inputInfo.m_Dimensions[2];
    const uint32_t minUpscaledHeight = upscaleFactor * inputInfo.m_Dimensions[1] - 1U;
    const uint32_t minUpscaledWidth  = upscaleFactor * inputInfo.m_Dimensions[2] - 1U;
    if (resizeInfo.m_NewHeight != maxUpscaledHeight && resizeInfo.m_NewHeight != minUpscaledHeight)
    {
        SetReason("Requested height isn't supported", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (resizeInfo.m_NewWidth != maxUpscaledWidth && resizeInfo.m_NewWidth != minUpscaledWidth)
    {
        SetReason("Requested width isn't supported", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if ((resizeInfo.m_NewWidth & 1) ^ (resizeInfo.m_NewHeight & 1))
    {
        SetReason("Requested width and height must be both even or both odd", reason, reasonMaxLength);
        return SupportedLevel::Unsupported;
    }

    if (outputInfo != nullptr)
    {
        // Note:
        //   Here we don't need to check IsTensorDepthSupported since calculated output of Resize results in an output
        //   tensor info that depth is the depth of input tensor info.
        //   This would lead to dead code as if we don't fail with input we will not fail on output.
        TensorInfo expectedOutputInfo = Resize::CalculateOutputTensorInfo(inputInfo, resizeInfo);

        if (utils::TotalSizeBytes(*outputInfo) != 0 && *outputInfo != expectedOutputInfo)
        {
            SetReason("Provided outputInfo is incorrect", reason, reasonMaxLength);
            return SupportedLevel::Unsupported;
        }
        *outputInfo = expectedOutputInfo;
    }

    return SupportedLevel::Supported;
}

}    // namespace support_library
}    // namespace ethosn
