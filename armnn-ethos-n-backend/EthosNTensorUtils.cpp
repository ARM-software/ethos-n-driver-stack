//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNTensorUtils.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnnUtils/Permute.hpp>

#include <algorithm>

namespace
{
using namespace armnn;

/// Utility function to convert armnn::DataLayout to ethosn_lib::DataFormat
ethosn_lib::DataFormat ConvertDataLayout(DataLayout dataLayout)
{
    switch (dataLayout)
    {
        case DataLayout::NCHW:
            return ethosn_lib::DataFormat::NCHW;
        case DataLayout::NHWC:
            return ethosn_lib::DataFormat::NHWC;
        default:
            throw InvalidArgumentException(std::string("Unsupported data layout: ") + GetDataLayoutName(dataLayout),
                                           CHECK_LOCATION());
    }
}

/// Utility function to convert armnn::DataType to ethosn_lib::DataType
ethosn_lib::DataType ConvertDataType(DataType dataType)
{
    switch (dataType)
    {
        case DataType::QAsymmU8:
            return ethosn_lib::DataType::UINT8_QUANTIZED;
        case DataType::QAsymmS8:
        case DataType::QSymmS8:
            return ethosn_lib::DataType::INT8_QUANTIZED;
        case DataType::Signed32:
            return ethosn_lib::DataType::INT32_QUANTIZED;
        default:
            throw InvalidArgumentException(std::string("Unsupported data type: ") + GetDataTypeName(dataType),
                                           CHECK_LOCATION());
    }
}

template <typename ArmnnDescriptor>
bool IsSupportedDilation(const ArmnnDescriptor& descriptor)
{
    return (descriptor.m_DilationX == 1U) && (descriptor.m_DilationY == 1U);
}
bool IsSupportedDilation(const armnn::TransposeConvolution2dDescriptor&)
{
    return true;
}

template <typename ArmnnDescriptor>
Optional<ethosn_lib::ConvolutionInfo> BuildEthosNConvolutionInfo(const ArmnnDescriptor& descriptor,
                                                                 float quantizationScale,
                                                                 int quantizationOffset,
                                                                 Optional<std::string&> reasonIfUnsupported = {})
{
    if (!IsSupportedDilation(descriptor))
    {
        if (reasonIfUnsupported)
        {
            reasonIfUnsupported.value() = "Dilation not supported";
        }
        return {};
    }
    using namespace ethosn_lib;
    const Padding pad(descriptor.m_PadTop, descriptor.m_PadBottom, descriptor.m_PadLeft, descriptor.m_PadRight);
    const Stride stride(descriptor.m_StrideX, descriptor.m_StrideY);
    const QuantizationInfo quantizationInfo(quantizationOffset, quantizationScale);
    return ConvolutionInfo(pad, stride, quantizationInfo);
}

bool IsTensorDataTypeSymmetric(const armnn::TensorInfo& tensorInfo)
{
    switch (tensorInfo.GetDataType())
    {
        case DataType::QSymmS8:
            return true;
        default:
            return false;
    }
}

template <typename T>
ethosn_lib::ReluInfo BuildEthosNReluInfoImpl(const armnn::ActivationDescriptor& descriptor,
                                             const float inputQuantizationScale,
                                             const int inputQuantizationOffset)
{
    ethosn_lib::ReluInfo reluInfo;

    if (descriptor.m_Function == ActivationFunction::BoundedReLu)
    {
        reluInfo.m_LowerBound = Quantize<T>(descriptor.m_B, inputQuantizationScale, inputQuantizationOffset);
        reluInfo.m_UpperBound = Quantize<T>(descriptor.m_A, inputQuantizationScale, inputQuantizationOffset);
    }
    else
    {
        ARMNN_ASSERT(descriptor.m_Function == ActivationFunction::ReLu);

        reluInfo.m_LowerBound = Quantize<T>(0, inputQuantizationScale, inputQuantizationOffset);
        reluInfo.m_UpperBound = std::numeric_limits<T>::max();
    }

    return reluInfo;
}

}    // namespace

namespace armnn
{
namespace ethosntensorutils
{

/// Utility function used to build a TensorShape object, that can be used to initialise
/// Ethos-N Tensor.
ethosn_lib::TensorShape BuildEthosNTensorShape(const armnn::TensorShape& tensorShape)
{
    constexpr size_t maxEthosNDims = ethosn_lib::TensorShape{}.size();

    const size_t numDims = tensorShape.GetNumDimensions();

    if (numDims > maxEthosNDims)
    {
        std::string msg = "Invalid TensorShape: max number of dimensions exceeded in EthosNAcc backend ";
        msg += std::to_string(numDims);
        msg += " > ";
        msg += std::to_string(maxEthosNDims);
        throw armnn::InvalidArgumentException(msg);
    }

    ethosn_lib::TensorShape shape;

    // Ethos-N tensors are always 4d, insert length 1 dimensions to pad.
    // We always pad the batch dimension first if the first dimension is >1 and any extra dimension
    // to the "right" (i.e. both 10x20 or 1x10x20 become 1x10x20x1) as this should result in more
    // favourable strategies being chosen. This is because the Height dimension will tend to stay large
    // which means strategy 0 can be used, instead of Height becoming 1 which could lead to strategy 4.
    const size_t dimOffset = ((tensorShape[0] > 1U) && (numDims < maxEthosNDims)) ? 1U : 0U;

    for (size_t i = 0; i < dimOffset; ++i)
    {
        shape[i] = 1;
    }
    for (unsigned int i = 0; i < numDims; ++i)
    {
        shape[i + dimOffset] = tensorShape[i];
    }
    for (size_t i = numDims + dimOffset; i < maxEthosNDims; ++i)
    {
        shape[i] = 1;
    }

    return shape;
}

/// Utility function used to build a TensorInfo object, that can be used to initialise a Ethos-N Tensor.
ethosn_lib::TensorInfo BuildEthosNTensorInfo(const armnn::TensorInfo& tensorInfo, DataLayout dataLayout)
{
    const ethosn_lib::DataFormat ethosnDataFormat = ConvertDataLayout(dataLayout);
    const ethosn_lib::DataType ethosnDataType     = ConvertDataType(tensorInfo.GetDataType());
    const int32_t zeroPoint = IsTensorDataTypeSymmetric(tensorInfo) ? 0 : tensorInfo.GetQuantizationOffset();
    ethosn_lib::QuantizationInfo ethosnQuantizationInfo(zeroPoint, { 1.0f });
    if (tensorInfo.HasMultipleQuantizationScales())
    {
        ethosnQuantizationInfo.SetScales(tensorInfo.GetQuantizationScales());
    }
    else
    {
        ethosnQuantizationInfo.SetScale(tensorInfo.GetQuantizationScale());
    }
    if (tensorInfo.GetQuantizationDim())
    {
        ethosnQuantizationInfo.SetQuantizationDim(tensorInfo.GetQuantizationDim().value());
    }

    const ethosn_lib::TensorShape EthosNTensorShape = BuildEthosNTensorShape(tensorInfo.GetShape());

    return ethosn_lib::TensorInfo(EthosNTensorShape, ethosnDataType, ethosnDataFormat, ethosnQuantizationInfo);
}

ethosn_lib::TensorInfo BuildEthosNTensorInfo(const Optional<TensorInfo>& tensorInfo, DataLayout dataLayout)
{
    if (tensorInfo)
    {
        return BuildEthosNTensorInfo(tensorInfo.value(), dataLayout);
    }

    // Return dummy tensor info for empty biases
    return ethosn_lib::TensorInfo();
}

PermutationVector GetEthosNConvolutionWeightsPermutationVector(DataLayout layerLayout, bool isDepthwiseConvolution)
{
    if (isDepthwiseConvolution)
    {
        // MIHW to HWIM
        return PermutationVector({ 3, 2, 0, 1 });
    }

    switch (layerLayout)
    {
        case DataLayout::NCHW:
            // OIHW to HWIO
            return PermutationVector({ 3, 2, 0, 1 });
        default:
            // OHWI to HWIO
            ARMNN_ASSERT(layerLayout == DataLayout::NHWC);
            return PermutationVector({ 3, 0, 1, 2 });
    }
}

ethosn_lib::TensorInfo BuildEthosNConvolutionWeightsInfo(const TensorInfo& weightsInfo,
                                                         DataLayout layerLayout,
                                                         bool isDepthwiseConvolution)
{
    const PermutationVector permutationVector =
        GetEthosNConvolutionWeightsPermutationVector(layerLayout, isDepthwiseConvolution);
    TensorInfo swizzledWeightsInfo = armnnUtils::Permuted(weightsInfo, permutationVector);

    ethosn_lib::TensorInfo ethosnWeightsInfo = BuildEthosNTensorInfo(swizzledWeightsInfo, DataLayout::NHWC);

    // NOTE: We need to specify HWIO as the DataFormat on the Ethos-N side
    // for all use cases except depthwise convolution, which requires HWIM
    ethosnWeightsInfo.m_DataFormat =
        isDepthwiseConvolution ? ethosn_lib::DataFormat::HWIM : ethosn_lib::DataFormat::HWIO;

    return ethosnWeightsInfo;
}

ethosn_lib::TensorInfo BuildEthosNFullyConnectedWeightsInfo(const TensorInfo& weightsInfo, bool transposeWeightMatrix)
{
    // Weight tensor is guaranteed to be 2D by the Arm NN validation (FullyConnectedQueueDescriptor::Validate).
    ARMNN_ASSERT(weightsInfo.GetNumDimensions() == 2);

    ethosn_lib::TensorInfo ethosnWeightsInfo;
    if (transposeWeightMatrix)
    {
        const TensorShape& weightsShape = weightsInfo.GetShape();

        // OI -> IO
        const TensorShape transposedWeightsShape = TensorShape({ weightsShape[1], weightsShape[0] });

        const TensorInfo transposedWeightsInfo(transposedWeightsShape, weightsInfo.GetDataType(),
                                               weightsInfo.GetQuantizationScale(), weightsInfo.GetQuantizationOffset());

        ethosnWeightsInfo = BuildEthosNTensorInfo(transposedWeightsInfo, DataLayout::NHWC);
        // The padding to 4D done by BuildEthosNTensorInfo is incorrect.
        ethosnWeightsInfo.m_Dimensions = { 1, 1, transposedWeightsInfo.GetShape()[0],
                                           transposedWeightsInfo.GetShape()[1] };
    }
    else
    {
        ethosnWeightsInfo = BuildEthosNTensorInfo(weightsInfo, DataLayout::NHWC);
        // The padding to 4D done by BuildEthosNTensorInfo is incorrect.
        ethosnWeightsInfo.m_Dimensions = { 1, 1, weightsInfo.GetShape()[0], weightsInfo.GetShape()[1] };
    }

    ethosnWeightsInfo.m_DataFormat = ethosn_lib::DataFormat::HWIO;
    return ethosnWeightsInfo;
}

ethosn_lib::TensorInfo
    BuildEthosNBiasesInfo(const TensorInfo& biasesInfo, const TensorInfo& inputInfo, const TensorInfo& weightsInfo)
{
    // The Arm NN bias tensor should be 1D, as validated by Arm NN's layer validation.
    // For unknown reasons however, there is a test that checks that a 4D bias works (1x1x1xN), so we must
    // make that work too :(
    auto ethosnBiasesInfo = BuildEthosNTensorInfo(biasesInfo, armnn::DataLayout::NHWC);
    // We must set the correct index for the QuantizationDim as well.
    if (ethosnBiasesInfo.m_QuantizationInfo.GetQuantizationDim())
    {
        ethosnBiasesInfo.m_QuantizationInfo.SetQuantizationDim(3U);
    }
    // The shape returned by BuildEthosNTensorInfo will be padded incorrectly.
    ethosnBiasesInfo.m_Dimensions = { 1, 1, 1, biasesInfo.GetNumElements() };
    // For some networks the quantisation scale of the bias may not be exactly correct, but the Ethos-N requires this.
    // Arm NN will validate that they are at least approximately correct, so we are safe to force this here.
    ethosn_lib::QuantizationScales inputScales(inputInfo.GetQuantizationScales());
    ethosn_lib::QuantizationScales weightScales(weightsInfo.GetQuantizationScales());
    ethosnBiasesInfo.m_QuantizationInfo.SetScales(inputScales * weightScales);
    return ethosnBiasesInfo;
}

armnn::ethosn_lib::TensorInfo
    BuildEthosNBiasesInfo(unsigned int numBiasElements, const TensorInfo& inputInfo, const TensorInfo& weightsInfo)
{
    TensorInfo biasInfo({ numBiasElements }, DataType::Signed32);

    return BuildEthosNBiasesInfo(biasInfo, inputInfo, weightsInfo);
}

Optional<ethosn_lib::ConvolutionInfo> BuildEthosNConvolutionInfo(const armnn::Convolution2dDescriptor& descriptor,
                                                                 int32_t quantizationOffset,
                                                                 float quantizationScale,
                                                                 Optional<std::string&> reasonIfUnsupported)
{
    return ::BuildEthosNConvolutionInfo(descriptor, quantizationScale, quantizationOffset, reasonIfUnsupported);
}

Optional<ethosn_lib::ConvolutionInfo>
    BuildEthosNConvolutionInfo(const armnn::DepthwiseConvolution2dDescriptor& descriptor,
                               int32_t quantizationOffset,
                               float quantizationScale,
                               Optional<std::string&> reasonIfUnsupported)
{
    return ::BuildEthosNConvolutionInfo(descriptor, quantizationScale, quantizationOffset, reasonIfUnsupported);
}

ethosn_lib::ConvolutionInfo BuildEthosNConvolutionInfo(const armnn::TransposeConvolution2dDescriptor& descriptor,
                                                       int32_t quantizationOffset,
                                                       float quantizationScale)
{
    return ::BuildEthosNConvolutionInfo(descriptor, quantizationScale, quantizationOffset, {}).value();
}

ethosn_lib::FullyConnectedInfo BuildEthosNFullyConnectedLayerInfo(const FullyConnectedDescriptor&,
                                                                  int32_t quantizationOffset,
                                                                  float quantizationScale)
{
    return ethosn_lib::FullyConnectedInfo(ethosn_lib::QuantizationInfo(quantizationOffset, quantizationScale));
}

ethosn_lib::PoolingInfo BuildEthosNPoolingLayerInfo(const armnn::Pooling2dDescriptor& descriptor)
{
    using ethosn_lib::Padding;
    using ethosn_lib::PoolingInfo;
    using ethosn_lib::PoolingType;

    const PoolingType poolingType = ConvertPoolingAlgorithmToEthosNPoolingType(descriptor.m_PoolType);
    const Padding padding         = ConvertPaddingToEthosNPadding(descriptor.m_PadTop, descriptor.m_PadBottom,
                                                          descriptor.m_PadLeft, descriptor.m_PadRight);

    const PoolingInfo poolingInfo(descriptor.m_PoolHeight, descriptor.m_PoolWidth, descriptor.m_StrideX,
                                  descriptor.m_StrideY, padding, poolingType);

    return ethosn_lib::PoolingInfo(poolingInfo);
}

Optional<ethosn_lib::ReluInfo> BuildEthosNReluInfo(const armnn::ActivationDescriptor& descriptor,
                                                   armnn::DataType inputDataType,
                                                   const float inputQuantizationScale,
                                                   const int inputQuantizationOffset)
{
    switch (inputDataType)
    {
        case DataType::QAsymmU8:
        {
            return BuildEthosNReluInfoImpl<uint8_t>(descriptor, inputQuantizationScale, inputQuantizationOffset);
        }
        case DataType::QAsymmS8:    // Intentional fallthrough
        case DataType::QSymmS8:
        {
            return BuildEthosNReluInfoImpl<int8_t>(descriptor, inputQuantizationScale, inputQuantizationOffset);
        }
        default:
        {
            return Optional<ethosn_lib::ReluInfo>();
        }
    }
}

ethosn_lib::LeakyReluInfo BuildEthosNLeakyReluInfo(const armnn::ActivationDescriptor& descriptor,
                                                   const float quantizationScale,
                                                   const int quantizationOffset)
{
    ethosn_lib::LeakyReluInfo leakyReluInfo;

    ARMNN_ASSERT(descriptor.m_Function == ActivationFunction::LeakyReLu);

    leakyReluInfo.m_Alpha                  = descriptor.m_A;
    leakyReluInfo.m_OutputQuantizationInfo = ethosn_lib::QuantizationInfo(quantizationOffset, quantizationScale);

    return leakyReluInfo;
}

Optional<ethosn_lib::TransposeInfo> BuildEthosNTransposeInfo(const armnn::PermutationVector& descriptor)
{
    if (descriptor.GetSize() != 4)
    {
        return Optional<ethosn_lib::TransposeInfo>();
    }

    std::array<uint32_t, 4> permutation;
    std::copy(descriptor.begin(), descriptor.begin() + descriptor.GetSize(), permutation.begin());

    Optional<ethosn_lib::TransposeInfo> transposeInfo(permutation);

    return transposeInfo;
}

ethosn_lib::RequantizeInfo BuildEthosNRequantizeInfo(const float quantizationScale, const int quantizationOffset)
{
    ethosn_lib::RequantizeInfo requantizeInfo;

    requantizeInfo.m_OutputQuantizationInfo = ethosn_lib::QuantizationInfo(quantizationOffset, quantizationScale);

    return requantizeInfo;
}

ethosn_lib::ResizeInfo
    BuildEthosNResizeInfo(const armnn::ResizeDescriptor& descriptor, float quantizationScale, int quantizationOffset)
{
    ethosn_lib::ResizeAlgorithm algo = ethosn_lib::ResizeAlgorithm::BILINEAR;

    switch (descriptor.m_Method)
    {
        case (ResizeMethod::Bilinear):
        {
            algo = ethosn_lib::ResizeAlgorithm::BILINEAR;
            break;
        }
        case (ResizeMethod::NearestNeighbor):
        {
            algo = ethosn_lib::ResizeAlgorithm::NEAREST_NEIGHBOUR;
            break;
        }
        default:
        {
            return ethosn_lib::ResizeInfo();
        }
    }

    return ethosn_lib::ResizeInfo(algo, descriptor.m_TargetHeight, descriptor.m_TargetWidth,
                                  ethosn_lib::QuantizationInfo(quantizationOffset, quantizationScale));
}

namespace
{

Optional<uint32_t> CalculateEthosNSplitAxis(const ViewsDescriptor& splitterDescriptor)
{
    ARMNN_ASSERT(splitterDescriptor.GetNumViews() >= 2);
    // The first view's origin should be at 0 in all dimensions.
    TensorShape origin0(splitterDescriptor.GetNumDimensions(), splitterDescriptor.GetViewOrigin(0));
    for (uint32_t d = 0; d < splitterDescriptor.GetNumDimensions(); ++d)
    {
        if (origin0[d] != 0)
        {
            return armnn::Optional<uint32_t>();
        }
    }

    // The second view's origin should be non-zero in exactly one dimension - the split axis
    Optional<uint32_t> result;
    TensorShape origin1(splitterDescriptor.GetNumDimensions(), splitterDescriptor.GetViewOrigin(1));
    for (uint32_t d = 0; d < splitterDescriptor.GetNumDimensions(); ++d)
    {
        if (origin1[d] != 0)
        {
            if (result.has_value())
            {
                // Not a single-axis split.
                return armnn::Optional<uint32_t>();
            }
            result = d;
        }
    }
    return result;
}

}    // namespace

armnn::Optional<armnn::ethosn_lib::SplitInfo> BuildEthosNSplitInfo(const TensorShape& inputShape,
                                                                   const ViewsDescriptor& splitterDescriptor)
{
    ARMNN_ASSERT(inputShape.GetNumDimensions() == splitterDescriptor.GetNumDimensions());
    // We need at least two views to determine a split axis
    if (splitterDescriptor.GetNumViews() < 2)
    {
        return armnn::Optional<armnn::ethosn_lib::SplitInfo>();
    }

    // First determine the split axis based on the origins.
    Optional<uint32_t> ethosnSplitAxis = CalculateEthosNSplitAxis(splitterDescriptor);
    if (!ethosnSplitAxis.has_value())
    {
        // No split axis detected
        return armnn::Optional<armnn::ethosn_lib::SplitInfo>();
    }

    // Now we know the split axis, calculate the size of each split and validate that they are contiguous and
    // monotonically increasing, and completely cover the input tensor.
    std::vector<uint32_t> ethosnSizes;
    uint32_t runningTotal = 0;
    for (uint32_t i = 0; i < splitterDescriptor.GetNumViews(); ++i)
    {
        TensorShape origin(splitterDescriptor.GetNumDimensions(), splitterDescriptor.GetViewOrigin(i));
        TensorShape size(splitterDescriptor.GetNumDimensions(), splitterDescriptor.GetViewSizes(i));

        // All views' origins should be zero on all except the split axis
        for (uint32_t d = 0; d < splitterDescriptor.GetNumDimensions(); ++d)
        {
            if (d != ethosnSplitAxis.value() && origin[d] != 0)
            {
                return armnn::Optional<armnn::ethosn_lib::SplitInfo>();
            }
        }
        // Check that the origins along the split axis are contiguous
        if (origin[ethosnSplitAxis.value()] != runningTotal)
        {
            return armnn::Optional<armnn::ethosn_lib::SplitInfo>();
        }
        // All sizes must be the full size of the input tensor, except along the split dimension.
        for (uint32_t d = 0; d < splitterDescriptor.GetNumDimensions(); ++d)
        {
            if (d != ethosnSplitAxis.value() && size[d] != inputShape[d])
            {
                return armnn::Optional<armnn::ethosn_lib::SplitInfo>();
            }
        }

        ethosnSizes.push_back(size[ethosnSplitAxis.value()]);
        runningTotal += size[ethosnSplitAxis.value()];
    }

    return ethosn_lib::SplitInfo(ethosnSplitAxis.value(), ethosnSizes);
}

bool IsDataTypeSupportedOnEthosN(const DataType dataType)
{
    switch (dataType)
    {
        case DataType::QAsymmU8:
        case DataType::QAsymmS8:
        case DataType::QSymmS8:
        case DataType::Signed32:
            return true;
        default:
        {
            return false;
        }
    }
}

namespace
{

template <typename T>
std::vector<int32_t> ConvertDataToInt32(const void* rawData,
                                        const uint32_t numElements,
                                        const float originalScale,
                                        const int32_t originalZeroPoint,
                                        const float newScale)
{
    const T* data = static_cast<const T*>(rawData);
    std::vector<int32_t> newConstantLayerData;
    newConstantLayerData.reserve(numElements);

    std::transform(data, data + numElements, std::back_inserter(newConstantLayerData), [=](const T x) {
        const float fpValue = originalScale * static_cast<float>(x - originalZeroPoint);
        return static_cast<int32_t>(std::round(fpValue / newScale));
    });

    return newConstantLayerData;
}

}    // namespace

armnn::Optional<std::vector<int32_t>> ConvertTensorValuesToSigned32(const void* srcData,
                                                                    const armnn::TensorInfo& srcInfo,
                                                                    const armnn::TensorInfo& dstInfo)
{
    if (dstInfo.GetDataType() != DataType::Signed32)
    {
        return EmptyOptional();
    }
    // The below rescaling function does not support a non-zero new offset
    if (dstInfo.GetQuantizationOffset() != 0)
    {
        return EmptyOptional();
    }
    switch (srcInfo.GetDataType())
    {
        case DataType::QAsymmU8:
            return ConvertDataToInt32<uint8_t>(srcData, srcInfo.GetNumElements(), srcInfo.GetQuantizationScale(),
                                               srcInfo.GetQuantizationOffset(), dstInfo.GetQuantizationScale());
        case DataType::QAsymmS8:
            return ConvertDataToInt32<int8_t>(srcData, srcInfo.GetNumElements(), srcInfo.GetQuantizationScale(),
                                              srcInfo.GetQuantizationOffset(), dstInfo.GetQuantizationScale());
        case DataType::QSymmS8:
            return ConvertDataToInt32<int8_t>(srcData, srcInfo.GetNumElements(), srcInfo.GetQuantizationScale(), 0,
                                              dstInfo.GetQuantizationScale());
        default:
            // Unsupported constant data type.
            // Note we should be able to easily add support by writing a conversion function like above
            return EmptyOptional();
    }
}

}    // namespace ethosntensorutils
}    // namespace armnn
