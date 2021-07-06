//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Optional.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnnUtils/Permute.hpp>
#include <ethosn_support_library/Support.hpp>

namespace armnn
{
class ITensorHandle;

namespace ethosn_lib = ethosn::support_library;

namespace ethosntensorutils
{

template <typename T>
void Swizzle1HWOToHWIM(const void* inputBuffer,
                       void* outputBuffer,
                       const armnn::TensorShape& inputShape,
                       unsigned int depthMultiplier)
{
    ARMNN_ASSERT(inputBuffer != nullptr);
    ARMNN_ASSERT(outputBuffer != nullptr);

    ARMNN_ASSERT(inputShape.GetNumDimensions() == 4);

    const T* typedInputData = reinterpret_cast<const T*>(inputBuffer);
    std::vector<T> output;
    output.reserve(inputShape.GetNumElements());

    uint32_t dimH = inputShape[1];
    uint32_t dimW = inputShape[2];
    uint32_t dimI = inputShape[3] / depthMultiplier;
    uint32_t dimM = depthMultiplier;

    for (unsigned int indH = 0; indH < dimH; indH++)
    {
        for (unsigned int indW = 0; indW < dimW; indW++)
        {
            for (unsigned int indI = 0; indI < dimI; indI++)
            {
                for (unsigned int indM = 0; indM < dimM; indM++)
                {
                    uint32_t flatIndex = (indH * dimW * dimI * dimM) + (indW * dimI * dimM) + (indI * dimM) + indM;
                    T elem             = typedInputData[flatIndex];
                    output.push_back(elem);
                }
            }
        }
    }

    memcpy(outputBuffer, output.data(), inputShape.GetNumElements() * sizeof(T));
}

template <typename T>
void SwizzleOHWIToHWIO(const void* inputBuffer, void* outputBuffer, const armnn::TensorShape& inputShape)
{
    ARMNN_ASSERT(inputBuffer != nullptr);
    ARMNN_ASSERT(outputBuffer != nullptr);

    ARMNN_ASSERT(inputShape.GetNumDimensions() == 4);

    const T* typedInputData = reinterpret_cast<const T*>(inputBuffer);
    std::vector<T> output;
    output.reserve(inputShape.GetNumElements());

    uint32_t dimO = inputShape[0];
    uint32_t dimH = inputShape[1];
    uint32_t dimW = inputShape[2];
    uint32_t dimI = inputShape[3];

    for (unsigned int indH = 0; indH < dimH; indH++)
    {
        for (unsigned int indW = 0; indW < dimW; indW++)
        {
            for (unsigned int indI = 0; indI < dimI; indI++)
            {
                for (unsigned int indO = 0; indO < dimO; indO++)
                {
                    uint32_t flatIndex = (indO * dimH * dimW * dimI) + (indH * dimW * dimI) + (indW * dimI) + indI;
                    T elem             = typedInputData[flatIndex];
                    output.push_back(elem);
                }
            }
        }
    }

    memcpy(outputBuffer, output.data(), inputShape.GetNumElements() * sizeof(T));
}

template <typename T>
void SwizzleOIHWToHWIO(const void* inputBuffer, void* outputBuffer, const armnn::TensorShape& inputShape)
{
    ARMNN_ASSERT(inputBuffer != nullptr);
    ARMNN_ASSERT(outputBuffer != nullptr);

    ARMNN_ASSERT(inputShape.GetNumDimensions() == 4);

    const T* typedInputData = reinterpret_cast<const T*>(inputBuffer);
    std::vector<T> output;
    output.reserve(inputShape.GetNumElements());

    uint32_t dimO = inputShape[0];
    uint32_t dimI = inputShape[1];
    uint32_t dimH = inputShape[2];
    uint32_t dimW = inputShape[3];

    for (unsigned int indH = 0; indH < dimH; indH++)
    {
        for (unsigned int indW = 0; indW < dimW; indW++)
        {
            for (unsigned int indI = 0; indI < dimI; indI++)
            {
                for (unsigned int indO = 0; indO < dimO; indO++)
                {
                    uint32_t flatIndex = (indO * dimI * dimH * dimW) + (indI * dimH * dimW) + (indH * dimW) + indW;
                    T elem             = typedInputData[flatIndex];
                    output.push_back(elem);
                }
            }
        }
    }

    memcpy(outputBuffer, output.data(), inputShape.GetNumElements() * sizeof(T));
}

template <typename T>
void SwizzleConvolutionWeightsData(const void* inputBuffer,
                                   void* outputBuffer,
                                   const armnn::TensorShape& inputShape,
                                   DataLayout layerLayout,
                                   bool isDepthwiseConvolution,
                                   unsigned int depthMultiplier)
{
    if (isDepthwiseConvolution)
    {
        /// Weights for depthwise have a datalayout of [1,H,W,O] = [1,H,W,I*M] -> HWIM
        Swizzle1HWOToHWIM<T>(inputBuffer, outputBuffer, inputShape, depthMultiplier);
    }
    else
    {
        // Convolution
        switch (layerLayout)
        {
            case DataLayout::NCHW:
                // OIHW -> HWIO
                SwizzleOIHWToHWIO<T>(inputBuffer, outputBuffer, inputShape);
                break;
            default:
                // OHWI -> HWIO
                ARMNN_ASSERT(layerLayout == DataLayout::NHWC);
                SwizzleOHWIToHWIO<T>(inputBuffer, outputBuffer, inputShape);
        }
    }
}

ethosn_lib::TensorShape BuildEthosNTensorShape(const armnn::TensorShape& tensorShape);

/// Utility function to setup a ethosn_lib::TensorInfo object.
/// Ethos-N tensors include the data layout (NHWC, NCHW etc.) which we require in addition to the Arm NN tensor description.
/// @{
ethosn_lib::TensorInfo BuildEthosNTensorInfo(const armnn::TensorInfo& tensorInfo, DataLayout dataLayout);
ethosn_lib::TensorInfo BuildEthosNTensorInfo(const Optional<TensorInfo>& tensorInfo, DataLayout dataLayout);
/// @}

ethosn_lib::TensorInfo BuildEthosNConvolutionWeightsInfo(const armnn::TensorInfo& weightsInfo,
                                                         const armnn::TensorInfo& inputInfo,
                                                         DataLayout layerLayout,
                                                         bool isDepthwiseConvolution);

ethosn_lib::TensorInfo BuildEthosNFullyConnectedWeightsInfo(const armnn::TensorInfo& weightsInfo,
                                                            bool transposeWeightMatrix = false);

ethosn_lib::TensorInfo
    BuildEthosNBiasesInfo(const TensorInfo& biasesInfo, const TensorInfo& inputInfo, const TensorInfo& weightsInfo);
ethosn_lib::TensorInfo
    BuildEthosNBiasesInfo(unsigned int numBiasElements, const TensorInfo& inputInfo, const TensorInfo& weightsInfo);

/// Utility function to set up a ethosn_lib::ConvolutionInfo object from armnn::Convolution2dDescriptor
Optional<ethosn_lib::ConvolutionInfo> BuildEthosNConvolutionInfo(const armnn::Convolution2dDescriptor& descriptor,
                                                                 int32_t quantizationOffset,
                                                                 float quantizationScale,
                                                                 Optional<std::string&> reasonIfUnsupported);

/// Utility function to set up a ethosn_lib::ConvolutionInfo object from armnn::DepthwiseConvolution2dDescriptor
Optional<ethosn_lib::ConvolutionInfo>
    BuildEthosNConvolutionInfo(const armnn::DepthwiseConvolution2dDescriptor& descriptor,
                               int32_t quantizationOffset,
                               float quantizationScale,
                               Optional<std::string&> reasonIfUnsupported);

/// Utility function to set up a ethosn_lib::ConvolutionInfo object from armnn::TransposeConvolution2dDescriptor
ethosn_lib::ConvolutionInfo BuildEthosNConvolutionInfo(const armnn::TransposeConvolution2dDescriptor& descriptor,
                                                       int32_t quantizationOffset,
                                                       float quantizationScale);

/// Utility function used to setup an ethosn_lib::FullyConnectedLayerInfo object from an armnn::FullyConnectedDescriptor
ethosn_lib::FullyConnectedInfo BuildEthosNFullyConnectedLayerInfo(const FullyConnectedDescriptor& descriptor,
                                                                  int32_t quantizationOffset,
                                                                  float quantizationScale);

/// Utility function used to setup an ethosn_lib::PoolingLayerInfo object from an armnn::Pooling2dDescriptor.
ethosn_lib::PoolingInfo BuildEthosNPoolingLayerInfo(const armnn::Pooling2dDescriptor& descriptor);

/// Utility function to setup a ethosn_lib::PoolingType object from an armnn::PoolingAlgorithm
inline ethosn_lib::PoolingType ConvertPoolingAlgorithmToEthosNPoolingType(PoolingAlgorithm poolingAlgorithm)
{
    using ethosn_lib::PoolingType;
    switch (poolingAlgorithm)
    {
        case PoolingAlgorithm::Max:
            return PoolingType::MAX;
        case PoolingAlgorithm::Average:
            return PoolingType::AVG;
        default:
            throw InvalidArgumentException("Unsupported pooling algorithm");
    }
}

/// Utility function to setup a ethosn_lib::Padding object from the given padding values
inline ethosn_lib::Padding ConvertPaddingToEthosNPadding(uint32_t top, uint32_t bottom, uint32_t left, uint32_t right)
{
    return ethosn_lib::Padding(top, bottom, left, right);
}

Optional<ethosn_lib::ReluInfo> BuildEthosNReluInfo(const armnn::ActivationDescriptor& descriptor,
                                                   armnn::DataType inputDataType,
                                                   float inputQuantizationScale,
                                                   int inputQuantizationOffset);

inline Optional<ethosn_lib::ReluInfo> BuildEthosNReluInfo(const armnn::ActivationDescriptor& descriptor,
                                                          const armnn::TensorInfo& inputInfo)
{
    return BuildEthosNReluInfo(descriptor, inputInfo.GetDataType(), inputInfo.GetQuantizationScale(),
                               inputInfo.GetQuantizationOffset());
}

ethosn_lib::LeakyReluInfo BuildEthosNLeakyReluInfo(const armnn::ActivationDescriptor& descriptor,
                                                   float quantizationScale,
                                                   int quantizationOffset);

inline ethosn_lib::LeakyReluInfo BuildEthosNLeakyReluInfo(const armnn::ActivationDescriptor& descriptor,
                                                          const armnn::TensorInfo& outputInfo)
{
    return BuildEthosNLeakyReluInfo(descriptor, outputInfo.GetQuantizationScale(), outputInfo.GetQuantizationOffset());
}

Optional<ethosn_lib::SplitInfo> BuildEthosNSplitInfo(const TensorShape& inputShape,
                                                     const ViewsDescriptor& splitterDescriptor);

Optional<ethosn_lib::TransposeInfo> BuildEthosNTransposeInfo(const armnn::PermutationVector& descriptor);

ethosn_lib::RequantizeInfo BuildEthosNRequantizeInfo(const float quantizationScale, const int quantizationOffset);

inline ethosn_lib::RequantizeInfo BuildEthosNRequantizeInfo(const armnn::TensorInfo& outputInfo)
{
    return BuildEthosNRequantizeInfo(outputInfo.GetQuantizationScale(), outputInfo.GetQuantizationOffset());
}

ethosn_lib::ReinterpretQuantizationInfo BuildEthosNReinterpretQuantizationInfo(const float quantizationScale,
                                                                               const int quantizationOffset);

inline ethosn_lib::ReinterpretQuantizationInfo
    BuildEthosNReinterpretQuantizationInfo(const armnn::TensorInfo& outputInfo)
{
    return BuildEthosNReinterpretQuantizationInfo(outputInfo.GetQuantizationScale(),
                                                  outputInfo.GetQuantizationOffset());
}

ethosn_lib::ResizeInfo
    BuildEthosNResizeInfo(const armnn::ResizeDescriptor& descriptor, float quantizationScale, int quantizationOffset);

inline ethosn_lib::ResizeInfo BuildEthosNResizeInfo(const armnn::ResizeDescriptor& descriptor,
                                                    const armnn::TensorInfo& outputInfo)
{
    return BuildEthosNResizeInfo(descriptor, outputInfo.GetQuantizationScale(), outputInfo.GetQuantizationOffset());
}

bool IsDataTypeSupportedOnEthosN(const armnn::DataType dataType);

/// Converts the values in the given tensor from the source tensor info to the dest tensor info.
/// This can change be used for example to change the data type and/or quantization params.
/// Returns an empty optional if the conversion is not supported.
Optional<std::vector<int32_t>> ConvertTensorValuesToSigned32(const void* srcData,
                                                             const armnn::TensorInfo& srcInfo,
                                                             const armnn::TensorInfo& dstInfo);

}    // namespace ethosntensorutils
}    // namespace armnn
