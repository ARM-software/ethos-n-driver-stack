//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
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

size_t DepthOf(const Operation& operation)
{
    size_t depth = 0;

    const std::vector<const Operand*> inputs = operation.GetInputs();

    if (inputs.size() > 0)
    {
        const auto maxDepth = [](const size_t depth, const Operand* input) {
            return std::max(depth, DepthOf(input->GetProducer()));
        };

        depth = 1 + std::accumulate(inputs.begin(), inputs.end(), size_t{ 0 }, maxDepth);
    }

    return depth;
}

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

void PrintOperation(std::ostream& os, const Operation& operation, const std::string& name, const std::string& extra)
{
    const size_t widthOfName = 12;

    const std::string postName = std::string(widthOfName - std::min(widthOfName, name.length()), ' ');

    os << "  " << std::setw(2) << DepthOf(operation) << ": ";
    os << name.substr(0, widthOfName) << postName << " ( ";

    for (const Operand* o : operation.GetInputs())
    {
        os << o << " ";
    }

    os << "->";

    for (const Operand& o : operation.GetOutputs())
    {
        os << " " << &o;
    }

    os << " )" << extra << std::endl;
}

void PrintOperation(std::ostream& os, const Operation& operation, const std::string& name)
{
    PrintOperation(os, operation, name, "");
}

}    // namespace

Input::Input(const detail::PosInNetwork pos, uint32_t id, const TensorInfo& info)
    : VisitableOperation<Input>(pos, id, {}, { info })
    , m_Info(info)
{}

void Input::Print(std::ostream& os)
{
    PrintOperation(os, *this, "Input");
}

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

void Output::Print(std::ostream& os)
{
    PrintOperation(os, *this, "Output");
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

template <typename T>
std::vector<T> Constant::GetDataVectorAs() const
{
    assert(m_Data.size() % sizeof(T) == 0);    // Otherwise won't fit exactly in result type.
    size_t numElements = m_Data.size() / sizeof(T);
    std::vector<T> result(numElements);
    std::memcpy(result.data(), m_Data.data(), m_Data.size());
    return result;
}

template std::vector<int32_t> Constant::GetDataVectorAs<int32_t>() const;

void Constant::Print(std::ostream& os)
{
    PrintOperation(os, *this, "Constant");
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

void Convolution::Print(std::ostream& os)
{
    std::stringstream ss;
    // clang-format off
    ss << " {"
        << " bias: " << &m_Bias.GetOutput(0)
        << ","
        << " weights: " << &m_Weights.GetOutput(0)
        << " }";
    // clang-format on
    PrintOperation(os, *this, "Convolution", ss.str());
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

void DepthwiseConvolution::Print(std::ostream& os)
{
    PrintOperation(os, *this, "DepthwiseConvolution");
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

void TransposeConvolution::Print(std::ostream& os)
{
    std::stringstream ss;
    // clang-format off
    ss << " {"
        << " bias: " << &m_Bias.GetOutput(0)
        << ","
        << " weights: " << &m_Weights.GetOutput(0)
        << " }";
    // clang-format on
    PrintOperation(os, *this, "TransposeConvolution", ss.str());
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

void Addition::Print(std::ostream& os)
{
    PrintOperation(os, *this, "Addition");
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

void FullyConnected::Print(std::ostream& os)
{
    std::stringstream ss;
    // clang-format off
            ss << " {"
               << " bias: " << &m_Bias.GetOutput(0)
               << ","
               << " weights: " << &m_Weights.GetOutput(0)
               << " }";
    // clang-format on
    PrintOperation(os, *this, "FullyConnected", ss.str());
}

Relu::Relu(const detail::PosInNetwork pos, uint32_t id, Operand& input, const ReluInfo& reluInfo)
    : VisitableOperation<Relu>(pos, id, { &input }, { input.GetTensorInfo() })
    , m_ReluInfo(reluInfo)
{}

void Relu::Print(std::ostream& os)
{
    PrintOperation(os, *this, "Relu");
}

Softmax::Softmax(const detail::PosInNetwork pos, uint32_t id, Operand& input)
    : VisitableOperation<Softmax>(pos, id, { &input }, { input.GetTensorInfo() })
{}

void Softmax::Print(std::ostream& os)
{
    PrintOperation(os, *this, "Softmax");
}

Sigmoid::Sigmoid(const detail::PosInNetwork pos, uint32_t id, Operand& input)
    : VisitableOperation<Sigmoid>(pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo()) })
{}

TensorInfo Sigmoid::CalculateOutputTensorInfo(const TensorInfo& inputInfo)
{
    TensorInfo outInfo         = inputInfo;
    outInfo.m_QuantizationInfo = QuantizationInfo(0, 1.f / 256);
    return outInfo;
}

void Sigmoid::Print(std::ostream& os)
{
    PrintOperation(os, *this, "Sigmoid");
}

Pooling::Pooling(const detail::PosInNetwork pos, uint32_t id, Operand& input, const PoolingInfo& poolingInfo)
    : VisitableOperation<Pooling>(
          pos, id, { &input }, { CalculateOutputTensorInfo(input.GetTensorInfo(), poolingInfo) })
    , m_PoolingInfo(poolingInfo)
{}

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

void Pooling::Print(std::ostream& os)
{
    std::stringstream ss;
    // clang-format off
    ss << " {"
        << " type: " << static_cast<int>(m_PoolingInfo.m_PoolingType)
        << " }";
    // clang-format on
    PrintOperation(os, *this, "Pooling", ss.str());
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

void Reshape::Print(std::ostream& os)
{
    PrintOperation(os, *this, "Reshape");
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

void Concatenation::Print(std::ostream& os)
{
    PrintOperation(os, *this, "Concatenation");
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

void Split::Print(std::ostream& os)
{
    PrintOperation(os, *this, "Split");
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

void DepthToSpace::Print(std::ostream& os)
{
    PrintOperation(os, *this, "DepthToSpace");
}

EstimateOnly::EstimateOnly(const detail::PosInNetwork pos,
                           uint32_t id,
                           const std::vector<Operand*>& inputs,
                           const EstimateOnlyInfo& info)
    : VisitableOperation<EstimateOnly>(pos, id, inputs, info.m_OutputInfos)
{}

void EstimateOnly::Print(std::ostream& os)
{
    PrintOperation(os, *this, "EstimateOnly");
}

}    // namespace support_library

}    // namespace ethosn
