//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNLayerSupport.hpp"

#include "EthosNBackend.hpp"
#include "EthosNConfig.hpp"
#include "EthosNReplaceUnsupported.hpp"
#include "EthosNTensorUtils.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/utility/Assert.hpp>
#include <ethosn_support_library/SupportQueries.hpp>

#include <algorithm>
#include <cstring>

using namespace armnn::ethosnbackend;

namespace armnn
{

namespace
{

class ReasonMessageHelper
{
public:
    explicit ReasonMessageHelper(unsigned int bufferSize = 1024u)
        : m_BufferSize(bufferSize)
        , m_Buffer(nullptr)
    {
        if (m_BufferSize > 0u)
        {
            m_Buffer    = new char[m_BufferSize];
            m_Buffer[0] = '\0';
        }
    }

    ~ReasonMessageHelper()
    {
        delete[] m_Buffer;
    }

    ReasonMessageHelper(const ReasonMessageHelper&) = delete;
    ReasonMessageHelper operator=(const ReasonMessageHelper&) = delete;

    char* GetBuffer() const
    {
        return m_Buffer;
    }

    unsigned int GetBufferSize() const
    {
        return m_BufferSize;
    }

    std::string GetString() const
    {
        if (m_Buffer)
        {
            return std::string(m_Buffer);
        }

        return std::string("");
    }

    void SetString(const std::string& string)
    {
        if (m_Buffer)
        {
            const unsigned int maxNumChars = m_BufferSize - 1u;
            strncpy(m_Buffer, string.c_str(), maxNumChars);
        }
    }

private:
    unsigned int m_BufferSize;
    char* m_Buffer;
};

void SetReason(Optional<std::string&> reasonIfUnsupported, const std::string& reason)
{
    if (reasonIfUnsupported)
    {
        reasonIfUnsupported.value() = reason;
    }
}

void SetReasonIfUnsupported(bool supported,
                            const ReasonMessageHelper& messageHelper,
                            Optional<std::string&> reasonIfUnsupported)
{
    if (!supported)
    {
        SetReason(reasonIfUnsupported, messageHelper.GetString());
    }
}

bool IsTensorSupportedOnEthosN(const TensorInfo& tensorInfo, Optional<std::string&> reasonIfUnsupported)
{
    using namespace ethosntensorutils;
    if (tensorInfo.GetNumDimensions() > 4)
    {
        SetReason(reasonIfUnsupported, std::string("The ethosn can only support up to 4D tensors"));
        return false;
    }
    if (!IsDataTypeSupportedOnEthosN(tensorInfo.GetDataType()))
    {
        SetReason(reasonIfUnsupported,
                  std::string("Unsupported data type: ") + GetDataTypeName(tensorInfo.GetDataType()));
        return false;
    }
    return true;
}

bool IsTensorSupportedOnEthosN(const Optional<TensorInfo>& tensorInfo, Optional<std::string&> reasonIfUnsupported)
{
    if (tensorInfo)
    {
        return IsTensorSupportedOnEthosN(tensorInfo.value(), reasonIfUnsupported);
    }
    else
    {
        return true;
    }
}

bool CheckSupportedLevel(ethosn_lib::SupportedLevel level, bool perfOnly)
{
    if (level == ethosn_lib::SupportedLevel::Supported)
    {
        return true;
    }
    else if (level == ethosn_lib::SupportedLevel::EstimateOnly && perfOnly)
    {
        return true;
    }
    return false;
}

}    // anonymous namespace

EthosNLayerSupport::EthosNLayerSupport(const EthosNConfig& config, const std::vector<char>& capabilities)
    : m_Config(config)
    , m_Queries(capabilities)
{}

using namespace ethosntensorutils;

bool EthosNLayerSupport::IsActivationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const ActivationDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return false;
    }

    ethosn_lib::TensorInfo ethosnInput  = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    ethosn_lib::TensorInfo ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    Optional<SupportedLevel> supportedLevel;
    ReasonMessageHelper messageHelper;

    switch (descriptor.m_Function)
    {
        case ActivationFunction::ReLu:
        case ActivationFunction::BoundedReLu:
        {
            const Optional<ethosn_lib::ReluInfo> reluInfo = BuildEthosNReluInfo(descriptor, input);
            if (!reluInfo.has_value())
            {
                SetReason(reasonIfUnsupported, "Cannot convert ReluInfo");
                return false;
            }

            supportedLevel = m_Queries.IsReluSupported(reluInfo.value(), ethosnInput, &ethosnOutput,
                                                       messageHelper.GetBuffer(), messageHelper.GetBufferSize());
            break;
        }
        case ActivationFunction::LeakyReLu:
        {
            const ethosn_lib::LeakyReluInfo leakyReluInfo = BuildEthosNLeakyReluInfo(descriptor, output);

            supportedLevel = m_Queries.IsLeakyReluSupported(leakyReluInfo, ethosnInput, &ethosnOutput,
                                                            messageHelper.GetBuffer(), messageHelper.GetBufferSize());
            break;
        }
        case ActivationFunction::Sigmoid:
        {
            supportedLevel = m_Queries.IsSigmoidSupported(ethosnInput, &ethosnOutput, messageHelper.GetBuffer(),
                                                          messageHelper.GetBufferSize());
            break;
        }
        case ActivationFunction::TanH:
        {
            supportedLevel = m_Queries.IsTanhSupported(ethosnInput, &ethosnOutput, messageHelper.GetBuffer(),
                                                       messageHelper.GetBufferSize());
            break;
        }
        default:
        {
            messageHelper.SetString("Unsupported activation function: " +
                                    std::string(GetActivationFunctionAsCString(descriptor.m_Function)));
            supportedLevel = SupportedLevel::EstimateOnly;
            break;
        }
    }

    bool supported = CheckSupportedLevel(supportedLevel.value(), m_Config.m_PerfOnly);

    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return GetAdditionSupportedMode(input0, input1, output, reasonIfUnsupported) != AdditionSupportedMode::None;
}

bool EthosNLayerSupport::IsAdditionSupportedByDepthwiseReplacement(const TensorInfo& input0,
                                                                   const TensorInfo& input1,
                                                                   const TensorInfo& output,
                                                                   const ethosn_lib::TensorInfo& ethosnInput0,
                                                                   const ethosn_lib::TensorInfo& ethosnInput1,
                                                                   Optional<std::string&> reasonIfUnsupported) const
{
    // If native addition is not supported, try substituting a pattern where a constant is broadcast-added for a DepthwiseConvolution2d.
    // Therefore we need to check if this is the case, and check the corresponding supportedness for
    // DepthwiseConvolution2d instead. Note that it is not possible at this stage to determine if one of the inputs
    // is constant, so we have to assume that it is. If it turns out to not be constant, then the replacement won't
    // take place and the support library will be asked to perform a broadcast add, which it will reject.
    const ethosn_lib::TensorShape& input0Shape = ethosnInput0.m_Dimensions;
    const ethosn_lib::TensorShape& input1Shape = ethosnInput1.m_Dimensions;

    bool isBroadcastShape0 = input0Shape == ethosn_lib::TensorShape{ 1, 1, 1, input0Shape[3] };
    bool isBroadcastShape1 = input1Shape == ethosn_lib::TensorShape{ 1, 1, 1, input1Shape[3] };

    if ((!isBroadcastShape0 && !isBroadcastShape1) || (input0Shape[3] != input1Shape[3]))
    {
        return false;
    }

    const TensorInfo& inputInfo    = isBroadcastShape0 ? input1 : input0;
    const TensorInfo& constantInfo = isBroadcastShape0 ? input0 : input1;

    // Check if the replacement is possible (e.g. the data types are compatible), and if so get the configuration of the new layer
    std::string failureReason;
    Optional<ConstantAddToDepthwiseReplacementConfig> configOpt =
        CalcConstantAddToDepthwiseReplacementConfig(inputInfo, constantInfo, output, failureReason);
    if (!configOpt.has_value())
    {
        ReasonMessageHelper messageHelper;
        messageHelper.SetString("Addition operation was attempted to be substituted for DepthwiseConvolution2d, "
                                "however the following error occurred in the substitution: " +
                                failureReason);
        SetReason(reasonIfUnsupported, messageHelper.GetString());
        return false;
    }
    const ConstantAddToDepthwiseReplacementConfig& config = configOpt.value();

    std::string depthwiseReasonIfUnsupported;
    bool supported = EthosNLayerSupport::IsDepthwiseConvolutionSupported(
        inputInfo, output, config.m_Desc, config.m_WeightsInfo, config.m_BiasInfo, depthwiseReasonIfUnsupported);

    ReasonMessageHelper messageHelper;
    messageHelper.SetString("Addition operation was attempted to be substituted for DepthwiseConvolution2d, "
                            "however the following error occurred when checking for Depthwise support: " +
                            depthwiseReasonIfUnsupported);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsAdditionSupportedByReinterpretQuantization(const TensorInfo& input0,
                                                                      const TensorInfo& input1,
                                                                      const TensorInfo& output,
                                                                      const ethosn_lib::TensorInfo& ethosnInput0,
                                                                      const ethosn_lib::TensorInfo& ethosnInput1,
                                                                      Optional<std::string&> reasonIfUnsupported) const
{
    // Support is claimed if a single input tensor is of shape {1,1,1,1}
    // When constant is of that shape, backend will substitute the Constant-Addition patterns
    // for ReinterpretQuantization.
    auto ethosnOutput                          = BuildEthosNTensorInfo(output, DataLayout::NHWC);
    const ethosn_lib::TensorShape& input0Shape = ethosnInput0.m_Dimensions;
    const ethosn_lib::TensorShape& input1Shape = ethosnInput1.m_Dimensions;
    bool isBroadcastShape0                     = input0Shape == ethosn_lib::TensorShape{ 1, 1, 1, 1 };
    bool isBroadcastShape1                     = input1Shape == ethosn_lib::TensorShape{ 1, 1, 1, 1 };
    bool supported                             = false;

    using ethosn_lib::SupportedLevel;
    if (isBroadcastShape0 || isBroadcastShape1)
    {
        auto reinterpretQuantizeInfo = BuildEthosNReinterpretQuantizationInfo(output);
        ReasonMessageHelper messageHelper;

        SupportedLevel supportedLevel =
            m_Queries.IsReinterpretQuantizationSupported(reinterpretQuantizeInfo, ethosnInput0, &ethosnOutput,
                                                         messageHelper.GetBuffer(), messageHelper.GetBufferSize());
        supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);

        if (supported)
        {
            // Checking if input and output scale quantities are equal (within margin of error)
            // as this is a required condition for scalar addition to be valid
            //
            // NOTE: input and output data types should also be equal but this condition
            // is already being checked by IsReinterpretQuantizationSupported
            auto input = (isBroadcastShape0) ? input1 : input0;
            supported  = std::abs(output.GetQuantizationScale() - input.GetQuantizationScale()) < 0.00001f;
            if (!supported)
                messageHelper.SetString("Input and output quantization scales are not equal");
        }

        SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
        return supported;
    }

    return false;
}

armnn::EthosNLayerSupport::AdditionSupportedMode
    EthosNLayerSupport::GetAdditionSupportedMode(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input0, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(input1, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return AdditionSupportedMode::None;
    }

    auto ethosnInput0 = BuildEthosNTensorInfo(input0, DataLayout::NHWC);
    auto ethosnInput1 = BuildEthosNTensorInfo(input1, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    // First try checking for support using a native addition
    ReasonMessageHelper messageHelper;
    SupportedLevel nativeSupportedLevel =
        m_Queries.IsAdditionSupported(ethosnInput0, ethosnInput1, ethosnOutput.m_QuantizationInfo, &ethosnOutput,
                                      messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool nativeSupported = CheckSupportedLevel(nativeSupportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(nativeSupported, messageHelper, reasonIfUnsupported);
    // If in perf-only mode, and we got EstimateOnly for native addition, don't early-out here but instead first check
    // if the depthwise replacement would give us full support, as that is preferable.
    if (nativeSupportedLevel == ethosn_lib::SupportedLevel::Supported)
    {
        return AdditionSupportedMode::Native;
    }

    // If native addition is not supported, try substituting a pattern where a constant is broadcast-added for a DepthwiseConvolution2d.
    if (IsAdditionSupportedByDepthwiseReplacement(input0, input1, output, ethosnInput0, ethosnInput1,
                                                  reasonIfUnsupported))
    {
        return AdditionSupportedMode::ReplaceWithDepthwise;
    }
    else if (IsAdditionSupportedByReinterpretQuantization(input0, input1, output, ethosnInput0, ethosnInput1,
                                                          reasonIfUnsupported))
    {
        return AdditionSupportedMode::ReplaceWithReinterpretQuantize;
    }
    else
    {
        return nativeSupported ? AdditionSupportedMode::Native : AdditionSupportedMode::None;
    }
}

bool EthosNLayerSupport::IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                                           const TensorInfo& output,
                                           const OriginsDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!IsTensorSupportedOnEthosN(output, reasonIfUnsupported))
    {
        return false;
    }

    const size_t numInputs = inputs.size();

    // construct temporary vector of converted Ethos-N input tensors
    std::vector<ethosn_lib::TensorInfo> ethosnInputs;
    ethosnInputs.reserve(numInputs);

    for (const TensorInfo* input : inputs)
    {
        if (!input || !IsTensorSupportedOnEthosN(*input, reasonIfUnsupported))
        {
            return false;
        }

        ethosnInputs.emplace_back(BuildEthosNTensorInfo(*input, DataLayout::NHWC));
    }

    ethosn_lib::TensorInfo ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    // The Ethos-N's concat axis is the same as Arm NN's even if the tensor shapes have been padded to 4D,
    // because we pad on the right hand side of the dimensions.
    // Note we ignore the "view origins" contained in OriginsDescriptor and use just the "concat axis".
    // This is a known issue/confusion in the Arm NN API - see Github Issue #234.
    uint32_t ethosnConcatAxis = descriptor.GetConcatAxis();

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsConcatenationSupported(
        ethosnInputs, ethosn_lib::ConcatenationInfo(ethosnConcatAxis, ethosnOutput.m_QuantizationInfo), &ethosnOutput,
        messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsConstantSupported(const TensorInfo& info, Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(info, reasonIfUnsupported)))
    {
        return false;
    }

    auto ethosnInfo = BuildEthosNTensorInfo(info, DataLayout::NHWC);

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel =
        m_Queries.IsConstantSupported(ethosnInfo, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const Convolution2dDescriptor& descriptor,
                                                  const TensorInfo& weights,
                                                  const Optional<TensorInfo>& biases,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(weights, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(biases, reasonIfUnsupported)))
    {
        return false;
    }
    if (descriptor.m_DataLayout != DataLayout::NHWC)
    {
        // In order to support other layouts we would need to do more than just use this layout when creating the
        // Ethos-N tensor infos, as the same tensor could be used for layers with different data layouts.
        SetReason(reasonIfUnsupported, "DataLayout must be NHWC");
        return false;
    }

    auto ethosnInput  = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    ethosn_lib::TensorInfo ethosnBias;
    try
    {
        ethosnBias = biases.has_value() ? BuildEthosNBiasesInfo(biases.value(), input, weights)
                                        : BuildEthosNBiasesInfo(ethosnOutput.m_Dimensions[3], input, weights);
    }
    catch (const InvalidArgumentException& e)
    {
        SetReason(reasonIfUnsupported, e.what());
        return false;
    }

    ethosn_lib::TensorInfo ethosnWeights;
    try
    {
        constexpr bool isDepthwiseConvolution = false;
        ethosnWeights =
            BuildEthosNConvolutionWeightsInfo(weights, input, descriptor.m_DataLayout, isDepthwiseConvolution);
    }
    catch (const InvalidArgumentException& e)
    {
        SetReason(reasonIfUnsupported, e.what());
        return false;
    }

    auto convolutionInfo = BuildEthosNConvolutionInfo(descriptor, output.GetQuantizationOffset(),
                                                      output.GetQuantizationScale(), reasonIfUnsupported);
    if (!convolutionInfo.has_value())
    {
        return false;
    }

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel =
        m_Queries.IsConvolutionSupported(ethosnBias, ethosnWeights, convolutionInfo.value(), ethosnInput, &ethosnOutput,
                                         messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                         const TensorInfo& output,
                                                         const DepthwiseConvolution2dDescriptor& descriptor,
                                                         const TensorInfo& weights,
                                                         const Optional<TensorInfo>& biases,
                                                         Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(weights, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(biases, reasonIfUnsupported)))
    {
        return false;
    }

    if (descriptor.m_DataLayout != DataLayout::NHWC)
    {
        // In order to support other layouts we would need to do more than just use this layout when creating the
        // Ethos-N tensor infos, as the same tensor could be used for layers with different data layouts.
        SetReason(reasonIfUnsupported, "DataLayout must be NHWC");
        return false;
    }

    auto ethosnInput  = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    ethosn_lib::TensorInfo ethosnBias;
    try
    {
        ethosnBias = biases.has_value() ? BuildEthosNBiasesInfo(biases.value(), input, weights)
                                        : BuildEthosNBiasesInfo(ethosnOutput.m_Dimensions[3], input, weights);
    }
    catch (const InvalidArgumentException& e)
    {
        SetReason(reasonIfUnsupported, e.what());
        return false;
    }

    ethosn_lib::TensorInfo ethosnWeights;
    try
    {
        constexpr bool isDepthwiseConvolution = true;
        ethosnWeights =
            BuildEthosNConvolutionWeightsInfo(weights, input, descriptor.m_DataLayout, isDepthwiseConvolution);
    }
    catch (const InvalidArgumentException& e)
    {
        SetReason(reasonIfUnsupported, e.what());
        return false;
    }

    auto convolutionInfo = BuildEthosNConvolutionInfo(descriptor, output.GetQuantizationOffset(),
                                                      output.GetQuantizationScale(), reasonIfUnsupported);
    if (!convolutionInfo.has_value())
    {
        return false;
    }

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsDepthwiseConvolutionSupported(
        ethosnBias, ethosnWeights, convolutionInfo.value(), ethosnInput, &ethosnOutput, messageHelper.GetBuffer(),
        messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsTransposeConvolution2dSupported(const TensorInfo& input,
                                                           const TensorInfo& output,
                                                           const TransposeConvolution2dDescriptor& descriptor,
                                                           const TensorInfo& weights,
                                                           const Optional<TensorInfo>& biases,
                                                           Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(weights, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(biases, reasonIfUnsupported)))
    {
        return false;
    }
    if (descriptor.m_DataLayout != DataLayout::NHWC)
    {
        // In order to support other layouts we would need to do more than just use this layout when creating the
        // Ethos-N tensor infos, as the same tensor could be used for layers with different data layouts.
        SetReason(reasonIfUnsupported, "DataLayout must be NHWC");
        return false;
    }

    auto ethosnInput  = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    ethosn_lib::TensorInfo ethosnBias;
    try
    {
        ethosnBias = biases.has_value() ? BuildEthosNBiasesInfo(biases.value(), input, weights)
                                        : BuildEthosNBiasesInfo(ethosnOutput.m_Dimensions[3], input, weights);
    }
    catch (const InvalidArgumentException& e)
    {
        SetReason(reasonIfUnsupported, e.what());
        return false;
    }

    ethosn_lib::TensorInfo ethosnWeights;
    try
    {
        constexpr bool isDepthwiseConvolution = false;
        ethosnWeights =
            BuildEthosNConvolutionWeightsInfo(weights, input, descriptor.m_DataLayout, isDepthwiseConvolution);
    }
    catch (const InvalidArgumentException& e)
    {
        SetReason(reasonIfUnsupported, e.what());
        return false;
    }

    auto convolutionInfo =
        BuildEthosNConvolutionInfo(descriptor, output.GetQuantizationOffset(), output.GetQuantizationScale());

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsTransposeConvolutionSupported(
        ethosnBias, ethosnWeights, convolutionInfo, ethosnInput, &ethosnOutput, messageHelper.GetBuffer(),
        messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const TensorInfo& weights,
                                                   const TensorInfo& biases,
                                                   const FullyConnectedDescriptor& descriptor,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(weights, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(biases, reasonIfUnsupported)))
    {
        return false;
    }

    // FullyConnected is defined to have the first dimension as batches, and all others are treated as a single
    // channels dimension. This is compatible with the Ethos-N's NHWC layout if the H and W dimensions are 1,
    // which we sort out below.
    constexpr DataLayout ethosnDataLayout = DataLayout::NHWC;

    auto ethosnInput  = BuildEthosNTensorInfo(input, ethosnDataLayout);
    auto ethosnOutput = BuildEthosNTensorInfo(output, ethosnDataLayout);

    // Override the input and output shape as the dimension padding performed in BuildEthosNTensorInfo
    // will result in N x C x 1 x 1 which is not valid for the Ethos-N.
    // We will handle this by adding reshapes when actually creating the Ethos-N network.
    // This also accounts for tensors with multiple channels dimensions
    ethosnInput.m_Dimensions  = { input.GetShape()[0], 1, 1, input.GetNumElements() / input.GetShape()[0] };
    ethosnOutput.m_Dimensions = { output.GetShape()[0], 1, 1, output.GetNumElements() / output.GetShape()[0] };

    ethosn_lib::TensorInfo ethosnBias;
    try
    {
        ethosnBias = BuildEthosNBiasesInfo(biases, input, weights);
    }
    catch (const InvalidArgumentException& e)
    {
        SetReason(reasonIfUnsupported, e.what());
        return false;
    }

    ethosn_lib::TensorInfo ethosnWeights;
    try
    {
        ethosnWeights = BuildEthosNFullyConnectedWeightsInfo(weights, descriptor.m_TransposeWeightMatrix);
    }
    catch (const InvalidArgumentException& e)
    {
        SetReason(reasonIfUnsupported, e.what());
        return false;
    }

    ethosn_lib::FullyConnectedInfo fullyConnectedInfo =
        BuildEthosNFullyConnectedLayerInfo(descriptor, output.GetQuantizationOffset(), output.GetQuantizationScale());

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel =
        m_Queries.IsFullyConnectedSupported(ethosnBias, ethosnWeights, fullyConnectedInfo, ethosnInput, &ethosnOutput,
                                            messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsInputSupported(const TensorInfo& input, Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported)))
    {
        return false;
    }

    auto ethosnInput = BuildEthosNTensorInfo(input, DataLayout::NHWC);

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel =
        m_Queries.IsInputSupported(ethosnInput, nullptr, messageHelper.GetBuffer(), messageHelper.GetBufferSize());
    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsMemCopySupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input);
    IgnoreUnused(output);
    IgnoreUnused(reasonIfUnsupported);
    return true;
}

bool EthosNLayerSupport::IsOutputSupported(const TensorInfo& output, Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return false;
    }

    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsOutputSupported(
        ethosnOutput, ethosnOutput.m_DataFormat, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const Pooling2dDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return false;
    }
    if (descriptor.m_DataLayout != DataLayout::NHWC)
    {
        // In order to support other layouts we would need to do more than just use this layout when creating the
        // Ethos-N tensor infos, as the same tensor could be used for layers with different data layouts.
        SetReason(reasonIfUnsupported, "DataLayout must be NHWC");
        return false;
    }

    auto ethosnInput  = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    auto poolingInfo = BuildEthosNPoolingLayerInfo(descriptor);

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsPoolingSupported(
        poolingInfo, ethosnInput, &ethosnOutput, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsPreCompiledSupported(const TensorInfo& input,
                                                const PreCompiledDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);

    return IsTensorSupportedOnEthosN(input, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsRankSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsReduceSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const ReduceDescriptor&,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsReshapeSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const ReshapeDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return false;
    }

    auto ethosnInput = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    auto ethosnShape = BuildEthosNTensorShape(descriptor.m_TargetShape);

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsReshapeSupported(
        ethosnShape, ethosnInput, nullptr, messageHelper.GetBuffer(), messageHelper.GetBufferSize());
    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const SoftmaxDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return false;
    }
    if (descriptor.m_Axis != -1 && descriptor.m_Axis != static_cast<int>(input.GetNumDimensions()) - 1)
    {
        SetReason(reasonIfUnsupported, "Softmax axis must be the last one");
        return false;
    }

    auto ethosnInput  = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsSoftmaxSupported(ethosnInput, &ethosnOutput, messageHelper.GetBuffer(),
                                                                 messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsSplitterSupported(const TensorInfo& input,
                                             const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                             const ViewsDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    ARMNN_ASSERT(outputs.size() == descriptor.GetNumViews());

    if (!IsTensorSupportedOnEthosN(input, reasonIfUnsupported))
    {
        return false;
    }
    ethosn_lib::TensorInfo ethosnInput = BuildEthosNTensorInfo(input, DataLayout::NHWC);

    // Convert output tensor infos to Ethos-N representation
    std::vector<ethosn_lib::TensorInfo> ethosnOutputs;
    ethosnOutputs.reserve(outputs.size());
    for (uint32_t i = 0; i < outputs.size(); ++i)
    {
        const TensorInfo& output = outputs[i];
        if (!IsTensorSupportedOnEthosN(output, reasonIfUnsupported))
        {
            return false;
        }

        ethosnOutputs.emplace_back(BuildEthosNTensorInfo(output, DataLayout::NHWC));
    }

    Optional<ethosn_lib::SplitInfo> ethosnSplitInfo = BuildEthosNSplitInfo(input.GetShape(), descriptor);
    if (!ethosnSplitInfo.has_value())
    {
        SetReason(reasonIfUnsupported, "Not a single-axis split");
        return false;
    }

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsSplitSupported(
        ethosnInput, ethosnSplitInfo.value(), &ethosnOutputs, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsDepthToSpaceSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const DepthToSpaceDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return false;
    }
    if (descriptor.m_DataLayout != DataLayout::NHWC)
    {
        // In order to support other layouts we would need to do more than just use this layout when creating the
        // Ethos-N tensor infos, as the same tensor could be used for layers with different data layouts.
        SetReason(reasonIfUnsupported, "DataLayout must be NHWC");
        return false;
    }

    auto ethosnInput  = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    ethosn_lib::DepthToSpaceInfo info(descriptor.m_BlockSize);
    if (descriptor.m_DataLayout != DataLayout::NHWC)
    {
        SetReason(reasonIfUnsupported, "Only NHWC data layout supported");
        return false;
    }

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsDepthToSpaceSupported(
        ethosnInput, info, &ethosnOutput, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::CheckEstimateOnlySupported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(std::vector<TensorInfo>{ input }, std::vector<TensorInfo>{ output },
                                      reasonIfUnsupported);
}

bool EthosNLayerSupport::CheckEstimateOnlySupported(const std::vector<TensorInfo>& inputs,
                                                    const std::vector<TensorInfo>& outputs,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    std::vector<ethosn_lib::TensorInfo> ethosnInputInfos;
    ethosnInputInfos.reserve(inputs.size());
    for (const auto& input : inputs)
    {
        if (!IsTensorSupportedOnEthosN(input, reasonIfUnsupported))
        {
            return false;
        }
        auto ethosnInput = BuildEthosNTensorInfo(input, DataLayout::NHWC);
        ethosnInputInfos.push_back(ethosnInput);
    }
    std::vector<ethosn_lib::TensorInfo> ethosnOutputInfos;
    ethosnOutputInfos.reserve(inputs.size());
    for (const auto& output : outputs)
    {
        if (!IsTensorSupportedOnEthosN(output, reasonIfUnsupported))
        {
            return false;
        }
        auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);
        ethosnOutputInfos.push_back(ethosnOutput);
    }

    ReasonMessageHelper messageHelper;
    ethosn_lib::EstimateOnlyInfo estimateInfo = ethosn_lib::EstimateOnlyInfo(ethosnOutputInfos);
    SupportedLevel supportedLevel             = m_Queries.IsEstimateOnlySupported(
        ethosnInputInfos, estimateInfo, nullptr, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    return supported;
}

bool EthosNLayerSupport::IsArgMinMaxSupported(const armnn::TensorInfo& input,
                                              const armnn::TensorInfo& output,
                                              const armnn::ArgMinMaxDescriptor&,
                                              armnn::Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsBatchNormalizationSupported(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const TensorInfo&,
                                                       const TensorInfo&,
                                                       const TensorInfo&,
                                                       const TensorInfo&,
                                                       const BatchNormalizationDescriptor&,
                                                       Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsBatchToSpaceNdSupported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const BatchToSpaceNdDescriptor&,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsCastSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsComparisonSupported(const TensorInfo& input0,
                                               const TensorInfo& input1,
                                               const TensorInfo& output,
                                               const ComparisonDescriptor&,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsConvertBf16ToFp32Supported(const TensorInfo&,
                                                      const TensorInfo&,
                                                      Optional<std::string&>) const
{
    // The Support Library does not support floating point types, even in performance-only mode.
    return false;
}

bool EthosNLayerSupport::IsConvertFp32ToBf16Supported(const TensorInfo&,
                                                      const TensorInfo&,
                                                      Optional<std::string&>) const
{
    // The Support Library does not support floating point types, even in performance-only mode.
    return false;
}

bool EthosNLayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo&,
                                                      const TensorInfo&,
                                                      Optional<std::string&>) const
{
    // The Support Library does not support floating point types, even in performance-only mode.
    return false;
}

bool EthosNLayerSupport::IsConvertFp32ToFp16Supported(const TensorInfo&,
                                                      const TensorInfo&,
                                                      Optional<std::string&>) const
{
    // The Support Library does not support floating point types, even in performance-only mode.
    return false;
}

bool EthosNLayerSupport::IsDebugSupported(const TensorInfo&, const TensorInfo&, Optional<std::string&>) const
{
    // The Support Library does not support floating point types, even in performance-only mode.
    return false;
}

bool EthosNLayerSupport::IsDequantizeSupported(const TensorInfo&, const TensorInfo&, Optional<std::string&>) const
{
    // The Support Library does not support floating point types, even in performance-only mode.
    return false;
}

bool EthosNLayerSupport::IsDetectionPostProcessSupported(const TensorInfo&,
                                                         const TensorInfo&,
                                                         const TensorInfo&,
                                                         const TensorInfo&,
                                                         const TensorInfo&,
                                                         const TensorInfo&,
                                                         const TensorInfo&,
                                                         const DetectionPostProcessDescriptor&,
                                                         Optional<std::string&>) const
{
    return false;
}

bool EthosNLayerSupport::IsDilatedDepthwiseConvolutionSupported(const TensorInfo& input,
                                                                const TensorInfo& output,
                                                                const DepthwiseConvolution2dDescriptor&,
                                                                const TensorInfo&,
                                                                const Optional<TensorInfo>&,
                                                                Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsElementwiseUnarySupported(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const ElementwiseUnaryDescriptor&,
                                                     Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsFakeQuantizationSupported(const TensorInfo& input,
                                                     const FakeQuantizationDescriptor&,
                                                     Optional<std::string&> reasonIfUnsupported) const
{
    // Even though this layer probably has minimal usefulness in an already-quantized context, the Ethos-N
    // could support it.
    return CheckEstimateOnlySupported({ input }, {}, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsFillSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const FillDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsFloorSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsGatherSupported(const armnn::TensorInfo& input0,
                                           const armnn::TensorInfo& input1,
                                           const armnn::TensorInfo& output,
                                           const GatherDescriptor& descriptor,
                                           armnn::Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsInstanceNormalizationSupported(const TensorInfo& input,
                                                          const TensorInfo& output,
                                                          const InstanceNormalizationDescriptor&,
                                                          Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const L2NormalizationDescriptor&,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsLogicalBinarySupported(const TensorInfo& input0,
                                                  const TensorInfo& input1,
                                                  const TensorInfo& output,
                                                  const LogicalBinaryDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsLogicalUnarySupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const ElementwiseUnaryDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsLogSoftmaxSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const LogSoftmaxDescriptor&,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsLstmSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const TensorInfo&,
                                         const TensorInfo&,
                                         const TensorInfo&,
                                         const TensorInfo&,
                                         const TensorInfo&,
                                         const LstmDescriptor&,
                                         const LstmInputParamsInfo&,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsMaximumSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsMeanSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const MeanDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;

    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        SetReason(reasonIfUnsupported, "Input/Output tensors are not supported");
        return false;
    }

    if (!descriptor.m_KeepDims)
    {
        // The dimensions need to be preserved.
        SetReason(reasonIfUnsupported, "The dimensions need to be preserved");
        return false;
    }

    auto ethosnInput  = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    if (!((ethosnOutput.m_Dimensions[1] == 1) && (ethosnOutput.m_Dimensions[2] == 1) &&
          (ethosnInput.m_Dimensions[3] == ethosnOutput.m_Dimensions[3]) &&
          (ethosnInput.m_Dimensions[0] == ethosnOutput.m_Dimensions[0])))
    {
        SetReason(reasonIfUnsupported, "Mean is supported for XY dimensions only");
        return false;
    }

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsMeanXySupported(ethosnInput, &ethosnOutput, messageHelper.GetBuffer(),
                                                                messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsMemImportSupported(const armnn::TensorInfo&,
                                              const armnn::TensorInfo&,
                                              armnn::Optional<std::string&>) const
{
    // This is a 'meta' layer type related to avoiding tensor copies between backends.
    // We should never receive this layer because we don't advertise support for this feature.
    return false;
}

bool EthosNLayerSupport::IsMergeSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsMinimumSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                                   const TensorInfo& input1,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return GetMultiplicationSupportedMode(input0, input1, output, reasonIfUnsupported) !=
           MultiplicationSupportedMode::None;
}

armnn::EthosNLayerSupport::MultiplicationSupportedMode
    EthosNLayerSupport::GetMultiplicationSupportedMode(const TensorInfo& input0,
                                                       const TensorInfo& input1,
                                                       const TensorInfo& output,
                                                       Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input0, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(input1, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return MultiplicationSupportedMode::None;
    }

    // Check first if multiplication is supported by depthwise replacement or not.
    if (IsMultiplicationSupportedByDepthwiseReplacement(input0, input1, output, reasonIfUnsupported))
    {
        return MultiplicationSupportedMode::ReplaceWithDepthwise;
    }

    // If multiplication by depthwise replacement is not supported, try substituting a pattern where a
    // constant is broadcast-multiplied with a ReinterpretQuantize.
    if (IsMultiplicationSupportedByReinterpretQuantizationReplacement(input0, input1, output, reasonIfUnsupported))
    {
        return MultiplicationSupportedMode::ReplaceWithReinterpretQuantize;
    }

    // If none of the replacements work, we check for estimate only support.
    if (CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported))
    {
        return MultiplicationSupportedMode::EstimateOnly;
    }

    return MultiplicationSupportedMode::None;
}

bool EthosNLayerSupport::IsMultiplicationSupportedByDepthwiseReplacement(
    const TensorInfo& input0,
    const TensorInfo& input1,
    const TensorInfo& output,
    Optional<std::string&> reasonIfUnsupported) const
{
    // Support for Multiplication operations is claimed where either of the input tensors has the
    // shape { 1, 1, 1, C }. When the input is a Constant of the said shape, the backend will then
    // substitute the Constant-Multiplication pattern for DepthwiseConvolution2d. Therfore, supportdness
    // for DepthwiseConvolution2D is checked. Note that it is not possible at this stage to determine
    // if one of the inputs is constant, so we have to assume that it is. If it turns out to not be
    // constant, then the replacement won't take place.

    auto ethosnInput0 = BuildEthosNTensorInfo(input0, DataLayout::NHWC);
    auto ethosnInput1 = BuildEthosNTensorInfo(input1, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    const ethosn_lib::TensorShape& input0Shape = ethosnInput0.m_Dimensions;
    const ethosn_lib::TensorShape& input1Shape = ethosnInput1.m_Dimensions;

    bool isBroadcastShape0 = input0Shape == ethosn_lib::TensorShape{ 1, 1, 1, input0Shape[3] };
    bool isBroadcastShape1 = input1Shape == ethosn_lib::TensorShape{ 1, 1, 1, input1Shape[3] };

    if ((isBroadcastShape0 || isBroadcastShape1) && (input0Shape[3] == input1Shape[3]))
    {
        const TensorInfo& inputInfo    = isBroadcastShape0 ? input1 : input0;
        const TensorInfo& constantInfo = isBroadcastShape0 ? input0 : input1;

        DepthwiseConvolution2dDescriptor desc;
        desc.m_DataLayout  = DataLayout::NHWC;
        desc.m_BiasEnabled = false;

        TensorInfo weightsInfo = constantInfo;
        unsigned int M         = output.GetShape()[3] / inputInfo.GetShape()[3];
        weightsInfo.SetShape({ 1, 1, 1, constantInfo.GetShape()[3] * M });    //1HW(I*M)

        std::string depthwiseReasonIfUnsupported;
        bool supported = EthosNLayerSupport::IsDepthwiseConvolutionSupported(
            inputInfo, output, desc, weightsInfo, EmptyOptional(), depthwiseReasonIfUnsupported);

        ReasonMessageHelper messageHelper;
        messageHelper.SetString("Multiplication operation is not supported on Arm Ethos-N NPU backend and an attempt "
                                "was made to substitute for DepthwiseConvolution2d, however the following error "
                                "occurred when checking for Depthwise support: " +
                                depthwiseReasonIfUnsupported);

        SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
        return supported;
    }

    return false;
}

bool EthosNLayerSupport::IsMultiplicationSupportedByReinterpretQuantizationReplacement(
    const TensorInfo& input0,
    const TensorInfo& input1,
    const TensorInfo& output,
    Optional<std::string&> reasonIfUnsupported) const
{
    // Support for Multiplication operations is claimed where either of the input tensors has the
    // shape { 1, 1, 1, 1 }. When the input is a Constant of the said shape, the backend will then
    // substitute the Constant-Multiplication pattern for ReinterpretQuantization. Therfore, supportdness
    // for ReinterpretQuantization is checked. Note that it is not possible at this stage to determine
    // if one of the inputs is constant, so we have to assume that it is. If it turns out to not be
    // constant, then the replacement won't take place.

    auto ethosnInput0 = BuildEthosNTensorInfo(input0, DataLayout::NHWC);
    auto ethosnInput1 = BuildEthosNTensorInfo(input1, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    const ethosn_lib::TensorShape& input0Shape = ethosnInput0.m_Dimensions;
    const ethosn_lib::TensorShape& input1Shape = ethosnInput1.m_Dimensions;

    bool isBroadcastShape0 = input0Shape == ethosn_lib::TensorShape{ 1, 1, 1, 1 };
    bool isBroadcastShape1 = input1Shape == ethosn_lib::TensorShape{ 1, 1, 1, 1 };

    using ethosn_lib::SupportedLevel;
    if ((isBroadcastShape0 || isBroadcastShape1) && (input0Shape[3] != input1Shape[3]))
    {
        auto reinterpretQuantizeInfo = BuildEthosNReinterpretQuantizationInfo(output);

        ReasonMessageHelper messageHelper;

        bool supported;
        auto ethosnInput = isBroadcastShape0 ? ethosnInput1 : ethosnInput0;
        auto input       = isBroadcastShape0 ? input1 : input0;

        SupportedLevel supportedLevel =
            m_Queries.IsReinterpretQuantizationSupported(reinterpretQuantizeInfo, ethosnInput, &ethosnOutput,
                                                         messageHelper.GetBuffer(), messageHelper.GetBufferSize());
        supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);

        if (supported)
        {
            // Checking if input and output zero points are equal
            // as this is a required condition for scalar multiplication to be valid
            //
            // NOTE: input and output data types should also be equal but this condition
            // is already being checked by IsReinterpretQuantizationSupported
            supported = output.GetQuantizationOffset() == input.GetQuantizationOffset();
            if (!supported)
                messageHelper.SetString("Input and output quantization offsets are not equal");
        }

        SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
        return supported;
    }

    return false;
}

bool EthosNLayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const NormalizationDescriptor&,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsPadSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const PadDescriptor&,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsPermuteSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const PermuteDescriptor&,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsPreluSupported(const TensorInfo& input,
                                          const TensorInfo&,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsQuantizeSupported(const armnn::TensorInfo& input,
                                             const armnn::TensorInfo& output,
                                             armnn::Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return false;
    }

    auto ethosnInput    = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    auto ethosnOutput   = BuildEthosNTensorInfo(output, DataLayout::NHWC);
    auto requantizeInfo = BuildEthosNRequantizeInfo(output);

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsRequantizeSupported(
        requantizeInfo, ethosnInput, &ethosnOutput, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);

    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsQLstmSupported(const TensorInfo& input,
                                          const TensorInfo&,
                                          const TensorInfo&,
                                          const TensorInfo&,
                                          const TensorInfo&,
                                          const TensorInfo& output,
                                          const QLstmDescriptor&,
                                          const LstmInputParamsInfo&,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsQuantizedLstmSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const TensorInfo&,
                                                  const TensorInfo&,
                                                  const TensorInfo&,
                                                  const QuantizedLstmInputParamsInfo&,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsResizeSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const ResizeDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;

    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return false;
    }

    if ((descriptor.m_Method != ResizeMethod::Bilinear) && (descriptor.m_Method != ResizeMethod::NearestNeighbor))
    {
        return false;
    }

    auto ethosnInput  = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    auto ethosResizeInfo = BuildEthosNResizeInfo(descriptor, output);

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsResizeSupported(
        ethosResizeInfo, ethosnInput, &ethosnOutput, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsShapeSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsSliceSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const SliceDescriptor&,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsSpaceToBatchNdSupported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const SpaceToBatchNdDescriptor&,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsSpaceToDepthSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const SpaceToDepthDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;

    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return false;
    }

    auto ethosnInput  = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    ethosn_lib::DepthToSpaceInfo info(descriptor.m_BlockSize);
    if (descriptor.m_DataLayout != DataLayout::NHWC)
    {
        SetReason(reasonIfUnsupported, "Only NHWC data layout supported");
        return false;
    }

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = m_Queries.IsSpaceToDepthSupported(
        ethosnInput, info, &ethosnOutput, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsStackSupported(const std::vector<const TensorInfo*>& inputs,
                                          const TensorInfo& output,
                                          const StackDescriptor&,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    std::vector<TensorInfo> inputTensorInfos;
    inputTensorInfos.reserve(inputs.size());
    for (auto it : inputs)
    {
        inputTensorInfos.push_back(*it);
    }
    return CheckEstimateOnlySupported(inputTensorInfos, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsStandInSupported(const std::vector<const TensorInfo*>& inputs,
                                            const std::vector<const TensorInfo*>& outputs,
                                            const StandInDescriptor&,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    std::vector<TensorInfo> inputTensorInfos;
    inputTensorInfos.reserve(inputs.size());
    for (auto it : inputs)
    {
        inputTensorInfos.push_back(*it);
    }
    std::vector<TensorInfo> outputTensorInfos;
    outputTensorInfos.reserve(outputs.size());
    for (auto it : outputs)
    {
        outputTensorInfos.push_back(*it);
    }
    return CheckEstimateOnlySupported(inputTensorInfos, outputTensorInfos, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsStridedSliceSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const StridedSliceDescriptor&,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsSwitchSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output0,
                                           const TensorInfo& output1,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output0, output1 }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsTransposeSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const TransposeDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;

    if (!(IsTensorSupportedOnEthosN(input, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return false;
    }

    auto ethosnInput  = BuildEthosNTensorInfo(input, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    auto ethosTransposeInfo = BuildEthosNTransposeInfo(descriptor.m_DimMappings);

    if (!ethosTransposeInfo.has_value())
    {
        return false;
    }

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel =
        m_Queries.IsTransposeSupported(ethosTransposeInfo.value(), ethosnInput, &ethosnOutput,
                                       messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsChannelShuffleSupported(const TensorInfo&,
                                                   const TensorInfo&,
                                                   const ChannelShuffleDescriptor&,
                                                   Optional<std::string&>) const
{
    return false;
}

bool EthosNLayerSupport::IsConvolution3dSupported(const TensorInfo&,
                                                  const TensorInfo&,
                                                  const Convolution3dDescriptor&,
                                                  const TensorInfo&,
                                                  const Optional<TensorInfo>&,
                                                  Optional<std::string&>) const
{
    return false;
}

bool EthosNLayerSupport::IsPooling3dSupported(const armnn::TensorInfo&,
                                              const armnn::TensorInfo&,
                                              const armnn::Pooling3dDescriptor&,
                                              Optional<std::string&>) const
{
    return false;
}

}    // namespace armnn
