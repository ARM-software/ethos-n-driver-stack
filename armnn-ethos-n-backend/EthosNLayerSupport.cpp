//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNLayerSupport.hpp"

#include "EthosNBackend.hpp"
#include "EthosNConfig.hpp"
#include "EthosNMapping.hpp"
#include "EthosNTensorUtils.hpp"
#include "InternalTypes.hpp"
#include "LayerSupportCommon.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>
#include <armnn/TypesUtils.hpp>
#include <boost/core/ignore_unused.hpp>
#include <ethosn_support_library/SupportQueries.hpp>

#include <algorithm>
#include <cstring>

using namespace boost;

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

bool IsLayerExcluded(const EthosNMappings& mappings, LayerType layerType, std::string func = std::string())
{
    const char* type = armnn::GetLayerTypeAsCString(layerType);

    auto IsExcluded = [&](EthosNMappings::const_reference mapping) -> bool {
        using MappedLayers = std::vector<SimpleLayer>;

        auto HasExclusion = [&](MappedLayers::const_reference layer) -> bool {
            return (layer.m_LayerTypeName == "Excluded");
        };

        auto IsExcludedLayer = [&](MappedLayers::const_reference layer) -> bool {
            bool functionIsExcluded = const_cast<SimpleLayer&>(layer).m_LayerParams["function"] == func;
            return (layer.m_LayerTypeName == type && functionIsExcluded);
        };

        const auto& replacements = mapping.m_ReplacementLayers;
        if (replacements.cend() != std::find_if(replacements.cbegin(), replacements.cend(), HasExclusion))
        {
            const auto& patterns = mapping.m_PatternLayers;
            return (patterns.cend() != std::find_if(patterns.cbegin(), patterns.cend(), IsExcludedLayer));
        }

        return false;
    };

    return (mappings.cend() != std::find_if(mappings.cbegin(), mappings.cend(), IsExcluded));
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

EthosNLayerSupport::EthosNLayerSupport()
{
    g_EthosNConfig   = GetEthosNConfig();
    g_EthosNMappings = GetMappings(g_EthosNConfig.m_PerfMappingFile);
}

using namespace ethosntensorutils;

bool EthosNLayerSupport::IsActivationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const ActivationDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    if (IsLayerExcluded(g_EthosNMappings, LayerType::Activation,
                        armnn::GetActivationFunctionAsCString(descriptor.m_Function)))
    {
        SetReason(reasonIfUnsupported, std::string("Layer declared excluded in mapping file"));
        return false;
    }

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
            const ethosn_lib::ReluInfo reluInfo = BuildEthosNReluInfo(descriptor, output);

            supportedLevel = ethosn_lib::IsReluSupported(reluInfo, ethosnInput, &ethosnOutput,
                                                         messageHelper.GetBuffer(), messageHelper.GetBufferSize());
            break;
        }
        case ActivationFunction::LeakyReLu:
        {
            const ethosn_lib::LeakyReluInfo leakyReluInfo = BuildEthosNLeakyReluInfo(descriptor, output);

            supportedLevel = ethosn_lib::IsLeakyReluSupported(leakyReluInfo, ethosnInput, &ethosnOutput,
                                                              messageHelper.GetBuffer(), messageHelper.GetBufferSize());
            break;
        }
        case ActivationFunction::Sigmoid:
        {
            supportedLevel = ethosn_lib::IsSigmoidSupported(ethosnInput, &ethosnOutput, messageHelper.GetBuffer(),
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

    bool supported = CheckSupportedLevel(supportedLevel.value(), g_EthosNConfig.m_PerfOnly);

    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    if (!(IsTensorSupportedOnEthosN(input0, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(input1, reasonIfUnsupported) &&
          IsTensorSupportedOnEthosN(output, reasonIfUnsupported)))
    {
        return false;
    }

    auto ethosnInput0 = BuildEthosNTensorInfo(input0, DataLayout::NHWC);
    auto ethosnInput1 = BuildEthosNTensorInfo(input1, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel =
        ethosn_lib::IsAdditionSupported(ethosnInput0, ethosnInput1, ethosnOutput.m_QuantizationInfo, &ethosnOutput,
                                        messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
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
    SupportedLevel supportedLevel = ethosn_lib::IsConcatenationSupported(
        ethosnInputs, ethosn_lib::ConcatenationInfo(ethosnConcatAxis, ethosnOutput.m_QuantizationInfo), &ethosnOutput,
        messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
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
        ethosn_lib::IsConstantSupported(ethosnInfo, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
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

    auto ethosnBias = biases.has_value() ? BuildEthosNBiasesInfo(biases.value(), input, weights)
                                         : BuildEthosNBiasesInfo(ethosnOutput.m_Dimensions[3], input, weights);

    constexpr bool isDepthwiseConvolution = false;
    auto ethosnWeights = BuildEthosNConvolutionWeightsInfo(weights, descriptor.m_DataLayout, isDepthwiseConvolution);

    auto convolutionInfo =
        BuildEthosNConvolutionInfo(descriptor, output.GetQuantizationOffset(), output.GetQuantizationScale());

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel =
        ethosn_lib::IsConvolutionSupported(ethosnBias, ethosnWeights, convolutionInfo, ethosnInput, &ethosnOutput,
                                           messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
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

    auto ethosnBias = biases.has_value() ? BuildEthosNBiasesInfo(biases.value(), input, weights)
                                         : BuildEthosNBiasesInfo(ethosnOutput.m_Dimensions[3], input, weights);

    constexpr bool isDepthwiseConvolution = true;
    auto ethosnWeights = BuildEthosNConvolutionWeightsInfo(weights, descriptor.m_DataLayout, isDepthwiseConvolution);

    auto convolutionInfo =
        BuildEthosNConvolutionInfo(descriptor, output.GetQuantizationOffset(), output.GetQuantizationScale());

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = ethosn_lib::IsDepthwiseConvolutionSupported(
        ethosnBias, ethosnWeights, convolutionInfo, ethosnInput, &ethosnOutput, messageHelper.GetBuffer(),
        messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
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

    auto ethosnBias = biases.has_value() ? BuildEthosNBiasesInfo(biases.value(), input, weights)
                                         : BuildEthosNBiasesInfo(ethosnOutput.m_Dimensions[3], input, weights);

    constexpr bool isDepthwiseConvolution = false;
    auto ethosnWeights = BuildEthosNConvolutionWeightsInfo(weights, descriptor.m_DataLayout, isDepthwiseConvolution);

    auto convolutionInfo =
        BuildEthosNConvolutionInfo(descriptor, output.GetQuantizationOffset(), output.GetQuantizationScale());

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel = ethosn_lib::IsTransposeConvolutionSupported(
        ethosnBias, ethosnWeights, convolutionInfo, ethosnInput, &ethosnOutput, messageHelper.GetBuffer(),
        messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
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

    auto ethosnBias = BuildEthosNBiasesInfo(biases, input, weights);

    auto ethosnWeights = BuildEthosNFullyConnectedWeightsInfo(weights, descriptor.m_TransposeWeightMatrix);

    ethosn_lib::FullyConnectedInfo fullyConnectedInfo =
        BuildEthosNFullyConnectedLayerInfo(descriptor, output.GetQuantizationOffset(), output.GetQuantizationScale());

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel =
        ethosn_lib::IsFullyConnectedSupported(ethosnBias, ethosnWeights, fullyConnectedInfo, ethosnInput, &ethosnOutput,
                                              messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
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
        ethosn_lib::IsInputSupported(ethosnInput, nullptr, messageHelper.GetBuffer(), messageHelper.GetBufferSize());
    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsMemCopySupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
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
    SupportedLevel supportedLevel = ethosn_lib::IsOutputSupported(
        ethosnOutput, ethosnOutput.m_DataFormat, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
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
    SupportedLevel supportedLevel = ethosn_lib::IsPoolingSupported(
        poolingInfo, ethosnInput, &ethosnOutput, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsPreCompiledSupported(const TensorInfo& input,
                                                const PreCompiledDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);

    return IsTensorSupportedOnEthosN(input, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsRankSupported(const TensorInfo& input,
                                         const TensorInfo& output,
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
    SupportedLevel supportedLevel = ethosn_lib::IsReshapeSupported(
        ethosnShape, ethosnInput, nullptr, messageHelper.GetBuffer(), messageHelper.GetBufferSize());
    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
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
    SupportedLevel supportedLevel = ethosn_lib::IsSoftmaxSupported(
        ethosnInput, &ethosnOutput, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsSplitterSupported(const TensorInfo& input,
                                             const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                             const ViewsDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    using ethosn_lib::SupportedLevel;
    BOOST_ASSERT(outputs.size() == descriptor.GetNumViews());

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
    SupportedLevel supportedLevel = ethosn_lib::IsSplitSupported(
        ethosnInput, ethosnSplitInfo.value(), &ethosnOutputs, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
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
    SupportedLevel supportedLevel = ethosn_lib::IsDepthToSpaceSupported(
        ethosnInput, info, &ethosnOutput, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
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
    SupportedLevel supportedLevel             = ethosn_lib::IsEstimateOnlySupported(
        ethosnInputInfos, estimateInfo, nullptr, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
    return supported;
}

bool EthosNLayerSupport::IsAbsSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
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

bool EthosNLayerSupport::IsEqualSupported(const armnn::TensorInfo& input0,
                                          const armnn::TensorInfo& input1,
                                          const armnn::TensorInfo& output,
                                          armnn::Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
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
    ignore_unused(descriptor);
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
                                           armnn::Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsGatherSupported(const armnn::TensorInfo& input0,
                                           const armnn::TensorInfo& input1,
                                           const armnn::TensorInfo& output,
                                           const GatherDescriptor& descriptor,
                                           armnn::Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsGreaterSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported) const
{
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
                                         const MeanDescriptor&,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
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

bool EthosNLayerSupport::IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                           const TensorInfo& output,
                                           const OriginsDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    // This is a depreceted version of IsConcatSupported, so forward to that.
    return IsConcatSupported(inputs, output, descriptor, reasonIfUnsupported);
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
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
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

bool EthosNLayerSupport::IsQuantizeSupported(const armnn::TensorInfo&,
                                             const armnn::TensorInfo&,
                                             armnn::Optional<std::string&>) const
{
    // Disabled for now to prevent an issue running Yolo v3.
    return false;
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

bool EthosNLayerSupport::IsResizeBilinearSupported(const TensorInfo& input,
                                                   const TensorInfo& output,
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
    SupportedLevel supportedLevel = ethosn_lib::IsResizeSupported(
        ethosResizeInfo, ethosnInput, &ethosnOutput, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsRsqrtSupported(const TensorInfo& input,
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
                                                 const SpaceToDepthDescriptor&,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsSplitterSupported(const TensorInfo& input,
                                             const ViewsDescriptor&,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input }, {}, reasonIfUnsupported);
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
    if (IsLayerExcluded(g_EthosNMappings, LayerType::StandIn))
    {
        SetReason(reasonIfUnsupported, std::string("Layer declared excluded in mapping file"));
        return false;
    }

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
        ethosn_lib::IsTransposeSupported(ethosTransposeInfo.value(), ethosnInput, &ethosnOutput,
                                         messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool supported = CheckSupportedLevel(supportedLevel, g_EthosNConfig.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

}    // namespace armnn
