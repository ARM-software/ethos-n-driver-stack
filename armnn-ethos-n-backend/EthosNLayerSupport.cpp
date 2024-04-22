//
// Copyright © 2018-2024 Arm Limited.
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
#include <armnn/utility/PolymorphicDowncast.hpp>
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
bool EthosNLayerSupport::IsLayerSupported(const LayerType& type,
                                          const std::vector<TensorInfo>& infos,
                                          const BaseDescriptor& descriptor,
                                          const Optional<LstmInputParamsInfo>& lstmParamsInfo,
                                          const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmParamsInfo,
                                          Optional<std::string&> reasonIfUnsupported) const
{

    switch (type)
    {
        case LayerType::Activation:
            return IsActivationSupportedImpl(infos[0], infos[1],
                                             *(PolymorphicDowncast<const ActivationDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        case LayerType::ArgMinMax:
            return IsArgMinMaxSupportedImpl(infos[0], infos[1],
                                            *(PolymorphicDowncast<const ArgMinMaxDescriptor*>(&descriptor)),
                                            reasonIfUnsupported);
        case LayerType::BatchNormalization:
            return IsBatchNormalizationSupportedImpl(
                infos[0], infos[1], infos[2], infos[3], infos[4], infos[5],
                *(PolymorphicDowncast<const BatchNormalizationDescriptor*>(&descriptor)), reasonIfUnsupported);
        case LayerType::BatchToSpaceNd:
            return IsBatchToSpaceNdSupportedImpl(infos[0], infos[1],
                                                 *(PolymorphicDowncast<const BatchToSpaceNdDescriptor*>(&descriptor)),
                                                 reasonIfUnsupported);
        case LayerType::Comparison:
        {
            return IsComparisonSupportedImpl(infos[0], infos[1], infos[2],
                                             *(PolymorphicDowncast<const ComparisonDescriptor*>(&descriptor)),
                                             reasonIfUnsupported);
        }
        case LayerType::Concat:
        {
            std::vector<const TensorInfo*> inputInfos;
            for (uint32_t i = 0; i < (infos.size() - 1); i++)
            {
                inputInfos.push_back(&infos[i]);
            }
            return IsConcatSupportedImpl(inputInfos, infos[infos.size() - 1],
                                         *(PolymorphicDowncast<const OriginsDescriptor*>(&descriptor)),
                                         reasonIfUnsupported);
        }
        case LayerType::Constant:
            return IsConstantSupportedImpl(infos[0], reasonIfUnsupported);
        case LayerType::Convolution2d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of Convolution2d "
                                               "TensorInfos. TensorInfos should be of format: "
                                               "{input, output, weights, biases}.");
            }

            auto desc = *(PolymorphicDowncast<const Convolution2dDescriptor*>(&descriptor));
            if (infos[3] == TensorInfo())
            {
                return IsConvolution2dSupportedImpl(infos[0], infos[1], desc, infos[2], EmptyOptional(),
                                                    reasonIfUnsupported);
            }
            else
            {
                return IsConvolution2dSupportedImpl(infos[0], infos[1], desc, infos[2], infos[3], reasonIfUnsupported);
            }
        }
        case LayerType::DepthToSpace:
            return IsDepthToSpaceSupportedImpl(infos[0], infos[1],
                                               *(PolymorphicDowncast<const DepthToSpaceDescriptor*>(&descriptor)),
                                               reasonIfUnsupported);
        case LayerType::DepthwiseConvolution2d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of DepthwiseConvolution2d "
                                               "TensorInfos. TensorInfos should be of format: "
                                               "{input, output, weights, biases}.");
            }

            auto desc = *(PolymorphicDowncast<const DepthwiseConvolution2dDescriptor*>(&descriptor));
            if (infos[3] == TensorInfo())
            {
                return IsDepthwiseConvolutionSupportedImpl(infos[0], infos[1], desc, infos[2], EmptyOptional(),
                                                           reasonIfUnsupported);
            }
            else
            {
                return IsDepthwiseConvolutionSupportedImpl(infos[0], infos[1], desc, infos[2], infos[3],
                                                           reasonIfUnsupported);
            }
        }
        case LayerType::Division:
            return IsDivisionSupportedImpl(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::ElementwiseUnary:
            return IsElementwiseUnarySupportedImpl(
                infos[0], infos[1], *(PolymorphicDowncast<const ElementwiseUnaryDescriptor*>(&descriptor)),
                reasonIfUnsupported);
        case LayerType::ElementwiseBinary:
        {
            const ElementwiseBinaryDescriptor& desc =
                *PolymorphicDowncast<const ElementwiseBinaryDescriptor*>(&descriptor);
            switch (desc.m_Operation)
            {
                case BinaryOperation::Add:
                    return IsAdditionSupportedImpl(infos[0], infos[1], infos[2], reasonIfUnsupported);
                case BinaryOperation::Mul:
                    return IsMultiplicationSupportedImpl(infos[0], infos[1], infos[2], reasonIfUnsupported);
                default:
                    return false;
            }
        }
        case LayerType::FakeQuantization:
            return IsFakeQuantizationSupportedImpl(
                infos[0], *(PolymorphicDowncast<const FakeQuantizationDescriptor*>(&descriptor)), reasonIfUnsupported);
        case LayerType::Fill:
            return IsFillSupportedImpl(infos[0], infos[1], *(PolymorphicDowncast<const FillDescriptor*>(&descriptor)),
                                       reasonIfUnsupported);
        case LayerType::Floor:
            return IsFloorSupportedImpl(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::FullyConnected:
            return IsFullyConnectedSupportedImpl(infos[0], infos[1], infos[2], infos[3],
                                                 *(PolymorphicDowncast<const FullyConnectedDescriptor*>(&descriptor)),
                                                 reasonIfUnsupported);
        case LayerType::Gather:
            return IsGatherSupportedImpl(infos[0], infos[1], infos[2],
                                         *(PolymorphicDowncast<const GatherDescriptor*>(&descriptor)),
                                         reasonIfUnsupported);
        case LayerType::Input:
            return IsInputSupportedImpl(infos[0], reasonIfUnsupported);
        case LayerType::InstanceNormalization:
            return IsInstanceNormalizationSupportedImpl(
                infos[0], infos[1], *(PolymorphicDowncast<const InstanceNormalizationDescriptor*>(&descriptor)),
                reasonIfUnsupported);
        case LayerType::L2Normalization:
            return IsL2NormalizationSupportedImpl(infos[0], infos[1],
                                                  *(PolymorphicDowncast<const L2NormalizationDescriptor*>(&descriptor)),
                                                  reasonIfUnsupported);
        case LayerType::LogicalBinary:
            return IsLogicalBinarySupportedImpl(infos[0], infos[1], infos[2],
                                                *(PolymorphicDowncast<const LogicalBinaryDescriptor*>(&descriptor)),
                                                reasonIfUnsupported);
        case LayerType::Lstm:
            return IsLstmSupportedImpl(infos[0], infos[1], infos[2], infos[3], infos[4], infos[5], infos[6],
                                       *(PolymorphicDowncast<const LstmDescriptor*>(&descriptor)),
                                       lstmParamsInfo.value(), reasonIfUnsupported);
        case LayerType::QLstm:
            return IsQLstmSupportedImpl(infos[0], infos[1], infos[2], infos[3], infos[4], infos[5],
                                        *(PolymorphicDowncast<const QLstmDescriptor*>(&descriptor)),
                                        lstmParamsInfo.value(), reasonIfUnsupported);
        case LayerType::Map:
            return true;
        case LayerType::Maximum:
            return IsMaximumSupportedImpl(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Mean:
            return IsMeanSupportedImpl(infos[0], infos[1], *(PolymorphicDowncast<const MeanDescriptor*>(&descriptor)),
                                       reasonIfUnsupported);
        case LayerType::MemCopy:
            return IsMemCopySupportedImpl(std::move(infos[0]), std::move(infos[1]), reasonIfUnsupported);
        case LayerType::Merge:
            return IsMergeSupportedImpl(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Minimum:
            return IsMinimumSupportedImpl(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Normalization:
            return IsNormalizationSupportedImpl(infos[0], infos[1],
                                                *(PolymorphicDowncast<const NormalizationDescriptor*>(&descriptor)),
                                                reasonIfUnsupported);
        case LayerType::Output:
            return IsOutputSupportedImpl(infos[0], reasonIfUnsupported);
        case LayerType::Pad:
            return IsPadSupportedImpl(infos[0], infos[1], *(PolymorphicDowncast<const PadDescriptor*>(&descriptor)),
                                      reasonIfUnsupported);
        case LayerType::Permute:
            return IsPermuteSupportedImpl(
                infos[0], infos[1], *(PolymorphicDowncast<const PermuteDescriptor*>(&descriptor)), reasonIfUnsupported);
        case LayerType::Pooling2d:
            return IsPooling2dSupportedImpl(infos[0], infos[1],
                                            *(PolymorphicDowncast<const Pooling2dDescriptor*>(&descriptor)),
                                            reasonIfUnsupported);
        case LayerType::PreCompiled:
            return IsPreCompiledSupportedImpl(
                infos[0], *(PolymorphicDowncast<const PreCompiledDescriptor*>(&descriptor)), reasonIfUnsupported);
        case LayerType::Prelu:
            return IsPreluSupportedImpl(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Quantize:
            return IsQuantizeSupportedImpl(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::QuantizedLstm:
            return IsQuantizedLstmSupportedImpl(infos[0], infos[1], infos[2], infos[3], infos[4],
                                                quantizedLstmParamsInfo.value(), reasonIfUnsupported);
        case LayerType::Reshape:
            return IsReshapeSupportedImpl(
                infos[0], infos[1], *(PolymorphicDowncast<const ReshapeDescriptor*>(&descriptor)), reasonIfUnsupported);
        case LayerType::Rank:
            return IsRankSupportedImpl(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::Resize:
            return IsResizeSupportedImpl(
                infos[0], infos[1], *(PolymorphicDowncast<const ResizeDescriptor*>(&descriptor)), reasonIfUnsupported);
        case LayerType::Reduce:
            return IsReduceSupportedImpl(
                infos[0], infos[1], *(PolymorphicDowncast<const ReduceDescriptor*>(&descriptor)), reasonIfUnsupported);
        case LayerType::Slice:
            return IsSliceSupportedImpl(infos[0], infos[1], *(PolymorphicDowncast<const SliceDescriptor*>(&descriptor)),
                                        reasonIfUnsupported);
        case LayerType::SpaceToBatchNd:
            return IsSpaceToBatchNdSupportedImpl(infos[0], infos[1],
                                                 *(PolymorphicDowncast<const SpaceToBatchNdDescriptor*>(&descriptor)),
                                                 reasonIfUnsupported);
        case LayerType::SpaceToDepth:
            return IsSpaceToDepthSupportedImpl(infos[0], infos[1],
                                               *(PolymorphicDowncast<const SpaceToDepthDescriptor*>(&descriptor)),
                                               reasonIfUnsupported);
        case LayerType::Splitter:
        {
            std::vector<TensorInfo> outputInfos;
            for (uint32_t i = 1; i < infos.size(); i++)
            {
                outputInfos.push_back(infos[i]);
            }
            return IsSplitterSupportedImpl(infos[0], { outputInfos.begin(), outputInfos.end() },
                                           *(PolymorphicDowncast<const ViewsDescriptor*>(&descriptor)),
                                           reasonIfUnsupported);
        }
        case LayerType::Stack:
        {
            std::vector<const TensorInfo*> inputInfos;
            for (uint32_t i = 0; i < infos.size() - 1; i++)
            {
                inputInfos.push_back(&infos[i]);
            }
            return IsStackSupportedImpl(inputInfos, infos[infos.size() - 1],
                                        *(PolymorphicDowncast<const StackDescriptor*>(&descriptor)),
                                        reasonIfUnsupported);
        }
        case LayerType::StandIn:
        {
            auto desc = *(PolymorphicDowncast<const StandInDescriptor*>(&descriptor));

            if (infos.size() != (desc.m_NumInputs + desc.m_NumOutputs))
            {
                throw InvalidArgumentException("Number of StandIn layer TensorInfos does not equal "
                                               "the combined number of input and output slots assigned "
                                               "to the StandIn descriptor");
            }

            std::vector<const TensorInfo*> inputInfos;
            for (uint32_t i = 0; i < desc.m_NumInputs; i++)
            {
                inputInfos.push_back(&infos[i]);
            }
            std::vector<const TensorInfo*> outputInfos;
            for (uint32_t i = desc.m_NumInputs; i < infos.size(); i++)
            {
                outputInfos.push_back(&infos[i]);
            }

            return IsStandInSupportedImpl(inputInfos, outputInfos, desc, reasonIfUnsupported);
        }
        case LayerType::StridedSlice:
            return IsStridedSliceSupportedImpl(infos[0], infos[1],
                                               *(PolymorphicDowncast<const StridedSliceDescriptor*>(&descriptor)),
                                               reasonIfUnsupported);
        case LayerType::Subtraction:
            return IsSubtractionSupportedImpl(infos[0], infos[1], infos[2], reasonIfUnsupported);
        case LayerType::Switch:
            return IsSwitchSupportedImpl(infos[0], infos[1], infos[2], infos[3], reasonIfUnsupported);
        case LayerType::Transpose:
            return IsTransposeSupportedImpl(infos[0], infos[1],
                                            *(PolymorphicDowncast<const TransposeDescriptor*>(&descriptor)),
                                            reasonIfUnsupported);
        case LayerType::TransposeConvolution2d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of TransposeConvolution2d "
                                               "TensorInfos. TensorInfos should be of format: "
                                               "{input, output, weights, biases}.");
            }

            auto desc = *(PolymorphicDowncast<const TransposeConvolution2dDescriptor*>(&descriptor));
            if (infos[3] == TensorInfo())
            {
                return IsTransposeConvolution2dSupportedImpl(infos[0], infos[1], desc, infos[2], EmptyOptional(),
                                                             reasonIfUnsupported);
            }
            else
            {
                return IsTransposeConvolution2dSupportedImpl(infos[0], infos[1], desc, infos[2], infos[3],
                                                             reasonIfUnsupported);
            }
        }
        case LayerType::Unmap:
            return true;
        case LayerType::Cast:
            return IsCastSupportedImpl(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::Shape:
            return IsShapeSupportedImpl(infos[0], infos[1], reasonIfUnsupported);
        case LayerType::ConvertFp16ToFp32:
            // Fall-through: The Support Library does not support floating point types, even in performance-only mode.
        case LayerType::ConvertFp32ToFp16:
            // Fall-through: The Support Library does not support floating point types, even in performance-only mode.
        case LayerType::LogSoftmax:
            // Fall-through: The Support Library does not support LogSoftmax.
        case LayerType::Softmax:
            // Fall-through: The Support Library does not support Softmax.
        case LayerType::Debug:
            // Fall-through: The Support Library does not support Debug
        case LayerType::Dequantize:
            // Fall-through: The Support Library does not support Dequantize
        case LayerType::MemImport:
            // Fall-through:
            // This is a 'meta' layer type related to avoiding tensor copies between backends.
            // We should never receive this layer because we don't advertise support for this feature.
        default:
            return false;
    }
}

bool EthosNLayerSupport::IsActivationSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsAdditionSupportedImpl(const TensorInfo& input0,
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
    bool supported = EthosNLayerSupport::IsDepthwiseConvolutionSupportedImpl(
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

bool EthosNLayerSupport::IsConcatSupportedImpl(const std::vector<const TensorInfo*>& inputs,
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

bool EthosNLayerSupport::IsConstantSupportedImpl(const TensorInfo& info,
                                                 Optional<std::string&> reasonIfUnsupported) const
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

bool EthosNLayerSupport::IsConvolution2dSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsDepthwiseConvolutionSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsTransposeConvolution2dSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsFullyConnectedSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsInputSupportedImpl(const TensorInfo& input, Optional<std::string&> reasonIfUnsupported) const
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

bool EthosNLayerSupport::IsMemCopySupportedImpl(const TensorInfo& input,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(input);
    IgnoreUnused(output);
    IgnoreUnused(reasonIfUnsupported);
    return true;
}

bool EthosNLayerSupport::IsOutputSupportedImpl(const TensorInfo& output,
                                               Optional<std::string&> reasonIfUnsupported) const
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

bool EthosNLayerSupport::IsPooling2dSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsPreCompiledSupportedImpl(const TensorInfo& input,
                                                    const PreCompiledDescriptor& descriptor,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);

    return IsTensorSupportedOnEthosN(input, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsRankSupportedImpl(const TensorInfo& input,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsReduceSupportedImpl(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const ReduceDescriptor&,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsReshapeSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsSplitterSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsDepthToSpaceSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsArgMinMaxSupportedImpl(const armnn::TensorInfo& input,
                                                  const armnn::TensorInfo& output,
                                                  const armnn::ArgMinMaxDescriptor&,
                                                  armnn::Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsBatchNormalizationSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsBatchToSpaceNdSupportedImpl(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const BatchToSpaceNdDescriptor&,
                                                       Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsCastSupportedImpl(const TensorInfo& input,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsComparisonSupportedImpl(const TensorInfo& input0,
                                                   const TensorInfo& input1,
                                                   const TensorInfo& output,
                                                   const ComparisonDescriptor&,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsDilatedDepthwiseConvolutionSupportedImpl(const TensorInfo& input,
                                                                    const TensorInfo& output,
                                                                    const DepthwiseConvolution2dDescriptor&,
                                                                    const TensorInfo&,
                                                                    const Optional<TensorInfo>&,
                                                                    Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsDivisionSupportedImpl(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsElementwiseUnarySupportedImpl(const TensorInfo& input,
                                                         const TensorInfo& output,
                                                         const ElementwiseUnaryDescriptor&,
                                                         Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsFakeQuantizationSupportedImpl(const TensorInfo& input,
                                                         const FakeQuantizationDescriptor&,
                                                         Optional<std::string&> reasonIfUnsupported) const
{
    // Even though this layer probably has minimal usefulness in an already-quantized context, the Ethos-N
    // could support it.
    return CheckEstimateOnlySupported({ input }, {}, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsFillSupportedImpl(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const FillDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsFloorSupportedImpl(const TensorInfo& input,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsGatherSupportedImpl(const armnn::TensorInfo& input0,
                                               const armnn::TensorInfo& input1,
                                               const armnn::TensorInfo& output,
                                               const GatherDescriptor& descriptor,
                                               armnn::Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsInstanceNormalizationSupportedImpl(const TensorInfo& input,
                                                              const TensorInfo& output,
                                                              const InstanceNormalizationDescriptor&,
                                                              Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsL2NormalizationSupportedImpl(const TensorInfo& input,
                                                        const TensorInfo& output,
                                                        const L2NormalizationDescriptor&,
                                                        Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsLogicalBinarySupportedImpl(const TensorInfo& input0,
                                                      const TensorInfo& input1,
                                                      const TensorInfo& output,
                                                      const LogicalBinaryDescriptor& descriptor,
                                                      Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsLogicalUnarySupportedImpl(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const ElementwiseUnaryDescriptor& descriptor,
                                                     Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(descriptor);
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsLstmSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsMaximumSupportedImpl(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsMeanSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsMergeSupportedImpl(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsMinimumSupportedImpl(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsMultiplicationSupportedImpl(const TensorInfo& input0,
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

    auto ethosnInput0 = BuildEthosNTensorInfo(input0, DataLayout::NHWC);
    auto ethosnInput1 = BuildEthosNTensorInfo(input1, DataLayout::NHWC);
    auto ethosnOutput = BuildEthosNTensorInfo(output, DataLayout::NHWC);

    // First try checking for support using a native addition
    ReasonMessageHelper messageHelper;
    SupportedLevel nativeSupportedLevel =
        m_Queries.IsMultiplicationSupported(ethosnInput0, ethosnInput1, ethosnOutput.m_QuantizationInfo, &ethosnOutput,
                                            messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    bool nativeSupported = CheckSupportedLevel(nativeSupportedLevel, m_Config.m_PerfOnly);

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

    if (nativeSupported)
    {
        return MultiplicationSupportedMode::Native;
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

        TensorInfo weightsInfo        = constantInfo;
        unsigned int outputChannels   = output.GetNumDimensions() > 2 ? output.GetShape()[3] : 1;
        unsigned int inputChannels    = inputInfo.GetNumDimensions() > 2 ? inputInfo.GetShape()[3] : 1;
        unsigned int constantChannels = constantInfo.GetNumDimensions() > 2 ? constantInfo.GetShape()[3] : 1;

        unsigned int M = outputChannels / inputChannels;
        weightsInfo.SetShape({ 1, 1, 1, constantChannels * M });    //1HW(I*M)

        std::string depthwiseReasonIfUnsupported;
        bool supported = EthosNLayerSupport::IsDepthwiseConvolutionSupportedImpl(
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

bool EthosNLayerSupport::IsNormalizationSupportedImpl(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const NormalizationDescriptor&,
                                                      Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsPadSupportedImpl(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const PadDescriptor& padding,
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

    if (padding.m_PaddingMode != PaddingMode::Constant)
    {
        SetReason(reasonIfUnsupported, "Only constant padding supported");
        return false;
    }

    if (std::abs(output.GetQuantizationScale() - input.GetQuantizationScale()) > 0.00001f)
    {
        SetReason(reasonIfUnsupported, "Input and output quantization scales are not equal");
        return false;
    }

    if (output.GetQuantizationOffset() != input.GetQuantizationOffset())
    {
        SetReason(reasonIfUnsupported, "Input and output quantization offsets are not equal");
        return false;
    }

    if (std::abs(padding.m_PadValue - static_cast<float>(input.GetQuantizationOffset())) > 0.00001f)
    {
        SetReason(reasonIfUnsupported, "Only zero (or zero point if quantized) padding supported");
        return false;
    }

    if (padding.m_PadList.size() > 4)
    {
        SetReason(reasonIfUnsupported, "Pad List contains more than 4 dimensions");
        return false;
    }

    std::pair<unsigned int, unsigned int> zeroPad = { 0, 0 };
    auto extendedPadList                          = ExtendPadList(padding.m_PadList, input.GetShape());
    if (extendedPadList[0] != zeroPad || extendedPadList[3] != zeroPad)
    {
        SetReason(reasonIfUnsupported, "Only padding in the middle two dimensions supported");
        return false;
    }

    ReasonMessageHelper messageHelper;
    SupportedLevel supportedLevel =
        m_Queries.IsStandalonePaddingSupported(BuildEthosNPaddingInfo(padding, input.GetShape()), ethosnInput,
                                               &ethosnOutput, messageHelper.GetBuffer(), messageHelper.GetBufferSize());

    if (!CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly))
    {
        SetReason(reasonIfUnsupported, "Padding config not supported");
    }

    bool supported = CheckSupportedLevel(supportedLevel, m_Config.m_PerfOnly);
    SetReasonIfUnsupported(supported, messageHelper, reasonIfUnsupported);
    return supported;
}

bool EthosNLayerSupport::IsPermuteSupportedImpl(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const PermuteDescriptor&,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsPreluSupportedImpl(const TensorInfo& input,
                                              const TensorInfo&,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsQuantizeSupportedImpl(const armnn::TensorInfo& input,
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

bool EthosNLayerSupport::IsQLstmSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsQuantizedLstmSupportedImpl(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const TensorInfo&,
                                                      const TensorInfo&,
                                                      const TensorInfo&,
                                                      const QuantizedLstmInputParamsInfo&,
                                                      Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsResizeSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsShapeSupportedImpl(const TensorInfo& input,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsSliceSupportedImpl(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const SliceDescriptor&,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsSpaceToBatchNdSupportedImpl(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const SpaceToBatchNdDescriptor&,
                                                       Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsSpaceToDepthSupportedImpl(const TensorInfo& input,
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

bool EthosNLayerSupport::IsStackSupportedImpl(const std::vector<const TensorInfo*>& inputs,
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

bool EthosNLayerSupport::IsStandInSupportedImpl(const std::vector<const TensorInfo*>& inputs,
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

bool EthosNLayerSupport::IsStridedSliceSupportedImpl(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const StridedSliceDescriptor&,
                                                     Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported(input, output, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsSubtractionSupportedImpl(const TensorInfo& input0,
                                                    const TensorInfo& input1,
                                                    const TensorInfo& output,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsSwitchSupportedImpl(const TensorInfo& input0,
                                               const TensorInfo& input1,
                                               const TensorInfo& output0,
                                               const TensorInfo& output1,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return CheckEstimateOnlySupported({ input0, input1 }, { output0, output1 }, reasonIfUnsupported);
}

bool EthosNLayerSupport::IsTransposeSupportedImpl(const TensorInfo& input,
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

}    // namespace armnn
