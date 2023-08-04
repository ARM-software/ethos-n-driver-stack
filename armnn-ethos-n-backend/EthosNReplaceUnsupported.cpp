//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNReplaceUnsupported.hpp"

#include "EthosNLayerSupport.hpp"
#include "EthosNTensorUtils.hpp"

#include <armnn/backends/OptimizationViews.hpp>
#include <armnn/backends/SubgraphView.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{
namespace ethosnbackend
{

// Replaces the pattern Constant-Multiplication with an optimized DepthwiseConvolution2d operation,
// if appropriate.
// Original pattern:
// Input    ->
//              Multiplication -> Output
// Constant ->
// Expected modified pattern:
// Input -> DepthwiseConvolution2d -> Output
//
bool ReplaceConstantMultiplicationWithDepthwise(
    SubgraphView& subgraph, IConnectableLayer* layer, INetwork& network, const EthosNConfig&, const std::vector<char>&)
{
    if (layer->GetType() == LayerType::ElementwiseBinary)
    {
        const ElementwiseBinaryDescriptor& desc =
            *PolymorphicDowncast<const ElementwiseBinaryDescriptor*>(&(layer->GetParameters()));
        if (desc.m_Operation != BinaryOperation::Mul)
        {
            return false;
        }

        IInputSlot* patternSubgraphInput = &layer->GetInputSlot(0);

        IOutputSlot* inputConnection = patternSubgraphInput->GetConnection();
        IOutputSlot* constConnection = layer->GetInputSlot(1).GetConnection();

        // Figure out which of the two inputs is the constant
        if (constConnection->GetOwningIConnectableLayer().GetType() != LayerType::Constant)
        {
            patternSubgraphInput = &layer->GetInputSlot(1);
            std::swap(inputConnection, constConnection);
        }

        if (constConnection->GetOwningIConnectableLayer().GetType() == LayerType::Constant)
        {
            const TensorInfo& inputInfo      = inputConnection->GetTensorInfo();
            IConnectableLayer* constantLayer = &constConnection->GetOwningIConnectableLayer();
            const TensorInfo& constInfo      = constantLayer->GetOutputSlot(0).GetTensorInfo();

            // Add a Depthwise only where the constant input is a scalar that takes the form { 1, 1, 1, C }.
            // The scalar is used as weights for the convolution.
            if (constInfo.GetShape() == TensorShape({ 1, 1, 1, inputInfo.GetShape()[3] }))
            {
                DepthwiseConvolution2dDescriptor desc;
                desc.m_DataLayout = DataLayout::NHWC;

                const auto depthwiseLayer =
                    network.AddDepthwiseConvolution2dLayer(desc, "Replacement for Constant-Multiplication");

                TensorInfo weightInfo        = constInfo;
                const TensorInfo& outputInfo = layer->GetOutputSlot(0).GetTensorInfo();
                unsigned int M               = outputInfo.GetShape()[3] / inputInfo.GetShape()[3];
                ARMNN_ASSERT_MSG(M == 1, "Constant multiplication only support 1x1x1xC, so M should always be 1 here");
                weightInfo.SetShape({ 1, 1, 1, constInfo.GetShape()[3] * M });    //1HW(I*M)

                const void* weightData = constantLayer->GetConstantTensorsByRef()[0].get()->GetConstTensor<void>();

                const ConstTensor weights(weightInfo, weightData);

                const auto weightsLayer =
                    network.AddConstantLayer(weights, "Replacement for Constant-Multiplication Weights");
                weightsLayer->GetOutputSlot(0).SetTensorInfo(weightInfo);
                weightsLayer->GetOutputSlot(0).Connect(depthwiseLayer->GetInputSlot(1));

                SubgraphView patternSubgraph({ layer, constantLayer }, { patternSubgraphInput },
                                             { &layer->GetOutputSlot(0) });

                /// Constructs a sub-graph view with the new weights, depthwise and the correct input and output slots.
                SubgraphView replacementSubgraph({ depthwiseLayer, weightsLayer }, { &depthwiseLayer->GetInputSlot(0) },
                                                 { &depthwiseLayer->GetOutputSlot(0) });

                subgraph.SubstituteSubgraph(patternSubgraph, replacementSubgraph);

                return true;
            }
        }
    }
    return false;
}

// Replaces the pattern Constant-Multiplication with a ReinterpretQuantize operation, if appropriate.
// Original pattern:
// Input    ->
//              Multiplication -> Output
// Constant ->
// Expected modified pattern:
// Input -> ReinterpretQuantize -> Output
//
bool ReplaceScalarMultiplicationWithReinterpretQuantization(SubgraphView& subgraph,
                                                            IConnectableLayer* layer,
                                                            INetwork& network,
                                                            const EthosNConfig&,
                                                            const std::vector<char>&,
                                                            std::string& outFailureReason)
{
    if (layer->GetType() == LayerType::ElementwiseBinary)
    {
        const ElementwiseBinaryDescriptor& desc =
            *PolymorphicDowncast<const ElementwiseBinaryDescriptor*>(&(layer->GetParameters()));
        if (desc.m_Operation != BinaryOperation::Mul)
        {
            return false;
        }

        IInputSlot* patternSubgraphInput = &layer->GetInputSlot(0);

        IOutputSlot* inputConnection = patternSubgraphInput->GetConnection();
        IOutputSlot* constConnection = layer->GetInputSlot(1).GetConnection();

        // Figure out which of the two inputs is the constant
        if (constConnection->GetOwningIConnectableLayer().GetType() != LayerType::Constant)
        {
            patternSubgraphInput = &layer->GetInputSlot(1);
            std::swap(inputConnection, constConnection);
        }

        if (constConnection->GetOwningIConnectableLayer().GetType() == LayerType::Constant)
        {
            IConnectableLayer& constantLayer = constConnection->GetOwningIConnectableLayer();
            const TensorInfo& constInfo      = constConnection->GetTensorInfo();
            const TensorInfo& outputInfo     = layer->GetOutputSlot(0).GetTensorInfo();
            const TensorInfo& inputInfo      = inputConnection->GetTensorInfo();

            // Add a ReinterpretQuantize only where the constant input is a scalar that takes the form { 1, 1, 1, 1 }.
            if (constInfo.GetShape() == TensorShape({ 1, 1, 1, 1 }))
            {
                auto ConvertDataToFloat = [](IConnectableLayer& layer, DataType dataType) {
                    ConstTensorHandle& tensorHandle = *layer.GetConstantTensorsByRef()[0].get();
                    switch (dataType)
                    {
                        case DataType::QAsymmU8:
                            return static_cast<float>(tensorHandle.GetConstTensor<uint8_t>()[0]);
                        case DataType::QSymmS8:
                        case DataType::QAsymmS8:
                            return static_cast<float>(tensorHandle.GetConstTensor<int8_t>()[0]);
                        case DataType::Signed32:
                            return static_cast<float>(tensorHandle.GetConstTensor<int32_t>()[0]);
                        default:
                            throw Exception("Data type not supported");
                    }
                };

                float data;
                try
                {
                    data = ConvertDataToFloat(constantLayer, constInfo.GetDataType());
                }
                catch (const std::exception&)
                {
                    // Data type is not supported
                    outFailureReason = "Data type is not supported";
                    return false;
                }

                float constQuantScale     = constInfo.GetQuantizationScale();
                float constQuantZeroPoint = static_cast<float>(constInfo.GetQuantizationOffset());
                float calculatedOutputQuantScale =
                    inputInfo.GetQuantizationScale() * constQuantScale * (data - constQuantZeroPoint);

                // This check will ensure that the quantisation info of the output, input and constant are coherent with each other.
                // The tolerance value has been selected as 0.004 as 1/255 = 0.0039 and after rounding off we get 0.004.
                if (std::abs(calculatedOutputQuantScale - outputInfo.GetQuantizationScale()) > 0.004f)
                {
                    outFailureReason = "Quantization info for input, scalar and output are not coherent";
                    return false;
                }

                StandInDescriptor desc;
                desc.m_NumInputs  = 1;
                desc.m_NumOutputs = 1;

                // We are using a StandIn layer here as a generic layer since Arm NN has no LayerType::ReinterpretQuantize
                // that we could directly add.
                // We set a custom value to name parameter of the StandIn layer which then is used to add the
                // ReinterpretQuantize layer from the Support Library.
                const auto standInLayer =
                    network.AddStandInLayer(desc, "EthosNBackend:ReplaceScalarMulWithReinterpretQuantization");

                SubgraphView patternSubgraph({ layer, &constantLayer }, { patternSubgraphInput },
                                             { &layer->GetOutputSlot(0) });

                subgraph.SubstituteSubgraph(patternSubgraph, SubgraphView{ standInLayer });

                return true;
            }
        }
    }
    return false;
}

// Replaces the pattern Constant-Multiplication with either a DepthwiseConvolution2d operation or a ReinterpretQuantize operation, whichever is appropriate.
// Original pattern:
// Input    ->
//              Multiplication -> Output
// Constant ->
// Expected modified pattern:
// Input -> DepthwiseConvolution2d -> Output
//                 OR
// Input -> ReinterpretQuantize -> Output
//
bool ReplaceMultiplication(SubgraphView& subgraph,
                           IConnectableLayer* layer,
                           INetwork& network,
                           const EthosNConfig& config,
                           const std::vector<char>& capabilities)
{
    if (layer->GetType() == LayerType::ElementwiseBinary)
    {
        const ElementwiseBinaryDescriptor& desc =
            *PolymorphicDowncast<const ElementwiseBinaryDescriptor*>(&(layer->GetParameters()));
        if (desc.m_Operation != BinaryOperation::Mul)
        {
            return false;
        }

        EthosNLayerSupport supportChecks(config, capabilities);

        const IOutputSlot* inputConnection0                           = layer->GetInputSlot(0).GetConnection();
        const IOutputSlot* inputConnection1                           = layer->GetInputSlot(1).GetConnection();
        EthosNLayerSupport::MultiplicationSupportedMode supportedMode = supportChecks.GetMultiplicationSupportedMode(
            inputConnection0->GetTensorInfo(), inputConnection1->GetTensorInfo(),
            layer->GetOutputSlot(0).GetTensorInfo());

        std::string failureReason;

        switch (supportedMode)
        {
            case EthosNLayerSupport::MultiplicationSupportedMode::None:
                return false;
                break;
            case EthosNLayerSupport::MultiplicationSupportedMode::EstimateOnly:
                return false;
                break;
            case EthosNLayerSupport::MultiplicationSupportedMode::ReplaceWithDepthwise:
                return ReplaceConstantMultiplicationWithDepthwise(subgraph, layer, network, config, capabilities);
                break;
            case EthosNLayerSupport::MultiplicationSupportedMode::ReplaceWithReinterpretQuantize:
                return ReplaceScalarMultiplicationWithReinterpretQuantization(subgraph, layer, network, config,
                                                                              capabilities, failureReason);
                break;
            default:
                throw Exception("Found unknown MultiplicationSupportedMode value");
                break;
        }
    }
    return false;
}

// Replaces the pattern Constant-Addition with an optimized DepthwiseConvolution2d operation, if appropriate.
// Original pattern:
// Input    ->
//              Addition -> Output
// Constant ->
// Expected modified pattern:
// Input -> DepthwiseConvolution2d -> Output
//
bool ReplaceConstantAdditionWithDepthwise(SubgraphView& subgraph, IConnectableLayer* layer, INetwork& network)
{
    if (layer->GetType() == LayerType::ElementwiseBinary)
    {
        const ElementwiseBinaryDescriptor& desc =
            *PolymorphicDowncast<const ElementwiseBinaryDescriptor*>(&(layer->GetParameters()));
        if (desc.m_Operation != BinaryOperation::Add)
        {
            return false;
        }

        // Figure out which of the two inputs is the constant
        IConnectableLayer* inputLayer    = nullptr;
        IConnectableLayer* constantLayer = nullptr;
        IInputSlot* subgraphInputSlot    = nullptr;
        IConnectableLayer* inputLayer0   = &layer->GetInputSlot(0).GetConnection()->GetOwningIConnectableLayer();
        IConnectableLayer* inputLayer1   = &layer->GetInputSlot(1).GetConnection()->GetOwningIConnectableLayer();
        if (inputLayer0->GetType() == LayerType::Constant)
        {
            inputLayer        = inputLayer1;
            constantLayer     = inputLayer0;
            subgraphInputSlot = &layer->GetInputSlot(1);
        }
        else if (inputLayer1->GetType() == LayerType::Constant)
        {
            inputLayer        = inputLayer0;
            constantLayer     = inputLayer1;
            subgraphInputSlot = &layer->GetInputSlot(0);
        }
        else
        {
            // Neither input is constant, so can't make the replacement
            return false;
        }

        const TensorInfo& inputInfo  = inputLayer->GetOutputSlot(0).GetTensorInfo();
        const TensorInfo& constInfo  = constantLayer->GetOutputSlot(0).GetTensorInfo();
        const TensorInfo& outputInfo = layer->GetOutputSlot(0).GetTensorInfo();

        // Get the configuration of the replacement layer.
        // Note that we expect this should always succeed, because otherwise the IsSupported check above would have failed.
        std::string failureReason;
        Optional<ConstantAddToDepthwiseReplacementConfig> replacementConfigOpt =
            CalcConstantAddToDepthwiseReplacementConfig(inputInfo, constInfo, outputInfo, failureReason);
        if (!replacementConfigOpt.has_value())
        {
            return false;
        }
        const ConstantAddToDepthwiseReplacementConfig& replacementConfig = replacementConfigOpt.value();

        const auto depthwiseLayer =
            network.AddDepthwiseConvolution2dLayer(replacementConfig.m_Desc, "Replacement for Constant-Addition");

        // Create identity weights
        const std::vector<uint8_t> weightsData(replacementConfig.m_WeightsInfo.GetNumElements(),
                                               replacementConfig.m_WeightsQuantizedValue);
        const ConstTensor weights(replacementConfig.m_WeightsInfo, weightsData);

        // Rescale the bias data
        const void* constData = constantLayer->GetConstantTensorsByRef()[0].get()->GetConstTensor<void>();
        Optional<std::vector<int32_t>> rescaledBiasData =
            ethosntensorutils::ConvertTensorValuesToSigned32(constData, constInfo, replacementConfig.m_BiasInfo);
        if (!rescaledBiasData.has_value())
        {
            // Unsupported conversion. This should have been checked by CalcConstantAddToDepthwiseReplacementConfig()
            // so we should never hit this in practice.
            return false;
        }
        const ConstTensor rescaledBias(replacementConfig.m_BiasInfo, rescaledBiasData.value());

        SubgraphView patternSubgraph({ layer, constantLayer }, { subgraphInputSlot }, { &layer->GetOutputSlot(0) });

        const auto weightsLayer =
            network.AddConstantLayer(weights, "Replacement for Constant-Addition Identity Weights");
        weightsLayer->GetOutputSlot(0).SetTensorInfo(replacementConfig.m_WeightsInfo);
        weightsLayer->GetOutputSlot(0).Connect(depthwiseLayer->GetInputSlot(1));

        const auto biasLayer = network.AddConstantLayer(rescaledBias, "Replacement for Constant-Addition Bias");
        biasLayer->GetOutputSlot(0).SetTensorInfo(replacementConfigOpt.value().m_BiasInfo);
        biasLayer->GetOutputSlot(0).Connect(depthwiseLayer->GetInputSlot(2));

        /// Constructs a sub-graph view with the depthwise, bias and weight layers with the correct input and output slots.
        SubgraphView view({ depthwiseLayer, weightsLayer, biasLayer }, { &depthwiseLayer->GetInputSlot(0) },
                          { &depthwiseLayer->GetOutputSlot(0) });

        subgraph.SubstituteSubgraph(patternSubgraph, view);

        return true;
    }
    return false;
}

bool ReplaceConstantAdditionWithReinterpretQuantization(SubgraphView& subgraph,
                                                        IConnectableLayer* layer,
                                                        INetwork& network,
                                                        std::string& outFailureReason)
{
    if (layer->GetType() == LayerType::ElementwiseBinary)
    {
        const ElementwiseBinaryDescriptor& desc =
            *PolymorphicDowncast<const ElementwiseBinaryDescriptor*>(&(layer->GetParameters()));
        if (desc.m_Operation != BinaryOperation::Add)
        {
            return false;
        }
        IInputSlot* patternSubgraphInput = &layer->GetInputSlot(0);

        // Figure out which of the two inputs is the constant, swap if necessary
        IConnectableLayer* inputLayer    = &patternSubgraphInput->GetConnection()->GetOwningIConnectableLayer();
        IConnectableLayer* constantLayer = &layer->GetInputSlot(1).GetConnection()->GetOwningIConnectableLayer();
        if (constantLayer->GetType() != LayerType::Constant)
        {
            patternSubgraphInput = &layer->GetInputSlot(1);
            std::swap(inputLayer, constantLayer);
        }

        // If still not constant, neither input is
        if (constantLayer->GetType() != LayerType::Constant)
        {
            // Neither Layer is constant
            return false;
        }

        // Get layer tensor info
        const TensorInfo& constInfo  = constantLayer->GetOutputSlot(0).GetTensorInfo();
        const TensorInfo& outputInfo = layer->GetOutputSlot(0).GetTensorInfo();
        const TensorInfo& inputInfo  = inputLayer->GetOutputSlot(0).GetTensorInfo();

        // Add a Reinterpret only where the constant input is a scalar that takes the form { 1, 1, 1, 1 }.
        // The scalar is used as weights for the convolution.
        if (constInfo.GetShape() == TensorShape({ 1, 1, 1, 1 }))
        {

            auto ConvertDataToFloat = [](IConnectableLayer* layer, DataType dataType) {
                switch (dataType)
                {
                    case DataType::QAsymmU8:
                        return static_cast<float>(
                            layer->GetConstantTensorsByRef()[0].get()->GetConstTensor<uint8_t>()[0]);
                    case DataType::QSymmS8:
                    case DataType::QAsymmS8:
                        return static_cast<float>(
                            layer->GetConstantTensorsByRef()[0].get()->GetConstTensor<int8_t>()[0]);
                    case DataType::Signed32:
                        return static_cast<float>(
                            layer->GetConstantTensorsByRef()[0].get()->GetConstTensor<int32_t>()[0]);
                    default:
                        throw Exception("Data type not supported");
                }
            };

            // Get single value in constant layer
            float data;
            try
            {
                data = ConvertDataToFloat(constantLayer, constInfo.GetDataType());
                data = data - static_cast<float>(constInfo.GetQuantizationOffset());
                data = data * constInfo.GetQuantizationScale();
            }
            catch (const std::exception&)
            {
                // Data type is not supported
                outFailureReason = "Data type is not supported";
                return false;
            }

            StandInDescriptor desc;
            desc.m_NumInputs  = 1;
            desc.m_NumOutputs = 1;

            // Ensure calculated 0 point equal to output zero point
            float outputOffset = static_cast<float>(outputInfo.GetQuantizationOffset());
            float inputOffset  = static_cast<float>(inputInfo.GetQuantizationOffset());

            float calculatedOutputOffset = (inputOffset - (data / inputInfo.GetQuantizationScale()));

            // If calculated and output offset values are outside margin of error, fail this replacement
            if (std::abs(calculatedOutputOffset - outputOffset) > 1.0f)
            {
                outFailureReason = "Quantization info for input, scalar and output are not coherent";
                return false;
            }

            // We are using a StandIn layer here as a generic layer since Arm NN has no LayerType::ReinterpretQuantize
            // that we could directly add.
            // We set a custom value to name parameter of the StandIn layer which then is used to add the
            // ReinterpretQuantize layer from the Support Library.
            const auto standInLayer =
                network.AddStandInLayer(desc, "EthosNBackend:ReplaceScalarAddWithReinterpretQuantization");

            SubgraphView patternSubgraph({ layer, constantLayer }, { patternSubgraphInput },
                                         { &layer->GetOutputSlot(0) });
            subgraph.SubstituteSubgraph(patternSubgraph, SubgraphView{ standInLayer });

            return true;
        }
    }
    return false;
}

bool ReplaceAddition(SubgraphView& subgraph,
                     IConnectableLayer* layer,
                     INetwork& network,
                     const EthosNConfig& config,
                     const std::vector<char>& capabilities)
{
    if (layer->GetType() == LayerType::ElementwiseBinary)
    {
        const ElementwiseBinaryDescriptor& desc =
            *PolymorphicDowncast<const ElementwiseBinaryDescriptor*>(&(layer->GetParameters()));
        if (desc.m_Operation != BinaryOperation::Add)
        {
            return false;
        }

        EthosNLayerSupport supportChecks(config, capabilities);
        const IOutputSlot* inputConnection0 = layer->GetInputSlot(0).GetConnection();
        const IOutputSlot* inputConnection1 = layer->GetInputSlot(1).GetConnection();
        auto supportedMode =
            supportChecks.GetAdditionSupportedMode(inputConnection0->GetTensorInfo(), inputConnection1->GetTensorInfo(),
                                                   layer->GetOutputSlot(0).GetTensorInfo());

        std::string failureReason;

        switch (supportedMode)
        {
            case EthosNLayerSupport::AdditionSupportedMode::None:
                return false;
                break;
            case EthosNLayerSupport::AdditionSupportedMode::Native:
                return false;
                break;
            case EthosNLayerSupport::AdditionSupportedMode::ReplaceWithDepthwise:
                return ReplaceConstantAdditionWithDepthwise(subgraph, layer, network);
                break;
            case EthosNLayerSupport::AdditionSupportedMode::ReplaceWithReinterpretQuantize:
                return ReplaceConstantAdditionWithReinterpretQuantization(subgraph, layer, network, failureReason);
                break;
            default:
                throw Exception("Found unknown AddSupportedMode value");
                break;
        }
    }
    return false;
}

void ReplaceUnsupportedLayers(SubgraphView& graph,
                              INetwork& network,
                              const EthosNConfig& config,
                              const std::vector<char>& capabilities)
{
    using ReplacementFunc =
        bool (*)(SubgraphView&, IConnectableLayer*, INetwork&, const EthosNConfig&, const std::vector<char>&);
    const ReplacementFunc replacementFuncs[] = {
        &ReplaceMultiplication,
        &ReplaceAddition,
    };

    bool madeChange;
    do
    {
        madeChange = false;
        for (auto it = graph.begin(); it != graph.end(); ++it)
        {
            for (const ReplacementFunc f : replacementFuncs)
            {
                madeChange = f(graph, *it, network, config, capabilities);
                if (madeChange)
                {
                    goto nextIteration;
                }
            }
        }
    nextIteration:;
    } while (madeChange);
}

Optional<ConstantAddToDepthwiseReplacementConfig>
    CalcConstantAddToDepthwiseReplacementConfig(const TensorInfo& inputInfo,
                                                const TensorInfo& constantInfo,
                                                const TensorInfo& outputInfo,
                                                std::string& outFailureReason)
{
    // Input and output must be quantized datatypes, as we use their quantization scale further down.
    // The constant could be any datatype, as it will get re-quantized anyway. However the requantizing function
    // in ReplaceConstantAdditionWithDepthwise supports only a limited set.
    if (!IsQuantizedType(inputInfo.GetDataType()) || !IsQuantizedType(outputInfo.GetDataType()) ||
        (constantInfo.GetDataType() != DataType::QAsymmU8 && constantInfo.GetDataType() != DataType::QAsymmS8 &&
         constantInfo.GetDataType() != DataType::QSymmS8))
    {
        outFailureReason = "Unsupported datatype";
        return {};
    }
    if (constantInfo.GetNumDimensions() != 4 || inputInfo.GetNumDimensions() != 4 ||
        constantInfo.GetShape() != TensorShape{ 1, 1, 1, inputInfo.GetShape()[3] })
    {
        outFailureReason = "Shapes not compatible";
        return {};
    }

    ConstantAddToDepthwiseReplacementConfig result;
    result.m_Desc.m_DataLayout  = DataLayout::NHWC;
    result.m_Desc.m_BiasEnabled = true;

    // The weights tensor must be set to identity (as we don't want to scale the input, only add the constant).
    // There are however many possible representations of identity weights because they are quantized.
    // The scale of the weights must be chosen such that:
    //   - the resulting quantized weight data for the identity convolution doesn't saturate
    //       (i.e. the quantized values must be between 1 and 255)
    //   - inputQuantScale * weightQuantScale needs to be less than the outputQuantScale
    //       (this is a limitation of the NPU)
    const float weightScaleUpperBound =
        std::min(outputInfo.GetQuantizationScale() / inputInfo.GetQuantizationScale(), 1.f);
    constexpr float weightScaleLowerBound = (1.f / 255.f);
    if (weightScaleUpperBound < weightScaleLowerBound)
    {
        outFailureReason = "Couldn't find valid weight scale";
        return {};
    }
    const float weightScaleTarget = (weightScaleUpperBound + weightScaleLowerBound) / 2.f;
    // The reciprical of the scale needs to be a whole number to minimize rounding error.
    const float weightScaleRecipRounded = std::round(1.f / weightScaleTarget);
    result.m_WeightsQuantizedValue      = static_cast<uint8_t>(weightScaleRecipRounded);
    const float weightScale             = 1.f / weightScaleRecipRounded;
    // The NPU requires the bias data to have a fixed quant scale, based on the input and weights.
    // Therefore the bias data needs to be rescaled to this.
    const float newConstantLayerScale = weightScale * inputInfo.GetQuantizationScale();

    unsigned int M = outputInfo.GetShape()[3] / inputInfo.GetShape()[3];
    ARMNN_ASSERT_MSG(M == 1, "Constant add only support 1x1x1xC, so M should always be 1 here");

    result.m_WeightsInfo = TensorInfo(TensorShape{ 1, 1, 1, inputInfo.GetShape()[3] * M }, DataType::QAsymmU8,
                                      weightScale, 0, true);    //1HW(I*M)

    result.m_BiasInfo = TensorInfo(constantInfo.GetShape(), DataType::Signed32, newConstantLayerScale, 0, true);

    return result;
}

}    // namespace ethosnbackend

}    // namespace armnn
