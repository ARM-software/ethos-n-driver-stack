//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNReplaceUnsupported.hpp"

#include "EthosNLayerSupport.hpp"
#include "EthosNTensorUtils.hpp"

#include <SubgraphView.hpp>
#include <backendsCommon/TensorHandle.hpp>

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
    Graph& graph, Layer* layer, const EthosNConfig&, const EthosNMappings&, const std::vector<char>&)
{
    if (layer->GetType() == LayerType::Multiplication)
    {
        InputSlot* patternSubgraphInput = &layer->GetInputSlot(0);

        Layer* inputLayer    = &patternSubgraphInput->GetConnectedOutputSlot()->GetOwningLayer();
        Layer* constantLayer = &layer->GetInputSlots()[1].GetConnectedOutputSlot()->GetOwningLayer();

        // Figure out which of the two inputs is the constant
        if (constantLayer->GetType() != LayerType::Constant)
        {
            patternSubgraphInput = &layer->GetInputSlot(1);
            std::swap(inputLayer, constantLayer);
        }

        if (constantLayer->GetType() == LayerType::Constant)
        {
            const TensorInfo& inputInfo = inputLayer->GetOutputSlot().GetTensorInfo();
            const TensorInfo& constInfo = constantLayer->GetOutputSlot().GetTensorInfo();

            // Add a Depthwise only where the constant input is a scalar that takes the form { 1, 1, 1, C }.
            // The scalar is used as weights for the convolution.
            if (constInfo.GetShape() == TensorShape({ 1, 1, 1, inputInfo.GetShape()[3] }))
            {
                Graph replacementGraph;

                DepthwiseConvolution2dDescriptor desc;
                desc.m_DataLayout = DataLayout::NHWC;

                const auto depthwiseLayer = replacementGraph.AddLayer<DepthwiseConvolution2dLayer>(
                    desc, "Replacement for Constant-Multiplication");

                TensorInfo weightInfo = constInfo;
                weightInfo.SetShape({ 1, constInfo.GetShape()[3], 1, 1 });

                const void* weightData = PolymorphicPointerDowncast<const ConstantLayer>(constantLayer)
                                             ->m_LayerOutput->GetConstTensor<void>();

                const ConstTensor weights(weightInfo, weightData);

                depthwiseLayer->m_Weight = std::make_unique<ScopedTensorHandle>(weights);

                SubgraphView patternSubgraph({ patternSubgraphInput }, { &layer->GetOutputSlot() },
                                             { layer, constantLayer });

                graph.SubstituteSubgraph(patternSubgraph, SubgraphView{ depthwiseLayer });

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
bool ReplaceScalarMultiplicationWithReinterpretQuantization(Graph& graph,
                                                            Layer* layer,
                                                            const EthosNConfig&,
                                                            const EthosNMappings&,
                                                            const std::vector<char>&,
                                                            std::string& outFailureReason)
{
    if (layer->GetType() == LayerType::Multiplication)
    {
        InputSlot* patternSubgraphInput = &layer->GetInputSlot(0);

        Layer* inputLayer    = &patternSubgraphInput->GetConnectedOutputSlot()->GetOwningLayer();
        Layer* constantLayer = &layer->GetInputSlots()[1].GetConnectedOutputSlot()->GetOwningLayer();

        // Figure out which of the two inputs is the constant
        if (constantLayer->GetType() != LayerType::Constant)
        {
            patternSubgraphInput = &layer->GetInputSlot(1);
            std::swap(inputLayer, constantLayer);
        }

        if (constantLayer->GetType() == LayerType::Constant)
        {
            const TensorInfo& constInfo  = constantLayer->GetOutputSlot().GetTensorInfo();
            const TensorInfo& outputInfo = layer->GetOutputSlot().GetTensorInfo();
            const TensorInfo& inputInfo  = inputLayer->GetOutputSlot().GetTensorInfo();

            // Add a ReinterpretQuantize only where the constant input is a scalar that takes the form { 1, 1, 1, 1 }.
            if (constInfo.GetShape() == TensorShape({ 1, 1, 1, 1 }))
            {
                auto ConvertDataToFloat = [](Layer* layer, DataType dataType) {
                    switch (dataType)
                    {
                        case DataType::QAsymmU8:
                            return static_cast<float>(PolymorphicPointerDowncast<const ConstantLayer>(layer)
                                                          ->m_LayerOutput->GetConstTensor<uint8_t>()[0]);
                        case DataType::QSymmS8:
                        case DataType::QAsymmS8:
                            return static_cast<float>(PolymorphicPointerDowncast<const ConstantLayer>(layer)
                                                          ->m_LayerOutput->GetConstTensor<int8_t>()[0]);
                        case DataType::Signed32:
                            return static_cast<float>(PolymorphicPointerDowncast<const ConstantLayer>(layer)
                                                          ->m_LayerOutput->GetConstTensor<int32_t>()[0]);
                        default:
                            throw Exception("Data type not supported");
                    }
                };

                float data;
                try
                {
                    data = ConvertDataToFloat(constantLayer, constInfo.GetDataType());
                }
                catch (const std::exception& e)
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

                Graph replacementGraph;

                StandInDescriptor desc;
                desc.m_NumInputs  = 1;
                desc.m_NumOutputs = 1;

                // We are using a StandIn layer here as a generic layer since Arm NN has no LayerType::ReinterpretQuantize
                // that we could directly add.
                // We set a custom value to name parameter of the StandIn layer which then is used to add the
                // ReinterpretQuantize layer from the Support Library.
                const auto standInLayer = replacementGraph.AddLayer<StandInLayer>(
                    desc, "EthosNBackend:ReplaceScalarMulWithReinterpretQuantization");

                SubgraphView patternSubgraph({ patternSubgraphInput }, { &layer->GetOutputSlot() },
                                             { layer, constantLayer });

                graph.SubstituteSubgraph(patternSubgraph, SubgraphView{ standInLayer });

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
bool ReplaceMultiplication(Graph& graph,
                           Layer* layer,
                           const EthosNConfig& config,
                           const EthosNMappings& mappings,
                           const std::vector<char>& capabilities)
{
    if (layer->GetType() == LayerType::Multiplication)
    {
        EthosNLayerSupport supportChecks(config, mappings, capabilities);

        EthosNLayerSupport::MultiplicationSupportedMode supportedMode = supportChecks.GetMultiplicationSupportedMode(
            layer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
            layer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(), layer->GetOutputSlot(0).GetTensorInfo());

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
                return ReplaceConstantMultiplicationWithDepthwise(graph, layer, config, mappings, capabilities);
                break;
            case EthosNLayerSupport::MultiplicationSupportedMode::ReplaceWithReinterpretQuantize:
                return ReplaceScalarMultiplicationWithReinterpretQuantization(graph, layer, config, mappings,
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
bool ReplaceConstantAdditionWithDepthwise(Graph& graph,
                                          Layer* layer,
                                          const EthosNConfig& config,
                                          const EthosNMappings& mappings,
                                          const std::vector<char>& capabilities)
{
    if (layer->GetType() != LayerType::Addition)
    {
        return false;
    }

    // Only attempt to replace the layer with a depthwise if we can't support it with a native addition
    EthosNLayerSupport supportChecks(config, mappings, capabilities);
    bool needsReplacement =
        supportChecks.GetAdditionSupportedMode(layer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                               layer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(),
                                               layer->GetOutputSlot(0).GetTensorInfo()) ==
        EthosNLayerSupport::AdditionSupportedMode::ReplaceWithDepthwise;
    if (!needsReplacement)
    {
        return false;
    }

    // Figure out which of the two inputs is the constant
    Layer* inputLayer            = nullptr;
    Layer* constantLayer         = nullptr;
    InputSlot* subgraphInputSlot = nullptr;
    Layer* inputLayer0           = &layer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
    Layer* inputLayer1           = &layer->GetInputSlot(1).GetConnectedOutputSlot()->GetOwningLayer();
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

    const TensorInfo& inputInfo  = inputLayer->GetOutputSlot().GetTensorInfo();
    const TensorInfo& constInfo  = constantLayer->GetOutputSlot().GetTensorInfo();
    const TensorInfo& outputInfo = layer->GetOutputSlot().GetTensorInfo();

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

    Graph replacementGraph;

    const auto depthwiseLayer = replacementGraph.AddLayer<DepthwiseConvolution2dLayer>(
        replacementConfig.m_Desc, "Replacement for Constant-Addition");

    // Create identity weights
    const std::vector<uint8_t> weightsData(replacementConfig.m_WeightsInfo.GetNumElements(),
                                           replacementConfig.m_WeightsQuantizedValue);
    const ConstTensor weights(replacementConfig.m_WeightsInfo, weightsData);
    depthwiseLayer->m_Weight = std::make_unique<ScopedTensorHandle>(weights);

    // Rescale the bias data
    const void* constData =
        PolymorphicPointerDowncast<const ConstantLayer>(constantLayer)->m_LayerOutput->GetConstTensor<void>();
    Optional<std::vector<int32_t>> rescaledBiasData =
        ethosntensorutils::ConvertTensorValuesToSigned32(constData, constInfo, replacementConfig.m_BiasInfo);
    if (!rescaledBiasData.has_value())
    {
        // Unsupported conversion. This should have been checked by CalcConstantAddToDepthwiseReplacementConfig()
        // so we should never hit this in practice.
        return false;
    }
    const ConstTensor rescaledBias(replacementConfig.m_BiasInfo, rescaledBiasData.value());
    depthwiseLayer->m_Bias = std::make_unique<ScopedTensorHandle>(rescaledBias);

    SubgraphView patternSubgraph({ subgraphInputSlot }, { &layer->GetOutputSlot() }, { layer, constantLayer });

    graph.SubstituteSubgraph(patternSubgraph, SubgraphView{ depthwiseLayer });

    return true;
}

void ReplaceUnsupportedLayers(Graph& graph,
                              const EthosNConfig& config,
                              const EthosNMappings& mappings,
                              const std::vector<char>& capabilities)
{
    using ReplacementFunc =
        bool (*)(Graph&, Layer*, const EthosNConfig&, const EthosNMappings&, const std::vector<char>&);
    const ReplacementFunc replacementFuncs[] = {
        &ReplaceMultiplication,
        &ReplaceConstantAdditionWithDepthwise,
    };

    bool madeChange;
    do
    {
        madeChange = false;
        for (Layer* layer : graph)
        {
            for (const ReplacementFunc f : replacementFuncs)
            {
                madeChange = f(graph, layer, config, mappings, capabilities);
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

    result.m_WeightsInfo =
        TensorInfo(TensorShape{ 1, inputInfo.GetShape()[3], 1, 1 }, DataType::QAsymmU8, weightScale, 0);

    result.m_BiasInfo = TensorInfo(constantInfo.GetShape(), DataType::Signed32, newConstantLayerScale, 0);

    return result;
}

}    // namespace ethosnbackend

}    // namespace armnn
