//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ArmnnParseRunner.hpp"

#include "ArmnnUtils.hpp"
#include "GgfParser.hpp"
#include "GlobalParameters.hpp"
#include "SystemTestsUtils.hpp"

using namespace armnn;

namespace ethosn
{
namespace system_tests
{

namespace
{

IConnectableLayer* AddConvolutionLayerToNetwork(INetwork& network,
                                                const Convolution2dDescriptor& descriptor,
                                                const ConstTensor& weights,
                                                const ConstTensor& biases,
                                                const std::string& name)
{
    ETHOSN_UNUSED(weights);
    ETHOSN_UNUSED(biases);
    return network.AddConvolution2dLayer(descriptor, name.c_str());
}

IConnectableLayer* AddConvolutionLayerToNetwork(INetwork& network,
                                                const DepthwiseConvolution2dDescriptor& descriptor,
                                                const ConstTensor& weights,
                                                const ConstTensor& biases,
                                                const std::string& name)
{
    ETHOSN_UNUSED(weights);
    ETHOSN_UNUSED(biases);
    return network.AddDepthwiseConvolution2dLayer(descriptor, name.c_str());
}

IConnectableLayer* AddConvolutionLayerToNetwork(INetwork& network,
                                                const TransposeConvolution2dDescriptor& descriptor,
                                                const ConstTensor& weights,
                                                const ConstTensor& biases,
                                                const std::string& name)
{
    return network.AddTransposeConvolution2dLayer(descriptor, weights, Optional<ConstTensor>(biases), name.c_str());
}

template <typename T>
OwnedTensor ConvertWeights(const BaseTensor& ethosnWeightData,
                           const support_library::TensorShape& ethosnWeightTensorShape,
                           bool isDepthwise)
{
    const MultiDimensionalArray<const T, 4> ethosnWeightTensor(ethosnWeightData.GetDataPtr<T>(),
                                                               ethosnWeightTensorShape);
    const std::vector<T> armnnWeightData = isDepthwise ? ConvertDepthwiseConvolutionWeightData(ethosnWeightTensor)
                                                       : ConvertConvolutionWeightData(ethosnWeightTensor);
    return MakeTensor(armnnWeightData);
}

}    // namespace

ArmnnParseRunner::ArmnnParseRunner(std::istream& ggfFile, LayerData& layerData)
    : GgfParser(ggfFile, layerData)
    , m_Network(armnn::INetwork::Create())
    , m_OutputMap()
{
    ParseNetwork();
}

void ArmnnParseRunner::AddInput(const std::string& name, ethosn::support_library::TensorShape shape)
{
    GgfParser::AddInput(name, shape);

    const ethosn::support_library::QuantizationInfo quantInfo = m_LayerData.GetInputQuantInfo(name);
    armnn::TensorShape tensorShape{ shape[0], shape[1], shape[2], shape[3] };
    armnn::TensorInfo inputTensorInfo{ tensorShape, m_LayerData.GetInputsDataType<armnn::DataType>(),
                                       quantInfo.GetScale(), quantInfo.GetZeroPoint() };

    IConnectableLayer* input = m_Network->AddInputLayer(GetInputLayerIndex(name));

    input->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    m_OutputMap[name] = &input->GetOutputSlot(0);    // Input has a single output with the same name as the layer
}

void ArmnnParseRunner::AddConstant(const std::string& name,
                                   ethosn::support_library::TensorShape shape,
                                   float constMin,
                                   float constMax)
{
    const void* constData = m_LayerData.GetConstantData(name, shape, constMin, constMax).GetByteData();

    // Create constant tensor
    const ethosn::support_library::QuantizationInfo constantQuantInfo =
        m_LayerData.GetConstantQuantInfo(name, constMin, constMax);
    const armnn::TensorInfo constTensorInfo(TensorShape({ shape[0], shape[1], shape[2], shape[3] }),
                                            m_LayerData.GetInputsDataType<armnn::DataType>(),
                                            constantQuantInfo.GetScale(), constantQuantInfo.GetZeroPoint(), true);
    const ConstTensor armnnConstantTensor{ constTensorInfo, constData };

    IConnectableLayer* constant = m_Network->AddConstantLayer(armnnConstantTensor);
    constant->GetOutputSlot(0).SetTensorInfo(constTensorInfo);

    m_OutputMap[name] = &constant->GetOutputSlot(0);    // Constant has a single output with the same name as the layer
}

template <typename ConvolutionDescriptor>
void ArmnnParseRunner::AddConvolution(const std::string& name,
                                      const std::string& inputName,
                                      const uint32_t kernelWidth,
                                      const uint32_t kernelHeight,
                                      const uint32_t strideWidth,
                                      const uint32_t strideHeight,
                                      uint32_t outputChannels,
                                      const bool biasEnable,
                                      const WeightParams& weightParams,
                                      const OutputParams& outputParams,
                                      const PaddingInfo padInfo)
{
    constexpr bool isConv2d    = std::is_same<ConvolutionDescriptor, Convolution2dDescriptor>::value;
    constexpr bool isDepthwise = std::is_same<ConvolutionDescriptor, DepthwiseConvolution2dDescriptor>::value;
    constexpr bool isTranspose = std::is_same<ConvolutionDescriptor, TransposeConvolution2dDescriptor>::value;

    IOutputSlot& input                  = *m_OutputMap[inputName];
    const TensorInfo& inputTensorInfo   = input.GetTensorInfo();
    const TensorShape& inputTensorShape = inputTensorInfo.GetShape();

    const uint32_t inputHeight   = inputTensorShape[1];
    const uint32_t inputWidth    = inputTensorShape[2];
    const uint32_t inputChannels = inputTensorShape[3];
    const float inputQuantScale  = inputTensorInfo.GetQuantizationScale();

    if (isDepthwise)
    {
        outputChannels *= inputChannels;
    }

    // Create weight tensor

    const uint32_t numSummedTerms =
        isDepthwise ? kernelWidth * kernelHeight : kernelWidth * kernelHeight * inputChannels;
    const ethosn::support_library::QuantizationInfo weightQuantInfo =
        m_LayerData.GetConvWeightQuantInfo(name, weightParams, numSummedTerms, outputChannels, isDepthwise);

    // weights layout for depthwise is [1,H,W,I*M]
    const TensorShape weightTensorShape = isDepthwise
                                              ? TensorShape{ 1, kernelHeight, kernelWidth, outputChannels }
                                              : TensorShape{ outputChannels, kernelHeight, kernelWidth, inputChannels };

    const ethosn::support_library::QuantizationScales& weightScales = weightQuantInfo.GetScales();

    TensorInfo weightTensorInfo;

    if (weightScales.size() == 1U)
    {
        weightTensorInfo = TensorInfo(weightTensorShape, m_LayerData.GetWeightDataType<armnn::DataType>(),
                                      weightQuantInfo.GetScale(), weightQuantInfo.GetZeroPoint(), true);
    }
    else
    {
        ethosn::support_library::DataType dType = m_LayerData.GetWeightDataType<ethosn::support_library::DataType>();
        if ((dType == ethosn::support_library::DataType::INT8_QUANTIZED) && (weightQuantInfo.GetZeroPoint() == 0U))
        {
            // Arm NN regular conv weights are OHWI, so the quant dim is 0
            // Arm NN depthwise conv weights are 1,H,W,I*M
            weightTensorInfo = TensorInfo(weightTensorShape, armnn::DataType::QSymmS8,
                                          std::vector<float>(std::begin(weightScales), std::end(weightScales)),
                                          isDepthwise ? 3U : 0U, true);
        }
        else
        {
            std::string errorMsg = "Error in " + std::string(__func__) +
                                   ": Weight dataType not supported or ZeroPoint nudged for per-channel quantization";
            throw std::invalid_argument(errorMsg);
        }
    }

    const ethosn::support_library::TensorShape ethosnWeightsShape =
        isDepthwise ? ethosn::support_library::TensorShape{ kernelHeight, kernelWidth, inputChannels,
                                                            outputChannels / inputChannels }
                    : ethosn::support_library::TensorShape{ weightTensorShape[1], weightTensorShape[2],
                                                            weightTensorShape[3], weightTensorShape[0] };

    const WeightTensor ethosnWeights =
        m_LayerData.GetConvWeightData(name, ethosnWeightsShape, weightQuantInfo, weightParams);

    // Convert weights from EthosN layout to Arm NN layout
    OwnedTensor armnnWeightTensorStorage;
    {
        switch (ethosnWeights->GetDataType())
        {
            case DataType::U8:
                armnnWeightTensorStorage = ConvertWeights<uint8_t>(*ethosnWeights, ethosnWeightsShape, isDepthwise);
                break;
            case DataType::S8:
                armnnWeightTensorStorage = ConvertWeights<int8_t>(*ethosnWeights, ethosnWeightsShape, isDepthwise);
                break;
            default:
                throw std::exception();
        }
    }
    const ConstTensor armnnWeightTensor{ weightTensorInfo, armnnWeightTensorStorage->GetByteData() };

    // Create bias tensor
    const ethosn::support_library::QuantizationInfo biasQuantInfo =
        m_LayerData.GetConvBiasQuantInfo(name, inputQuantScale, weightScales);

    TensorInfo biasInfo;
    const ethosn::support_library::QuantizationScales& biasScales = biasQuantInfo.GetScales();
    if (biasScales.size() == 1U)
    {
        biasInfo = TensorInfo({ outputChannels }, armnn::DataType::Signed32, biasQuantInfo.GetScale(),
                              biasQuantInfo.GetZeroPoint(), true);
    }
    else
    {
        biasInfo = TensorInfo({ outputChannels }, armnn::DataType::Signed32,
                              std::vector<float>(std::begin(biasScales), std::end(biasScales)), 0U, true);
    }

    const BaseTensor& biasData = m_LayerData.GetConvBiasData(name, outputChannels);

    const ConstTensor bias(biasInfo, biasData.GetByteData());

    // Create convolution layer
    ConvolutionDescriptor desc{};
    desc.m_BiasEnabled = biasEnable;
    desc.m_DataLayout  = armnn::DataLayout::NHWC;
    desc.m_StrideX     = strideWidth;
    desc.m_StrideY     = strideHeight;

    uint32_t outputHeight;
    uint32_t outputWidth;

    if (padInfo.alg != PaddingAlgorithm::EXPLICIT)
    {
        const bool padSame = padInfo.alg == PaddingAlgorithm::SAME;
        auto padY          = std::tie(desc.m_PadTop, desc.m_PadBottom);
        auto padX          = std::tie(desc.m_PadLeft, desc.m_PadRight);

        std::tie(outputHeight, padY) =
            CalcConvOutSizeAndPadding(inputHeight, kernelHeight, strideHeight, padSame, isTranspose);
        std::tie(outputWidth, padX) =
            CalcConvOutSizeAndPadding(inputWidth, kernelWidth, strideWidth, padSame, isTranspose);
    }
    else
    {
        desc.m_PadTop    = padInfo.info.padTop;
        desc.m_PadBottom = padInfo.info.padBottom;
        desc.m_PadLeft   = padInfo.info.padLeft;
        desc.m_PadRight  = padInfo.info.padRight;

        outputHeight =
            CalcConvOutSize(inputHeight, kernelHeight, strideHeight, desc.m_PadTop, desc.m_PadBottom, isTranspose);
        outputWidth =
            CalcConvOutSize(inputWidth, kernelWidth, strideWidth, desc.m_PadLeft, desc.m_PadRight, isTranspose);
    }

    IConnectableLayer& output = *AddConvolutionLayerToNetwork(*m_Network, desc, armnnWeightTensor, bias, name);

    const ethosn::support_library::QuantizationInfo inputQuantInfo(inputTensorInfo.GetQuantizationOffset(),
                                                                   inputTensorInfo.GetQuantizationScale());
    const ethosn::support_library::QuantizationInfo outputQuantInfo =
        m_LayerData.GetConvOutputQuantInfo(name, inputQuantInfo, weightQuantInfo, numSummedTerms, outputParams);

    const TensorShape outputTensorShape{ 1, outputHeight, outputWidth, outputChannels };
    const TensorInfo outputTensorInfo{ outputTensorShape, m_LayerData.GetInputsDataType<armnn::DataType>(),
                                       outputQuantInfo.GetScale(), outputQuantInfo.GetZeroPoint() };

    output.GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    input.Connect(output.GetInputSlot(0));

    if (isDepthwise || isConv2d)
    {
        armnn::IConnectableLayer* weightsLayer =
            m_Network->AddConstantLayer(armnnWeightTensor, (name + "Weights").c_str());
        weightsLayer->GetOutputSlot(0).SetTensorInfo(weightTensorInfo);
        weightsLayer->GetOutputSlot(0).Connect(output.GetInputSlot(1));
        if (biasEnable)
        {
            armnn::IConnectableLayer* biasLayer = m_Network->AddConstantLayer(bias, (name + "Bias").c_str());
            biasLayer->GetOutputSlot(0).SetTensorInfo(biasInfo);
            biasLayer->GetOutputSlot(0).Connect(output.GetInputSlot(2));
        }
    }

    m_OutputMap[name] = &output.GetOutputSlot(0);    // Conv has a single output with the same name as the layer
}

void ArmnnParseRunner::AddDepthwiseConvolution(const std::string& name,
                                               const std::string& inputName,
                                               uint32_t kernelWidth,
                                               uint32_t kernelHeight,
                                               uint32_t strideWidth,
                                               uint32_t strideHeight,
                                               uint32_t channelMultiplier,
                                               bool biasEnable,
                                               const WeightParams& weightParams,
                                               const OutputParams& outputParams,
                                               PaddingInfo padInfo)
{
    AddConvolution<DepthwiseConvolution2dDescriptor>(name, inputName, kernelWidth, kernelHeight, strideWidth,
                                                     strideHeight, channelMultiplier, biasEnable, weightParams,
                                                     outputParams, padInfo);
}

void ArmnnParseRunner::AddStandalonePadding(const std::string& name, const std::string& inputName, PaddingInfo padInfo)
{
    IOutputSlot* input = m_OutputMap[inputName];
    armnn::PadDescriptor padDesc;
    const armnn::TensorInfo inputTensorInfo = input->GetTensorInfo();
    const armnn::TensorShape inputShape     = inputTensorInfo.GetShape();

    // Only constant padding supported
    padDesc.m_PaddingMode = armnn::PaddingMode::Constant;

    padDesc.m_PadList = { { 0, 0 },
                          { padInfo.info.padTop, padInfo.info.padBottom },
                          { padInfo.info.padLeft, padInfo.info.padRight },
                          { 0, 0 } };

    padDesc.m_PadValue = static_cast<float>(inputTensorInfo.GetQuantizationOffset());

    armnn::IConnectableLayer* padLayer = m_Network->AddPadLayer(padDesc, name.c_str());

    uint32_t outputHeight = CalcConvOutSize(inputShape[1], 1, 1, padInfo.info.padTop, padInfo.info.padBottom, false);
    uint32_t outputWidth  = CalcConvOutSize(inputShape[2], 1, 1, padInfo.info.padLeft, padInfo.info.padRight, false);

    const TensorShape outputTensorShape{ inputShape[0], outputHeight, outputWidth, inputShape[3] };

    armnn::TensorInfo outputTensorInfo(outputTensorShape, inputTensorInfo.GetDataType(),
                                       inputTensorInfo.GetQuantizationScale(), inputTensorInfo.GetQuantizationOffset());

    padLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
    input->Connect(padLayer->GetInputSlot(0));

    // requantize has a single output with the same name as the layer
    m_OutputMap[name] = &padLayer->GetOutputSlot(0);
}

void ArmnnParseRunner::AddConvolution(const std::string& name,
                                      const std::string& inputName,
                                      uint32_t kernelWidth,
                                      uint32_t kernelHeight,
                                      uint32_t strideWidth,
                                      uint32_t strideHeight,
                                      uint32_t numOutput,
                                      bool biasEnable,
                                      const WeightParams& weightParams,
                                      const OutputParams& outputParams,
                                      PaddingInfo padInfo)
{
    AddConvolution<Convolution2dDescriptor>(name, inputName, kernelWidth, kernelHeight, strideWidth, strideHeight,
                                            numOutput, biasEnable, weightParams, outputParams, padInfo);
}

void ArmnnParseRunner::AddTransposeConvolution(const std::string& name,
                                               const std::string& inputName,
                                               uint32_t kernelWidth,
                                               uint32_t kernelHeight,
                                               uint32_t strideWidth,
                                               uint32_t strideHeight,
                                               uint32_t numOutput,
                                               bool biasEnable,
                                               const WeightParams& weightParams,
                                               const OutputParams& outputParams,
                                               PaddingInfo padInfo)
{
    AddConvolution<TransposeConvolution2dDescriptor>(name, inputName, kernelWidth, kernelHeight, strideWidth,
                                                     strideHeight, numOutput, biasEnable, weightParams, outputParams,
                                                     padInfo);
}

void ArmnnParseRunner::AddMeanXy(const std::string& name, const std::string& inputName)
{
    IOutputSlot* input = m_OutputMap[inputName];
    MeanDescriptor desc;
    const armnn::TensorInfo inputTensorInfo   = input->GetTensorInfo();
    const armnn::TensorShape inputTensorShape = inputTensorInfo.GetShape();

    // In our Ggf parser, we support keep_dims 1 only
    desc.m_KeepDims = true;

    // In our Ggf parser, we support dimension 2_3 only (ie mean across width and height)
    desc.m_Axis = { 1, 2 };

    armnn::TensorShape outputTensorShape = CalcTensorShapeForMeanXy(inputTensorShape);
    armnn::TensorInfo outputTensorInfo{ outputTensorShape, inputTensorInfo.GetDataType(),
                                        inputTensorInfo.GetQuantizationScale(),
                                        inputTensorInfo.GetQuantizationOffset() };

    IConnectableLayer* mean = m_Network->AddMeanLayer(desc, name.c_str());

    input->Connect(mean->GetInputSlot(0));
    m_OutputMap[name] = &mean->GetOutputSlot(0);    // Mean has a single output with the same name as the layer
    mean->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
}

void ArmnnParseRunner::AddFullyConnected(const std::string& name,
                                         const std::string& inputName,
                                         uint32_t numOutput,
                                         const WeightParams& weightParams,
                                         const OutputParams& outputParams)

{
    IOutputSlot* input                        = m_OutputMap[inputName];
    const armnn::TensorInfo inputTensorInfo   = input->GetTensorInfo();
    const armnn::TensorShape inputTensorShape = inputTensorInfo.GetShape();

    // Create weight tensor
    uint32_t numInputs = inputTensorShape.GetNumElements();
    ethosn::support_library::QuantizationInfo weightQuantInfo =
        m_LayerData.GetFCWeightQuantInfo(name, weightParams, numInputs);
    armnn::TensorInfo weightsInfo(armnn::TensorShape({ numInputs, numOutput }),
                                  m_LayerData.GetWeightDataType<armnn::DataType>(), weightQuantInfo.GetScale(),
                                  weightQuantInfo.GetZeroPoint(), true);
    ethosn::support_library::TensorShape weightTensorShape{ numInputs, numOutput };
    const WeightTensor weightsTensor =
        m_LayerData.GetFullyConnectedWeightData(name, weightTensorShape, weightQuantInfo, weightParams);
    armnn::ConstTensor weights(weightsInfo, weightsTensor->GetByteData());
    IConnectableLayer* const weightsLayer = m_Network->AddConstantLayer(weights, ("weights for " + name).c_str());

    // Create bias tensor
    ethosn::support_library::QuantizationInfo biasQuantInfo = m_LayerData.GetFCBiasQuantInfo(
        name, inputTensorInfo.GetQuantizationScale(), weightsInfo.GetQuantizationScale());
    uint32_t biasDims[1] = { numOutput };
    armnn::TensorInfo biasInfo(armnn::TensorShape(1, biasDims), armnn::DataType::Signed32, biasQuantInfo.GetScale(),
                               biasQuantInfo.GetZeroPoint(), true);
    const void* biasData = m_LayerData.GetFullyConnectedBiasData(name, numOutput).GetByteData();
    ConstTensor bias(biasInfo, biasData);
    IConnectableLayer* const biasLayer = m_Network->AddConstantLayer(bias, ("bias for " + name).c_str());

    // Create Fully Connected layer
    armnn::FullyConnectedDescriptor fullyConnectedDesc;
    fullyConnectedDesc.m_BiasEnabled = true;

    IConnectableLayer* fullyConnected = m_Network->AddFullyConnectedLayer(fullyConnectedDesc, name.c_str());

    weightsLayer->GetOutputSlot(0).Connect(fullyConnected->GetInputSlot(1));
    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightsInfo);

    biasLayer->GetOutputSlot(0).Connect(fullyConnected->GetInputSlot(2));
    biasLayer->GetOutputSlot(0).SetTensorInfo(biasInfo);

    ethosn::support_library::QuantizationInfo outputQuantInfo = m_LayerData.GetFCOutputQuantInfo(
        name,
        ethosn::support_library::QuantizationInfo(inputTensorInfo.GetQuantizationOffset(),
                                                  inputTensorInfo.GetQuantizationScale()),
        weightQuantInfo, numInputs, outputParams);
    armnn::TensorInfo outputTensorInfo(armnn::TensorShape({ 1, numOutput }),
                                       m_LayerData.GetInputsDataType<armnn::DataType>(), outputQuantInfo.GetScale(),
                                       outputQuantInfo.GetZeroPoint());
    fullyConnected->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    input->Connect(fullyConnected->GetInputSlot(0));
    m_OutputMap[name] = &fullyConnected->GetOutputSlot(0);    // FC has a single output with the same name as the layer
}

void ArmnnParseRunner::AddActivation(const std::string& name,
                                     const std::string& inputName,
                                     const ActivationDescriptor& desc,
                                     armnn::TensorInfo outputTensorInfo)
{
    IOutputSlot* input            = m_OutputMap[inputName];
    IConnectableLayer* activation = m_Network->AddActivationLayer(desc, name.c_str());

    activation->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    input->Connect(activation->GetInputSlot(0));
    m_OutputMap[inputName] = &activation->GetOutputSlot(0);    // Activations "modify" their input layer
    m_OutputMap[name] =
        &activation->GetOutputSlot(0);    // Activations have a single output with the same name as the layer
}

void ArmnnParseRunner::AddRelu(const std::string& name, const std::string& inputName)
{
    IOutputSlot* input                               = m_OutputMap[inputName];
    const ethosn::support_library::ReluInfo reluInfo = m_LayerData.GetReluInfo(name);
    const armnn::TensorInfo inputTensorInfo          = input->GetTensorInfo();
    const float prevQuantScale                       = inputTensorInfo.GetQuantizationScale();

    ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::BoundedReLu;
    desc.m_A        = armnn::Dequantize(reluInfo.m_UpperBound, prevQuantScale, inputTensorInfo.GetQuantizationOffset());
    desc.m_B        = armnn::Dequantize(reluInfo.m_LowerBound, prevQuantScale, inputTensorInfo.GetQuantizationOffset());

    AddActivation(name, inputName, desc, input->GetTensorInfo());
}

void ArmnnParseRunner::AddLeakyRelu(const std::string& name, const std::string& inputName, const float alpha)
{
    IOutputSlot* input = m_OutputMap[inputName];

    ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::LeakyReLu;
    desc.m_A        = alpha;

    const armnn::TensorInfo inputTensorInfo = input->GetTensorInfo();
    auto prevQuantInfo = ethosn::support_library::QuantizationInfo(inputTensorInfo.GetQuantizationOffset(),
                                                                   inputTensorInfo.GetQuantizationScale());

    ethosn::support_library::QuantizationInfo outputQuantInfo =
        m_LayerData.GetLeakyReluOutputQuantInfo(name, prevQuantInfo, alpha);

    armnn::TensorInfo outputTensorInfo = inputTensorInfo;
    outputTensorInfo.SetQuantizationOffset(outputQuantInfo.GetZeroPoint());
    outputTensorInfo.SetQuantizationScale(outputQuantInfo.GetScale());

    AddActivation(name, inputName, desc, outputTensorInfo);
}

void ArmnnParseRunner::AddRequantize(const std::string& name,
                                     const std::string& inputName,
                                     ethosn::support_library::RequantizeInfo& requantizeInfo)
{
    IOutputSlot* input                      = m_OutputMap[inputName];
    const armnn::TensorInfo inputTensorInfo = input->GetTensorInfo();
    const armnn::TensorShape inputShape     = inputTensorInfo.GetShape();

    armnn::IConnectableLayer* requantize = m_Network->AddQuantizeLayer(name.c_str());

    armnn::TensorInfo outputTensorInfo(inputShape, inputTensorInfo.GetDataType(),
                                       requantizeInfo.m_OutputQuantizationInfo.GetScale(),
                                       requantizeInfo.m_OutputQuantizationInfo.GetZeroPoint());

    requantize->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
    input->Connect(requantize->GetInputSlot(0));

    // requantize has a single output with the same name as the layer
    m_OutputMap[name] = &requantize->GetOutputSlot(0);
}

void ArmnnParseRunner::AddSigmoid(const std::string& name, const std::string& inputName)
{
    IOutputSlot* input          = m_OutputMap[inputName];
    const TensorInfo& inputInfo = input->GetTensorInfo();

    const int32_t zeroPoint = (inputInfo.GetDataType() == armnn::DataType::QAsymmS8) ? -128 : 0;
    const TensorInfo outputInfo(inputInfo.GetShape(), inputInfo.GetDataType(), 1. / 256, zeroPoint);

    ActivationDescriptor desc;
    desc.m_Function = ActivationFunction::Sigmoid;

    AddActivation(name, inputName, desc, outputInfo);
}

void ArmnnParseRunner::AddTanh(const std::string& name, const std::string& inputName)
{
    IOutputSlot* input          = m_OutputMap[inputName];
    const TensorInfo& inputInfo = input->GetTensorInfo();

    const int32_t zeroPoint = (inputInfo.GetDataType() == armnn::DataType::QAsymmS8) ? 0 : 128;
    const TensorInfo outputInfo(inputInfo.GetShape(), inputInfo.GetDataType(), 1. / 128, zeroPoint);

    ActivationDescriptor desc;
    desc.m_Function = ActivationFunction::TanH;
    desc.m_A        = 1.0f;
    desc.m_B        = 1.0f;

    AddActivation(name, inputName, desc, outputInfo);
}

void ArmnnParseRunner::AddReshape(const std::string& name,
                                  const std::string& inputName,
                                  ethosn::support_library::TensorShape shape)
{
    IOutputSlot* input                      = m_OutputMap[inputName];
    const armnn::TensorInfo inputTensorInfo = input->GetTensorInfo();

    ReshapeDescriptor desc;
    desc.m_TargetShape = { shape[0], shape[1], shape[2], shape[3] };

    IConnectableLayer* reshape = m_Network->AddReshapeLayer(desc, name.c_str());

    ethosn::support_library::QuantizationInfo reshapeQuantInfo(inputTensorInfo.GetQuantizationOffset(),
                                                               inputTensorInfo.GetQuantizationScale());
    armnn::TensorInfo tensorinfo(armnn::TensorShape({ shape[0], shape[1], shape[2], shape[3] }),
                                 m_LayerData.GetInputsDataType<armnn::DataType>(), reshapeQuantInfo.GetScale(),
                                 reshapeQuantInfo.GetZeroPoint());
    reshape->GetOutputSlot(0).SetTensorInfo(tensorinfo);

    input->Connect(reshape->GetInputSlot(0));
    m_OutputMap[name] = &reshape->GetOutputSlot(0);    // Reshape has a single output with the same name as the layer
}

void ArmnnParseRunner::AddConcatenation(const std::string& name,
                                        const std::vector<std::string>& inputNames,
                                        uint32_t axis)
{
    uint32_t mergeDimPosition = 0;
    uint32_t numTensorDims    = 4;
    uint32_t numInputs        = static_cast<uint32_t>(inputNames.size());
    std::vector<ethosn::support_library::QuantizationInfo> inputQuantInfos;

    std::vector<IOutputSlot*> inputs;
    for (auto it = inputNames.begin(); it != inputNames.end(); ++it)
    {
        IOutputSlot* input = m_OutputMap[*it];
        inputs.push_back(input);
        inputQuantInfos.push_back(
            { input->GetTensorInfo().GetQuantizationOffset(), input->GetTensorInfo().GetQuantizationScale() });
    }

    OriginsDescriptor descriptor(numInputs, numTensorDims);
    descriptor.SetConcatAxis(axis);

    for (uint32_t i = 0; i < numInputs; ++i)
    {
        descriptor.SetViewOriginCoord(i, axis, mergeDimPosition);
        mergeDimPosition += inputs[i]->GetTensorInfo().GetShape()[axis];
    }

    IConnectableLayer* concatenation = m_Network->AddConcatLayer(descriptor, name.c_str());

    for (uint32_t i = 0; i < numInputs; ++i)
    {
        inputs[i]->Connect(concatenation->GetInputSlot(i));
    }

    armnn::TensorShape tensorShape = inputs[0]->GetTensorInfo().GetShape();
    tensorShape[axis]              = mergeDimPosition;
    armnn::TensorInfo tensorInfo   = inputs[0]->GetTensorInfo();
    tensorInfo.SetShape(tensorShape);
    ethosn::support_library::QuantizationInfo outputQuantInfo =
        m_LayerData.GetConcatOutputQuantInfo(name, inputQuantInfos);
    tensorInfo.SetQuantizationScale(outputQuantInfo.GetScale());
    tensorInfo.SetQuantizationOffset(outputQuantInfo.GetZeroPoint());
    concatenation->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    m_OutputMap[name] =
        &concatenation->GetOutputSlot(0);    // Concat has a single output with the same name as the layer
}

void ArmnnParseRunner::AddSplit(const std::string& name,
                                const std::string& inputName,
                                uint32_t axis,
                                std::vector<uint32_t> sizes)
{
    IOutputSlot* input                        = m_OutputMap[inputName];
    const armnn::TensorInfo inputTensorInfo   = input->GetTensorInfo();
    const armnn::TensorShape inputTensorShape = inputTensorInfo.GetShape();
    const size_t numOutputs                   = sizes.size();

    // Build Arm NN descriptor the splitter
    ViewsDescriptor desc(static_cast<uint32_t>(numOutputs), 4);
    uint32_t runningTotal = 0;
    for (uint32_t i = 0; i < numOutputs; ++i)
    {
        for (uint32_t d = 0; d < 4; ++d)
        {
            if (d == axis)
            {
                desc.SetViewOriginCoord(i, d, runningTotal);
                runningTotal += sizes[i];
                desc.SetViewSize(i, d, sizes[i]);
            }
            else
            {
                desc.SetViewOriginCoord(i, d, 0);
                desc.SetViewSize(i, d, inputTensorShape[d]);
            }
        }
    }

    // Add the layer to the Network
    IConnectableLayer* split = m_Network->AddSplitterLayer(desc, name.c_str());

    // Set output tensor infos and store output slots
    for (uint32_t i = 0; i < numOutputs; ++i)
    {
        TensorShape shape = inputTensorShape;
        shape[axis]       = sizes[i];
        split->GetOutputSlot(i).SetTensorInfo(TensorInfo(shape, inputTensorInfo.GetDataType(),
                                                         inputTensorInfo.GetQuantizationScale(),
                                                         inputTensorInfo.GetQuantizationOffset()));

        m_OutputMap[name + "_" + std::to_string(i)] = &split->GetOutputSlot(i);
    }

    // Connect to our input
    input->Connect(split->GetInputSlot(0));
}

void ArmnnParseRunner::AddAddition(const std::string& name,
                                   const std::string& firstInputName,
                                   const std::string& secondInputName)
{
    IOutputSlot* inputOne = m_OutputMap[firstInputName];
    IOutputSlot* inputTwo = m_OutputMap[secondInputName];
    IConnectableLayer* addition =
        m_Network->AddElementwiseBinaryLayer(ElementwiseBinaryDescriptor(BinaryOperation::Add), name.c_str());

    inputOne->Connect(addition->GetInputSlot(0));
    inputTwo->Connect(addition->GetInputSlot(1));

    // The tensor info is duplicated from the first layer except for the quant info and the shape
    armnn::TensorInfo outputTensorInfo = inputOne->GetTensorInfo();

    const ethosn::support_library::QuantizationInfo inputOneQuantInfo = {
        inputOne->GetTensorInfo().GetQuantizationOffset(), inputOne->GetTensorInfo().GetQuantizationScale()
    };
    const ethosn::support_library::QuantizationInfo inputTwoQuantInfo = {
        inputTwo->GetTensorInfo().GetQuantizationOffset(), inputTwo->GetTensorInfo().GetQuantizationScale()
    };
    const std::vector<ethosn::support_library::QuantizationInfo> addQuantInfos{ inputOneQuantInfo, inputTwoQuantInfo };
    const ethosn::support_library::QuantizationInfo addQuantInfo =
        m_LayerData.GetAdditionQuantInfo(name, addQuantInfos);

    outputTensorInfo.SetQuantizationScale(addQuantInfo.GetScale());
    outputTensorInfo.SetQuantizationOffset(addQuantInfo.GetZeroPoint());
    outputTensorInfo.SetShape(
        TensorShape({ std::max(inputOne->GetTensorInfo().GetShape()[0], inputTwo->GetTensorInfo().GetShape()[0]),
                      std::max(inputOne->GetTensorInfo().GetShape()[1], inputTwo->GetTensorInfo().GetShape()[1]),
                      std::max(inputOne->GetTensorInfo().GetShape()[2], inputTwo->GetTensorInfo().GetShape()[2]),
                      std::max(inputOne->GetTensorInfo().GetShape()[3], inputTwo->GetTensorInfo().GetShape()[3]) }));

    addition->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    m_OutputMap[name] = &addition->GetOutputSlot(0);    // Addition has a single output with the same name as the layer
}

void ArmnnParseRunner::AddMultiplication(const std::string& name,
                                         const std::string& firstInputName,
                                         const std::string& secondInputName)
{
    IOutputSlot* inputOne = m_OutputMap[firstInputName];
    IOutputSlot* inputTwo = m_OutputMap[secondInputName];
    IConnectableLayer* multiplication =
        m_Network->AddElementwiseBinaryLayer(ElementwiseBinaryDescriptor(BinaryOperation::Mul), name.c_str());

    inputOne->Connect(multiplication->GetInputSlot(0));
    inputTwo->Connect(multiplication->GetInputSlot(1));

    // The tensor info is duplicated from the first layer except for the quant info and the shape
    armnn::TensorInfo outputTensorInfo = inputOne->GetTensorInfo();

    const ethosn::support_library::QuantizationInfo inputOneQuantInfo = {
        inputOne->GetTensorInfo().GetQuantizationOffset(), inputOne->GetTensorInfo().GetQuantizationScale()
    };
    const ethosn::support_library::QuantizationInfo inputTwoQuantInfo = {
        inputTwo->GetTensorInfo().GetQuantizationOffset(), inputTwo->GetTensorInfo().GetQuantizationScale()
    };

    const std::vector<ethosn::support_library::QuantizationInfo> mulQuantInfos{ inputOneQuantInfo, inputTwoQuantInfo };
    const ethosn::support_library::QuantizationInfo mulQuantInfo =
        m_LayerData.GetMultiplicationQuantInfo(name, mulQuantInfos);

    outputTensorInfo.SetQuantizationScale(mulQuantInfo.GetScale());
    outputTensorInfo.SetQuantizationOffset(mulQuantInfo.GetZeroPoint());
    outputTensorInfo.SetShape(
        TensorShape({ std::max(inputOne->GetTensorInfo().GetShape()[0], inputTwo->GetTensorInfo().GetShape()[0]),
                      std::max(inputOne->GetTensorInfo().GetShape()[1], inputTwo->GetTensorInfo().GetShape()[1]),
                      std::max(inputOne->GetTensorInfo().GetShape()[2], inputTwo->GetTensorInfo().GetShape()[2]),
                      std::max(inputOne->GetTensorInfo().GetShape()[3], inputTwo->GetTensorInfo().GetShape()[3]) }));

    multiplication->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    m_OutputMap[name] =
        &multiplication->GetOutputSlot(0);    // Multiplication has a single output with the same name as the layer
}

void ArmnnParseRunner::AddPooling(const std::string& name,
                                  const std::string& inputName,
                                  ethosn::support_library::PoolingInfo poolInfo,
                                  PaddingAlgorithm paddingAlgorithm)
{
    IOutputSlot* input                        = m_OutputMap[inputName];
    const armnn::TensorInfo inputTensorInfo   = input->GetTensorInfo();
    const armnn::TensorShape inputTensorShape = inputTensorInfo.GetShape();

    uint32_t prevHeight   = inputTensorShape[1];
    uint32_t prevWidth    = inputTensorShape[2];
    uint32_t prevChannels = inputTensorShape[3];

    PoolingAlgorithm poolingAlgorithm = poolInfo.m_PoolingType == ethosn::support_library::PoolingType::AVG
                                            ? PoolingAlgorithm::Average
                                            : PoolingAlgorithm::Max;

    uint32_t outHeight;
    uint32_t outWidth;

    Pooling2dDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;
    desc.m_PoolType   = poolingAlgorithm;
    desc.m_PoolWidth  = poolInfo.m_PoolingSizeX;
    desc.m_PoolHeight = poolInfo.m_PoolingSizeY;
    desc.m_StrideX    = poolInfo.m_PoolingStrideX;
    desc.m_StrideY    = poolInfo.m_PoolingStrideY;

    const bool padSame = paddingAlgorithm == PaddingAlgorithm::SAME;
    auto padY          = std::tie(desc.m_PadTop, desc.m_PadBottom);
    auto padX          = std::tie(desc.m_PadLeft, desc.m_PadRight);

    std::tie(outHeight, padY) =
        CalcConvOutSizeAndPadding(prevHeight, poolInfo.m_PoolingSizeY, poolInfo.m_PoolingStrideY, padSame);
    std::tie(outWidth, padX) =
        CalcConvOutSizeAndPadding(prevWidth, poolInfo.m_PoolingSizeX, poolInfo.m_PoolingStrideX, padSame);

    IConnectableLayer* pool = m_Network->AddPooling2dLayer(desc, name.c_str());

    ethosn::support_library::QuantizationInfo poolQuantInfo(inputTensorInfo.GetQuantizationOffset(),
                                                            inputTensorInfo.GetQuantizationScale());
    armnn::TensorShape poolTensorShape{ 1, outHeight, outWidth, prevChannels };
    armnn::TensorInfo poolTensorInfo{ poolTensorShape, m_LayerData.GetInputsDataType<armnn::DataType>(),
                                      poolQuantInfo.GetScale(), poolQuantInfo.GetZeroPoint() };
    pool->GetOutputSlot(0).SetTensorInfo(poolTensorInfo);

    input->Connect(pool->GetInputSlot(0));
    m_OutputMap[name] = &pool->GetOutputSlot(0);    // Pool has a single output with the same name as the layer
}

void ArmnnParseRunner::AddDepthToSpace(const std::string& name, const std::string& inputName, uint32_t blockSize)
{
    IOutputSlot* input                      = m_OutputMap[inputName];
    const armnn::TensorInfo inputTensorInfo = input->GetTensorInfo();

    DepthToSpaceDescriptor desc;
    desc.m_BlockSize                = blockSize;
    desc.m_DataLayout               = DataLayout::NHWC;
    IConnectableLayer* depthToSpace = m_Network->AddDepthToSpaceLayer(desc, name.c_str());

    armnn::TensorInfo outputTensorInfo(
        armnn::TensorShape({ inputTensorInfo.GetShape()[0], inputTensorInfo.GetShape()[1] * blockSize,
                             inputTensorInfo.GetShape()[2] * blockSize,
                             inputTensorInfo.GetShape()[3] / (blockSize * blockSize) }),
        m_LayerData.GetInputsDataType<armnn::DataType>(), inputTensorInfo.GetQuantizationScale(),
        inputTensorInfo.GetQuantizationOffset());
    depthToSpace->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    input->Connect(depthToSpace->GetInputSlot(0));
    m_OutputMap[name] =
        &depthToSpace->GetOutputSlot(0);    // DepthToSpace has a single output with the same name as the layer
}

void ArmnnParseRunner::AddSpaceToDepth(const std::string& name, const std::string& inputName, uint32_t blockSize)
{
    IOutputSlot* input                      = m_OutputMap[inputName];
    const armnn::TensorInfo inputTensorInfo = input->GetTensorInfo();

    SpaceToDepthDescriptor desc;
    desc.m_BlockSize                = blockSize;
    desc.m_DataLayout               = DataLayout::NHWC;
    IConnectableLayer* spaceToDepth = m_Network->AddSpaceToDepthLayer(desc, name.c_str());

    // Note the output data type follows the input's.
    armnn::TensorInfo outputTensorInfo(
        armnn::TensorShape({ inputTensorInfo.GetShape()[0], inputTensorInfo.GetShape()[1] / blockSize,
                             inputTensorInfo.GetShape()[2] / blockSize,
                             inputTensorInfo.GetShape()[3] * blockSize * blockSize }),
        m_LayerData.GetInputsDataType<armnn::DataType>(), inputTensorInfo.GetQuantizationScale(),
        inputTensorInfo.GetQuantizationOffset());
    spaceToDepth->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    input->Connect(spaceToDepth->GetInputSlot(0));
    m_OutputMap[name] =
        &spaceToDepth->GetOutputSlot(0);    // SpaceToDepth has a single output with the same name as the layer
}

void ArmnnParseRunner::AddOutput(const std::string& name, const std::string& inputName)
{
    GgfParser::AddOutput(name, inputName);

    IOutputSlot* input                    = m_OutputMap[inputName];
    armnn::IConnectableLayer* outputLayer = m_Network->AddOutputLayer(GetOutputLayerIndex(name));
    input->Connect(outputLayer->GetInputSlot(0));
}

void ArmnnParseRunner::AddTranspose(const std::string& name,
                                    const std::string& inputName,
                                    const std::array<uint32_t, 4>& permutation)
{
    IOutputSlot* input                      = m_OutputMap[inputName];
    const armnn::TensorInfo inputTensorInfo = input->GetTensorInfo();
    const armnn::TensorShape inputShape     = inputTensorInfo.GetShape();

    // Transpose input tensor shape
    armnn::TensorShape outputShape(4);
    outputShape[0] = inputShape[permutation[0]];
    outputShape[1] = inputShape[permutation[1]];
    outputShape[2] = inputShape[permutation[2]];
    outputShape[3] = inputShape[permutation[3]];

    armnn::TransposeDescriptor descriptor({ permutation[0], permutation[1], permutation[2], permutation[3] });
    armnn::IConnectableLayer* transpose = m_Network->AddTransposeLayer(descriptor);

    armnn::TensorInfo outputTensorInfo(outputShape, inputTensorInfo.GetDataType(),
                                       inputTensorInfo.GetQuantizationScale(), inputTensorInfo.GetQuantizationOffset());

    transpose->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
    input->Connect(transpose->GetInputSlot(0));

    // Transpose has a single output with the same name as the layer
    m_OutputMap[name] = &transpose->GetOutputSlot(0);
}

void ArmnnParseRunner::AddResize(const std::string& name, const std::string& inputName, const ResizeParams& params)
{
    IOutputSlot* input                      = m_OutputMap[inputName];
    const armnn::TensorInfo inputTensorInfo = input->GetTensorInfo();
    const armnn::TensorShape inputShape     = inputTensorInfo.GetShape();

    armnn::ResizeMethod resizeMethod;

    switch (params.m_Algo)
    {
        case (ethosn::support_library::ResizeAlgorithm::BILINEAR):
        {
            resizeMethod = ResizeMethod::Bilinear;
            break;
        }
        case (ethosn::support_library::ResizeAlgorithm::NEAREST_NEIGHBOUR):
        {
            resizeMethod = ResizeMethod::NearestNeighbor;
            break;
        }
        default:
        {
            std::string errorMessage = "Error: Resize Algorithm not supported";
            throw std::invalid_argument(errorMessage);
        }
    }

    armnn::ResizeDescriptor desc;
    desc.m_Method       = resizeMethod;
    desc.m_TargetHeight = CalcUpsampleOutputSize(params.m_Height, inputShape[1]);
    desc.m_TargetWidth  = CalcUpsampleOutputSize(params.m_Width, inputShape[2]);
    desc.m_DataLayout   = armnn::DataLayout::NHWC;

    if (params.m_Height.m_Mode != params.m_Width.m_Mode)
    {
        std::string errorMessage = "Error: Resize width and height must be both even or both odd";
        throw std::invalid_argument(errorMessage);
    }

    desc.m_AlignCorners = (params.m_Height.m_Mode == ResizeMode::DROP) &&
                          (params.m_Algo == ethosn::support_library::ResizeAlgorithm::BILINEAR);

    armnn::IConnectableLayer* resize = m_Network->AddResizeLayer(desc);

    armnn::TensorShape outputShape(4);
    outputShape[0] = inputShape[0];
    outputShape[1] = desc.m_TargetHeight;
    outputShape[2] = desc.m_TargetWidth;
    outputShape[3] = inputShape[3];

    armnn::TensorInfo outputTensorInfo(outputShape, inputTensorInfo.GetDataType(),
                                       inputTensorInfo.GetQuantizationScale(), inputTensorInfo.GetQuantizationOffset());

    resize->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
    input->Connect(resize->GetInputSlot(0));

    m_OutputMap[name] = &resize->GetOutputSlot(0);
}

InferenceOutputs ArmnnParseRunner::RunNetwork(const std::vector<armnn::BackendId>& backends)
{
    // Prepare inputs
    std::vector<armnn::LayerBindingId> inputBindings;
    std::vector<std::string> inputNames = GetInputLayerNames();
    const size_t numInputLayers         = inputNames.size();
    InferenceInputs inputData(numInputLayers);
    for (size_t i = 0; i < numInputLayers; ++i)
    {
        LayerBindingId armnnLayerBindingId = static_cast<LayerBindingId>(i);
        g_Logger.Debug("ArmnnParseRunner::%s input[%zu] name=%s", __func__, i, inputNames[i].c_str());

        ethosn::support_library::TensorShape ethosnInputShape = GetInputLayerShapes()[i];

        InputTensor rawInputData = m_LayerData.GetInputData(inputNames[i], ethosnInputShape);
        if (m_LayerData.GetInputTensorFormat() == ethosn::support_library::DataFormat::NHWCB)
        {
            inputData[i] =
                ConvertNhwcbToNhwc(*rawInputData, ethosnInputShape[1], ethosnInputShape[2], ethosnInputShape[3]);
        }
        else
        {
            inputData[i] = std::move(rawInputData);
        }

        inputBindings.push_back(armnnLayerBindingId);
    }

    // Prepare outputs
    std::vector<armnn::LayerBindingId> outputBindings;
    std::vector<std::string> outputLayerNames = GetOutputLayerNames();
    const size_t numOutputLayers              = outputLayerNames.size();
    for (size_t i = 0; i < numOutputLayers; ++i)
    {
        LayerBindingId armnnLayerBindingId = static_cast<LayerBindingId>(i);
        outputBindings.push_back(armnnLayerBindingId);
    }

    std::vector<armnn::BackendOptions> backendOptions = g_ArmnnBackendOptions;

    // The reference backend doesn't support importing (protected or non-protected)
    bool containsCpuRef    = false;
    bool containsEthosNAcc = false;
    for (auto it : backends)
    {
        if (it == "CpuRef")
        {
            containsCpuRef = true;
        }
        else if (it == "EthosNAcc")
        {
            containsEthosNAcc = true;
        }
    }
    bool useDmaBuf = (!containsCpuRef && (g_UseDmaBuf || g_RunProtectedInference));

    // g_RunProtectedInference overrides g_DmaBuf
    const char* dmaBufHeap =
        useDmaBuf ? (g_RunProtectedInference ? g_DmaBufProtected.c_str() : g_DmaBufHeap.c_str()) : nullptr;

    // Even if multiple runs were requested, ignore this for (e.g.) CpuRef because we are generally only interested
    // in doing multiple inferences on the NPU.
    size_t numInferences = containsEthosNAcc ? g_NumberRuns : 1;

    return ArmnnRunNetwork(m_Network.get(), backends, inputBindings, outputBindings, inputData, backendOptions,
                           dmaBufHeap, g_RunProtectedInference && useDmaBuf, numInferences);
}

}    // namespace system_tests

}    // namespace ethosn
