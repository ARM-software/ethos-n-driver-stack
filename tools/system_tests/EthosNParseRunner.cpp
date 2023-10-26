//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNParseRunner.hpp"

#include "SystemTestsUtils.hpp"

#include <ethosn_driver_library/Network.hpp>
#include <ethosn_driver_library/ProcMemAllocator.hpp>
#include <ethosn_utils/Strings.hpp>
#include <ethosn_utils/VectorStream.hpp>
#if defined(__unix__)
#include <poll.h>
#include <unistd.h>
#elif defined(_MSC_VER)
#define O_CLOEXEC 0
#endif

using namespace ethosn::support_library;

namespace ethosn
{
namespace system_tests
{

EthosNParseRunner::CreationOptions EthosNParseRunner::CreationOptions::CreateWithGlobalOptions(std::istream& ggfFile,
                                                                                               LayerData& layerData)
{
    CreationOptions result(ggfFile, layerData);
    result.m_StrictPrecision = g_StrictPrecision;
    if (g_Debug.find("dump-support-library-debug-files=None") != std::string::npos)
    {
        result.m_DumpDebugFiles = CompilationOptions::DebugLevel::None;
    }
    else if (g_Debug.find("dump-support-library-debug-files=Medium") != std::string::npos)
    {
        result.m_DumpDebugFiles = CompilationOptions::DebugLevel::Medium;
    }
    else if (g_Debug.find("dump-support-library-debug-files=High") != std::string::npos)
    {
        result.m_DumpDebugFiles = CompilationOptions::DebugLevel::High;
    }
    result.m_NumberRuns   = g_NumberRuns;
    result.m_RunBatchSize = g_RunBatchSize;
    return result;
}

EthosNParseRunner::EthosNParseRunner(const EthosNParseRunner::CreationOptions& creationOptions)
    : GgfParser(creationOptions.m_GgfFile, creationOptions.m_LayerData)
    , m_Network()
    , m_OutputToOperand()
    , m_OperandToOperationIdAndIndex()
    , m_OutputNameToOperationIdAndIndex()
    , m_OutputLayerToOperand()
    , m_Options()
    , m_EstimationOptions(creationOptions.m_EstimationOptions)
    , m_NumberRuns(creationOptions.m_NumberRuns)
    , m_RunBatchSize(creationOptions.m_RunBatchSize)
{
    if (!ethosn::driver_library::VerifyKernel())
    {
        throw std::runtime_error("Kernel version is not supported");
    }

    std::vector<char> fwAndHwCapabilities(ethosn::driver_library::GetFirmwareAndHardwareCapabilities());
    m_Network = creationOptions.m_EstimationMode ? CreateEstimationNetwork(fwAndHwCapabilities)
                                                 : CreateNetwork(fwAndHwCapabilities);
    ParseNetwork();
    m_Options.m_EnableIntermediateCompression = creationOptions.m_LayerData.GetIntermediateCompression();
    m_Options.m_DebugInfo.m_DumpDebugFiles    = creationOptions.m_DumpDebugFiles;
    m_Options.m_StrictPrecision               = creationOptions.m_StrictPrecision;
    if (creationOptions.m_LayerData.GetConvolutionAlgorithm() == ConvolutionAlgorithm::Direct)
    {
        m_Options.m_DisableWinograd = true;
    }
    else if (creationOptions.m_LayerData.GetConvolutionAlgorithm() == ConvolutionAlgorithm::BestEffort)
    {
        m_Options.m_DisableWinograd = false;
    }
    else
    {
        // cppcheck-suppress assertWithSideEffect
        assert(creationOptions.m_LayerData.GetConvolutionAlgorithm() == ConvolutionAlgorithm::SupportLibraryDefault);
    }
    SetBlockConfigs(creationOptions.m_LayerData.GetBlockConfigs());
}

void EthosNParseRunner::RecordAddedLayerSingleOutput(const std::string& name, TensorAndId<Operand> ethosnOutput)
{
    RecordAddedLayerSingleOutput(name, ethosnOutput.tensor, ethosnOutput.operationId);
}

void EthosNParseRunner::RecordAddedLayerSingleOutput(const std::string& name, TensorAndId<Constant> ethosnOutput)
{
    // Constant has a single operand.
    RecordAddedLayerSingleOutput(name, GetOperand(ethosnOutput.tensor), ethosnOutput.operationId);
}

void EthosNParseRunner::RecordAddedLayerSingleOutput(const std::string& name,
                                                     std::shared_ptr<ethosn::support_library::Operand> operand,
                                                     uint32_t operationId)
{
    m_OutputToOperand[name]                 = operand;
    m_OperandToOperationIdAndIndex[operand] = { operationId, 0 };
}

void EthosNParseRunner::RecordAddedLayerMultipleOutput(const std::string& name, TensorsAndId ethosnOutput)
{
    for (uint32_t i = 0; i < ethosnOutput.tensors.size(); ++i)
    {
        m_OutputToOperand[name + "_" + std::to_string(i)]       = ethosnOutput.tensors[i];
        m_OperandToOperationIdAndIndex[ethosnOutput.tensors[i]] = { ethosnOutput.operationId, i };
    }
}

void EthosNParseRunner::AddInput(const std::string& name, TensorShape shape)
{
    GgfParser::AddInput(name, shape);

    TensorInfo inputTensorInfo{ shape, m_LayerData.GetInputsDataType<ethosn::support_library::DataType>(),
                                m_LayerData.GetInputTensorFormat(), m_LayerData.GetInputQuantInfo(name) };
    TensorAndId<Operand> input = ethosn::support_library::AddInput(m_Network, inputTensorInfo);
    RecordAddedLayerSingleOutput(name, input);
    // Record this input for later lookup when matching up Ethos-N inputs to GGF inputs.
    // Note this is extra information specific to input layers, not recorded by the above RecordAddedLayerSingleOutput.
    m_OperationIdAndIndexToInputName[{ input.operationId, 0 }] = name;
}

void EthosNParseRunner::AddConstant(const std::string& name, TensorShape shape, float constMin, float constMax)
{
    const BaseTensor& constData = m_LayerData.GetConstantData(name, shape, constMin, constMax);

    // Create constant tensor
    const QuantizationInfo constantQuantInfo = m_LayerData.GetConstantQuantInfo(name, constMin, constMax);
    const TensorInfo constTensorInfo{ shape, m_LayerData.GetInputsDataType<ethosn::support_library::DataType>(),
                                      DataFormat::NHWC, constantQuantInfo };

    TensorAndId<Constant> constant =
        ethosn::support_library::AddConstant(m_Network, constTensorInfo, constData.GetByteData());
    RecordAddedLayerSingleOutput(name, constant);
}

void EthosNParseRunner::AddConvolution(const std::string& name,
                                       const std::string& inputName,
                                       const uint32_t kernelWidth,
                                       const uint32_t kernelHeight,
                                       const uint32_t strideWidth,
                                       const uint32_t strideHeight,
                                       uint32_t outputChannels,
                                       const bool biasEnable,
                                       const WeightParams& weightParams,
                                       const OutputParams& outputParams,
                                       const PaddingInfo padInfo,
                                       decltype(::AddConvolution)& addConv)
{
    const bool isDepthwise = &addConv == &ethosn::support_library::AddDepthwiseConvolution;
    const bool isTranspose = &addConv == &ethosn::support_library::AddTransposeConvolution;

    const TensorInfo inputTensorInfo = GetTensorInfo(m_OutputToOperand.at(inputName));

    const uint32_t inputHeight   = inputTensorInfo.m_Dimensions[1];
    const uint32_t inputWidth    = inputTensorInfo.m_Dimensions[2];
    const uint32_t inputChannels = inputTensorInfo.m_Dimensions[3];
    const float inputQuantScale  = inputTensorInfo.m_QuantizationInfo.GetScale();

    if (isDepthwise)
    {
        outputChannels *= inputChannels;
    }

    // Create weight tensor
    const uint32_t numSummedTerms =
        isDepthwise ? kernelWidth * kernelHeight : kernelWidth * kernelHeight * inputChannels;
    const QuantizationInfo weightQuantInfo =
        m_LayerData.GetConvWeightQuantInfo(name, weightParams, numSummedTerms, outputChannels, isDepthwise);

    const TensorShape weightTensorShape{
        kernelHeight,
        kernelWidth,
        inputChannels,
        isDepthwise ? outputChannels / inputChannels : outputChannels,
    };
    const TensorInfo weightTensorInfo{
        weightTensorShape,
        m_LayerData.GetWeightDataType<ethosn::support_library::DataType>(),
        isDepthwise ? DataFormat::HWIM : DataFormat::HWIO,
        weightQuantInfo,
    };

    const WeightTensor weightData =
        m_LayerData.GetConvWeightData(name, weightTensorShape, weightQuantInfo, weightParams);

    // Create bias tensor
    const QuantizationInfo biasQuantInfo =
        m_LayerData.GetConvBiasQuantInfo(name, inputQuantScale, weightQuantInfo.GetScales());

    const TensorShape biasTensorShape{ 1, 1, 1, outputChannels };
    const TensorInfo biasTensorInfo{ biasTensorShape, ethosn::support_library::DataType::INT32_QUANTIZED,
                                     DataFormat::NHWC, biasQuantInfo };

    const OwnedTensor biasData = biasEnable ? MakeTensor(m_LayerData.GetConvBiasData(name, outputChannels))
                                            : MakeTensor(std::vector<int32_t>(outputChannels * sizeof(int32_t), 0));

    // Create convolution layer
    ConvolutionInfo convInfo{};
    ethosn::support_library::Padding& padding = convInfo.m_Padding;

    if (padInfo.alg != PaddingAlgorithm::EXPLICIT)
    {
        const bool padSame = padInfo.alg == PaddingAlgorithm::SAME;
        auto padY          = std::tie(padding.m_Top, padding.m_Bottom);
        auto padX          = std::tie(padding.m_Left, padding.m_Right);

        std::tie(std::ignore, padY) =
            CalcConvOutSizeAndPadding(inputHeight, kernelHeight, strideHeight, padSame, isTranspose);
        std::tie(std::ignore, padX) =
            CalcConvOutSizeAndPadding(inputWidth, kernelWidth, strideWidth, padSame, isTranspose);
    }
    else
    {
        padding.m_Top    = padInfo.info.padTop;
        padding.m_Bottom = padInfo.info.padBottom;
        padding.m_Left   = padInfo.info.padLeft;
        padding.m_Right  = padInfo.info.padRight;
    }

    convInfo.m_OutputQuantizationInfo = m_LayerData.GetConvOutputQuantInfo(
        name, inputTensorInfo.m_QuantizationInfo, weightQuantInfo, numSummedTerms, outputParams);
    convInfo.m_Stride = Stride{ strideWidth, strideHeight };

    const std::shared_ptr<Constant> bias =
        ethosn::support_library::AddConstant(m_Network, biasTensorInfo, biasData->GetByteData()).tensor;
    const std::shared_ptr<Constant> weight =
        ethosn::support_library::AddConstant(m_Network, weightTensorInfo, weightData->GetByteData()).tensor;

    RecordAddedLayerSingleOutput(name, addConv(m_Network, *m_OutputToOperand.at(inputName), *bias, *weight, convInfo));

    m_LayerData.SetMaxKernelSize(std::max(kernelHeight, kernelWidth));
}

void EthosNParseRunner::AddDepthwiseConvolution(const std::string& name,
                                                const std::string& inputName,
                                                const uint32_t kernelWidth,
                                                const uint32_t kernelHeight,
                                                const uint32_t strideWidth,
                                                const uint32_t strideHeight,
                                                const uint32_t channelMultiplier,
                                                const bool biasEnable,
                                                const WeightParams& weightParams,
                                                const OutputParams& outputParams,
                                                const PaddingInfo padInfo)
{
    AddConvolution(name, inputName, kernelWidth, kernelHeight, strideWidth, strideHeight, channelMultiplier, biasEnable,
                   weightParams, outputParams, padInfo, ethosn::support_library::AddDepthwiseConvolution);
}

void EthosNParseRunner::AddStandalonePadding(const std::string& name, const std::string& inputName, PaddingInfo padInfo)
{
    ethosn::support_library::Padding padding(padInfo.info.padTop, padInfo.info.padBottom, padInfo.info.padLeft,
                                             padInfo.info.padRight);

    auto standalonePadding =
        ethosn::support_library::AddStandalonePadding(m_Network, *m_OutputToOperand.at(inputName), padding);
    RecordAddedLayerSingleOutput(name, standalonePadding);
}

void EthosNParseRunner::AddConvolution(const std::string& name,
                                       const std::string& inputName,
                                       const uint32_t kernelWidth,
                                       const uint32_t kernelHeight,
                                       const uint32_t strideWidth,
                                       const uint32_t strideHeight,
                                       const uint32_t numOutput,
                                       const bool biasEnable,
                                       const WeightParams& weightParams,
                                       const OutputParams& outputParams,
                                       const PaddingInfo padInfo)
{
    AddConvolution(name, inputName, kernelWidth, kernelHeight, strideWidth, strideHeight, numOutput, biasEnable,
                   weightParams, outputParams, padInfo, ethosn::support_library::AddConvolution);
}

void EthosNParseRunner::AddTransposeConvolution(const std::string& name,
                                                const std::string& inputName,
                                                const uint32_t kernelWidth,
                                                const uint32_t kernelHeight,
                                                const uint32_t strideWidth,
                                                const uint32_t strideHeight,
                                                const uint32_t numOutput,
                                                const bool biasEnable,
                                                const WeightParams& weightParams,
                                                const OutputParams& outputParams,
                                                const PaddingInfo padInfo)
{
    AddConvolution(name, inputName, kernelWidth, kernelHeight, strideWidth, strideHeight, numOutput, biasEnable,
                   weightParams, outputParams, padInfo, ethosn::support_library::AddTransposeConvolution);
}

void EthosNParseRunner::AddFullyConnected(const std::string& name,
                                          const std::string& inputName,
                                          uint32_t numOutput,
                                          const WeightParams& weightParams,
                                          const OutputParams& outputParams)
{
    TensorInfo prevTensorInfo = GetTensorInfo(m_OutputToOperand.at(inputName));

    const uint32_t height   = prevTensorInfo.m_Dimensions[1];
    const uint32_t width    = prevTensorInfo.m_Dimensions[2];
    const uint32_t channels = prevTensorInfo.m_Dimensions[3];

    // Create weight tensor
    uint32_t numInputs               = width * height * channels;
    QuantizationInfo weightQuantInfo = m_LayerData.GetFCWeightQuantInfo(name, weightParams, numInputs);
    TensorInfo weightInfo{ { 1, 1, numInputs, numOutput },
                           m_LayerData.GetWeightDataType<ethosn::support_library::DataType>(),
                           DataFormat::HWIO,
                           weightQuantInfo };

    // Create bias tensor
    QuantizationInfo biasQuantInfo =
        m_LayerData.GetFCBiasQuantInfo(name, prevTensorInfo.m_QuantizationInfo.GetScale(), weightQuantInfo.GetScale());
    TensorInfo biasInfo{
        { 1, 1, 1, numOutput }, ethosn::support_library::DataType::INT32_QUANTIZED, DataFormat::NHWC, biasQuantInfo
    };
    const BaseTensor& biasData = m_LayerData.GetFullyConnectedBiasData(name, numOutput);
    std::shared_ptr<Constant> bias =
        ethosn::support_library::AddConstant(m_Network, biasInfo, biasData.GetByteData()).tensor;

    // Create fully connected layer
    QuantizationInfo outputQuantInfo = m_LayerData.GetFCOutputQuantInfo(name, prevTensorInfo.m_QuantizationInfo,
                                                                        weightQuantInfo, numInputs, outputParams);
    ethosn::support_library::TensorShape weightTensorShape{ 1, 1, numInputs, numOutput };
    const WeightTensor weightsData =
        m_LayerData.GetFullyConnectedWeightData(name, weightTensorShape, weightQuantInfo, weightParams);
    std::shared_ptr<Constant> weights =
        ethosn::support_library::AddConstant(m_Network, weightInfo, weightsData->GetByteData()).tensor;
    FullyConnectedInfo fullyConnectedInfo{ outputQuantInfo };
    RecordAddedLayerSingleOutput(name,
                                 ethosn::support_library::AddFullyConnected(m_Network, *m_OutputToOperand.at(inputName),
                                                                            *bias, *weights, fullyConnectedInfo));
}

void EthosNParseRunner::AddRelu(const std::string& name, const std::string& inputName)
{
    const ReluInfo reluInfo = m_LayerData.GetReluInfo(name);
    auto relu               = ethosn::support_library::AddRelu(m_Network, *m_OutputToOperand.at(inputName), reluInfo);
    RecordAddedLayerSingleOutput(inputName, relu);    // Relu "modifies" its input layer
    RecordAddedLayerSingleOutput(name, relu);
}

void EthosNParseRunner::AddLeakyRelu(const std::string& name, const std::string& inputName, const float alpha)
{
    TensorInfo prevTensorInfo = GetTensorInfo(m_OutputToOperand.at(inputName));

    ethosn::support_library::QuantizationInfo outputQuantInfo =
        m_LayerData.GetLeakyReluOutputQuantInfo(name, prevTensorInfo.m_QuantizationInfo, alpha);

    auto leakyrelu = ethosn::support_library::AddLeakyRelu(m_Network, *m_OutputToOperand.at(inputName),
                                                           LeakyReluInfo(alpha, outputQuantInfo));
    RecordAddedLayerSingleOutput(inputName,
                                 leakyrelu);    // Leakyrelu "modifies" its input layer
    RecordAddedLayerSingleOutput(name, leakyrelu);
}

void EthosNParseRunner::AddRequantize(const std::string& name,
                                      const std::string& inputName,
                                      ethosn::support_library::RequantizeInfo& requantizeInfo)
{
    auto requantize =
        ethosn::support_library::AddRequantize(m_Network, *m_OutputToOperand.at(inputName), requantizeInfo);
    RecordAddedLayerSingleOutput(name, requantize);
}

void EthosNParseRunner::AddSigmoid(const std::string& name, const std::string& inputName)
{
    auto sigmoid = ethosn::support_library::AddSigmoid(m_Network, *m_OutputToOperand.at(inputName));
    RecordAddedLayerSingleOutput(inputName,
                                 sigmoid);    // Sigmoid "modifies" its input layer
    RecordAddedLayerSingleOutput(name, sigmoid);
}

void EthosNParseRunner::AddTanh(const std::string& name, const std::string& inputName)
{
    auto tanh = ethosn::support_library::AddTanh(m_Network, *m_OutputToOperand.at(inputName));
    RecordAddedLayerSingleOutput(inputName,
                                 tanh);    // Tanh "modifies" its input layer
    RecordAddedLayerSingleOutput(name, tanh);
}

void EthosNParseRunner::AddReshape(const std::string& name, const std::string& inputName, TensorShape shape)
{
    RecordAddedLayerSingleOutput(
        name, ethosn::support_library::AddReshape(m_Network, *m_OutputToOperand.at(inputName), shape));
}

void EthosNParseRunner::AddConcatenation(const std::string& name,
                                         const std::vector<std::string>& inputNames,
                                         uint32_t axis)
{
    std::vector<Operand*> input;
    std::vector<std::string>::const_iterator it;
    std::vector<QuantizationInfo> inputQuantInfos;

    for (it = inputNames.begin(); it != inputNames.end(); ++it)
    {
        auto operand = m_OutputToOperand.find((*it).c_str())->second;
        input.push_back(&(*operand));
        inputQuantInfos.push_back(GetTensorInfo(operand).m_QuantizationInfo);
    }
    QuantizationInfo outputQuantInfo = m_LayerData.GetConcatOutputQuantInfo(name, inputQuantInfos);
    RecordAddedLayerSingleOutput(
        name, ethosn::support_library::AddConcatenation(m_Network, input, ConcatenationInfo(axis, outputQuantInfo)));
}

void EthosNParseRunner::AddSplit(const std::string& name,
                                 const std::string& inputName,
                                 uint32_t axis,
                                 std::vector<uint32_t> sizes)
{
    RecordAddedLayerMultipleOutput(
        name, ethosn::support_library::AddSplit(m_Network, *m_OutputToOperand.at(inputName), SplitInfo(axis, sizes)));
}

void EthosNParseRunner::AddAddition(const std::string& name,
                                    const std::string& firstInputName,
                                    const std::string& secondInputName)
{
    TensorInfo firstTensorInfo  = GetTensorInfo(m_OutputToOperand.at(firstInputName));
    TensorInfo secondTensorInfo = GetTensorInfo(m_OutputToOperand.at(secondInputName));

    // the quantization info from the first layer is used as default value
    const std::vector<ethosn::support_library::QuantizationInfo> addQuantInfos{ firstTensorInfo.m_QuantizationInfo,
                                                                                secondTensorInfo.m_QuantizationInfo };
    QuantizationInfo addQuantInfo = m_LayerData.GetAdditionQuantInfo(name, addQuantInfos);

    RecordAddedLayerSingleOutput(name, ethosn::support_library::AddAddition(
                                           m_Network, *m_OutputToOperand.find(firstInputName.c_str())->second,
                                           *m_OutputToOperand.find(secondInputName.c_str())->second, addQuantInfo));
}

void EthosNParseRunner::AddMultiplication(const std::string& name,
                                          const std::string& firstInputName,
                                          const std::string& secondInputName)
{
    TensorInfo firstTensorInfo  = GetTensorInfo(m_OutputToOperand.at(firstInputName));
    TensorInfo secondTensorInfo = GetTensorInfo(m_OutputToOperand.at(secondInputName));

    // the quantization info from the first layer is used as default value
    const std::vector<ethosn::support_library::QuantizationInfo> quantInfos{ firstTensorInfo.m_QuantizationInfo,
                                                                             secondTensorInfo.m_QuantizationInfo };
    QuantizationInfo mulQuantInfo = m_LayerData.GetMultiplicationQuantInfo(name, quantInfos);

    RecordAddedLayerSingleOutput(name, ethosn::support_library::AddMultiplication(
                                           m_Network, *m_OutputToOperand.find(firstInputName.c_str())->second,
                                           *m_OutputToOperand.find(secondInputName.c_str())->second, mulQuantInfo));
}

void EthosNParseRunner::AddMeanXy(const std::string& name, const std::string& inputName)
{
    RecordAddedLayerSingleOutput(name, ethosn::support_library::AddMeanXy(m_Network, *m_OutputToOperand.at(inputName)));
}

void EthosNParseRunner::AddPooling(const std::string& name,
                                   const std::string& inputName,
                                   PoolingInfo poolingInfo,
                                   PaddingAlgorithm paddingAlgorithm)
{
    TensorInfo prevTensorInfo = GetTensorInfo(m_OutputToOperand.at(inputName));

    const uint32_t prevHeight = prevTensorInfo.m_Dimensions[1];
    const uint32_t prevWidth  = prevTensorInfo.m_Dimensions[2];

    const bool padSame                        = paddingAlgorithm == PaddingAlgorithm::SAME;
    ethosn::support_library::Padding& padding = poolingInfo.m_Padding;

    auto padY = std::tie(padding.m_Top, padding.m_Bottom);
    auto padX = std::tie(padding.m_Left, padding.m_Right);

    std::tie(std::ignore, padY) =
        CalcConvOutSizeAndPadding(prevHeight, poolingInfo.m_PoolingSizeY, poolingInfo.m_PoolingStrideY, padSame);
    std::tie(std::ignore, padX) =
        CalcConvOutSizeAndPadding(prevWidth, poolingInfo.m_PoolingSizeX, poolingInfo.m_PoolingStrideX, padSame);

    RecordAddedLayerSingleOutput(
        name, ethosn::support_library::AddPooling(m_Network, *m_OutputToOperand.at(inputName), poolingInfo));
}

void EthosNParseRunner::AddDepthToSpace(const std::string& name, const std::string& inputName, uint32_t blockSize)
{
    RecordAddedLayerSingleOutput(
        name, ethosn::support_library::AddDepthToSpace(m_Network, *m_OutputToOperand.at(inputName), blockSize));
}

void EthosNParseRunner::AddSpaceToDepth(const std::string& name, const std::string& inputName, uint32_t blockSize)
{
    RecordAddedLayerSingleOutput(
        name, ethosn::support_library::AddSpaceToDepth(m_Network, *m_OutputToOperand.at(inputName), blockSize));
}

void EthosNParseRunner::AddOutput(const std::string& name, const std::string& inputName)
{
    GgfParser::AddOutput(name, inputName);

    std::shared_ptr<Operand> input = m_OutputToOperand.at(inputName);
    TensorAndId<Output> output =
        ethosn::support_library::AddOutput(m_Network, *input, m_LayerData.GetOutputTensorFormat());
    // Record this output for later lookup when matching up Ethos-N outputs to GGF outputs.
    m_OutputNameToOperationIdAndIndex[name] = m_OperandToOperationIdAndIndex[input];
    // Record this output for later lookup when querying the output shape
    m_OutputLayerToOperand[name] = input;
}

void EthosNParseRunner::AddTranspose(const std::string& name,
                                     const std::string& inputName,
                                     const std::array<uint32_t, 4>& permutation)
{
    RecordAddedLayerSingleOutput(
        name, ethosn::support_library::AddTranspose(m_Network, *m_OutputToOperand.at(inputName),
                                                    ethosn::support_library::TransposeInfo(permutation)));
}

void EthosNParseRunner::AddResize(const std::string& name, const std::string& inputName, const ResizeParams& params)
{
    TensorInfo prevTensorInfo = GetTensorInfo(m_OutputToOperand.at(inputName));

    const uint32_t prevHeight = prevTensorInfo.m_Dimensions[1];
    const uint32_t prevWidth  = prevTensorInfo.m_Dimensions[2];
    const ethosn::support_library::ResizeInfo resizeInfo(
        params.m_Algo, CalcUpsampleOutputSize(params.m_Height, prevHeight),
        CalcUpsampleOutputSize(params.m_Width, prevWidth), prevTensorInfo.m_QuantizationInfo);

    RecordAddedLayerSingleOutput(
        name, ethosn::support_library::AddResize(m_Network, *m_OutputToOperand.at(inputName), resizeInfo));
}

void EthosNParseRunner::SetStrategies(const std::string& strategies)
{
    if (!strategies.empty())
    {
        m_Options.m_Strategy0 = false;
        m_Options.m_Strategy1 = false;
        m_Options.m_Strategy3 = false;
        m_Options.m_Strategy4 = false;
        m_Options.m_Strategy6 = false;
        m_Options.m_Strategy7 = false;

        // Split strategies by comma
        std::string part;
        std::size_t pos = 0;
        while (!(part = Split(strategies, ",", pos)).empty())
        {
            part = ethosn::utils::Trim(part);
            if (part == "0")
            {
                m_Options.m_Strategy0 = true;
            }
            else if (part == "1")
            {
                m_Options.m_Strategy1 = true;
            }
            else if (part == "3")
            {
                m_Options.m_Strategy3 = true;
            }
            else if (part == "4")
            {
                m_Options.m_Strategy4 = true;
            }
            else if (part == "6")
            {
                m_Options.m_Strategy6 = true;
            }
            else if (part == "7")
            {
                m_Options.m_Strategy7 = true;
            }
        }
    }
}

void EthosNParseRunner::SetBlockConfigs(const std::string& blockConfigs)
{
    if (!blockConfigs.empty())
    {
        m_Options.m_BlockConfig16x16 = false;
        m_Options.m_BlockConfig32x8  = false;
        m_Options.m_BlockConfig8x32  = false;
        m_Options.m_BlockConfig16x8  = false;
        m_Options.m_BlockConfig8x16  = false;
        m_Options.m_BlockConfig8x8   = false;

        // Split block configs by comma
        std::string part;
        std::size_t pos = 0;
        while (!(part = Split(blockConfigs, ",", pos)).empty())
        {
            part = ethosn::utils::Trim(part);
            g_Logger.Debug("EthosNParseRunner::BlockConfig=%s", part.c_str());
            if (part == "16x16")
            {
                m_Options.m_BlockConfig16x16 = true;
            }
            else if (part == "32x8")
            {
                m_Options.m_BlockConfig32x8 = true;
            }
            else if (part == "8x32")
            {
                m_Options.m_BlockConfig8x32 = true;
            }
            else if (part == "16x8")
            {
                m_Options.m_BlockConfig16x8 = true;
            }
            else if (part == "8x16")
            {
                m_Options.m_BlockConfig8x16 = true;
            }
            else if (part == "8x8")
            {
                m_Options.m_BlockConfig8x8 = true;
            }
        }
    }
}

void EthosNParseRunner::SetActionCallback(ActionsCallback callback)
{
    m_Callbacks = std::move(callback);
}

float EthosNParseRunner::GetComparisonTolerance()
{
    uint32_t kernelSize = m_LayerData.GetMaxKernelSize();
    float tol           = 0;

    if (m_Options.m_DisableWinograd == 0)
    {
        // Winograd and wide kernel will be enabled
        if (kernelSize > 3)
        {
            tol = 3.f;
        }
        else if (kernelSize == 1)
        {
            tol = 1.f;
        }
        else
        {
            tol = 2.f;
        }
    }
    else
    {
        tol = 1.f;
    }
    g_Logger.Debug("EthosNParseRunner::comparisonTolerance=%f", tol);
    return tol;
}

std::vector<std::unique_ptr<CompiledNetwork>> EthosNParseRunner::GetCompiledNetworks()
{
    return Compile(*m_Network, m_Options);
}

ethosn::support_library::TensorShape EthosNParseRunner::GetLayerOutputShape(const std::string& layerName)
{
    return GetTensorInfo(m_OutputLayerToOperand.at(layerName)).m_Dimensions;
}

int EthosNParseRunner::GetEthosNIndex(std::vector<ethosn::support_library::OutputBufferInfo> outputBufferInfos,
                                      std::pair<uint32_t, uint32_t> operand)
{
    auto lambda = [operand](auto& bufferInfo) {
        return (bufferInfo.m_SourceOperationId == operand.first &&
                bufferInfo.m_SourceOperationOutputIndex == operand.second);
    };
    std::vector<ethosn::support_library::OutputBufferInfo>::iterator it =
        std::find_if(outputBufferInfos.begin(), outputBufferInfos.end(), lambda);
    return static_cast<int>(std::distance(outputBufferInfos.begin(), it));
}

const ethosn::driver_library::IntermediateBufferReq
    EthosNParseRunner::GetIntermediateBufferReq(const ethosn::system_tests::DmaBuffer* intermediateDmaBuf,
                                                uint32_t intermediateBufferSize)
{
    if (intermediateBufferSize == 0)
    {
        return ethosn::driver_library::IntermediateBufferReq(ethosn::driver_library::MemType::NONE);
    }
    if (intermediateDmaBuf != nullptr)
    {
        return ethosn::driver_library::IntermediateBufferReq(ethosn::driver_library::MemType::IMPORT,
                                                             intermediateDmaBuf->GetFd(), O_RDWR | O_CLOEXEC);
    }
    else
    {
        return ethosn::driver_library::IntermediateBufferReq();
    }
}

InferenceOutputs EthosNParseRunner::RunNetwork(int timeoutSeconds)
{
    std::unique_ptr<DmaBufferDevice> dmaBufHeap;
    const char* dmaBufferDeviceFile;

    if (g_RunProtectedInference)
        dmaBufferDeviceFile = g_DmaBufProtected.c_str();
    else if (g_UseDmaBuf)
        dmaBufferDeviceFile = g_DmaBufHeap.c_str();

    if (g_UseDmaBuf || g_RunProtectedInference)
    {
        dmaBufHeap = std::make_unique<DmaBufferDevice>(dmaBufferDeviceFile);
    }

    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetworks = Compile(*m_Network, m_Options);

    if (compiledNetworks.size() == 0)
    {
        throw std::runtime_error("Compilation failed");
    }
    else if (compiledNetworks.size() > 1)
    {
        throw std::runtime_error("Multiple compiled networks not supported");
    }

    CompiledNetwork& compiledNetwork = *compiledNetworks[0].get();
    std::vector<char> compiledNetworkData;
    {
        ethosn::utils::VectorStream compiledNetworkStream(compiledNetworkData);
        compiledNetwork.Serialize(compiledNetworkStream);
    }

    InferenceDmaBuffers intermediateDmaBuf(1);

    if ((g_UseDmaBuf || g_RunProtectedInference) && compiledNetwork.GetIntermediateBufferSize() > 0)
    {
        intermediateDmaBuf[0] = std::make_shared<DmaBuffer>(dmaBufHeap, compiledNetwork.GetIntermediateBufferSize());
    }

    std::unique_ptr<ethosn::driver_library::ProcMemAllocator> processMemAllocator =
        std::make_unique<ethosn::driver_library::ProcMemAllocator>(g_RunProtectedInference);

    const ethosn::driver_library::IntermediateBufferReq intermediateBuffReq = GetIntermediateBufferReq(
        intermediateDmaBuf[0].get(), static_cast<uint32_t>(compiledNetwork.GetIntermediateBufferSize()));

    ethosn::driver_library::Network netInst =
        processMemAllocator->CreateNetwork(compiledNetworkData.data(), compiledNetworkData.size(), intermediateBuffReq);
    netInst.SetDebugName("Ggf");

    // Create input buffers
    const std::vector<std::string> inputLayerNames = GetInputLayerNames();
    const size_t numInputLayers                    = inputLayerNames.size();
    assert(numInputLayers == compiledNetwork.GetInputBufferInfos().size());
    g_Logger.Debug("EthosNParseRunner::%s numInputLayers=%zu", __func__, numInputLayers);
    InferenceInputs inputData(numInputLayers);
    InferenceDmaBuffers ifmDmaBuf(numInputLayers);
    InferenceInputBuffers ifmBuffer(numInputLayers);
    InferenceInputBuffersPtr ifmRawBuffer(numInputLayers);

    for (size_t i = 0; i < numInputLayers; ++i)
    {
        uint32_t operationId         = compiledNetwork.GetInputBufferInfos()[i].m_SourceOperationId;
        uint32_t outputIndex         = compiledNetwork.GetInputBufferInfos()[i].m_SourceOperationOutputIndex;
        const std::string& inputName = m_OperationIdAndIndexToInputName.at(std::make_pair(operationId, outputIndex));
        g_Logger.Debug("EthosNParseRunner::%s input[%zu] name=%s", __func__, i, inputName.c_str());

        ethosn::support_library::TensorInfo inputTensorInfo = GetTensorInfo(m_OutputToOperand.at(inputName));

        inputData[i] = m_LayerData.GetInputData(inputName, inputTensorInfo.m_Dimensions);

        if (g_UseDmaBuf || g_RunProtectedInference)
        {
            // Use the buffer size returned from the compiler to allocate the input buffer
            ifmDmaBuf[i] = std::make_shared<DmaBuffer>(dmaBufHeap, compiledNetwork.GetInputBufferInfos()[i].m_Size);
            ifmDmaBuf[i]->PopulateData(const_cast<uint8_t*>(inputData[i]->GetByteData()), inputData[i]->GetNumBytes());
            ethosn::driver_library::Buffer bufInst = processMemAllocator->ImportBuffer(
                ifmDmaBuf[i]->GetFd(), static_cast<uint32_t>(ifmDmaBuf[i]->GetSize()));
            ifmBuffer[i] = std::make_shared<ethosn::driver_library::Buffer>(std::move(bufInst));
        }
        else
        {
            ethosn::driver_library::Buffer bufInst = processMemAllocator->CreateBuffer(
                const_cast<uint8_t*>(inputData[i]->GetByteData()), compiledNetwork.GetInputBufferInfos()[i].m_Size);
            // Use the buffer size returned from the compiler to allocate the input buffer
            ifmBuffer[i] = std::make_shared<ethosn::driver_library::Buffer>(std::move(bufInst));
        }

        ifmRawBuffer[i] = ifmBuffer[i].get();
    }

    // Create output buffers
    const std::vector<std::string> outputLayerNames = GetOutputLayerNames();
    const size_t numOutputLayers                    = outputLayerNames.size();
    assert(numOutputLayers == compiledNetwork.GetOutputBufferInfos().size());
    g_Logger.Debug("EthosNParseRunner::%s numOutputLayers=%zu", __func__, numOutputLayers);

    InferenceOutputs outputData(numOutputLayers);
    InferenceOutputsPtr outputDataRaw(numOutputLayers);
    InferenceOutputs firstOutputData(numOutputLayers);
    for (size_t i = 0; i < numOutputLayers; ++i)
    {
        uint32_t outputSize = compiledNetwork.GetOutputBufferInfos()[i].m_Size;
        DataType dataType   = m_LayerData.GetInputsDataType<DataType>();
        outputData[i]       = MakeTensor(dataType, outputSize / GetNumBytes(dataType));
        outputDataRaw[i]    = outputData[i]->GetByteData();
    }

    size_t numRunsInBatch = m_RunBatchSize > 0 ? m_RunBatchSize : m_NumberRuns;
    size_t numBatches     = DivRoundUp(m_NumberRuns, numRunsInBatch);
    size_t completedRuns  = 0;
    g_Logger.Debug("Inference will run %zu times, split into %zu batches", m_NumberRuns, numBatches);
    for (size_t batch = 0; batch < numBatches; ++batch)
    {
        // Calculate number of runs in batch from the total number of runs left
        numRunsInBatch = std::min(numRunsInBatch, m_NumberRuns - completedRuns);

        MultipleInferenceDmaBuffers ofmDmaBuf(numRunsInBatch);
        MultipleInferenceOutputBuffers ofmBuffer(numRunsInBatch);
        MultipleInferenceOutputBuffersPtr ofmRawBuffer(numRunsInBatch);
        InferenceResult result(numRunsInBatch);

        for (size_t run = 0; run < numRunsInBatch; ++run)
        {
            ofmDmaBuf[run].resize(numOutputLayers);
            ofmBuffer[run].resize(numOutputLayers);
            ofmRawBuffer[run].resize(numOutputLayers);

            for (size_t i = 0; i < numOutputLayers; ++i)
            {
                g_Logger.Debug("EthosNParseRunner::%s output[%zu]", __func__, i);
                if (g_UseDmaBuf || g_RunProtectedInference)
                {
                    ofmDmaBuf[run][i] = std::make_shared<DmaBuffer>(dmaBufHeap, outputData[i]->GetNumBytes());
                    ethosn::driver_library::Buffer bufInst = processMemAllocator->ImportBuffer(
                        ofmDmaBuf[run][i]->GetFd(), static_cast<uint32_t>(ofmDmaBuf[run][i]->GetSize()));
                    ofmBuffer[run][i] = std::make_shared<ethosn::driver_library::Buffer>(std::move(bufInst));
                }
                else
                {

                    ethosn::driver_library::Buffer bufInst =
                        processMemAllocator->CreateBuffer(outputData[i]->GetNumBytes());
                    ofmBuffer[run][i] = std::make_shared<ethosn::driver_library::Buffer>(std::move(bufInst));
                }
                ofmRawBuffer[run][i] = ofmBuffer[run][i].get();
            }
        }

        g_Logger.Debug("Running %zu inferences for batch %zu", numRunsInBatch, batch);
        for (size_t run = 0; run < numRunsInBatch; ++run)
        {
            // Schedule the inference.
            g_Logger.Debug("EthosNParseRunner::%s ScheduleInference", __func__);

            std::unique_ptr<ethosn::driver_library::Inference> tmp(
                netInst.ScheduleInference(ifmRawBuffer.data(), static_cast<uint32_t>(ifmRawBuffer.size()),
                                          ofmRawBuffer[run].data(), static_cast<uint32_t>(ofmRawBuffer[run].size())));

            result[run] = std::move(tmp);

            if (!result[run])
            {
                throw std::runtime_error("ScheduleInference failed.");
            }
        }

        if (m_Callbacks.afterScheduleCallback)
        {
            m_Callbacks.afterScheduleCallback.value()(result);
        }

        for (size_t run = 0; run < numRunsInBatch; ++run)
        {
            ethosn::driver_library::InferenceResult inferenceResult = result[run]->Wait(timeoutSeconds * 1000);
            switch (inferenceResult)
            {
                case ethosn::driver_library::InferenceResult::Scheduled:
                    // Intentional fallthrough
                case ethosn::driver_library::InferenceResult::Running:
                    throw std::runtime_error("Inference timed out after " + std::to_string(timeoutSeconds) + "s");
                case ethosn::driver_library::InferenceResult::Completed:
                    // Yay!
                    break;
                case ethosn::driver_library::InferenceResult::Error:
                    throw std::runtime_error("Inference error");
                default:
                    throw std::runtime_error("Unknown inference result");
            }
            CopyBuffers(ofmRawBuffer[run], outputDataRaw);

            g_Logger.Info("Cycle count: %lu", result[run]->GetCycleCount());

            for (size_t i = 0; i < numOutputLayers; ++i)
            {
                // Store the result from the first inference separately to be
                // used as a reference for the other inferences.
                if (batch == 0 && run == 0)
                {
                    firstOutputData[i] = MakeTensor(*outputData[i]);
                }
                else if (!CompareTensors(*outputData[i], *firstOutputData[i], 0.f))
                {
                    std::string res = DumpOutputToFiles(*outputData[i], *firstOutputData[i], "EthosN",
                                                        outputLayerNames[i], run + completedRuns);

                    throw std::runtime_error(res);
                }
            }
        }

        completedRuns += numRunsInBatch;
    }

    InferenceOutputs ggfOutputData(numOutputLayers);
    // Re-order output buffers into the GGF order (which may be different from the Ethos-N order)
    auto outputBufferInfos = compiledNetwork.GetOutputBufferInfos();
    for (size_t ggfIdx = 0; ggfIdx < numOutputLayers; ++ggfIdx)
    {
        const std::string ggfOutputName       = GetGgfOutputLayerName(ggfIdx);
        std::pair<uint32_t, uint32_t> operand = m_OutputNameToOperationIdAndIndex[ggfOutputName];
        int ethosNIdx                         = GetEthosNIndex(outputBufferInfos, operand);
        outputBufferInfos[ethosNIdx]          = {};
        ggfOutputData[ggfIdx]                 = std::move(outputData[ethosNIdx]);
    }

    return ggfOutputData;
}

NetworkPerformanceData EthosNParseRunner::EstimateNetwork()
{
    return EstimatePerformance(*m_Network, m_Options, m_EstimationOptions);
}

const EstimationOptions& EthosNParseRunner::GetEstimationOptions() const
{
    return m_EstimationOptions;
}

}    // namespace system_tests
}    // namespace ethosn
