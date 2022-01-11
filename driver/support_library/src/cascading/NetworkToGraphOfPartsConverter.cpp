//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "NetworkToGraphOfPartsConverter.hpp"

#include "ConcatPart.hpp"
#include "EstimateOnlyPart.hpp"
#include "FullyConnectedPart.hpp"
#include "FusedPlePart.hpp"
#include "GraphNodes.hpp"
#include "InputPart.hpp"
#include "McePart.hpp"
#include "OutputPart.hpp"
#include "Part.hpp"
#include "ReshapePart.hpp"
#include "StandalonePlePart.hpp"
#include "Utils.hpp"
#include "cascading/MceEstimationUtils.hpp"
#include <fstream>

using namespace ethosn::support_library::utils;

namespace ethosn
{
namespace support_library
{

std::unique_ptr<McePart> NetworkToGraphOfPartsConverter::CreateIdentityMcePart(const TensorShape& shape,
                                                                               const QuantizationInfo& inputQuantInfo,
                                                                               uint32_t operationId,
                                                                               command_stream::DataType dataType,
                                                                               const EstimationOptions& estOpt,
                                                                               const CompilationOptions& compOpt,
                                                                               const HardwareCapabilities& capabilities)
{

    McePart::ConstructionParams params(estOpt, compOpt, capabilities);
    params.m_Id                     = m_GraphOfParts.GeneratePartId();
    params.m_InputTensorShape       = shape;
    params.m_OutputTensorShape      = shape;
    params.m_InputQuantizationInfo  = inputQuantInfo;
    params.m_OutputQuantizationInfo = inputQuantInfo;
    const uint32_t numIfm           = shape[3];
    const float weightScale         = 0.5f;
    params.m_WeightsInfo   = { { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale } };
    params.m_WeightsData   = std::vector<uint8_t>(1 * 1 * 1 * numIfm, 2);
    const float biasScale  = weightScale * inputQuantInfo.GetScale();
    params.m_BiasInfo      = { { 1, 1, 1, numIfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };
    params.m_BiasData      = std::vector<int32_t>(numIfm, 0);
    params.m_Op            = command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    params.m_OperationIds  = std::set<uint32_t>{ operationId };
    params.m_DataType      = dataType;
    params.m_UpscaleFactor = 1;
    params.m_UpsampleType  = command_stream::UpsampleType::OFF;
    params.m_LowerBound    = dataType == command_stream::DataType::U8 ? 0 : -128;
    params.m_UpperBound    = dataType == command_stream::DataType::U8 ? 255 : 127;
    auto mcePart           = std::make_unique<McePart>(std::move(params));
    return mcePart;
}

NetworkToGraphOfPartsConverter::NetworkToGraphOfPartsConverter(const Network& network,
                                                               const HardwareCapabilities& capabilities,
                                                               const EstimationOptions& estimationOptions,
                                                               const CompilationOptions& compilationOptions)
    : m_Capabilities(capabilities)
    , m_EstimationOptions(estimationOptions)
    , m_CompilationOptions(compilationOptions)
{
    network.Accept(*this);
}

NetworkToGraphOfPartsConverter::~NetworkToGraphOfPartsConverter()
{}

void NetworkToGraphOfPartsConverter::Visit(Input& input)
{
    std::vector<BasePart*> parts;
    // Convert from DataFormat to CompilerFormat needed for the InputPart.
    CompilerDataFormat compilerDataFormat = ConvertExternalToCompilerDataFormat(input.GetTensorInfo().m_DataFormat);
    auto inputPart = std::make_unique<InputPart>(m_GraphOfParts.GeneratePartId(), input.GetTensorInfo().m_Dimensions,
                                                 compilerDataFormat, input.GetTensorInfo().m_QuantizationInfo,
                                                 std::set<uint32_t>{ input.GetId() }, m_EstimationOptions.value(),
                                                 m_CompilationOptions, m_Capabilities);
    parts.push_back(std::move(inputPart.get()));
    m_GraphOfParts.m_Parts.push_back(std::move(inputPart));
    ConnectParts(input, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Output& output)
{
    std::vector<BasePart*> parts;
    CompilerDataFormat compilerDataFormat = ConvertExternalToCompilerDataFormat(output.GetTensorInfo().m_DataFormat);
    auto outputPart = std::make_unique<OutputPart>(m_GraphOfParts.GeneratePartId(), output.GetTensorInfo().m_Dimensions,
                                                   compilerDataFormat, output.GetTensorInfo().m_QuantizationInfo,
                                                   std::set<uint32_t>{ output.GetId() }, m_EstimationOptions.value(),
                                                   m_CompilationOptions, m_Capabilities);
    parts.push_back(std::move(outputPart.get()));
    m_GraphOfParts.m_Parts.push_back(std::move(outputPart));
    ConnectParts(output, parts);
}

void NetworkToGraphOfPartsConverter::Visit(DepthwiseConvolution& depthwise)
{
    std::vector<BasePart*> parts;
    auto convInfo = depthwise.GetConvolutionInfo();

    TensorInfo mceOperationInput = depthwise.GetInput(0).GetTensorInfo();
    // Check if it is a strided depthwise and add a FusedPlePart.
    if (convInfo.m_Stride.m_X > 1 || convInfo.m_Stride.m_Y > 1)
    {
        // Create additional layer before strided convolution
        // Only supports stride 2x2 for now
        assert(convInfo.m_Stride.m_X == 2 && convInfo.m_Stride.m_Y == 2);

        uint32_t h = DivRoundUp(depthwise.GetInput(0).GetTensorInfo().m_Dimensions[1], convInfo.m_Stride.m_Y);
        uint32_t w = DivRoundUp(depthwise.GetInput(0).GetTensorInfo().m_Dimensions[2], convInfo.m_Stride.m_X);
        uint32_t c = GetNumSubmapChannels(depthwise.GetInput(0).GetTensorInfo().m_Dimensions[3], convInfo.m_Stride.m_X,
                                          convInfo.m_Stride.m_Y, m_Capabilities);

        TensorInfo mceOperationInput = TensorInfo({ depthwise.GetInput(0).GetTensorInfo().m_Dimensions[0], h, w, c },
                                                  depthwise.GetInput(0).GetTensorInfo().m_DataType,
                                                  depthwise.GetInput(0).GetTensorInfo().m_DataFormat,
                                                  depthwise.GetInput(0).GetTensorInfo().m_QuantizationInfo);

        auto fusedPlePart = std::make_unique<FusedPlePart>(
            m_GraphOfParts.GeneratePartId(), depthwise.GetInput(0).GetTensorInfo().m_Dimensions,
            mceOperationInput.m_Dimensions, depthwise.GetInput(0).GetTensorInfo().m_QuantizationInfo,
            mceOperationInput.m_QuantizationInfo, command_stream::PleOperation::INTERLEAVE_2X2_2_2,
            utils::ShapeMultiplier{ { 1, convInfo.m_Stride.m_Y },
                                    { 1, convInfo.m_Stride.m_X },
                                    { convInfo.m_Stride.m_X * convInfo.m_Stride.m_Y } },
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
            std::set<uint32_t>{ depthwise.GetId(), depthwise.GetBias().GetId(), depthwise.GetWeights().GetId() },
            GetCommandDataType(depthwise.GetOutput(0).GetTensorInfo().m_DataType));

        parts.push_back(std::move(fusedPlePart.get()));
        m_GraphOfParts.m_Parts.push_back(std::move(fusedPlePart));
    }

    command_stream::MceOperation operation                = command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    ethosn::support_library::TensorInfo weightsTensorInfo = depthwise.GetWeights().GetTensorInfo();
    weightsTensorInfo.m_DataFormat                        = DataFormat::HWIM;
    // We support channel multiplier > 1 if there is only 1 input channel because
    // a depthwise convolution with 1 input channel is equivalent to a normal convolution
    if (depthwise.GetWeights().GetTensorInfo().m_Dimensions[3] > 1)
    {
        assert(depthwise.GetWeights().GetTensorInfo().m_Dimensions[2] == 1);
        weightsTensorInfo.m_DataFormat = DataFormat::HWIO;
        operation                      = command_stream::MceOperation::CONVOLUTION;
    }

    // We don't use winograd for depthwise convolution
    auto mcePart = std::make_unique<McePart>(
        m_GraphOfParts.GeneratePartId(), mceOperationInput.m_Dimensions,
        depthwise.GetOutput(0).GetTensorInfo().m_Dimensions, mceOperationInput.m_QuantizationInfo,
        depthwise.GetOutput(0).GetTensorInfo().m_QuantizationInfo, depthwise.GetWeights().GetTensorInfo(),
        OverrideWeights(depthwise.GetWeights().GetDataVector(), weightsTensorInfo), depthwise.GetBias().GetTensorInfo(),
        GetDataVectorAs<int32_t, uint8_t>(depthwise.GetBias().GetDataVector()), depthwise.GetConvolutionInfo().m_Stride,
        depthwise.GetConvolutionInfo().m_Padding.m_Top, depthwise.GetConvolutionInfo().m_Padding.m_Left, operation,
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
        std::set<uint32_t>{ depthwise.GetId(), depthwise.GetBias().GetId(), depthwise.GetWeights().GetId() },
        GetCommandDataType(depthwise.GetOutput(0).GetTensorInfo().m_DataType));

    parts.push_back(std::move(mcePart.get()));
    m_GraphOfParts.m_Parts.push_back(std::move(mcePart));
    ConnectParts(depthwise, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Convolution& convolution)
{
    std::vector<BasePart*> parts;
    auto convInfo = convolution.GetConvolutionInfo();
    TensorInfo mcePartInputTensor;

    // Check if it is a strided convolution and add a FusedPlePart.
    if (convInfo.m_Stride.m_X > 1 || convInfo.m_Stride.m_Y > 1)
    {
        // Only stride 2x2 is supported for now.
        // Winograd is not considered for strided convolution.
        assert(convInfo.m_Stride.m_X == 2 && convInfo.m_Stride.m_Y == 2);

        uint32_t h = DivRoundUp(convolution.GetInput(0).GetTensorInfo().m_Dimensions[1], convInfo.m_Stride.m_Y);
        uint32_t w = DivRoundUp(convolution.GetInput(0).GetTensorInfo().m_Dimensions[2], convInfo.m_Stride.m_X);
        uint32_t c = GetNumSubmapChannels(convolution.GetInput(0).GetTensorInfo().m_Dimensions[3],
                                          convInfo.m_Stride.m_X, convInfo.m_Stride.m_Y, m_Capabilities);
        TensorInfo interleaveOutput = TensorInfo({ convolution.GetInput(0).GetTensorInfo().m_Dimensions[0], h, w, c },
                                                 convolution.GetInput(0).GetTensorInfo().m_DataType,
                                                 convolution.GetInput(0).GetTensorInfo().m_DataFormat,
                                                 convolution.GetInput(0).GetTensorInfo().m_QuantizationInfo);

        auto fusedPlePart = std::make_unique<FusedPlePart>(
            m_GraphOfParts.GeneratePartId(), convolution.GetInput(0).GetTensorInfo().m_Dimensions,
            interleaveOutput.m_Dimensions, convolution.GetInput(0).GetTensorInfo().m_QuantizationInfo,
            interleaveOutput.m_QuantizationInfo, command_stream::PleOperation::INTERLEAVE_2X2_2_2,
            utils::ShapeMultiplier{ { 1, convInfo.m_Stride.m_Y },
                                    { 1, convInfo.m_Stride.m_X },
                                    { convInfo.m_Stride.m_X * convInfo.m_Stride.m_Y } },
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
            std::set<uint32_t>{ convolution.GetId(), convolution.GetBias().GetId(), convolution.GetWeights().GetId() },
            GetCommandDataType(convolution.GetOutput(0).GetTensorInfo().m_DataType));
        parts.push_back(std::move(fusedPlePart.get()));
        m_GraphOfParts.m_Parts.push_back(std::move(fusedPlePart));

        // Pass interleaved Output as Input Tensor to subsequent McePart
        mcePartInputTensor = interleaveOutput;
    }
    else
    {
        // Pass default convolution Input Tensor
        mcePartInputTensor = convolution.GetInput(0).GetTensorInfo();
    }

    auto mcePart = std::make_unique<McePart>(
        m_GraphOfParts.GeneratePartId(), mcePartInputTensor.m_Dimensions,
        convolution.GetOutput(0).GetTensorInfo().m_Dimensions, mcePartInputTensor.m_QuantizationInfo,
        convolution.GetOutput(0).GetTensorInfo().m_QuantizationInfo, convolution.GetWeights().GetTensorInfo(),
        OverrideWeights(convolution.GetWeights().GetDataVector(), convolution.GetWeights().GetTensorInfo()),
        convolution.GetBias().GetTensorInfo(), GetDataVectorAs<int32_t, uint8_t>(convolution.GetBias().GetDataVector()),
        convolution.GetConvolutionInfo().m_Stride, convolution.GetConvolutionInfo().m_Padding.m_Top,
        convolution.GetConvolutionInfo().m_Padding.m_Left, command_stream::MceOperation::CONVOLUTION,
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
        std::set<uint32_t>{ convolution.GetId(), convolution.GetBias().GetId(), convolution.GetWeights().GetId() },
        GetCommandDataType(convolution.GetOutput(0).GetTensorInfo().m_DataType));

    parts.push_back(std::move(mcePart.get()));
    m_GraphOfParts.m_Parts.push_back(std::move(mcePart));
    ConnectParts(convolution, parts);
}

void NetworkToGraphOfPartsConverter::Visit(FullyConnected& fullyConnected)
{
    std::vector<BasePart*> parts;
    parts.reserve(1);
    const TensorInfo& inputTensorInfo     = fullyConnected.GetInput(0).GetTensorInfo();
    const std::set<uint32_t> operationIds = { fullyConnected.GetId(), fullyConnected.GetBias().GetId(),
                                              fullyConnected.GetWeights().GetId() };

    // However we interpret it as NHWCB so that it gets copied without conversion into SRAM.
    // We choose the smallest shape that will encompass all the data when it is interpreted in brick format.
    auto GetShapeContainingLinearElements = [](const TensorShape& brickGroupShape, uint32_t numElements) {
        const uint32_t brickGroupHeight           = brickGroupShape[1];
        const uint32_t brickGroupWidth            = brickGroupShape[2];
        const uint32_t brickGroupChannels         = brickGroupShape[3];
        const uint32_t patchHeight                = 4;
        const uint32_t patchWidth                 = 4;
        const uint32_t patchesPerBrickGroupHeight = brickGroupHeight / patchHeight;
        const uint32_t patchesPerBrickGroupWidth  = brickGroupWidth / patchWidth;
        const uint32_t patchesPerBrickGroup =
            patchesPerBrickGroupHeight * patchesPerBrickGroupWidth * brickGroupChannels;

        // If there are less than one bricks worth of elements then we can have a tensor with a single patch in XY
        // and up to 16 channels.
        // If there are between one and two bricks worth of elements then we can have a tensor with a column of two
        // patches in XY and 16 channels. Note we always need 16 channels in this case as the first brick is full.
        // If there are between two and four bricks worth of elements then we can have a tensor of a full brick group.
        // Again note we always need 16 channels in this case as the first two brick are full.
        // If we have more than four bricks of elements then we add brick groups behind the first one (i.e. stacking
        // along depth). The number of channels in the final brick group may be less than 16 if there is less
        // than a full bricks worth of elements in that final brick group.
        const uint32_t numPatches = DivRoundUp(numElements, patchWidth * patchHeight);
        const uint32_t reinterpretedWidth =
            numPatches <= brickGroupChannels * patchesPerBrickGroupHeight ? patchWidth : brickGroupWidth;
        const uint32_t reinterpretedHeight = numPatches <= brickGroupChannels ? patchHeight : brickGroupHeight;
        const uint32_t numFullBrickGroups  = numPatches / patchesPerBrickGroup;
        const uint32_t reinterpretedChannels =
            brickGroupChannels * numFullBrickGroups + std::min(brickGroupChannels, numPatches % patchesPerBrickGroup);
        return TensorShape{ 1, reinterpretedHeight, reinterpretedWidth, reinterpretedChannels };
    };

    const TensorShape reinterpretedInput =
        GetShapeContainingLinearElements(m_Capabilities.GetBrickGroupShape(), inputTensorInfo.m_Dimensions[3]);

    // The weight encoder for fully connected requires the input channel to be a multiple of 1024.
    // It is easier to make this adjustment here rather than the WeightEncoder itself, even though
    // it is less desirable.
    TensorInfo weightsInfo      = fullyConnected.GetWeights().GetTensorInfo();
    weightsInfo.m_Dimensions[2] = RoundUpToNearestMultiple(weightsInfo.m_Dimensions[2], g_WeightsChannelVecProd);
    std::vector<uint8_t> paddedWeightsData = fullyConnected.GetWeights().GetDataVector();
    paddedWeightsData.resize(TotalSizeBytes(weightsInfo),
                             static_cast<uint8_t>(weightsInfo.m_QuantizationInfo.GetZeroPoint()));

    auto fcPart = std::make_unique<FullyConnectedPart>(
        m_GraphOfParts.GeneratePartId(), reinterpretedInput, fullyConnected.GetOutput(0).GetTensorInfo().m_Dimensions,
        fullyConnected.GetInput(0).GetTensorInfo().m_QuantizationInfo,
        fullyConnected.GetOutput(0).GetTensorInfo().m_QuantizationInfo, weightsInfo, paddedWeightsData,
        fullyConnected.GetBias().GetTensorInfo(),
        GetDataVectorAs<int32_t, uint8_t>(fullyConnected.GetBias().GetDataVector()), m_EstimationOptions.value(),
        m_CompilationOptions, m_Capabilities, operationIds,
        GetCommandDataType(fullyConnected.GetOutput(0).GetTensorInfo().m_DataType));
    parts.push_back(fcPart.get());
    m_GraphOfParts.m_Parts.push_back(std::move(fcPart));

    ConnectParts(fullyConnected, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Pooling& pooling)
{
    std::vector<BasePart*> parts;
    const PoolingInfo& poolingInfo = pooling.GetPoolingInfo();
    if (poolingInfo == PoolingInfo{ 2, 2, 2, 2, poolingInfo.m_Padding, PoolingType::MAX })
    {
        auto poolingFusedPlePart = std::make_unique<FusedPlePart>(
            m_GraphOfParts.GeneratePartId(), pooling.GetInput(0).GetTensorInfo().m_Dimensions,
            pooling.GetOutput(0).GetTensorInfo().m_Dimensions, pooling.GetInput(0).GetTensorInfo().m_QuantizationInfo,
            pooling.GetOutput(0).GetTensorInfo().m_QuantizationInfo, command_stream::PleOperation::MAXPOOL_2X2_2_2,
            utils::ShapeMultiplier{
                { 1, pooling.GetPoolingInfo().m_PoolingStrideY }, { 1, pooling.GetPoolingInfo().m_PoolingStrideX }, 1 },
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities, std::set<uint32_t>{ pooling.GetId() },
            GetCommandDataType(pooling.GetOutput(0).GetTensorInfo().m_DataType));
        parts.push_back(std::move(poolingFusedPlePart.get()));
        m_GraphOfParts.m_Parts.push_back(std::move(poolingFusedPlePart));
        ConnectParts(pooling, parts);
    }
    else if (poolingInfo == PoolingInfo{ 3, 3, 1, 1, poolingInfo.m_Padding, PoolingType::AVG })
    {
        const std::vector<QuantizationInfo> inputQuantizations = {
            pooling.GetInput(0).GetTensorInfo().m_QuantizationInfo
        };
        const std::vector<TensorShape> inputShapes = { pooling.GetInput(0).GetTensorInfo().m_Dimensions };
        auto poolingStandalonePlePart              = std::make_unique<StandalonePlePart>(
            m_GraphOfParts.GeneratePartId(), inputShapes, pooling.GetOutput(0).GetTensorInfo().m_Dimensions,
            inputQuantizations, pooling.GetOutput(0).GetTensorInfo().m_QuantizationInfo,
            command_stream::PleOperation::AVGPOOL_3X3_1_1_UDMA, m_EstimationOptions.value(), m_CompilationOptions,
            m_Capabilities, std::set<uint32_t>{ pooling.GetId() },
            GetCommandDataType(pooling.GetOutput(0).GetTensorInfo().m_DataType));
        parts.push_back(std::move(poolingStandalonePlePart.get()));
        m_GraphOfParts.m_Parts.push_back(std::move(poolingStandalonePlePart));
        ConnectParts(pooling, parts);
    }
    else
    {
        throw InternalErrorException("Only PoolingType::MAX and AVG are supported at the moment");
    }
}

void NetworkToGraphOfPartsConverter::Visit(Reshape& reshape)
{
    std::vector<BasePart*> parts;
    auto reshapePart = std::make_unique<ReshapePart>(
        m_GraphOfParts.GeneratePartId(), reshape.GetInput(0).GetTensorInfo().m_Dimensions,
        reshape.GetOutput(0).GetTensorInfo().m_Dimensions, CompilerDataFormat::NHWC,
        reshape.GetOutput(0).GetTensorInfo().m_QuantizationInfo, std::set<uint32_t>{ reshape.GetId() },
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
    parts.push_back(std::move(reshapePart.get()));
    m_GraphOfParts.m_Parts.push_back(std::move(reshapePart));
    ConnectParts(reshape, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Addition& addition)
{
    std::vector<BasePart*> parts;

    const auto& inputInfo0 = addition.GetInput(0).GetTensorInfo();
    const auto& inputInfo1 = addition.GetInput(1).GetTensorInfo();
    const auto& outputInfo = addition.GetOutput(0).GetTensorInfo();

    const QuantizationInfo& quantInfoInput0 = inputInfo0.m_QuantizationInfo;
    const QuantizationInfo& quantInfoInput1 = inputInfo1.m_QuantizationInfo;
    const QuantizationInfo& quantInfoOutput = outputInfo.m_QuantizationInfo;

    bool isQuantInfoIdentical = (quantInfoInput0 == quantInfoInput1) && (quantInfoInput0 == quantInfoOutput);

    // use non-scaling PLE kernel if all quant info is identical for both inputs and output
    command_stream::PleOperation pleOp =
        isQuantInfoIdentical ? command_stream::PleOperation::ADDITION : command_stream::PleOperation::ADDITION_RESCALE;

    const std::vector<QuantizationInfo> inputQuantizations = { quantInfoInput0, quantInfoInput1 };
    const std::vector<TensorShape> inputShapes             = { addition.GetInput(0).GetTensorInfo().m_Dimensions,
                                                   addition.GetInput(1).GetTensorInfo().m_Dimensions };
    auto additionStandalonePlePart                         = std::make_unique<StandalonePlePart>(
        m_GraphOfParts.GeneratePartId(), inputShapes, addition.GetOutput(0).GetTensorInfo().m_Dimensions,
        inputQuantizations, addition.GetOutput(0).GetTensorInfo().m_QuantizationInfo, pleOp,
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities, std::set<uint32_t>{ addition.GetId() },
        GetCommandDataType(addition.GetOutput(0).GetTensorInfo().m_DataType));
    parts.push_back(std::move(additionStandalonePlePart.get()));
    m_GraphOfParts.m_Parts.push_back(std::move(additionStandalonePlePart));
    ConnectParts(addition, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Concatenation& concat)
{
    size_t numInputs     = concat.GetInputs().size();
    auto outputQuantInfo = concat.GetOutput(0).GetTensorInfo().m_QuantizationInfo;
    std::map<uint32_t, PartId> mcePartIds;

    // The ConcatPart assumes that all Inputs and the Output have the same quantization information.
    // If that is not the case, a requantize McePart is generated for any Inputs that are different to the Output.
    // Subsequently, all generated MceParts, as well as the ConcatPart are connected to the GraphOfParts.
    for (uint32_t i = 0; i < numInputs; i++)
    {
        Operand& inputOperand = concat.GetInput(i);
        if (inputOperand.GetTensorInfo().m_QuantizationInfo != outputQuantInfo)
        {
            auto mcePart = CreateIdentityMcePart(inputOperand.GetTensorInfo().m_Dimensions,
                                                 inputOperand.GetTensorInfo().m_QuantizationInfo, concat.GetId(),
                                                 GetCommandDataType(concat.GetOutput(0).GetTensorInfo().m_DataType),
                                                 m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

            // Add the connection to the GraphOfParts, then store the new PartId in a temporary map and then add the McePart to the GraphOfParts.
            m_GraphOfParts.AddConnection({ mcePart->GetPartId(), 0 }, { m_OperandToPart.at(&inputOperand)->GetPartId(),
                                                                        inputOperand.GetProducerOutputIndex() });
            mcePartIds[i] = mcePart->GetPartId();
            m_GraphOfParts.m_Parts.push_back(std::move(mcePart));
        }
    }

    // Create a ConcatPart for the GraphOfParts
    std::vector<TensorInfo> inputTensorsInfo;
    inputTensorsInfo.reserve(numInputs);
    for (uint32_t i = 0; i < numInputs; i++)
    {
        inputTensorsInfo.push_back(concat.GetInput(i).GetTensorInfo());
    }

    auto concatInfo = concat.GetConcatenationInfo();
    auto concatPart = std::make_unique<ConcatPart>(
        m_GraphOfParts.GeneratePartId(), inputTensorsInfo, concat.GetConcatenationInfo(), CompilerDataFormat::NHWCB,
        std::set<uint32_t>{ concat.GetId() }, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

    // Mark the ConcatPart Output for connection with any subsequent Parts.
    m_OperandToPart[&concat.GetOutput(0)] = concatPart.get();

    // Connect ConcatPart to the GraphOfParts. Loop through all Inputs of the ConcatPart and determine whether:
    // 1. There is a direct connection of ConcatPart with the preceding Part.
    // 2. There is a connection of ConcatPart with the respective requantise McePart.
    for (uint32_t i = 0; i < numInputs; i++)
    {
        Operand& inputOperand = concat.GetInput(i);
        if (mcePartIds.find(i) != mcePartIds.end())
        {
            m_GraphOfParts.AddConnection({ concatPart->GetPartId(), i }, { mcePartIds[i], 0 });
        }
        else
        {
            m_GraphOfParts.AddConnection(
                { concatPart->GetPartId(), i },
                { m_OperandToPart.at(&inputOperand)->GetPartId(), inputOperand.GetProducerOutputIndex() });
        }
    }

    // Add the ConcatPart to the GraphOfParts
    m_GraphOfParts.m_Parts.push_back(std::move(concatPart));
}

void NetworkToGraphOfPartsConverter::Visit(LeakyRelu& leakyRelu)
{
    std::vector<BasePart*> parts;
    auto leakyReluPart = std::make_unique<FusedPlePart>(
        m_GraphOfParts.GeneratePartId(), leakyRelu.GetInput(0).GetTensorInfo().m_Dimensions,
        leakyRelu.GetOutput(0).GetTensorInfo().m_Dimensions, leakyRelu.GetInput(0).GetTensorInfo().m_QuantizationInfo,
        leakyRelu.GetOutput(0).GetTensorInfo().m_QuantizationInfo, command_stream::PleOperation::LEAKY_RELU,
        g_IdentityShapeMultiplier, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
        std::set<uint32_t>{ leakyRelu.GetId() }, GetCommandDataType(leakyRelu.GetOutput(0).GetTensorInfo().m_DataType));
    parts.push_back(std::move(leakyReluPart.get()));
    m_GraphOfParts.m_Parts.push_back(std::move(leakyReluPart));
    ConnectParts(leakyRelu, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Sigmoid& sigmoid)
{
    std::vector<BasePart*> parts;
    auto sigmoidPart = std::make_unique<FusedPlePart>(
        m_GraphOfParts.GeneratePartId(), sigmoid.GetInput(0).GetTensorInfo().m_Dimensions,
        sigmoid.GetOutput(0).GetTensorInfo().m_Dimensions, sigmoid.GetInput(0).GetTensorInfo().m_QuantizationInfo,
        sigmoid.GetOutput(0).GetTensorInfo().m_QuantizationInfo, command_stream::PleOperation::SIGMOID,
        g_IdentityShapeMultiplier, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
        std::set<uint32_t>{ sigmoid.GetId() }, GetCommandDataType(sigmoid.GetOutput(0).GetTensorInfo().m_DataType));
    parts.push_back(std::move(sigmoidPart.get()));
    m_GraphOfParts.m_Parts.push_back(std::move(sigmoidPart));
    ConnectParts(sigmoid, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Tanh& tanh)
{
    // Note that Tanh and Sigmoid share the same PLE operation
    // The differences are:
    // (1) Input scaling factor
    // (2) Output quantization
    // The differences are handled later on when generating the command stream, based on the quantization info bounds.
    std::vector<BasePart*> parts;
    auto tanhPart = std::make_unique<FusedPlePart>(
        m_GraphOfParts.GeneratePartId(), tanh.GetInput(0).GetTensorInfo().m_Dimensions,
        tanh.GetOutput(0).GetTensorInfo().m_Dimensions, tanh.GetInput(0).GetTensorInfo().m_QuantizationInfo,
        tanh.GetOutput(0).GetTensorInfo().m_QuantizationInfo, command_stream::PleOperation::SIGMOID,
        g_IdentityShapeMultiplier, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
        std::set<uint32_t>{ tanh.GetId() }, GetCommandDataType(tanh.GetOutput(0).GetTensorInfo().m_DataType));
    parts.push_back(std::move(tanhPart.get()));
    m_GraphOfParts.m_Parts.push_back(std::move(tanhPart));
    ConnectParts(tanh, parts);
}

void NetworkToGraphOfPartsConverter::Visit(MeanXy& meanxy)
{
    std::vector<BasePart*> parts;
    ShapeMultiplier shapeMultiplier = { 1, 1, 1 };
    command_stream::PleOperation cmd_stream;
    if (meanxy.GetInput(0).GetTensorInfo().m_Dimensions[1] == 7)
    {
        cmd_stream = command_stream::PleOperation::MEAN_XY_7X7;
    }
    else
    {
        cmd_stream = command_stream::PleOperation::MEAN_XY_8X8;
    }
    auto meanxyPart = std::make_unique<FusedPlePart>(
        m_GraphOfParts.GeneratePartId(), meanxy.GetInput(0).GetTensorInfo().m_Dimensions,
        meanxy.GetOutput(0).GetTensorInfo().m_Dimensions, meanxy.GetInput(0).GetTensorInfo().m_QuantizationInfo,
        meanxy.GetOutput(0).GetTensorInfo().m_QuantizationInfo, cmd_stream, shapeMultiplier,
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities, std::set<uint32_t>{ meanxy.GetId() },
        GetCommandDataType(meanxy.GetOutput(0).GetTensorInfo().m_DataType));
    parts.push_back(std::move(meanxyPart.get()));
    m_GraphOfParts.m_Parts.push_back(std::move(meanxyPart));
    ConnectParts(meanxy, parts);
}

void NetworkToGraphOfPartsConverter::Visit(EstimateOnly& estimateOnly)
{
    std::vector<BasePart*> parts;
    // Convert from DataFormat to CompilerFormat needed for the EstimateOnly.
    CompilerDataFormat compilerDataFormat =
        ConvertExternalToCompilerDataFormat(estimateOnly.GetEstimateOnlyInfo().m_OutputInfos[0].m_DataFormat);
    auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
        m_GraphOfParts.GeneratePartId(), estimateOnly.GetEstimateOnlyInfo().m_ReasonForEstimateOnly,
        estimateOnly.GetEstimateOnlyInfo().m_OutputInfos, estimateOnly.GetEstimateOnlyInfo().m_OutputInfos,
        compilerDataFormat, std::set<uint32_t>{ estimateOnly.GetId() }, m_EstimationOptions.value(),
        m_CompilationOptions, m_Capabilities);

    parts.push_back(std::move(estimateOnlyPart.get()));
    m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
    ConnectParts(estimateOnly, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Resize& resize)
{
    const TensorInfo& inputInfo   = resize.GetInput(0).GetTensorInfo();
    const TensorShape& inputShape = inputInfo.m_Dimensions;
    const TensorInfo& outputInfo  = resize.GetOutput(0).GetTensorInfo();
    const ResizeInfo& resizeInfo  = resize.GetResizeInfo();

    // This is checked in IsSupported but let's make sure that here it using the only
    // upscale factor supported which is 2U for height and width.
    const uint32_t upscaleFactorHeight = DivRoundUp(GetHeight(outputInfo.m_Dimensions), GetHeight(inputShape));
    const uint32_t upscaleFactorWidth  = DivRoundUp(GetWidth(outputInfo.m_Dimensions), GetWidth(inputShape));
    ETHOSN_UNUSED(upscaleFactorWidth);
    assert((upscaleFactorHeight == upscaleFactorWidth) && (upscaleFactorHeight == 2U));

    McePart::ConstructionParams params(m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
    params.m_Id                     = m_GraphOfParts.GeneratePartId();
    params.m_InputTensorShape       = inputShape;
    params.m_OutputTensorShape      = outputInfo.m_Dimensions;
    params.m_InputQuantizationInfo  = inputInfo.m_QuantizationInfo;
    params.m_OutputQuantizationInfo = outputInfo.m_QuantizationInfo;
    const uint32_t numIfm           = inputShape[3];
    const float weightScale         = 0.5f;
    params.m_WeightsInfo   = { { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale } };
    params.m_WeightsData   = std::vector<uint8_t>(1 * 1 * 1 * numIfm, 2);
    const float biasScale  = weightScale * inputInfo.m_QuantizationInfo.GetScale();
    params.m_BiasInfo      = { { 1, 1, 1, numIfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };
    params.m_BiasData      = std::vector<int32_t>(numIfm, 0);
    params.m_Op            = command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    params.m_OperationIds  = std::set<uint32_t>{ resize.GetId() };
    params.m_DataType      = GetCommandDataType(outputInfo.m_DataType);
    params.m_UpscaleFactor = upscaleFactorHeight;
    params.m_UpsampleType  = ConvertResizeAlgorithmToCommand(resizeInfo.m_Algo);
    auto mcePart           = std::make_unique<McePart>(std::move(params));

    std::vector<BasePart*> parts;
    parts.push_back(mcePart.get());
    m_GraphOfParts.m_Parts.push_back(std::move(mcePart));
    ConnectParts(resize, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Relu& relu)
{
    ReluInfo info               = relu.GetReluInfo();
    const Operand& inputOperand = relu.GetInput(0);

    std::vector<BasePart*> parts;

    auto iterator = m_OperandToPart.find(&inputOperand);
    assert(iterator != m_OperandToPart.end());
    BasePart* inputPart = iterator->second;
    assert(inputPart);

    // Multiple cases. Mce -> Relu. We need to update the relu bounds in the mce op
    // not mce -> Relu. We need to insert an identity mce operation with new relu bounds
    if (!inputPart->HasActivationBounds())
    {
        std::unique_ptr<McePart> mcePart = CreateIdentityMcePart(
            inputOperand.GetTensorInfo().m_Dimensions, inputOperand.GetTensorInfo().m_QuantizationInfo, relu.GetId(),
            GetCommandDataType(inputOperand.GetTensorInfo().m_DataType), m_EstimationOptions.value(),
            m_CompilationOptions, m_Capabilities);

        inputPart = mcePart.get();
        parts.push_back(mcePart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(mcePart));
        ConnectParts(relu, parts);
    }

    // If the input to the relu has activations we need to modify them
    inputPart->ModifyActivationBounds(info.m_LowerBound, info.m_UpperBound);
    m_OperandToPart[&relu.GetOutput(0)] = inputPart;
}

std::vector<uint8_t> NetworkToGraphOfPartsConverter::OverrideWeights(const std::vector<uint8_t>& userWeights,
                                                                     const TensorInfo& weightsInfo) const
{
    if (m_EstimationOptions.has_value() && m_EstimationOptions.value().m_UseWeightCompressionOverride)
    {
        std::vector<uint8_t> dummyWeightData =
            GenerateCompressibleData(userWeights.size(), m_EstimationOptions.value().m_WeightCompressionSaving,
                                     weightsInfo.m_QuantizationInfo.GetZeroPoint());
        return dummyWeightData;
    }
    else
    {
        return userWeights;
    }
}

GraphOfParts NetworkToGraphOfPartsConverter::ReleaseGraphOfParts()
{
    return std::move(m_GraphOfParts);
}

void NetworkToGraphOfPartsConverter::ConnectParts(Operation& operation, std::vector<BasePart*>& m_Part)
{
    // This function currently supports Operations with no/single Output.
    assert(operation.GetOutputs().size() <= 1);

    // Loop through all parts in the vector of BaseParts and connect them together.
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_Part.size()) - 1; i++)
    {
        m_GraphOfParts.AddConnection({ m_Part[i + 1]->GetPartId(), 0 }, { m_Part[i]->GetPartId(), 0 });
    }

    uint32_t i = 0;
    // Loop through all input Operands of current Operation and connect first Part in vector of BaseParts with
    // the preceding Part that has the same Operand as output.
    for (const Operand* op : operation.GetInputs())
    {
        m_GraphOfParts.AddConnection({ m_Part.front()->GetPartId(), i },
                                     { m_OperandToPart.at(op)->GetPartId(), op->GetProducerOutputIndex() });
        i += 1;
    }

    // Check if current operation has outputs and if so mark them for connection with the subsequent operation.
    if (operation.GetOutputs().size() > 0)
    {
        m_OperandToPart[&operation.GetOutput(0)] = m_Part.back();
    }
}

}    // namespace support_library
}    // namespace ethosn
