//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "NetworkToGraphOfPartsConverter.hpp"

#include "ConcatPart.hpp"
#include "ConstantPart.hpp"
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
                                                                               const QuantizationInfo& outputQuantInfo,
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
    params.m_OutputQuantizationInfo = outputQuantInfo;
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
    , m_Queries(capabilities.GetData())
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
    parts.push_back(inputPart.get());
    m_GraphOfParts.m_Parts.push_back(std::move(inputPart));
    ConnectParts(input, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Output& output)
{
    std::vector<BasePart*> parts;
    CompilerDataFormat compilerDataFormat = ConvertExternalToCompilerDataFormat(output.GetTensorInfo().m_DataFormat);

    // Note that we return the ID of the *producer* that feeds in to the output node, not the ID of the output
    // node itself. This is for consistency when we start splitting the network and need to identify network outputs
    // that do not have their own unique node. See documentation on InputBufferInfo struct in Support.hpp for details.
    auto outputPart = std::make_unique<OutputPart>(
        m_GraphOfParts.GeneratePartId(), output.GetTensorInfo().m_Dimensions, compilerDataFormat,
        output.GetTensorInfo().m_QuantizationInfo, std::set<uint32_t>{ output.GetInput(0).GetProducer().GetId() },
        output.GetInput(0).GetProducerOutputIndex(), m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
    parts.push_back(outputPart.get());
    m_GraphOfParts.m_Parts.push_back(std::move(outputPart));
    ConnectParts(output, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Constant& constant)
{
    if (constant.GetInputs().size() == 0 && constant.GetOutputs().size() == 1 &&
        constant.GetOutput(0).GetConsumers().size() == 0)
    {
        // Weights/Bias are Constant Operations in the Network, but are typically not connected to other Operations and so will
        // never be relevant in the GraphOfParts. Creating a Part for constant weights is not supported by the
        // ConstantPart code anyway, so we skip these. This also makes the GraphOfParts simpler.
        return;
    }

    std::vector<BasePart*> parts;
    CompilerDataFormat compilerDataFormat = ConvertExternalToCompilerDataFormat(constant.GetTensorInfo().m_DataFormat);
    auto constPart                        = std::make_unique<ConstantPart>(
        m_GraphOfParts.GeneratePartId(), constant.GetTensorInfo().m_Dimensions, compilerDataFormat,
        constant.GetTensorInfo().m_QuantizationInfo, std::set<uint32_t>{ constant.GetId() },
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
    parts.push_back(constPart.get());
    m_GraphOfParts.m_Parts.push_back(std::move(constPart));
    ConnectParts(constant, parts);
}

void NetworkToGraphOfPartsConverter::Visit(DepthwiseConvolution& depthwise)
{
    std::vector<BasePart*> parts;
    auto convInfo = depthwise.GetConvolutionInfo();

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    char reason[1024];
    const SupportedLevel supportedLevel = m_Queries.IsDepthwiseConvolutionSupported(
        depthwise.GetBias().GetTensorInfo(), depthwise.GetWeights().GetTensorInfo(), convInfo,
        depthwise.GetInput(0).GetTensorInfo(), nullptr, reason, sizeof(reason));

    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        const TensorInfo& outputInfo          = depthwise.GetOutput(0).GetTensorInfo();
        const std::set<uint32_t> operationIds = { depthwise.GetId(), depthwise.GetBias().GetId(),
                                                  depthwise.GetWeights().GetId() };

        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ depthwise.GetInput(0).GetTensorInfo() },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            operationIds, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
    }
    else
    {
        TensorInfo mceOperationInput        = depthwise.GetInput(0).GetTensorInfo();
        TensorShape uninterleavedInputShape = depthwise.GetInput(0).GetTensorInfo().m_Dimensions;

        // Check if it is a strided depthwise and add a FusedPlePart.
        if (convInfo.m_Stride.m_X > 1 || convInfo.m_Stride.m_Y > 1)
        {
            // Create additional layer before strided convolution
            // Only supports stride 2x2 for now
            assert(convInfo.m_Stride.m_X == 2 && convInfo.m_Stride.m_Y == 2);

            uint32_t h = DivRoundUp(depthwise.GetInput(0).GetTensorInfo().m_Dimensions[1], convInfo.m_Stride.m_Y);
            uint32_t w = DivRoundUp(depthwise.GetInput(0).GetTensorInfo().m_Dimensions[2], convInfo.m_Stride.m_X);
            uint32_t c = GetNumSubmapChannels(depthwise.GetInput(0).GetTensorInfo().m_Dimensions[3],
                                              convInfo.m_Stride.m_X, convInfo.m_Stride.m_Y, m_Capabilities);

            mceOperationInput = TensorInfo({ depthwise.GetInput(0).GetTensorInfo().m_Dimensions[0], h, w, c },
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

            parts.push_back(fusedPlePart.get());
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
            OverrideWeights(depthwise.GetWeights().GetDataVector(), weightsTensorInfo),
            depthwise.GetBias().GetTensorInfo(), GetDataVectorAs<int32_t, uint8_t>(depthwise.GetBias().GetDataVector()),
            depthwise.GetConvolutionInfo().m_Stride, depthwise.GetConvolutionInfo().m_Padding.m_Top,
            depthwise.GetConvolutionInfo().m_Padding.m_Left, operation, m_EstimationOptions.value(),
            m_CompilationOptions, m_Capabilities,
            std::set<uint32_t>{ depthwise.GetId(), depthwise.GetBias().GetId(), depthwise.GetWeights().GetId() },
            GetCommandDataType(depthwise.GetOutput(0).GetTensorInfo().m_DataType));

        if (convInfo.m_Stride.m_X > 1 || convInfo.m_Stride.m_Y > 1)
        {
            mcePart->setUninterleavedInputShape(uninterleavedInputShape);
        }

        parts.push_back(mcePart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(mcePart));
    }

    ConnectParts(depthwise, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Convolution& convolution)
{
    std::vector<BasePart*> parts;
    auto convInfo = convolution.GetConvolutionInfo();
    TensorInfo mcePartInputTensor;

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    char reason[1024];
    const SupportedLevel supportedLevel = m_Queries.IsConvolutionSupported(
        convolution.GetBias().GetTensorInfo(), convolution.GetWeights().GetTensorInfo(), convInfo,
        convolution.GetInput(0).GetTensorInfo(), nullptr, reason, sizeof(reason));

    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        const TensorInfo& outputInfo          = convolution.GetOutput(0).GetTensorInfo();
        const std::set<uint32_t> operationIds = { convolution.GetId(), convolution.GetBias().GetId(),
                                                  convolution.GetWeights().GetId() };

        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ convolution.GetInput(0).GetTensorInfo() },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            operationIds, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
    }
    else
    {
        TensorShape uninterleavedInputShape = convolution.GetInput(0).GetTensorInfo().m_Dimensions;

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
            TensorInfo interleaveOutput =
                TensorInfo({ convolution.GetInput(0).GetTensorInfo().m_Dimensions[0], h, w, c },
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
                std::set<uint32_t>{ convolution.GetId(), convolution.GetBias().GetId(),
                                    convolution.GetWeights().GetId() },
                GetCommandDataType(convolution.GetOutput(0).GetTensorInfo().m_DataType));
            parts.push_back(fusedPlePart.get());
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
            convolution.GetBias().GetTensorInfo(),
            GetDataVectorAs<int32_t, uint8_t>(convolution.GetBias().GetDataVector()),
            convolution.GetConvolutionInfo().m_Stride, convolution.GetConvolutionInfo().m_Padding.m_Top,
            convolution.GetConvolutionInfo().m_Padding.m_Left, command_stream::MceOperation::CONVOLUTION,
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
            std::set<uint32_t>{ convolution.GetId(), convolution.GetBias().GetId(), convolution.GetWeights().GetId() },
            GetCommandDataType(convolution.GetOutput(0).GetTensorInfo().m_DataType));

        if (convInfo.m_Stride.m_X > 1 || convInfo.m_Stride.m_Y > 1)
        {
            mcePart->setUninterleavedInputShape(uninterleavedInputShape);
        }

        parts.push_back(mcePart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(mcePart));
    }

    ConnectParts(convolution, parts);
}

void NetworkToGraphOfPartsConverter::Visit(FullyConnected& fullyConnected)
{
    std::vector<BasePart*> parts;
    parts.reserve(1);
    const TensorInfo& inputTensorInfo     = fullyConnected.GetInput(0).GetTensorInfo();
    const std::set<uint32_t> operationIds = { fullyConnected.GetId(), fullyConnected.GetBias().GetId(),
                                              fullyConnected.GetWeights().GetId() };

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    char reason[1024];
    const SupportedLevel supportedLevel = m_Queries.IsFullyConnectedSupported(
        fullyConnected.GetBias().GetTensorInfo(), fullyConnected.GetWeights().GetTensorInfo(),
        fullyConnected.GetFullyConnectedInfo(), inputTensorInfo, nullptr, reason, sizeof(reason));

    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        const TensorInfo& outputTensorInfo = fullyConnected.GetOutput(0).GetTensorInfo();

        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ inputTensorInfo },
            std::vector<TensorInfo>{ outputTensorInfo },
            ConvertExternalToCompilerDataFormat(outputTensorInfo.m_DataFormat), operationIds,
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
    }
    else
    {

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
            const uint32_t reinterpretedHeight   = numPatches <= brickGroupChannels ? patchHeight : brickGroupHeight;
            const uint32_t numFullBrickGroups    = numPatches / patchesPerBrickGroup;
            const uint32_t reinterpretedChannels = brickGroupChannels * numFullBrickGroups +
                                                   std::min(brickGroupChannels, numPatches % patchesPerBrickGroup);
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
            m_GraphOfParts.GeneratePartId(), inputTensorInfo.m_Dimensions, reinterpretedInput,
            fullyConnected.GetOutput(0).GetTensorInfo().m_Dimensions,
            fullyConnected.GetInput(0).GetTensorInfo().m_QuantizationInfo,
            fullyConnected.GetOutput(0).GetTensorInfo().m_QuantizationInfo, weightsInfo, paddedWeightsData,
            fullyConnected.GetBias().GetTensorInfo(),
            GetDataVectorAs<int32_t, uint8_t>(fullyConnected.GetBias().GetDataVector()), m_EstimationOptions.value(),
            m_CompilationOptions, m_Capabilities, operationIds,
            GetCommandDataType(fullyConnected.GetOutput(0).GetTensorInfo().m_DataType));
        parts.push_back(fcPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(fcPart));
    }

    ConnectParts(fullyConnected, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Pooling& pooling)
{
    std::vector<BasePart*> parts;

    const uint32_t inputHeight = pooling.GetInput(0).GetTensorInfo().m_Dimensions[1];
    const uint32_t inputWidth  = pooling.GetInput(0).GetTensorInfo().m_Dimensions[2];

    const bool isInputEven = (((inputWidth % 2U) == 0) && ((inputHeight % 2U) == 0));
    const bool isInputOdd  = (((inputWidth % 2U) != 0) && ((inputHeight % 2U) != 0));

    const PoolingInfo& poolingInfo      = pooling.GetPoolingInfo();
    const PoolingInfo poolingInfoMeanXy = {
        inputWidth,
        inputHeight,
        poolingInfo.m_PoolingStrideX,
        poolingInfo.m_PoolingStrideY,
        Padding{ 0, 0, 0, 0 },
        PoolingType::AVG,
    };

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    char reason[1024];
    const SupportedLevel supportedLevel =
        m_Queries.IsPoolingSupported(poolingInfo, pooling.GetInput(0).GetTensorInfo(), nullptr, reason, sizeof(reason));

    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        const TensorInfo& outputInfo = pooling.GetOutput(0).GetTensorInfo();

        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ pooling.GetInput(0).GetTensorInfo() },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            std::set<uint32_t>{ pooling.GetId() }, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
    }
    else
    {
        auto createPoolingPart = [&](command_stream::PleOperation op) {
            auto poolingFusedPlePart = std::make_unique<FusedPlePart>(
                m_GraphOfParts.GeneratePartId(), pooling.GetInput(0).GetTensorInfo().m_Dimensions,
                pooling.GetOutput(0).GetTensorInfo().m_Dimensions,
                pooling.GetInput(0).GetTensorInfo().m_QuantizationInfo,
                pooling.GetOutput(0).GetTensorInfo().m_QuantizationInfo, op,
                utils::ShapeMultiplier{ { 1, poolingInfo.m_PoolingStrideY }, { 1, poolingInfo.m_PoolingStrideX }, 1 },
                m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                std::set<uint32_t>{ pooling.GetId() },
                GetCommandDataType(pooling.GetOutput(0).GetTensorInfo().m_DataType));
            parts.push_back(poolingFusedPlePart.get());
            m_GraphOfParts.m_Parts.push_back(std::move(poolingFusedPlePart));
        };

        // Pooling Visitor decoder, creating appropriate FusedPle Parts for supported Operations.
        // Handle MeanXy Operations with 7x7, 8x8 sizes.
        if (inputHeight == 7U && inputWidth == 7U && poolingInfo == poolingInfoMeanXy)
        {
            createPoolingPart(command_stream::PleOperation::MEAN_XY_7X7);
        }
        else if (inputHeight == 8U && inputWidth == 8U && poolingInfo == poolingInfoMeanXy)
        {
            createPoolingPart(command_stream::PleOperation::MEAN_XY_8X8);
        }
        // Handle MaxPool Operations of supported kernel sizes, strides and padding.
        else if (poolingInfo == PoolingInfo{ 2, 2, 2, 2, poolingInfo.m_Padding, PoolingType::MAX })
        {
            createPoolingPart(command_stream::PleOperation::MAXPOOL_2X2_2_2);
        }
        else if (isInputOdd && poolingInfo == PoolingInfo{ 3, 3, 2, 2, poolingInfo.m_Padding, PoolingType::MAX })
        {
            createPoolingPart(command_stream::PleOperation::MAXPOOL_3X3_2_2_ODD);
        }
        else if (isInputEven && poolingInfo == PoolingInfo{ 3, 3, 2, 2, poolingInfo.m_Padding, PoolingType::MAX })
        {
            createPoolingPart(command_stream::PleOperation::MAXPOOL_3X3_2_2_EVEN);
        }
        else if (poolingInfo == PoolingInfo{ 1, 1, 2, 2, poolingInfo.m_Padding, PoolingType::MAX })
        {
            createPoolingPart(command_stream::PleOperation::DOWNSAMPLE_2X2);
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
            parts.push_back(poolingStandalonePlePart.get());
            m_GraphOfParts.m_Parts.push_back(std::move(poolingStandalonePlePart));
        }
        else
        {
            throw InternalErrorException(
                "Only PoolingType::MAX 2x2_2_2, 3x3_2_2_even/odd and PoolingType::AVG 3x3_1_1, "
                "7x7_2_2, 8x8_2_2 are supported at the moment");
        }
    }

    ConnectParts(pooling, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Reshape& reshape)
{
    std::vector<BasePart*> parts;
    auto reshapePart = std::make_unique<ReshapePart>(
        m_GraphOfParts.GeneratePartId(), reshape.GetInput(0).GetTensorInfo().m_Dimensions,
        reshape.GetOutput(0).GetTensorInfo().m_Dimensions, CompilerDataFormat::NHWC,
        reshape.GetOutput(0).GetTensorInfo().m_QuantizationInfo, std::set<uint32_t>{ reshape.GetId() },
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
    parts.push_back(reshapePart.get());
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

    char reason[1024];

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    const SupportedLevel supportedLevel =
        m_Queries.IsAdditionSupported(inputInfo0, inputInfo1, quantInfoOutput, nullptr, reason, sizeof(reason));

    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ inputInfo0, inputInfo1 },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            std::set<uint32_t>{ addition.GetId() }, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
    }
    else
    {
        bool isQuantInfoIdentical = (quantInfoInput0 == quantInfoInput1) && (quantInfoInput0 == quantInfoOutput);

        // use non-scaling PLE kernel if all quant info is identical for both inputs and output
        command_stream::PleOperation pleOp = isQuantInfoIdentical ? command_stream::PleOperation::ADDITION
                                                                  : command_stream::PleOperation::ADDITION_RESCALE;

        const std::vector<QuantizationInfo> inputQuantizations = { quantInfoInput0, quantInfoInput1 };
        const std::vector<TensorShape> inputShapes             = { addition.GetInput(0).GetTensorInfo().m_Dimensions,
                                                       addition.GetInput(1).GetTensorInfo().m_Dimensions };
        auto additionStandalonePlePart                         = std::make_unique<StandalonePlePart>(
            m_GraphOfParts.GeneratePartId(), inputShapes, addition.GetOutput(0).GetTensorInfo().m_Dimensions,
            inputQuantizations, addition.GetOutput(0).GetTensorInfo().m_QuantizationInfo, pleOp,
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities, std::set<uint32_t>{ addition.GetId() },
            GetCommandDataType(addition.GetOutput(0).GetTensorInfo().m_DataType));
        parts.push_back(additionStandalonePlePart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(additionStandalonePlePart));
    }

    ConnectParts(addition, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Concatenation& concat)
{
    size_t numInputs     = concat.GetInputs().size();
    auto outputQuantInfo = concat.GetOutput(0).GetTensorInfo().m_QuantizationInfo;
    std::map<uint32_t, PartId> mcePartIds;

    // Create a ConcatPart for the GraphOfParts
    std::vector<TensorInfo> inputTensorsInfo;
    inputTensorsInfo.reserve(numInputs);
    for (uint32_t i = 0; i < numInputs; i++)
    {
        inputTensorsInfo.push_back(concat.GetInput(i).GetTensorInfo());
    }

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    char reason[1024];
    const SupportedLevel supportedLevel = m_Queries.IsConcatenationSupported(
        inputTensorsInfo, concat.GetConcatenationInfo(), nullptr, reason, sizeof(reason));

    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        std::vector<BasePart*> parts;
        const TensorInfo& outputInfo = concat.GetOutput(0).GetTensorInfo();

        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, inputTensorsInfo, std::vector<TensorInfo>{ outputInfo },
            ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat), std::set<uint32_t>{ concat.GetId() },
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
        ConnectParts(concat, parts);
    }
    else
    {
        // The ConcatPart assumes that all Inputs and the Output have the same quantization information.
        // If that is not the case, a requantize McePart is generated for any Inputs that are different to the Output.
        // Subsequently, all generated MceParts, as well as the ConcatPart are connected to the GraphOfParts.
        for (uint32_t i = 0; i < numInputs; i++)
        {
            Operand& inputOperand = concat.GetInput(i);
            if (inputOperand.GetTensorInfo().m_QuantizationInfo != outputQuantInfo)
            {
                auto mcePart = CreateIdentityMcePart(
                    inputOperand.GetTensorInfo().m_Dimensions, inputOperand.GetTensorInfo().m_QuantizationInfo,
                    outputQuantInfo, concat.GetId(), GetCommandDataType(concat.GetOutput(0).GetTensorInfo().m_DataType),
                    m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

                // Add the connection to the GraphOfParts, then store the new PartId in a temporary map and then add the McePart to the GraphOfParts.
                m_GraphOfParts.AddConnection(
                    { mcePart->GetPartId(), 0 },
                    { m_OperandToPart.at(&inputOperand)->GetPartId(), inputOperand.GetProducerOutputIndex() });
                mcePartIds[i] = mcePart->GetPartId();
                m_GraphOfParts.m_Parts.push_back(std::move(mcePart));
            }
        }

        auto concatInfo = concat.GetConcatenationInfo();

        // Figure out if we need to use NHWC or if we can get away with NHWCB (which should be more efficient).
        // We can use NHWCB if the dimensions along the concat axis are all multiples of the brick group size, so
        // that the DMA is capable of placing the tensors correctly in DRAM.
        CompilerDataFormat format = CompilerDataFormat::NHWCB;
        for (uint32_t i = 0; i < numInputs; ++i)
        {
            if (concat.GetInput(i).GetTensorInfo().m_Dimensions[concatInfo.m_Axis] %
                    m_Capabilities.GetBrickGroupShape()[concatInfo.m_Axis] !=
                0)
            {
                format = CompilerDataFormat::NHWC;
                break;
            }
        }

        auto concatPart = std::make_unique<ConcatPart>(
            m_GraphOfParts.GeneratePartId(), inputTensorsInfo, concat.GetConcatenationInfo(), format,
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
}

void NetworkToGraphOfPartsConverter::Visit(Requantize& requantize)
{
    std::vector<BasePart*> parts;

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    char reason[1024];
    const SupportedLevel supportedLevel = m_Queries.IsRequantizeSupported(
        requantize.GetRequantizeInfo(), requantize.GetInput(0).GetTensorInfo(), nullptr, reason, sizeof(reason));

    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        const TensorInfo& outputInfo = requantize.GetOutput(0).GetTensorInfo();

        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ requantize.GetInput(0).GetTensorInfo() },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            std::set<uint32_t>{ requantize.GetId() }, m_EstimationOptions.value(), m_CompilationOptions,
            m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
        ConnectParts(requantize, parts);
    }
    else
    {
        auto inputQuantInfo  = requantize.GetInput(0).GetTensorInfo().m_QuantizationInfo;
        auto outputQuantInfo = requantize.GetOutput(0).GetTensorInfo().m_QuantizationInfo;

        // If input and output quantizations are different, an McePart is added to the GraphOfParts to perform requantization,
        // otherwise the requantize operation is optimized out (no requantization needed)
        Operand& inputOperand = requantize.GetInput(0);
        if (inputQuantInfo != outputQuantInfo)
        {
            auto mcePart = CreateIdentityMcePart(inputOperand.GetTensorInfo().m_Dimensions, inputQuantInfo,
                                                 outputQuantInfo, requantize.GetId(),
                                                 GetCommandDataType(requantize.GetOutput(0).GetTensorInfo().m_DataType),
                                                 m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

            parts.push_back(mcePart.get());
            m_GraphOfParts.m_Parts.push_back(std::move(mcePart));
            ConnectParts(requantize, parts);
        }
        else
        {
            ConnectNoOp(requantize);
        }
    }
}

void NetworkToGraphOfPartsConverter::Visit(LeakyRelu& leakyRelu)
{
    std::vector<BasePart*> parts;
    char reason[1024];

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    const SupportedLevel supportedLevel = m_Queries.IsLeakyReluSupported(
        leakyRelu.GetLeakyReluInfo(), leakyRelu.GetInput(0).GetTensorInfo(), nullptr, reason, sizeof(reason));

    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        const TensorInfo& outputInfo = leakyRelu.GetOutput(0).GetTensorInfo();

        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ leakyRelu.GetInput(0).GetTensorInfo() },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            std::set<uint32_t>{ leakyRelu.GetId() }, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
    }
    else
    {
        auto leakyReluPart = std::make_unique<FusedPlePart>(
            m_GraphOfParts.GeneratePartId(), leakyRelu.GetInput(0).GetTensorInfo().m_Dimensions,
            leakyRelu.GetOutput(0).GetTensorInfo().m_Dimensions,
            leakyRelu.GetInput(0).GetTensorInfo().m_QuantizationInfo,
            leakyRelu.GetOutput(0).GetTensorInfo().m_QuantizationInfo, command_stream::PleOperation::LEAKY_RELU,
            g_IdentityShapeMultiplier, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
            std::set<uint32_t>{ leakyRelu.GetId() },
            GetCommandDataType(leakyRelu.GetOutput(0).GetTensorInfo().m_DataType));
        parts.push_back(leakyReluPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(leakyReluPart));
    }

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
    parts.push_back(sigmoidPart.get());
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
    parts.push_back(tanhPart.get());
    m_GraphOfParts.m_Parts.push_back(std::move(tanhPart));
    ConnectParts(tanh, parts);
}

void NetworkToGraphOfPartsConverter::Visit(MeanXy& meanxy)
{
    std::vector<BasePart*> parts;
    ShapeMultiplier shapeMultiplier = { 1, 1, 1 };
    command_stream::PleOperation pleOperation;
    if (meanxy.GetInput(0).GetTensorInfo().m_Dimensions[1] == 7)
    {
        pleOperation = command_stream::PleOperation::MEAN_XY_7X7;
    }
    else
    {
        pleOperation = command_stream::PleOperation::MEAN_XY_8X8;
    }
    auto meanxyPart = std::make_unique<FusedPlePart>(
        m_GraphOfParts.GeneratePartId(), meanxy.GetInput(0).GetTensorInfo().m_Dimensions,
        meanxy.GetOutput(0).GetTensorInfo().m_Dimensions, meanxy.GetInput(0).GetTensorInfo().m_QuantizationInfo,
        meanxy.GetOutput(0).GetTensorInfo().m_QuantizationInfo, pleOperation, shapeMultiplier,
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities, std::set<uint32_t>{ meanxy.GetId() },
        GetCommandDataType(meanxy.GetOutput(0).GetTensorInfo().m_DataType));
    parts.push_back(meanxyPart.get());
    m_GraphOfParts.m_Parts.push_back(std::move(meanxyPart));
    ConnectParts(meanxy, parts);
}

void NetworkToGraphOfPartsConverter::Visit(EstimateOnly& estimateOnly)
{
    std::vector<BasePart*> parts;
    // Convert from DataFormat to CompilerFormat needed for the EstimateOnly.
    CompilerDataFormat compilerDataFormat =
        ConvertExternalToCompilerDataFormat(estimateOnly.GetEstimateOnlyInfo().m_OutputInfos[0].m_DataFormat);
    std::vector<TensorInfo> inputInfos;
    for (const Operand* input : estimateOnly.GetInputs())
    {
        inputInfos.push_back(input->GetTensorInfo());
    }

    auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
        m_GraphOfParts.GeneratePartId(), estimateOnly.GetEstimateOnlyInfo().m_ReasonForEstimateOnly, inputInfos,
        estimateOnly.GetEstimateOnlyInfo().m_OutputInfos, compilerDataFormat,
        std::set<uint32_t>{ estimateOnly.GetId() }, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

    parts.push_back(estimateOnlyPart.get());
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
            inputOperand.GetTensorInfo().m_Dimensions, inputOperand.GetTensorInfo().m_QuantizationInfo,
            inputOperand.GetTensorInfo().m_QuantizationInfo, relu.GetId(),
            GetCommandDataType(inputOperand.GetTensorInfo().m_DataType), m_EstimationOptions.value(),
            m_CompilationOptions, m_Capabilities);

        inputPart = mcePart.get();
        parts.push_back(mcePart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(mcePart));
        ConnectParts(relu, parts);
    }

    // If the input to the relu has activations we need to modify them
    inputPart->ModifyActivationBounds(info.m_LowerBound, info.m_UpperBound);
    inputPart->AddOperationId(relu.GetId());
    m_OperandToPart[&relu.GetOutput(0)] = inputPart;
}

std::vector<BasePart*> NetworkToGraphOfPartsConverter::CreateTransposeConv(const Stride& stride,
                                                                           const TensorInfo& weightsInfo,
                                                                           const std::vector<uint8_t>& weightsData,
                                                                           const TensorInfo& biasInfo,
                                                                           std::vector<int32_t> biasData,
                                                                           const Padding& padding,
                                                                           const TensorInfo& inputInfo,
                                                                           const TensorInfo& outputInfo,
                                                                           const std::set<uint32_t>& operationIds)
{
    std::vector<BasePart*> parts;

    // TransposeConvolution is implemented as an upscale (padding) operation + a convolution.
    // The stride parameter of a TransposeConvolution represents the upscaling factor.
    // The stride of the convolution operation underneath is always 1.
    // The stride comes in as a vector {x, y} where x = y (validated by IsSupported checks)
    assert(stride.m_X == stride.m_Y);
    uint32_t upscaleFactor                            = stride.m_X;
    ethosn::command_stream::UpsampleType upsampleType = ethosn::command_stream::UpsampleType::TRANSPOSE;
    const TensorShape& weightsShape                   = weightsInfo.m_Dimensions;

    // The padding of a TransposeConvolution affects the convolution operation underneath, but requires modification.
    // This means there is a restriction on the size of the padding such that our internal padding cannot be negative,
    // which is checked in IsTransposeConvolutionSupported (by virtue of supporting only same/valid padding).

    // The user-specified padding applies to the *output* of the transpose conv rather than the input like in a regular
    // convolution (see below example of output tensor with 1 padding on top/left). The padding is essentially cropping
    // the output tensor.
    //
    // When the padding is specified as zero the output tensor is not cropped at all, meaning that the top-left-most
    // (s_x, s_y) elements (where s_x, s_y are the strides) are equal to top-left (s_x, s_y) portion of the kernel
    // multiplied by the top-left input value.
    //
    // In order to get this same result from our internal convolution we need to add enough padding so that as we slide
    // the kernel over the upscaled-and-padded input, the first (s_x, s_y) output elements depend only on the top-left
    // input value. Here is an example showing that we need 2 padding for a 3x3 kernel with stride 2. The highlighted
    // window shows the values used to calculate the (1,1) output value and it depends only on I0 as required.
    // The same is true for the (0,0), (0,1) and (1,0) output values.
    //
    // +---+---+----+---+----+---+
    // | P | P | P  | P | P  | P |
    // +---╬═══╬════╬═══╬----+---+
    // | P ║ P | P  | P ║ P  | P |
    // +---╬---+----+---╬----+---+
    // | P ║ P | I0 | 0 ║ I1 | 0 |
    // +---╬---+----+---╬----+---+
    // | P ║ P | 0  | 0 ║ 0  | 0 |
    // +---╬═══╬════╬═══╬----+---+
    // | P | P | I2 | 0 | I3 | 0 |
    // +---+---+----+---+----+---+
    // | P | P | 0  | 0 | 0  | 0 |
    // +---+---+----+---+----+---+
    //
    // The amount of padding required for the zero-padding case is therefore kernel_size - 1.
    // Increasing the padding on the transpose convolution crops pixels from the output, which means that the region of
    // the output which depends only on the first input value gets smaller. This means that for our internal convolution
    // we must *decrease* the padding by the same amount. At the extreme this means that we will have zero padding
    // on our internal convolution so that *only* the first output value will depend on the first input value.
    // This corresponds to a padding/cropping of kernel_size - 1 on the transpose convolution.
    //
    // From this, we can calculate the internal convolution padding as: kernel_size - 1 - original_padding.
    const uint32_t topMcePadding  = weightsShape[0] - 1 - padding.m_Top;
    const uint32_t leftMcePadding = weightsShape[1] - 1 - padding.m_Left;

    TensorShape inputShape = inputInfo.m_Dimensions;

    // We can't do upscaling with a large kernel size, so we have to do the upscaling in a separate pass beforehand
    // with an identity (1x1) kernel. The convolution is then performed in another pass.
    if (weightsShape[0] > 7 || weightsShape[1] > 7)
    {
        const TensorShape& intermediateOutputShape = { inputShape[0], inputShape[1] * upscaleFactor,
                                                       inputShape[2] * upscaleFactor, inputShape[3] };

        const uint32_t numIfm   = inputShape[3];
        const float weightScale = 0.5f;
        const float biasScale   = weightScale * inputInfo.m_QuantizationInfo.GetScale();

        std::vector<uint8_t> weightsData(1 * 1 * 1 * numIfm, 2);
        std::vector<int32_t> biasData(numIfm, 0);

        TensorInfo weightInfo{ { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale } };
        TensorInfo biasInfo{ { 1, 1, 1, numIfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };

        McePart::ConstructionParams params(m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
        params.m_Id                     = m_GraphOfParts.GeneratePartId();
        params.m_InputTensorShape       = inputShape;
        params.m_OutputTensorShape      = intermediateOutputShape;
        params.m_InputQuantizationInfo  = inputInfo.m_QuantizationInfo;
        params.m_OutputQuantizationInfo = inputInfo.m_QuantizationInfo;
        params.m_WeightsInfo            = weightInfo;
        params.m_WeightsData            = std::move(weightsData);
        params.m_BiasInfo               = biasInfo;
        params.m_BiasData               = std::move(biasData);
        params.m_Stride                 = Stride(1, 1);
        params.m_PadTop                 = 0;
        params.m_PadLeft                = 0;
        params.m_Op                     = command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
        params.m_OperationIds           = operationIds;
        params.m_DataType               = GetCommandDataType(inputInfo.m_DataType);
        params.m_UpscaleFactor          = upscaleFactor;
        params.m_UpsampleType           = upsampleType;

        auto identityDepthwisePart = std::make_unique<McePart>(std::move(params));
        parts.push_back(identityDepthwisePart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(identityDepthwisePart));

        upscaleFactor = 1;
        upsampleType  = ethosn::command_stream::UpsampleType::OFF;
        inputShape    = intermediateOutputShape;
    }

    // Rotate weights by 180 in the XY plane.
    // This is needed for the internal convolution to produce the same result as the transpose convolution.
    ConstTensorData originalWeights(weightsData.data(), weightsShape);
    std::vector<uint8_t> flippedWeightsData(weightsData.size());
    TensorData flippedWeights(flippedWeightsData.data(), weightsShape);
    for (uint32_t y = 0; y < weightsShape[0]; ++y)
    {
        for (uint32_t x = 0; x < weightsShape[1]; ++x)
        {
            // The other two dimensions are irrelevant and we can copy them together as a contiguous block
            const uint32_t n   = weightsShape[2] * weightsShape[3];
            const uint8_t& src = originalWeights.GetElementRef(y, x, 0, 0);
            uint8_t& dst       = flippedWeights.GetElementRef(weightsShape[0] - 1 - y, weightsShape[1] - 1 - x, 0, 0);
            std::copy_n(&src, n, &dst);
        }
    }

    McePart::ConstructionParams params(m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
    params.m_Id                     = m_GraphOfParts.GeneratePartId();
    params.m_InputTensorShape       = inputShape;
    params.m_OutputTensorShape      = outputInfo.m_Dimensions;
    params.m_InputQuantizationInfo  = inputInfo.m_QuantizationInfo;
    params.m_OutputQuantizationInfo = outputInfo.m_QuantizationInfo;
    params.m_WeightsInfo            = weightsInfo;
    params.m_WeightsData            = std::move(flippedWeightsData);
    params.m_BiasInfo               = biasInfo;
    params.m_BiasData               = std::move(biasData);
    params.m_Stride                 = Stride(1, 1);
    params.m_PadTop                 = topMcePadding;
    params.m_PadLeft                = leftMcePadding;
    params.m_Op                     = command_stream::MceOperation::CONVOLUTION;
    params.m_OperationIds           = operationIds;
    params.m_DataType               = GetCommandDataType(outputInfo.m_DataType);
    params.m_UpscaleFactor          = upscaleFactor;
    params.m_UpsampleType           = upsampleType;
    auto mcePart                    = std::make_unique<McePart>(std::move(params));

    parts.push_back(mcePart.get());
    m_GraphOfParts.m_Parts.push_back(std::move(mcePart));

    return parts;
}

void NetworkToGraphOfPartsConverter::Visit(ReinterpretQuantization& reinterpretQuantization)
{
    // Reinterpret quantization doesn't "do" anything
    // The operations which follow the reinterpret quantization will
    // pick up the new input quantization from the reinterpret quantization
    const Operand* inputOperand = &reinterpretQuantization.GetInput(0);
    assert(inputOperand);
    BasePart* inputPart                                    = m_OperandToPart[inputOperand];
    m_OperandToPart[&reinterpretQuantization.GetOutput(0)] = inputPart;
}

void NetworkToGraphOfPartsConverter::Visit(TransposeConvolution& transposeConvolution)
{
    const Stride& stride                    = transposeConvolution.GetConvolutionInfo().m_Stride;
    const TensorInfo& weightsInfo           = transposeConvolution.GetWeights().GetTensorInfo();
    const std::vector<uint8_t>& weightsData = transposeConvolution.GetWeights().GetDataVector();
    const TensorInfo& biasInfo              = transposeConvolution.GetBias().GetTensorInfo();
    std::vector<int32_t> biasData = GetDataVectorAs<int32_t, uint8_t>(transposeConvolution.GetBias().GetDataVector());
    const Padding& padding        = transposeConvolution.GetConvolutionInfo().m_Padding;
    const TensorInfo& inputInfo   = transposeConvolution.GetInput(0).GetTensorInfo();
    const TensorInfo& outputInfo  = transposeConvolution.GetOutput(0).GetTensorInfo();
    const std::set<uint32_t> operationIds = { transposeConvolution.GetId(), transposeConvolution.GetBias().GetId(),
                                              transposeConvolution.GetWeights().GetId() };

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    char reason[1024];
    const SupportedLevel supportedLevel = m_Queries.IsTransposeConvolutionSupported(
        transposeConvolution.GetBias().GetTensorInfo(), transposeConvolution.GetWeights().GetTensorInfo(),
        transposeConvolution.GetConvolutionInfo(), transposeConvolution.GetInput(0).GetTensorInfo(), nullptr, reason,
        sizeof(reason));
    std::vector<BasePart*> parts;
    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ inputInfo },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            operationIds, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
    }
    else
    {
        parts = CreateTransposeConv(stride, weightsInfo, weightsData, biasInfo, std::move(biasData), padding, inputInfo,
                                    outputInfo, operationIds);
    }

    ConnectParts(transposeConvolution, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Softmax& softmax)
{
    std::vector<BasePart*> parts;

    std::string reasonForEstimateOnly = "softmax is not supported by ethosn NPU";

    auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
        m_GraphOfParts.GeneratePartId(), reasonForEstimateOnly,
        std::vector<TensorInfo>{ softmax.GetInput(0).GetTensorInfo() },
        std::vector<TensorInfo>{ softmax.GetOutput(0).GetTensorInfo() }, CompilerDataFormat::NHWCB,
        std::set<uint32_t>{ softmax.GetId() }, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

    parts.push_back(estimateOnlyPart.get());
    m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
    ConnectParts(softmax, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Split& split)
{
    const TensorInfo& inputInfo           = split.GetInput(0).GetTensorInfo();
    const TensorInfo& outputInfo          = split.GetOutput(0).GetTensorInfo();
    const std::set<uint32_t> operationIds = { split.GetId() };

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    char reason[1024];
    const SupportedLevel supportedLevel =
        m_Queries.IsSplitSupported(inputInfo, split.GetSplitInfo(), nullptr, reason, sizeof(reason));

    std::vector<BasePart*> parts;
    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ inputInfo },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            operationIds, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
    }

    // Loop through all parts in the vector of BaseParts and connect them together.
    for (uint32_t i = 0; i < static_cast<uint32_t>(parts.size()) - 1; i++)
    {
        m_GraphOfParts.AddConnection({ parts[i + 1]->GetPartId(), 0 }, { parts[i]->GetPartId(), 0 });
    }

    Operand& operand = split.GetInput(0);
    m_GraphOfParts.AddConnection({ parts.front()->GetPartId(), 0 },
                                 { m_OperandToPart.at(&operand)->GetPartId(), operand.GetProducerOutputIndex() });

    // Check if current operation has outputs and if so mark them for connection with the subsequent operation.
    for (auto&& outputOperand : split.GetOutputs())
    {
        m_OperandToPart[&outputOperand] = parts.back();
    }
}

void NetworkToGraphOfPartsConverter::Visit(Transpose& transpose)
{
    const TensorInfo& inputInfo           = transpose.GetInput(0).GetTensorInfo();
    const TensorInfo& outputInfo          = transpose.GetOutput(0).GetTensorInfo();
    const std::set<uint32_t> operationIds = { transpose.GetId() };

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    char reason[1024];
    const SupportedLevel supportedLevel =
        m_Queries.IsTransposeSupported(transpose.GetTransposeInfo(), inputInfo, nullptr, reason, sizeof(reason));
    std::vector<BasePart*> parts;
    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ inputInfo },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            operationIds, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
    }
    ConnectParts(transpose, parts);
}

void NetworkToGraphOfPartsConverter::Visit(DepthToSpace& depthToSpace)
{
    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    char reason[1024];

    const SupportedLevel supportedLevel = m_Queries.IsDepthToSpaceSupported(
        depthToSpace.GetInput(0).GetTensorInfo(), depthToSpace.GetDepthToSpaceInfo(), nullptr, reason, sizeof(reason));

    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        std::vector<BasePart*> parts;
        const TensorInfo& outputTensorInfo = depthToSpace.GetOutput(0).GetTensorInfo();

        auto estimateOnlyPart =
            std::make_unique<EstimateOnlyPart>(m_GraphOfParts.GeneratePartId(), reason,
                                               std::vector<TensorInfo>{ depthToSpace.GetInput(0).GetTensorInfo() },
                                               std::vector<TensorInfo>{ outputTensorInfo },
                                               ConvertExternalToCompilerDataFormat(outputTensorInfo.m_DataFormat),
                                               std::set<uint32_t>{ depthToSpace.GetId() }, m_EstimationOptions.value(),
                                               m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
        ConnectParts(depthToSpace, parts);
    }
    else
    {
        // We implement depth-to-space (block-size 2) with a transpose convolution (stride 2) with a 2x2
        // kernel, where the weights are used to 'select' which elements of the input are placed into each
        // element of the output.
        // By setting the stride and kernel size the same, the output is made by multiplying the kernel
        // by each IFM (x, y) position and tiling the resulting tensors.
        // The weight vector along input-channels at each (u, v) position in the kernel will be dotted
        // with the IFM along channels at each (x, y) position.
        // This means that we can choose different weight vectors to be dotted with the IFM vectors for
        // each of the four output pixels that we want to derive from each input pixel,
        // so that we can select the correct IFM channel for each.
        // The weight vectors at each (u, v) are therefore simple "one-hot" vectors.
        // Below is an example for a 1x1x4 input being turned into a 2x2x1 output.
        //
        //  Input:                     Output:                       Weights:
        // (with padding)
        //
        //  Channel 0:                Channel 0:                  Input channel 0:
        //     I0                       I0   I1                        1   0
        //                              I2   I3                        0   0
        //
        //  Channel 1:                                            Input channel 1:
        //     I1                                                      0   1
        //                                                             0   0
        //
        //  Channel 2:                                            Input channel 2:
        //     I2                                                      0   0
        //                                                             1   0
        //
        //  Channel 3:                                            Input channel 3:
        //     I3                                                      0   0
        //                                                             0   1
        //
        uint32_t blockSize = depthToSpace.GetDepthToSpaceInfo().m_BlockSize;
        assert(blockSize == 2);    // Checked by IsDepthToSpaceSupported
        uint32_t ifmChannelsPerOfm = blockSize * blockSize;

        const TensorShape& inputShape  = depthToSpace.GetInput(0).GetTensorInfo().m_Dimensions;
        const TensorShape& outputShape = depthToSpace.GetOutput(0).GetTensorInfo().m_Dimensions;

        // Set weights according to the above explanation
        const float weightsScale =
            0.5f;    // We can't use a scale of 1.0 as that would cause an overall multiplier >= 1.
        TensorInfo weightsInfo({ blockSize, blockSize, inputShape[3], outputShape[3] }, DataType::UINT8_QUANTIZED,
                               DataFormat::HWIO, QuantizationInfo(0, weightsScale));
        std::vector<uint8_t> weightsData(GetNumElements(weightsInfo.m_Dimensions), 0);
        TensorData weights(weightsData.data(), weightsInfo.m_Dimensions);
        for (uint32_t ofmIdx = 0; ofmIdx < outputShape[3]; ++ofmIdx)
        {
            // Each OFM is derived from 4 IFMs which are distributed across the channels.
            // All of the top-left elements come first, then all the top-right, bottom-left then finally
            // bottom-right.
            // This means that the IFMs for a particular OFM start at the same index as the OFM
            // and are separated from each other by the number of blocks.
            const uint32_t ifmBase   = ofmIdx;
            const uint32_t ifmStride = inputShape[3] / ifmChannelsPerOfm;
            // Set the weight vectors for each of the (u, v) positions, each of which will contain just
            // one non-zero value
            for (uint32_t v = 0; v < blockSize; ++v)
            {
                for (uint32_t u = 0; u < blockSize; ++u)
                {
                    // Calculate which IFM we want this weight vector to select
                    const uint32_t ifmWithinBlock = v * blockSize + u;
                    const uint32_t ifmIdx         = ifmBase + ifmWithinBlock * ifmStride;
                    weights.SetElement(v, u, ifmIdx, ofmIdx, static_cast<uint8_t>(1.0f / weightsScale));
                }
            }
        }

        // Set biases to all zero (we don't need a bias)
        const float biasScale = weightsScale * depthToSpace.GetInput(0).GetTensorInfo().m_QuantizationInfo.GetScale();
        TensorInfo biasInfo({ 1, 1, 1, outputShape[3] }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                            QuantizationInfo(0, biasScale));
        std::vector<int32_t> biasData(GetNumElements(biasInfo.m_Dimensions), 0);

        const std::set<uint32_t> operationId = { depthToSpace.GetId() };
        std::vector<BasePart*> transposeConv =
            CreateTransposeConv(Stride(blockSize, blockSize), weightsInfo, std::move(weightsData), biasInfo,
                                std::move(biasData), Padding(0, 0), depthToSpace.GetInput(0).GetTensorInfo(),
                                depthToSpace.GetOutput(0).GetTensorInfo(), operationId);

        ConnectParts(depthToSpace, transposeConv);
    }
}

void NetworkToGraphOfPartsConverter::Visit(SpaceToDepth& spaceToDepth)
{
    const TensorInfo& inputInfo           = spaceToDepth.GetInput(0).GetTensorInfo();
    const TensorInfo& outputInfo          = spaceToDepth.GetOutput(0).GetTensorInfo();
    const std::set<uint32_t> operationIds = { spaceToDepth.GetId() };

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    char reason[1024];
    const SupportedLevel supportedLevel = m_Queries.IsSpaceToDepthSupported(
        inputInfo, spaceToDepth.GetSpaceToDepthInfo(), nullptr, reason, sizeof(reason));
    std::vector<BasePart*> parts;
    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ inputInfo },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            operationIds, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.m_Parts.push_back(std::move(estimateOnlyPart));
    }

    ConnectParts(spaceToDepth, parts);
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
void NetworkToGraphOfPartsConverter::ConnectNoOp(Operation& operation)
{
    // Sanity check for single input support
    assert(operation.GetInputs().size() == 1);

    for (size_t i = 0; i < operation.GetOutputs().size(); ++i)
    {
        m_OperandToPart[&operation.GetOutput(i)] = m_OperandToPart[&operation.GetInput(0)];
    }
}
}    // namespace support_library
}    // namespace ethosn
