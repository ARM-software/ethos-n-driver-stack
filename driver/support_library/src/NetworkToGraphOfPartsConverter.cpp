//
// Copyright © 2021-2025 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "NetworkToGraphOfPartsConverter.hpp"

#include "ConcatPart.hpp"
#include "ConcreteOperations.hpp"
#include "ConstantPart.hpp"
#include "DebuggingContext.hpp"
#include "EstimateOnlyPart.hpp"
#include "FullyConnectedPart.hpp"
#include "FusedPlePart.hpp"
#include "InputPart.hpp"
#include "MceEstimationUtils.hpp"
#include "McePart.hpp"
#include "OutputPart.hpp"
#include "Part.hpp"
#include "ReformatPart.hpp"
#include "ReshapePart.hpp"
#include "SplitPart.hpp"
#include "StandalonePlePart.hpp"
#include "Utils.hpp"
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
                                                                               DataType inputDataType,
                                                                               DataType outputDataType,
                                                                               const EstimationOptions& estOpt,
                                                                               const CompilationOptions& compOpt,
                                                                               const HardwareCapabilities& capabilities)
{
    McePart::ConstructionParams params(estOpt, compOpt, capabilities, m_DebuggingContext, m_ThreadPool);
    params.m_Id                     = m_GraphOfParts.GeneratePartId();
    params.m_InputTensorShape       = shape;
    params.m_OutputTensorShape      = shape;
    params.m_InputQuantizationInfo  = inputQuantInfo;
    params.m_OutputQuantizationInfo = outputQuantInfo;
    const uint32_t numIfm           = shape[3];
    const float weightScale         = 0.5f;
    params.m_WeightsInfo    = { { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale } };
    params.m_WeightsData    = std::vector<uint8_t>(1 * 1 * 1 * numIfm, 2);
    const float biasScale   = weightScale * inputQuantInfo.GetScale();
    params.m_BiasInfo       = { { 1, 1, 1, numIfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };
    params.m_BiasData       = std::vector<int32_t>(numIfm, 0);
    params.m_Op             = command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    params.m_OperationIds   = std::set<uint32_t>{ operationId };
    params.m_UpscaleFactor  = 1;
    params.m_UpsampleType   = MceUpsampleType::OFF;
    params.m_InputDataType  = inputDataType;
    params.m_OutputDataType = outputDataType;
    params.m_LowerBound     = outputDataType == DataType::UINT8_QUANTIZED ? 0 : -128;
    params.m_UpperBound     = outputDataType == DataType::UINT8_QUANTIZED ? 255 : 127;
    params.m_IsChannelSelector = (inputQuantInfo == outputQuantInfo);
    auto mcePart               = std::make_unique<McePart>(std::move(params));
    return mcePart;
}

std::unique_ptr<McePart>
    CreateIdentityMcePartWithPaddedOutputChannels(PartId partId,
                                                  const TensorShape& shape,
                                                  const QuantizationInfo& inputQuantInfo,
                                                  const QuantizationInfo& outputQuantInfo,
                                                  uint32_t operationId,
                                                  DataType inputDataType,
                                                  DataType outputDataType,
                                                  const EstimationOptions& estOpt,
                                                  const CompilationOptions& compOpt,
                                                  const HardwareCapabilities& capabilities,
                                                  const std::vector<std::pair<uint32_t, uint32_t>>& padAmounts,
                                                  DebuggingContext& debuggingContext,
                                                  ThreadPool& threadPool)
{
    uint32_t numOfm = GetChannels(shape);
    for (size_t i = 0; i < padAmounts.size(); ++i)
    {
        numOfm += padAmounts[i].second;
    }

    McePart::ConstructionParams params(estOpt, compOpt, capabilities, debuggingContext, threadPool);
    params.m_Id                     = partId;
    params.m_InputTensorShape       = shape;
    params.m_OutputTensorShape      = { shape[0], shape[1], shape[2], numOfm };
    params.m_InputQuantizationInfo  = inputQuantInfo;
    params.m_OutputQuantizationInfo = outputQuantInfo;
    const uint32_t numIfm           = shape[3];
    const float weightScale         = 0.5f;
    params.m_WeightsInfo            = {
        { 1, 1, numIfm, numOfm }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, weightScale }
    };

    params.m_WeightsData.reserve(GetNumElements(params.m_WeightsInfo.m_Dimensions));
    for (uint32_t i = 0; i < numIfm; ++i)
    {
        uint32_t padIdx  = 0;
        uint32_t origIdx = 0;
        while (true)
        {
            if (padIdx < padAmounts.size() && origIdx >= padAmounts[padIdx].first)
            {
                for (uint32_t p = 0; p < padAmounts[padIdx].second; ++p)
                {
                    params.m_WeightsData.push_back(0);
                }
                padIdx++;
            }
            if (origIdx >= shape[3])
            {
                break;
            }
            params.m_WeightsData.push_back(origIdx == i ? 2 : 0);
            ++origIdx;
        }
    }

    const float biasScale      = weightScale * inputQuantInfo.GetScale();
    params.m_BiasInfo          = { { 1, 1, 1, numOfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };
    params.m_BiasData          = std::vector<int32_t>(numOfm, 0);
    params.m_Op                = command_stream::MceOperation::CONVOLUTION;
    params.m_OperationIds      = std::set<uint32_t>{ operationId };
    params.m_UpscaleFactor     = 1;
    params.m_UpsampleType      = MceUpsampleType::OFF;
    params.m_InputDataType     = inputDataType;
    params.m_OutputDataType    = outputDataType;
    params.m_LowerBound        = outputDataType == DataType::UINT8_QUANTIZED ? 0 : -128;
    params.m_UpperBound        = outputDataType == DataType::UINT8_QUANTIZED ? 255 : 127;
    params.m_IsChannelSelector = (inputQuantInfo == outputQuantInfo);
    auto mcePart               = std::make_unique<McePart>(std::move(params));
    return mcePart;
}

std::unique_ptr<McePart>
    CreateIdentityMcePartWithRemovedInputChannels(PartId partId,
                                                  const TensorShape& shape,
                                                  const QuantizationInfo& inputQuantInfo,
                                                  const QuantizationInfo& outputQuantInfo,
                                                  uint32_t operationId,
                                                  DataType inputDataType,
                                                  DataType outputDataType,
                                                  const EstimationOptions& estOpt,
                                                  const CompilationOptions& compOpt,
                                                  const HardwareCapabilities& capabilities,
                                                  const std::vector<std::pair<uint32_t, uint32_t>>& removeAmounts,
                                                  DebuggingContext& debuggingContext,
                                                  ThreadPool& threadPool)
{
    uint32_t numOfm = GetChannels(shape);
    for (size_t i = 0; i < removeAmounts.size(); ++i)
    {
        numOfm -= removeAmounts[i].second;
    }

    McePart::ConstructionParams params(estOpt, compOpt, capabilities, debuggingContext, threadPool);
    params.m_Id                     = partId;
    params.m_InputTensorShape       = shape;
    params.m_OutputTensorShape      = { shape[0], shape[1], shape[2], numOfm };
    params.m_InputQuantizationInfo  = inputQuantInfo;
    params.m_OutputQuantizationInfo = outputQuantInfo;
    const uint32_t numIfm           = shape[3];
    const float weightScale         = 0.5f;
    params.m_WeightsInfo            = {
        { 1, 1, numIfm, numOfm }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, weightScale }
    };

    params.m_WeightsData.reserve(GetNumElements(params.m_WeightsInfo.m_Dimensions));
    for (uint32_t i = 0; i < numIfm; ++i)
    {
        uint32_t removeIdx = 0;
        for (uint32_t o = 0; o < numIfm; ++o)
        {
            if (removeIdx < removeAmounts.size() && o == removeAmounts[removeIdx].first)
            {
                o += removeAmounts[removeIdx].second;
                removeIdx++;
            }
            if (o >= numIfm)
            {
                break;
            }
            params.m_WeightsData.push_back(o == i ? 2 : 0);
        }
    }

    const float biasScale      = weightScale * inputQuantInfo.GetScale();
    params.m_BiasInfo          = { { 1, 1, 1, numOfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };
    params.m_BiasData          = std::vector<int32_t>(numOfm, 0);
    params.m_Op                = command_stream::MceOperation::CONVOLUTION;
    params.m_OperationIds      = std::set<uint32_t>{ operationId };
    params.m_UpscaleFactor     = 1;
    params.m_UpsampleType      = MceUpsampleType::OFF;
    params.m_InputDataType     = inputDataType;
    params.m_OutputDataType    = outputDataType;
    params.m_LowerBound        = outputDataType == DataType::UINT8_QUANTIZED ? 0 : -128;
    params.m_UpperBound        = outputDataType == DataType::UINT8_QUANTIZED ? 255 : 127;
    params.m_IsChannelSelector = (inputQuantInfo == outputQuantInfo);
    auto mcePart               = std::make_unique<McePart>(std::move(params));
    return mcePart;
}

NetworkToGraphOfPartsConverter::NetworkToGraphOfPartsConverter(const Network& network,
                                                               const HardwareCapabilities& capabilities,
                                                               const EstimationOptions& estimationOptions,
                                                               const CompilationOptions& compilationOptions,
                                                               DebuggingContext& debuggingContext,
                                                               ThreadPool& threadPool)
    : m_Capabilities(capabilities)
    , m_EstimationOptions(estimationOptions)
    , m_CompilationOptions(compilationOptions)
    , m_DebuggingContext(debuggingContext)
    , m_Queries(capabilities.GetData(), true)
    , m_ThreadPool(threadPool)
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
                                                 input.GetTensorInfo().m_DataType, std::set<uint32_t>{ input.GetId() },
                                                 m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
    parts.push_back(inputPart.get());
    m_GraphOfParts.AddPart(std::move(inputPart));
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
        output.GetTensorInfo().m_QuantizationInfo, output.GetTensorInfo().m_DataType,
        std::set<uint32_t>{ output.GetInput(0).GetProducer().GetId() }, output.GetInput(0).GetProducerOutputIndex(),
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
    parts.push_back(outputPart.get());
    m_GraphOfParts.AddPart(std::move(outputPart));
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
        constant.GetTensorInfo().m_QuantizationInfo, constant.GetTensorInfo().m_DataType,
        std::set<uint32_t>{ constant.GetId() }, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
        constant.GetDataVector());
    parts.push_back(constPart.get());
    m_GraphOfParts.AddPart(std::move(constPart));
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
        const TensorInfo& outputInfo    = depthwise.GetOutput(0).GetTensorInfo();
        std::set<uint32_t> operationIds = { depthwise.GetId(), depthwise.GetBias().GetId(),
                                            depthwise.GetWeights().GetId() };

        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ depthwise.GetInput(0).GetTensorInfo() },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            std::move(operationIds), m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
    }
    else
    {
        TensorInfo mceOperationInput        = depthwise.GetInput(0).GetTensorInfo();
        TensorInfo mceOperationOutput       = depthwise.GetOutput(0).GetTensorInfo();
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
                mceOperationInput.m_QuantizationInfo, PleOperation::INTERLEAVE_2X2_2_2,
                utils::ShapeMultiplier{ { 1, convInfo.m_Stride.m_Y },
                                        { 1, convInfo.m_Stride.m_X },
                                        { convInfo.m_Stride.m_X * convInfo.m_Stride.m_Y } },
                m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                std::set<uint32_t>{ depthwise.GetId(), depthwise.GetBias().GetId(), depthwise.GetWeights().GetId() },
                mceOperationInput.m_DataType, mceOperationOutput.m_DataType, m_DebuggingContext, m_ThreadPool,
                std::map<std::string, std::string>{}, std::map<std::string, int>{}, std::map<std::string, int>{});

            parts.push_back(fusedPlePart.get());
            m_GraphOfParts.AddPart(std::move(fusedPlePart));
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
        McePart::ConstructionParams params(m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                                           m_DebuggingContext, m_ThreadPool);
        params.m_Id                     = m_GraphOfParts.GeneratePartId();
        params.m_InputTensorShape       = mceOperationInput.m_Dimensions;
        params.m_OutputTensorShape      = depthwise.GetOutput(0).GetTensorInfo().m_Dimensions;
        params.m_InputQuantizationInfo  = mceOperationInput.m_QuantizationInfo;
        params.m_OutputQuantizationInfo = depthwise.GetOutput(0).GetTensorInfo().m_QuantizationInfo;
        params.m_WeightsInfo            = weightsTensorInfo;
        params.m_WeightsData            = OverrideWeights(depthwise.GetWeights().GetDataVector(), weightsTensorInfo);
        params.m_BiasInfo               = depthwise.GetBias().GetTensorInfo();
        params.m_BiasData               = GetDataVectorAs<int32_t, uint8_t>(depthwise.GetBias().GetDataVector());
        params.m_Op                     = operation;
        params.m_OperationIds =
            std::set<uint32_t>{ depthwise.GetId(), depthwise.GetBias().GetId(), depthwise.GetWeights().GetId() };
        params.m_Stride            = depthwise.GetConvolutionInfo().m_Stride;
        params.m_Padding           = depthwise.GetConvolutionInfo().m_Padding;
        params.m_UpscaleFactor     = 1;
        params.m_UpsampleType      = MceUpsampleType::OFF;
        params.m_InputDataType     = mceOperationInput.m_DataType;
        params.m_OutputDataType    = mceOperationOutput.m_DataType;
        params.m_LowerBound        = mceOperationOutput.m_DataType == DataType::UINT8_QUANTIZED ? 0 : -128;
        params.m_UpperBound        = mceOperationOutput.m_DataType == DataType::UINT8_QUANTIZED ? 255 : 127;
        params.m_IsChannelSelector = false;
        auto mcePart               = std::make_unique<McePart>(std::move(params));

        if (convInfo.m_Stride.m_X > 1 || convInfo.m_Stride.m_Y > 1)
        {
            mcePart->setUninterleavedInputShape(uninterleavedInputShape);
        }

        parts.push_back(mcePart.get());
        m_GraphOfParts.AddPart(std::move(mcePart));
    }

    ConnectParts(depthwise, parts);
}

void NetworkToGraphOfPartsConverter::Visit(StandalonePadding& padding)
{
    std::vector<BasePart*> parts;
    const Padding& paddingInfo   = padding.GetPadding();
    const TensorInfo& inputInfo  = padding.GetInput(0).GetTensorInfo();
    const TensorInfo& outputInfo = padding.GetOutput(0).GetTensorInfo();

    const uint32_t numIfm                = inputInfo.m_Dimensions[3];
    const float weightScale              = 0.5f;
    const TensorInfo identityWeightsInfo = {
        { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale }
    };

    const float biasScale             = weightScale * inputInfo.m_QuantizationInfo.GetScale();
    const TensorInfo identityBiasInfo = {
        { 1, 1, 1, numIfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale }
    };

    McePart::ConstructionParams params(m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                                       m_DebuggingContext, m_ThreadPool);
    params.m_Id                     = m_GraphOfParts.GeneratePartId();
    params.m_InputTensorShape       = inputInfo.m_Dimensions;
    params.m_OutputTensorShape      = outputInfo.m_Dimensions;
    params.m_InputQuantizationInfo  = inputInfo.m_QuantizationInfo;
    params.m_OutputQuantizationInfo = outputInfo.m_QuantizationInfo;
    params.m_WeightsInfo            = identityWeightsInfo;
    params.m_WeightsData            = std::vector<uint8_t>(1 * 1 * 1 * numIfm, 2);
    params.m_BiasInfo               = identityBiasInfo;
    params.m_BiasData               = std::vector<int32_t>(numIfm, 0);
    params.m_Padding                = paddingInfo;
    params.m_Op                     = command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    params.m_OperationIds           = std::set<uint32_t>{ padding.GetId() };
    params.m_UpscaleFactor          = 1;
    params.m_UpsampleType           = MceUpsampleType::OFF;
    params.m_InputDataType          = inputInfo.m_DataType;
    params.m_OutputDataType         = outputInfo.m_DataType;
    params.m_LowerBound             = outputInfo.m_DataType == DataType::UINT8_QUANTIZED ? 0 : -128;
    params.m_UpperBound             = outputInfo.m_DataType == DataType::UINT8_QUANTIZED ? 255 : 127;
    params.m_IsChannelSelector      = false;

    auto mcePart = std::make_unique<McePart>(std::move(params));
    parts.push_back(mcePart.get());
    m_GraphOfParts.AddPart(std::move(mcePart));

    ConnectParts(padding, parts);
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
        const TensorInfo& outputInfo    = convolution.GetOutput(0).GetTensorInfo();
        std::set<uint32_t> operationIds = { convolution.GetId(), convolution.GetBias().GetId(),
                                            convolution.GetWeights().GetId() };

        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ convolution.GetInput(0).GetTensorInfo() },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            std::move(operationIds), m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
    }
    else
    {
        TensorShape uninterleavedInputShape = convolution.GetInput(0).GetTensorInfo().m_Dimensions;
        TensorInfo mceOperationInput        = convolution.GetInput(0).GetTensorInfo();
        TensorInfo mceOperationOutput       = convolution.GetOutput(0).GetTensorInfo();

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
                interleaveOutput.m_QuantizationInfo, PleOperation::INTERLEAVE_2X2_2_2,
                utils::ShapeMultiplier{ { 1, convInfo.m_Stride.m_Y },
                                        { 1, convInfo.m_Stride.m_X },
                                        { convInfo.m_Stride.m_X * convInfo.m_Stride.m_Y } },
                m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                std::set<uint32_t>{ convolution.GetId(), convolution.GetBias().GetId(),
                                    convolution.GetWeights().GetId() },
                mceOperationInput.m_DataType, mceOperationOutput.m_DataType, m_DebuggingContext, m_ThreadPool,
                std::map<std::string, std::string>{}, std::map<std::string, int>{}, std::map<std::string, int>{});
            parts.push_back(fusedPlePart.get());
            m_GraphOfParts.AddPart(std::move(fusedPlePart));

            // Pass interleaved Output as Input Tensor to subsequent McePart
            mcePartInputTensor = interleaveOutput;
        }
        else
        {
            // Pass default convolution Input Tensor
            mcePartInputTensor = convolution.GetInput(0).GetTensorInfo();
        }

        McePart::ConstructionParams params(m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                                           m_DebuggingContext, m_ThreadPool);
        params.m_Id                     = m_GraphOfParts.GeneratePartId();
        params.m_InputTensorShape       = mcePartInputTensor.m_Dimensions;
        params.m_OutputTensorShape      = convolution.GetOutput(0).GetTensorInfo().m_Dimensions;
        params.m_InputQuantizationInfo  = mcePartInputTensor.m_QuantizationInfo;
        params.m_OutputQuantizationInfo = convolution.GetOutput(0).GetTensorInfo().m_QuantizationInfo;
        params.m_WeightsInfo            = convolution.GetWeights().GetTensorInfo();
        params.m_WeightsData =
            OverrideWeights(convolution.GetWeights().GetDataVector(), convolution.GetWeights().GetTensorInfo());
        params.m_BiasInfo = convolution.GetBias().GetTensorInfo();
        params.m_BiasData = GetDataVectorAs<int32_t, uint8_t>(convolution.GetBias().GetDataVector());
        params.m_Op       = command_stream::MceOperation::CONVOLUTION;
        params.m_OperationIds =
            std::set<uint32_t>{ convolution.GetId(), convolution.GetBias().GetId(), convolution.GetWeights().GetId() };
        params.m_Stride            = convolution.GetConvolutionInfo().m_Stride;
        params.m_Padding           = convolution.GetConvolutionInfo().m_Padding;
        params.m_UpscaleFactor     = 1;
        params.m_UpsampleType      = MceUpsampleType::OFF;
        params.m_InputDataType     = mceOperationInput.m_DataType;
        params.m_OutputDataType    = mceOperationOutput.m_DataType;
        params.m_LowerBound        = mceOperationOutput.m_DataType == DataType::UINT8_QUANTIZED ? 0 : -128;
        params.m_UpperBound        = mceOperationOutput.m_DataType == DataType::UINT8_QUANTIZED ? 255 : 127;
        params.m_IsChannelSelector = false;
        auto mcePart               = std::make_unique<McePart>(std::move(params));

        if (convInfo.m_Stride.m_X > 1 || convInfo.m_Stride.m_Y > 1)
        {
            mcePart->setUninterleavedInputShape(uninterleavedInputShape);
        }

        parts.push_back(mcePart.get());
        m_GraphOfParts.AddPart(std::move(mcePart));
    }

    ConnectParts(convolution, parts);
}

void NetworkToGraphOfPartsConverter::Visit(FullyConnected& fullyConnected)
{
    std::vector<BasePart*> parts;
    parts.reserve(1);
    const TensorInfo& inputTensorInfo = fullyConnected.GetInput(0).GetTensorInfo();
    std::set<uint32_t> operationIds   = { fullyConnected.GetId(), fullyConnected.GetBias().GetId(),
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
            ConvertExternalToCompilerDataFormat(outputTensorInfo.m_DataFormat), std::move(operationIds),
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
    }
    else
    {
        TensorInfo mceOperationInput  = fullyConnected.GetInput(0).GetTensorInfo();
        TensorInfo mceOperationOutput = fullyConnected.GetOutput(0).GetTensorInfo();
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
            GetShapeContainingLinearElements(g_BrickGroupShape, inputTensorInfo.m_Dimensions[3]);

        // The weight encoder for fully connected requires the input channel to be a multiple of 1024.
        // It is easier to make this adjustment here rather than the WeightEncoder itself, even though
        // it is less desirable.
        TensorInfo weightsInfo      = fullyConnected.GetWeights().GetTensorInfo();
        weightsInfo.m_Dimensions[2] = RoundUpToNearestMultiple(weightsInfo.m_Dimensions[2], g_WeightsChannelVecProd);
        std::vector<uint8_t> paddedWeightsData = fullyConnected.GetWeights().GetDataVector();
        paddedWeightsData.resize(TotalSizeBytes(weightsInfo),
                                 static_cast<uint8_t>(weightsInfo.m_QuantizationInfo.GetZeroPoint()));

        FullyConnectedPart::ConstructionParams params(m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                                                      m_DebuggingContext, m_ThreadPool);
        params.m_Id                            = m_GraphOfParts.GeneratePartId();
        params.m_InputTensorShape              = inputTensorInfo.m_Dimensions;
        params.m_ReinterpretedInputTensorShape = reinterpretedInput;
        params.m_OutputTensorShape             = fullyConnected.GetOutput(0).GetTensorInfo().m_Dimensions;
        params.m_InputQuantizationInfo         = fullyConnected.GetInput(0).GetTensorInfo().m_QuantizationInfo;
        params.m_OutputQuantizationInfo        = fullyConnected.GetOutput(0).GetTensorInfo().m_QuantizationInfo;
        params.m_WeightsInfo                   = weightsInfo;
        params.m_WeightsData                   = std::move(paddedWeightsData);
        params.m_BiasInfo                      = fullyConnected.GetBias().GetTensorInfo();
        params.m_BiasData       = GetDataVectorAs<int32_t, uint8_t>(fullyConnected.GetBias().GetDataVector());
        params.m_OperationIds   = std::move(operationIds);
        params.m_InputDataType  = mceOperationInput.m_DataType;
        params.m_OutputDataType = mceOperationOutput.m_DataType;
        auto fcPart             = std::make_unique<FullyConnectedPart>(std::move(params));

        parts.push_back(fcPart.get());
        m_GraphOfParts.AddPart(std::move(fcPart));
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
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
    }
    else
    {
        TensorInfo inputInfo  = pooling.GetInput(0).GetTensorInfo();
        TensorInfo outputInfo = pooling.GetOutput(0).GetTensorInfo();

        // Create the appropriate fused or standalone PLE parts, based on the type of pooling

        auto createFusedPoolingPart = [&](PleOperation op) {
            std::map<std::string, std::string> selectionStringParams;
            if (op != PleOperation::DOWNSAMPLE_2X2)    // Downsample is sign-agnostic
            {
                selectionStringParams["datatype"] = (outputInfo.m_DataType == DataType::INT8_QUANTIZED) ? "s8" : "u8";
            }
            auto poolingFusedPlePart = std::make_unique<FusedPlePart>(
                m_GraphOfParts.GeneratePartId(), pooling.GetInput(0).GetTensorInfo().m_Dimensions,
                pooling.GetOutput(0).GetTensorInfo().m_Dimensions,
                pooling.GetInput(0).GetTensorInfo().m_QuantizationInfo,
                pooling.GetOutput(0).GetTensorInfo().m_QuantizationInfo, op,
                utils::ShapeMultiplier{ { 1, poolingInfo.m_PoolingStrideY }, { 1, poolingInfo.m_PoolingStrideX }, 1 },
                m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                std::set<uint32_t>{ pooling.GetId() }, inputInfo.m_DataType, outputInfo.m_DataType, m_DebuggingContext,
                m_ThreadPool, selectionStringParams, std::map<std::string, int>{}, std::map<std::string, int>{});
            parts.push_back(poolingFusedPlePart.get());
            m_GraphOfParts.AddPart(std::move(poolingFusedPlePart));
        };

        // MeanXy
        if (inputHeight == 7U && inputWidth == 7U && poolingInfo == poolingInfoMeanXy)
        {
            createFusedPoolingPart(PleOperation::MEAN_XY_7X7);
        }
        else if (inputHeight == 8U && inputWidth == 8U && poolingInfo == poolingInfoMeanXy)
        {
            createFusedPoolingPart(PleOperation::MEAN_XY_8X8);
        }
        // MaxPool with stride 2
        else if (poolingInfo == PoolingInfo{ 2, 2, 2, 2, poolingInfo.m_Padding, PoolingType::MAX })
        {
            createFusedPoolingPart(PleOperation::MAXPOOL_2X2_2_2);
        }
        else if (isInputOdd && poolingInfo == PoolingInfo{ 3, 3, 2, 2, poolingInfo.m_Padding, PoolingType::MAX })
        {
            createFusedPoolingPart(PleOperation::MAXPOOL_3X3_2_2_ODD);
        }
        else if (isInputEven && poolingInfo == PoolingInfo{ 3, 3, 2, 2, poolingInfo.m_Padding, PoolingType::MAX })
        {
            createFusedPoolingPart(PleOperation::MAXPOOL_3X3_2_2_EVEN);
        }
        else if (poolingInfo == PoolingInfo{ 1, 1, 2, 2, poolingInfo.m_Padding, PoolingType::MAX })
        {
            createFusedPoolingPart(PleOperation::DOWNSAMPLE_2X2);
        }
        // AvgPool
        else if (poolingInfo == PoolingInfo{ 3, 3, 1, 1, poolingInfo.m_Padding, PoolingType::AVG })
        {
            const std::vector<QuantizationInfo> inputQuantizations = {
                pooling.GetInput(0).GetTensorInfo().m_QuantizationInfo
            };
            const std::vector<TensorShape> inputShapes = { pooling.GetInput(0).GetTensorInfo().m_Dimensions };
            std::map<std::string, std::string> selectionStringParams = {
                { "datatype", outputInfo.m_DataType == DataType::INT8_QUANTIZED ? "s8" : "u8" }
            };
            auto poolingStandalonePlePart = std::make_unique<StandalonePlePart>(
                m_GraphOfParts.GeneratePartId(), inputShapes, pooling.GetOutput(0).GetTensorInfo().m_Dimensions,
                inputQuantizations, pooling.GetOutput(0).GetTensorInfo().m_QuantizationInfo,
                PleOperation::AVGPOOL_3X3_1_1_UDMA, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                std::set<uint32_t>{ pooling.GetId() }, pooling.GetOutput(0).GetTensorInfo().m_DataType,
                selectionStringParams, std::map<std::string, int>{}, std::map<std::string, int>{});
            parts.push_back(poolingStandalonePlePart.get());
            m_GraphOfParts.AddPart(std::move(poolingStandalonePlePart));
        }
        // MaxPool with stride 1
        else if (poolingInfo.m_PoolingType == PoolingType::MAX && poolingInfo.m_PoolingStrideX == 1 &&
                 poolingInfo.m_PoolingStrideY == 1)
        {
            std::map<std::string, std::string> selectionStringParams = {
                { "datatype", outputInfo.m_DataType == DataType::INT8_QUANTIZED ? "s8" : "u8" }
            };
            const std::vector<QuantizationInfo> inputQuantizations = {
                pooling.GetInput(0).GetTensorInfo().m_QuantizationInfo
            };
            // Decompose a 2D pooling into 2 x 1D pooling (first X then Y)
            TensorShape intermediateTensorShape = pooling.GetInput(0).GetTensorInfo().m_Dimensions;
            intermediateTensorShape[2]          = GetWidth(pooling.GetOutput(0).GetTensorInfo().m_Dimensions);

            if (poolingInfo.m_PoolingSizeX > 1)
            {
                const std::vector<TensorShape> inputShapes    = { pooling.GetInput(0).GetTensorInfo().m_Dimensions };
                std::map<std::string, int> selectionIntParams = { { "is_direction_x", 1 } };
                std::map<std::string, int> runtimeParams      = {
                    { "pooling_size", poolingInfo.m_PoolingSizeX },
                    { "pad_before", poolingInfo.m_Padding.m_Left },
                };

                auto poolingStandalonePlePartX = std::make_unique<StandalonePlePart>(
                    m_GraphOfParts.GeneratePartId(), inputShapes, intermediateTensorShape, inputQuantizations,
                    pooling.GetOutput(0).GetTensorInfo().m_QuantizationInfo, PleOperation::MAXPOOL1D,
                    m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                    std::set<uint32_t>{ pooling.GetId() }, pooling.GetOutput(0).GetTensorInfo().m_DataType,
                    selectionStringParams, selectionIntParams, runtimeParams);
                parts.push_back(poolingStandalonePlePartX.get());
                m_GraphOfParts.AddPart(std::move(poolingStandalonePlePartX));
            }

            if (poolingInfo.m_PoolingSizeY > 1)
            {
                const std::vector<TensorShape> inputShapes    = { intermediateTensorShape };
                std::map<std::string, int> selectionIntParams = { { "is_direction_y", 1 } };
                std::map<std::string, int> runtimeParams      = {
                    { "pooling_size", poolingInfo.m_PoolingSizeY },
                    { "pad_before", poolingInfo.m_Padding.m_Top },
                };
                auto poolingStandalonePlePartY = std::make_unique<StandalonePlePart>(
                    m_GraphOfParts.GeneratePartId(), inputShapes, pooling.GetOutput(0).GetTensorInfo().m_Dimensions,
                    inputQuantizations, pooling.GetOutput(0).GetTensorInfo().m_QuantizationInfo,
                    PleOperation::MAXPOOL1D, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                    std::set<uint32_t>{ pooling.GetId() }, pooling.GetOutput(0).GetTensorInfo().m_DataType,
                    selectionStringParams, selectionIntParams, runtimeParams);
                parts.push_back(poolingStandalonePlePartY.get());
                m_GraphOfParts.AddPart(std::move(poolingStandalonePlePartY));
            }
        }
        else
        {
            // This should have already been caught by the supported checks
            throw InternalErrorException("Unsupported pooling configuration");
        }
    }

    ConnectParts(pooling, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Reshape& reshape)
{
    std::vector<BasePart*> parts;
    auto reshapePart = std::make_unique<ReshapePart>(
        m_GraphOfParts.GeneratePartId(), reshape.GetInput(0).GetTensorInfo().m_Dimensions,
        reshape.GetOutput(0).GetTensorInfo().m_Dimensions, reshape.GetOutput(0).GetTensorInfo().m_QuantizationInfo,
        reshape.GetOutput(0).GetTensorInfo().m_DataType, std::set<uint32_t>{ reshape.GetId() },
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
    parts.push_back(reshapePart.get());
    m_GraphOfParts.AddPart(std::move(reshapePart));
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
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
    }
    else
    {
        std::map<std::string, std::string> selectionStringParams = {
            { "datatype", outputInfo.m_DataType == DataType::INT8_QUANTIZED ? "s8" : "u8" }
        };

        bool isQuantInfoIdentical = (quantInfoInput0 == quantInfoInput1) && (quantInfoInput0 == quantInfoOutput);

        // use non-scaling PLE kernel if all quant info is identical for both inputs and output
        PleOperation pleOp = isQuantInfoIdentical ? PleOperation::ADDITION : PleOperation::ADDITION_RESCALE;

        const std::vector<QuantizationInfo> inputQuantizations = { quantInfoInput0, quantInfoInput1 };
        const std::vector<TensorShape> inputShapes             = { addition.GetInput(0).GetTensorInfo().m_Dimensions,
                                                       addition.GetInput(1).GetTensorInfo().m_Dimensions };

        // Addition still uses the definition of blocks even though it doesn't come from the MCE
        std::map<std::string, int> selectionIntParams = {
            { "block_width", 16 },
            { "block_height", 16 },
        };

        const double outputScale = quantInfoOutput.GetScale();

        uint16_t input0Multiplier = 0;
        uint16_t input0Shift      = 0;
        const double inputScale0  = quantInfoInput0.GetScale();
        utils::CalculateRescaleMultiplierAndShift(inputScale0 / outputScale, input0Multiplier, input0Shift);

        uint16_t input1Multiplier = 0;
        uint16_t input1Shift      = 0;
        const double inputScale1  = quantInfoInput1.GetScale();
        utils::CalculateRescaleMultiplierAndShift(inputScale1 / outputScale, input1Multiplier, input1Shift);

        std::map<std::string, int> runtimeParams = {
            { "input0_multiplier", input0Multiplier },
            { "input0_shift", input0Shift },
            { "input1_multiplier", input1Multiplier },
            { "input1_shift", input1Shift },
        };

        auto additionStandalonePlePart = std::make_unique<StandalonePlePart>(
            m_GraphOfParts.GeneratePartId(), inputShapes, addition.GetOutput(0).GetTensorInfo().m_Dimensions,
            inputQuantizations, addition.GetOutput(0).GetTensorInfo().m_QuantizationInfo, pleOp,
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities, std::set<uint32_t>{ addition.GetId() },
            addition.GetOutput(0).GetTensorInfo().m_DataType, selectionStringParams, selectionIntParams, runtimeParams);
        parts.push_back(additionStandalonePlePart.get());
        m_GraphOfParts.AddPart(std::move(additionStandalonePlePart));
    }

    ConnectParts(addition, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Multiplication& multiplication)
{
    std::vector<BasePart*> parts;

    const auto& inputInfo0 = multiplication.GetInput(0).GetTensorInfo();
    const auto& inputInfo1 = multiplication.GetInput(1).GetTensorInfo();
    const auto& outputInfo = multiplication.GetOutput(0).GetTensorInfo();

    const QuantizationInfo& quantInfoInput0 = inputInfo0.m_QuantizationInfo;
    const QuantizationInfo& quantInfoInput1 = inputInfo1.m_QuantizationInfo;
    const QuantizationInfo& quantInfoOutput = outputInfo.m_QuantizationInfo;

    char reason[1024];

    // Check if this is supported only as an estimate-only, and if so use an EstimateOnlyPart
    const SupportedLevel supportedLevel =
        m_Queries.IsMultiplicationSupported(inputInfo0, inputInfo1, quantInfoOutput, nullptr, reason, sizeof(reason));

    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ inputInfo0, inputInfo1 },
            std::vector<TensorInfo>{ outputInfo }, ConvertExternalToCompilerDataFormat(outputInfo.m_DataFormat),
            std::set<uint32_t>{ multiplication.GetId() }, m_EstimationOptions.value(), m_CompilationOptions,
            m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
    }
    else
    {
        std::map<std::string, std::string> selectionStringParams = {
            { "datatype", outputInfo.m_DataType == DataType::INT8_QUANTIZED ? "s8" : "u8" }
        };

        PleOperation pleOp = PleOperation::MULTIPLICATION;

        const std::vector<QuantizationInfo> inputQuantizations = { quantInfoInput0, quantInfoInput1 };
        const std::vector<TensorShape> inputShapes = { multiplication.GetInput(0).GetTensorInfo().m_Dimensions,
                                                       multiplication.GetInput(1).GetTensorInfo().m_Dimensions };

        const double outputScale  = quantInfoOutput.GetScale();
        const double overallScale = quantInfoInput0.GetScale() * quantInfoInput1.GetScale() / outputScale;

        uint16_t overallMultiplier = 0;
        uint16_t overallShift      = 0;
        uint16_t maxPrecision      = (outputInfo.m_DataType == DataType::INT8_QUANTIZED ? 15 : 16);
        utils::CalculateRescaleMultiplierAndShift(overallScale, overallMultiplier, overallShift, maxPrecision);

        std::map<std::string, int> runtimeParams = {
            { "overall_multiplier", overallMultiplier },
            { "overall_shift", overallShift },
            { "input0_zeropoint", quantInfoInput0.GetZeroPoint() },
            { "input1_zeropoint", quantInfoInput1.GetZeroPoint() },
            { "output_zeropoint", quantInfoOutput.GetZeroPoint() },
        };
        std::map<std::string, int> selectionIntParams = {};

        auto multiplicationStandalonePlePart = std::make_unique<StandalonePlePart>(
            m_GraphOfParts.GeneratePartId(), inputShapes, multiplication.GetOutput(0).GetTensorInfo().m_Dimensions,
            inputQuantizations, multiplication.GetOutput(0).GetTensorInfo().m_QuantizationInfo, pleOp,
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
            std::set<uint32_t>{ multiplication.GetId() }, multiplication.GetOutput(0).GetTensorInfo().m_DataType,
            selectionStringParams, selectionIntParams, runtimeParams);
        parts.push_back(multiplicationStandalonePlePart.get());
        m_GraphOfParts.AddPart(std::move(multiplicationStandalonePlePart));
    }

    ConnectParts(multiplication, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Concatenation& concat)
{
    const size_t numInputs                  = concat.GetInputs().size();
    const QuantizationInfo& outputQuantInfo = concat.GetOutput(0).GetTensorInfo().m_QuantizationInfo;
    const DataType outputDataType           = concat.GetOutput(0).GetTensorInfo().m_DataType;
    const ConcatenationInfo& concatInfo     = concat.GetConcatenationInfo();

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
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
        ConnectParts(concat, parts);
    }
    else
    {
        // The ConcatPart assumes that all Inputs and the Output have the same quantization information.
        // If that is not the case, a requantize McePart is generated for any Inputs that are different to the Output.
        // Subsequently, all generated MceParts, as well as the ConcatPart are connected to the GraphOfParts.
        std::map<uint32_t, PartId> mcePartIds;
        std::vector<uint32_t> offsets;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < numInputs; i++)
        {
            offsets.push_back(offset);
            offset += concat.GetInput(i).GetTensorInfo().m_Dimensions[concatInfo.m_Axis];

            TensorInfo mceOperationInput  = concat.GetInput(i).GetTensorInfo();
            TensorInfo mceOperationOutput = mceOperationInput;
            Operand& inputOperand         = concat.GetInput(i);
            if (inputOperand.GetTensorInfo().m_QuantizationInfo != outputQuantInfo)
            {
                auto mcePart = CreateIdentityMcePart(
                    inputOperand.GetTensorInfo().m_Dimensions, inputOperand.GetTensorInfo().m_QuantizationInfo,
                    outputQuantInfo, concat.GetId(), mceOperationInput.m_DataType, mceOperationOutput.m_DataType,
                    m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

                // Add the connection to the GraphOfParts, then store the new PartId in a temporary map and then add the McePart to the GraphOfParts.
                m_GraphOfParts.AddConnection(
                    { mcePart->GetPartId(), 0 },
                    { m_OperandToPart.at(&inputOperand)->GetPartId(), inputOperand.GetProducerOutputIndex() });
                mcePartIds[i] = mcePart->GetPartId();
                m_GraphOfParts.AddPart(std::move(mcePart));

                inputTensorsInfo[i].m_QuantizationInfo = outputQuantInfo;
            }
        }

        // Optimisation: if we are concating in channels with any non-multiples of the brick-group-depth (16),
        // then this can be very slow for the firmware because it needs to split into lots of chunks. Instead,
        // we pad the output tensor so that we can concat on multiples of 16 instead (i.e. aligning the join points)
        // and then add a following conv layer that removes these padding channels for the next layer to consume
        TensorInfo concatOutputTensorInfo = concat.GetOutput(0).GetTensorInfo();
        std::vector<std::pair<uint32_t, uint32_t>> removeAmounts;
        if (concatInfo.m_Axis == 3)
        {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < numInputs; ++i)
            {
                offsets[i] = offset;
                offset += concat.GetInput(i).GetTensorInfo().m_Dimensions[3];
                uint32_t rem = concat.GetInput(i).GetTensorInfo().m_Dimensions[3] % g_BrickGroupShape[3];
                if (rem != 0)
                {
                    uint32_t numPadChannels = g_BrickGroupShape[3] - rem;
                    removeAmounts.push_back(std::make_pair(offset, numPadChannels));
                    offset += numPadChannels;
                }
            }
            concatOutputTensorInfo.m_Dimensions[3] = offset;
        }

        // Check whether we should prefer to use NHWC
        // Generally we prefer to use NHWCB if we can, as it should be the more efficient format.
        // However, if all our inputs are likely to produce NHWC outputs, then it is probably better
        // to use NHWC, as it avoids the need for conversion.
        bool allInputsPreferNhwc = true;
        for (uint32_t i = 0; i < numInputs; ++i)
        {
            if (!m_OperandToPart.at(&concat.GetInput(i))->IsOutputGuaranteedNhwc())
            {
                allInputsPreferNhwc = false;
            }
        }

        std::vector<BasePart*> parts;

        auto concatPart = std::make_unique<ConcatPart>(
            m_GraphOfParts.GeneratePartId(), inputTensorsInfo, concatOutputTensorInfo, concatInfo.m_Axis, offsets,
            allInputsPreferNhwc, std::set<uint32_t>{ concat.GetId() }, m_EstimationOptions.value(),
            m_CompilationOptions, m_Capabilities);
        ConcatPart* concatPartRaw = concatPart.get();
        parts.push_back(concatPartRaw);
        m_GraphOfParts.AddPart(std::move(concatPart));

        if (!removeAmounts.empty())
        {
            std::unique_ptr<McePart> paddingPart = CreateIdentityMcePartWithRemovedInputChannels(
                m_GraphOfParts.GeneratePartId(), concatOutputTensorInfo.m_Dimensions, outputQuantInfo, outputQuantInfo,
                concat.GetId(), outputDataType, outputDataType, m_EstimationOptions.value(), m_CompilationOptions,
                m_Capabilities, removeAmounts, m_DebuggingContext, m_ThreadPool);
            parts.push_back(paddingPart.get());
            m_GraphOfParts.AddConnection({ paddingPart->GetPartId(), 0 }, { concatPartRaw->GetPartId(), 0 });
            m_GraphOfParts.AddPart(std::move(paddingPart));
        }

        // Connect ConcatPart to the GraphOfParts. Loop through all Inputs of the ConcatPart and determine whether:
        // 1. There is a direct connection of ConcatPart with the preceding Part.
        // 2. There is a connection of ConcatPart with the respective requantise McePart.
        for (uint32_t i = 0; i < numInputs; i++)
        {
            Operand& inputOperand = concat.GetInput(i);
            if (mcePartIds.find(i) != mcePartIds.end())
            {
                m_GraphOfParts.AddConnection({ concatPartRaw->GetPartId(), i }, { mcePartIds[i], 0 });
            }
            else
            {
                m_GraphOfParts.AddConnection(
                    { concatPartRaw->GetPartId(), i },
                    { m_OperandToPart.at(&inputOperand)->GetPartId(), inputOperand.GetProducerOutputIndex() });
            }
        }

        // Mark the ConcatPart Output for connection with any subsequent Parts.
        m_OperandToPart[&concat.GetOutput(0)] = parts.back();
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
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
        ConnectParts(requantize, parts);
    }
    else
    {
        auto inputQuantInfo  = requantize.GetInput(0).GetTensorInfo().m_QuantizationInfo;
        auto outputQuantInfo = requantize.GetOutput(0).GetTensorInfo().m_QuantizationInfo;

        TensorInfo inputInfo  = requantize.GetInput(0).GetTensorInfo();
        TensorInfo outputInfo = requantize.GetOutput(0).GetTensorInfo();

        // If input and output quantizations are different, an McePart is added to the GraphOfParts to perform requantization,
        // otherwise the requantize operation is optimized out (no requantization needed)
        Operand& inputOperand = requantize.GetInput(0);
        if (inputQuantInfo != outputQuantInfo)
        {
            auto mcePart =
                CreateIdentityMcePart(inputOperand.GetTensorInfo().m_Dimensions, inputQuantInfo, outputQuantInfo,
                                      requantize.GetId(), inputInfo.m_DataType, outputInfo.m_DataType,
                                      m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

            parts.push_back(mcePart.get());
            m_GraphOfParts.AddPart(std::move(mcePart));
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
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
    }
    else
    {
        TensorInfo inputInfo  = leakyRelu.GetInput(0).GetTensorInfo();
        TensorInfo outputInfo = leakyRelu.GetOutput(0).GetTensorInfo();

        std::map<std::string, std::string> selectionStringParams = {
            { "datatype", outputInfo.m_DataType == DataType::INT8_QUANTIZED ? "s8" : "u8" }
        };

        const double alphaRescaleFactor =
            leakyRelu.GetLeakyReluInfo().m_Alpha *
            (inputInfo.m_QuantizationInfo.GetScale() / outputInfo.m_QuantizationInfo.GetScale());
        uint16_t alphaMult;
        uint16_t alphaShift;
        CalculateRescaleMultiplierAndShift(alphaRescaleFactor, alphaMult, alphaShift);

        const double inputToOutputRescaleFactor =
            (inputInfo.m_QuantizationInfo.GetScale() / outputInfo.m_QuantizationInfo.GetScale());
        uint16_t inputToOutputMult;
        uint16_t inputToOutputShift;
        CalculateRescaleMultiplierAndShift(inputToOutputRescaleFactor, inputToOutputMult, inputToOutputShift);

        std::map<std::string, int> runtimeParams = {
            { "input0_multiplier", inputToOutputMult },
            { "input0_shift", inputToOutputShift },
            // We "misuse" the input1 multiplier/shift here
            { "input1_multiplier", alphaMult },
            { "input1_shift", alphaShift },
        };

        auto leakyReluPart = std::make_unique<FusedPlePart>(
            m_GraphOfParts.GeneratePartId(), leakyRelu.GetInput(0).GetTensorInfo().m_Dimensions,
            leakyRelu.GetOutput(0).GetTensorInfo().m_Dimensions,
            leakyRelu.GetInput(0).GetTensorInfo().m_QuantizationInfo,
            leakyRelu.GetOutput(0).GetTensorInfo().m_QuantizationInfo, PleOperation::LEAKY_RELU,
            g_IdentityShapeMultiplier, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
            std::set<uint32_t>{ leakyRelu.GetId() }, inputInfo.m_DataType, outputInfo.m_DataType, m_DebuggingContext,
            m_ThreadPool, selectionStringParams, std::map<std::string, int>{}, runtimeParams);
        parts.push_back(leakyReluPart.get());
        m_GraphOfParts.AddPart(std::move(leakyReluPart));
    }

    ConnectParts(leakyRelu, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Sigmoid& sigmoid)
{
    std::vector<BasePart*> parts;

    TensorInfo inputInfo  = sigmoid.GetInput(0).GetTensorInfo();
    TensorInfo outputInfo = sigmoid.GetOutput(0).GetTensorInfo();

    std::map<std::string, std::string> selectionStringParams = {
        { "datatype", outputInfo.m_DataType == DataType::INT8_QUANTIZED ? "s8" : "u8" }
    };

    constexpr double log2e = 1.4426950408889634;

    const double inputScale = inputInfo.m_QuantizationInfo.GetScale();

    const double rescaleFactor = inputScale * (log2e * 256.0);

    assert(outputInfo.m_QuantizationInfo.GetScale() == (1.0f / 256.0f));

    uint16_t input0Multiplier = 0;
    uint16_t input0Shift      = 0;
    utils::CalculateRescaleMultiplierAndShift(rescaleFactor, input0Multiplier, input0Shift);

    int absMax = static_cast<int>(std::ceil(std::ldexp(1.0, 15U + input0Shift) / input0Multiplier)) - 1;

    if (absMax == 0)
    {
        absMax = 1;

        input0Multiplier = INT16_MAX;
        input0Shift      = 0;
    }

    std::map<std::string, int> runtimeParams = {
        { "input0_multiplier", input0Multiplier },
        { "input0_shift", input0Shift },
    };

    auto sigmoidPart = std::make_unique<FusedPlePart>(
        m_GraphOfParts.GeneratePartId(), sigmoid.GetInput(0).GetTensorInfo().m_Dimensions,
        sigmoid.GetOutput(0).GetTensorInfo().m_Dimensions, sigmoid.GetInput(0).GetTensorInfo().m_QuantizationInfo,
        sigmoid.GetOutput(0).GetTensorInfo().m_QuantizationInfo, PleOperation::SIGMOID, g_IdentityShapeMultiplier,
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities, std::set<uint32_t>{ sigmoid.GetId() },
        inputInfo.m_DataType, outputInfo.m_DataType, m_DebuggingContext, m_ThreadPool, selectionStringParams,
        std::map<std::string, int>{}, runtimeParams);
    parts.push_back(sigmoidPart.get());
    m_GraphOfParts.AddPart(std::move(sigmoidPart));
    ConnectParts(sigmoid, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Tanh& tanh)
{
    // Note that Tanh and Sigmoid share the same PLE operation
    // The differences are:
    // (1) Input scaling factor
    // (2) Output quantization
    std::vector<BasePart*> parts;

    TensorInfo inputInfo  = tanh.GetInput(0).GetTensorInfo();
    TensorInfo outputInfo = tanh.GetOutput(0).GetTensorInfo();

    std::map<std::string, std::string> selectionStringParams = {
        { "datatype", outputInfo.m_DataType == DataType::INT8_QUANTIZED ? "s8" : "u8" }
    };

    constexpr double log2e = 1.4426950408889634;

    const double inputScale = inputInfo.m_QuantizationInfo.GetScale();

    const double rescaleFactor = inputScale * (log2e * 256.0) * 2.0;

    assert(outputInfo.m_QuantizationInfo.GetScale() == (1.0f / 128));

    uint16_t input0Multiplier = 0;
    uint16_t input0Shift      = 0;
    utils::CalculateRescaleMultiplierAndShift(rescaleFactor, input0Multiplier, input0Shift);

    int absMax = static_cast<int>(std::ceil(std::ldexp(1.0, 15U + input0Shift) / input0Multiplier)) - 1;

    if (absMax == 0)
    {
        absMax = 1;

        input0Multiplier = INT16_MAX;
        input0Shift      = 0;
    }

    std::map<std::string, int> runtimeParams = {
        { "input0_multiplier", input0Multiplier },
        { "input0_shift", input0Shift },
    };

    auto tanhPart = std::make_unique<FusedPlePart>(
        m_GraphOfParts.GeneratePartId(), tanh.GetInput(0).GetTensorInfo().m_Dimensions,
        tanh.GetOutput(0).GetTensorInfo().m_Dimensions, tanh.GetInput(0).GetTensorInfo().m_QuantizationInfo,
        tanh.GetOutput(0).GetTensorInfo().m_QuantizationInfo, PleOperation::SIGMOID, g_IdentityShapeMultiplier,
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities, std::set<uint32_t>{ tanh.GetId() },
        inputInfo.m_DataType, outputInfo.m_DataType, m_DebuggingContext, m_ThreadPool, selectionStringParams,
        std::map<std::string, int>{}, runtimeParams);
    parts.push_back(tanhPart.get());
    m_GraphOfParts.AddPart(std::move(tanhPart));
    ConnectParts(tanh, parts);
}

void NetworkToGraphOfPartsConverter::Visit(MeanXy& meanxy)
{
    std::vector<BasePart*> parts;
    ShapeMultiplier shapeMultiplier = { 1, 1, 1 };
    PleOperation pleOperation;
    if (meanxy.GetInput(0).GetTensorInfo().m_Dimensions[1] == 7)
    {
        pleOperation = PleOperation::MEAN_XY_7X7;
    }
    else
    {
        pleOperation = PleOperation::MEAN_XY_8X8;
    }

    TensorInfo inputInfo  = meanxy.GetInput(0).GetTensorInfo();
    TensorInfo outputInfo = meanxy.GetOutput(0).GetTensorInfo();

    std::map<std::string, std::string> selectionStringParams = {
        { "datatype", outputInfo.m_DataType == DataType::INT8_QUANTIZED ? "s8" : "u8" }
    };

    auto meanxyPart = std::make_unique<FusedPlePart>(
        m_GraphOfParts.GeneratePartId(), meanxy.GetInput(0).GetTensorInfo().m_Dimensions,
        meanxy.GetOutput(0).GetTensorInfo().m_Dimensions, meanxy.GetInput(0).GetTensorInfo().m_QuantizationInfo,
        meanxy.GetOutput(0).GetTensorInfo().m_QuantizationInfo, pleOperation, shapeMultiplier,
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities, std::set<uint32_t>{ meanxy.GetId() },
        inputInfo.m_DataType, outputInfo.m_DataType, m_DebuggingContext, m_ThreadPool, selectionStringParams,
        std::map<std::string, int>{}, std::map<std::string, int>{});
    parts.push_back(meanxyPart.get());
    m_GraphOfParts.AddPart(std::move(meanxyPart));
    ConnectParts(meanxy, parts);
}

void NetworkToGraphOfPartsConverter::Visit(EstimateOnly& estimateOnly)
{
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

    EstimateOnlyPart* estimateOnlyPartRaw = estimateOnlyPart.get();
    m_GraphOfParts.AddPart(std::move(estimateOnlyPart));

    // Connect to inputs
    for (uint32_t inputSlot = 0; inputSlot < estimateOnly.GetInputs().size(); ++inputSlot)
    {
        const Operand* op = estimateOnly.GetInputs()[inputSlot];
        m_GraphOfParts.AddConnection({ estimateOnlyPartRaw->GetPartId(), inputSlot },
                                     { m_OperandToPart.at(op)->GetPartId(), op->GetProducerOutputIndex() });
    }

    for (const Operand& outputOperand : estimateOnly.GetOutputs())
    {
        m_OperandToPart[&outputOperand] = estimateOnlyPartRaw;
    }
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

    McePart::ConstructionParams params(m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                                       m_DebuggingContext, m_ThreadPool);
    params.m_Id                     = m_GraphOfParts.GeneratePartId();
    params.m_InputTensorShape       = inputShape;
    params.m_OutputTensorShape      = outputInfo.m_Dimensions;
    params.m_InputQuantizationInfo  = inputInfo.m_QuantizationInfo;
    params.m_OutputQuantizationInfo = outputInfo.m_QuantizationInfo;
    const uint32_t numIfm           = inputShape[3];
    const float weightScale         = 0.5f;
    params.m_WeightsInfo    = { { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale } };
    params.m_WeightsData    = std::vector<uint8_t>(1 * 1 * 1 * numIfm, 2);
    const float biasScale   = weightScale * inputInfo.m_QuantizationInfo.GetScale();
    params.m_BiasInfo       = { { 1, 1, 1, numIfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };
    params.m_BiasData       = std::vector<int32_t>(numIfm, 0);
    params.m_Op             = command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    params.m_OperationIds   = std::set<uint32_t>{ resize.GetId() };
    params.m_InputDataType  = inputInfo.m_DataType;
    params.m_OutputDataType = outputInfo.m_DataType;
    params.m_UpscaleFactor  = upscaleFactorHeight;
    params.m_UpsampleType   = ConvertResizeAlgorithmToMceUpsampleType(resizeInfo.m_Algo);
    params.m_LowerBound     = outputInfo.m_DataType == DataType::UINT8_QUANTIZED ? 0 : -128;
    params.m_UpperBound     = outputInfo.m_DataType == DataType::UINT8_QUANTIZED ? 255 : 127;
    auto mcePart            = std::make_unique<McePart>(std::move(params));

    std::vector<BasePart*> parts;
    parts.push_back(mcePart.get());
    m_GraphOfParts.AddPart(std::move(mcePart));
    ConnectParts(resize, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Relu& relu)
{
    ReluInfo info               = relu.GetReluInfo();
    const Operand& inputOperand = relu.GetInput(0);

    TensorInfo inputInfo  = relu.GetInput(0).GetTensorInfo();
    TensorInfo outputInfo = relu.GetOutput(0).GetTensorInfo();

    std::vector<BasePart*> parts;

    auto iterator = m_OperandToPart.find(&inputOperand);
    assert(iterator != m_OperandToPart.end());
    BasePart* inputPart = iterator->second;
    assert(inputPart);

    // Multiple cases:
    //    * Mce -> Relu, and no other consumers of the Mce: We need to update the relu bounds in the mce op.
    //    * Otherwise: We need to insert an identity mce operation with new relu bounds
    if (!inputPart->HasActivationBounds() || inputOperand.GetConsumers().size() > 1)
    {
        std::unique_ptr<McePart> mcePart = CreateIdentityMcePart(
            inputOperand.GetTensorInfo().m_Dimensions, inputOperand.GetTensorInfo().m_QuantizationInfo,
            inputOperand.GetTensorInfo().m_QuantizationInfo, relu.GetId(), inputInfo.m_DataType, outputInfo.m_DataType,
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        inputPart = mcePart.get();
        parts.push_back(mcePart.get());
        m_GraphOfParts.AddPart(std::move(mcePart));
        ConnectParts(relu, parts);
    }

    // If the input to the relu has activations we need to modify them
    inputPart->ApplyActivationBounds(info.m_LowerBound, info.m_UpperBound);
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
    uint32_t upscaleFactor          = stride.m_X;
    MceUpsampleType upsampleType    = MceUpsampleType::TRANSPOSE;
    const TensorShape& weightsShape = weightsInfo.m_Dimensions;

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

        McePart::ConstructionParams params(m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                                           m_DebuggingContext, m_ThreadPool);
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
        params.m_Padding                = Padding();
        params.m_Op                     = command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
        params.m_OperationIds           = operationIds;
        params.m_InputDataType          = inputInfo.m_DataType;
        params.m_OutputDataType         = inputInfo.m_DataType;
        params.m_UpscaleFactor          = upscaleFactor;
        params.m_UpsampleType           = upsampleType;
        params.m_LowerBound             = inputInfo.m_DataType == DataType::UINT8_QUANTIZED ? 0 : -128;
        params.m_UpperBound             = inputInfo.m_DataType == DataType::UINT8_QUANTIZED ? 255 : 127;

        auto identityDepthwisePart = std::make_unique<McePart>(std::move(params));
        parts.push_back(identityDepthwisePart.get());
        m_GraphOfParts.AddPart(std::move(identityDepthwisePart));

        upscaleFactor = 1;
        upsampleType  = MceUpsampleType::OFF;
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

    McePart::ConstructionParams params(m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
                                       m_DebuggingContext, m_ThreadPool);
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
    params.m_Padding                = Padding(topMcePadding, 0, leftMcePadding, 0);
    params.m_Op                     = command_stream::MceOperation::CONVOLUTION;
    params.m_OperationIds           = operationIds;
    params.m_InputDataType          = inputInfo.m_DataType;
    params.m_OutputDataType         = outputInfo.m_DataType;
    params.m_UpscaleFactor          = upscaleFactor;
    params.m_UpsampleType           = upsampleType;
    params.m_LowerBound             = outputInfo.m_DataType == DataType::UINT8_QUANTIZED ? 0 : -128;
    params.m_UpperBound             = outputInfo.m_DataType == DataType::UINT8_QUANTIZED ? 255 : 127;
    auto mcePart                    = std::make_unique<McePart>(std::move(params));

    parts.push_back(mcePart.get());
    m_GraphOfParts.AddPart(std::move(mcePart));

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
    std::vector<int32_t> biasData   = GetDataVectorAs<int32_t, uint8_t>(transposeConvolution.GetBias().GetDataVector());
    const Padding& padding          = transposeConvolution.GetConvolutionInfo().m_Padding;
    const TensorInfo& inputInfo     = transposeConvolution.GetInput(0).GetTensorInfo();
    const TensorInfo& outputInfo    = transposeConvolution.GetOutput(0).GetTensorInfo();
    std::set<uint32_t> operationIds = { transposeConvolution.GetId(), transposeConvolution.GetBias().GetId(),
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
            std::move(operationIds), m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
    }
    else
    {
        parts = CreateTransposeConv(stride, weightsInfo, weightsData, biasInfo, std::move(biasData), padding, inputInfo,
                                    outputInfo, operationIds);
    }

    ConnectParts(transposeConvolution, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Split& split)
{
    const SplitInfo& splitInfo = split.GetSplitInfo();
    const size_t numOutputs    = split.GetOutputs().size();

    TensorInfo inputInfo                  = split.GetInput(0).GetTensorInfo();
    const std::set<uint32_t> operationIds = { split.GetId() };

    std::vector<BasePart*> parts;
    std::vector<uint32_t> offsets;
    std::vector<TensorInfo> outputTensorInfos;
    {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < numOutputs; ++i)
        {
            outputTensorInfos.push_back(split.GetOutput(i).GetTensorInfo());

            offsets.push_back(offset);
            offset += splitInfo.m_Sizes[i];
        }
    }

    // Optimisation: if we are splitting in channels with any non-multiples of the brick-group-depth (16),
    // then this can be very slow for the firmware because it needs to split into lots of chunks. Instead,
    // we insert a conv layer that "pads" the output channels of the previous layer so that we can split on
    // multiples of 16 instead (i.e. aligning the split points)
    McePart* paddingPartRaw = nullptr;
    if (splitInfo.m_Axis == 3)
    {
        std::vector<std::pair<uint32_t, uint32_t>> padAmounts;
        uint32_t origOffset = 0;
        uint32_t newOffset  = 0;
        for (uint32_t i = 0; i < numOutputs; ++i)
        {
            offsets[i] = newOffset;
            origOffset += split.GetOutput(i).GetTensorInfo().m_Dimensions[3];
            newOffset += split.GetOutput(i).GetTensorInfo().m_Dimensions[3];
            uint32_t rem = split.GetOutput(i).GetTensorInfo().m_Dimensions[3] % g_BrickGroupShape[3];
            if (rem != 0)
            {
                uint32_t numPadChannels = g_BrickGroupShape[3] - rem;
                padAmounts.push_back(std::make_pair(origOffset, numPadChannels));
                newOffset += numPadChannels;
            }
        }
        const uint32_t newInputDepth = newOffset;

        if (!padAmounts.empty())
        {
            std::unique_ptr<McePart> paddingPart = CreateIdentityMcePartWithPaddedOutputChannels(
                m_GraphOfParts.GeneratePartId(), inputInfo.m_Dimensions, inputInfo.m_QuantizationInfo,
                inputInfo.m_QuantizationInfo, split.GetId(), inputInfo.m_DataType, inputInfo.m_DataType,
                m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities, padAmounts, m_DebuggingContext,
                m_ThreadPool);
            paddingPartRaw = paddingPart.get();
            parts.push_back(paddingPartRaw);
            m_GraphOfParts.AddPart(std::move(paddingPart));

            inputInfo.m_Dimensions[3] = newInputDepth;
        }
    }

    auto splitPart = std::make_unique<SplitPart>(m_GraphOfParts.GeneratePartId(), inputInfo, outputTensorInfos,
                                                 splitInfo.m_Axis, offsets, std::set<uint32_t>{ split.GetId() },
                                                 m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

    parts.push_back(splitPart.get());
    if (paddingPartRaw)
    {
        m_GraphOfParts.AddConnection({ splitPart->GetPartId(), 0 }, { paddingPartRaw->GetPartId(), 0 });
    }

    auto inputQuantInfo = split.GetInput(0).GetTensorInfo().m_QuantizationInfo;
    // The SplitPart assumes that all Inputs and the Output have the same quantization information.
    // If that is not the case, a requantize McePart is generated for any Outputs that are different to the Input.
    // Subsequently, all generated MceParts, as well as the SplitPart are connected to the GraphOfParts.
    for (uint32_t i = 0; i < numOutputs; ++i)
    {
        Operand& outputOperand = split.GetOutput(i);
        if (outputOperand.GetTensorInfo().m_QuantizationInfo != inputQuantInfo)
        {
            std::map<uint32_t, PartId> mcePartIds;

            // Note the dimensions used here deliberately do not account for any padding channels, as they should be implicitly
            // removed at this point.
            auto mcePart = CreateIdentityMcePart(outputOperand.GetTensorInfo().m_Dimensions,
                                                 outputOperand.GetTensorInfo().m_QuantizationInfo, inputQuantInfo,
                                                 split.GetId(), split.GetOutput(0).GetTensorInfo().m_DataType,
                                                 split.GetOutput(0).GetTensorInfo().m_DataType,
                                                 m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);

            // Add the connection to the GraphOfParts, then store the new PartId in a temporary map and then add the McePart to the GraphOfParts.
            m_GraphOfParts.AddConnection({ mcePart->GetPartId(), 0 },
                                         { splitPart->GetPartId(), outputOperand.GetProducerOutputIndex() });
            mcePartIds[i] = mcePart->GetPartId();

            parts.push_back(mcePart.get());
            m_GraphOfParts.AddPart(std::move(mcePart));

            m_OperandToPart[&outputOperand] = parts.back();
        }
        else
        {
            // If no mcePart required then simply connect outputParts to Split op
            m_OperandToPart[&outputOperand] = parts.back();
        }
    }
    m_GraphOfParts.AddPart(std::move(splitPart));

    Operand& operand = split.GetInput(0);
    m_GraphOfParts.AddConnection({ parts.front()->GetPartId(), 0 },
                                 { m_OperandToPart.at(&operand)->GetPartId(), operand.GetProducerOutputIndex() });
}

void NetworkToGraphOfPartsConverter::Visit(Transpose& transpose)
{
    std::vector<Node*> nodes;

    const auto& inputOperand    = transpose.GetInput(0);
    const auto& inputTensorInfo = inputOperand.GetTensorInfo();
    auto& outputTensorInfo      = transpose.GetOutput(0).GetTensorInfo();
    auto& permutation           = transpose.GetTransposeInfo().m_Permutation;
    // Figure out if transpose can be performed via data conversion node
    // transposeInfo contains the tensor reordering in <> format for output
    // i.e <0, 3, 1, 2> means N->N, C->H, W->H, H->C <N,H,W,C> becomes <N,C,W,H>.

    char reason[1024];
    const SupportedLevel supportedLevel =
        m_Queries.IsTransposeSupported(transpose.GetTransposeInfo(), inputTensorInfo, nullptr, reason, sizeof(reason));
    std::vector<BasePart*> parts;

    if (supportedLevel == SupportedLevel::EstimateOnly)
    {
        auto estimateOnlyPart = std::make_unique<EstimateOnlyPart>(
            m_GraphOfParts.GeneratePartId(), reason, std::vector<TensorInfo>{ inputTensorInfo },
            std::vector<TensorInfo>{ outputTensorInfo },
            ConvertExternalToCompilerDataFormat(outputTensorInfo.m_DataFormat), std::set<uint32_t>{ transpose.GetId() },
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
    }

    // Transpose to 0 3 1 2 can be performed via converting between NHWC and NCHW formats.
    // 0 3 1 2 => Data in NHWC in DRAM => Load NHWC => NHWCB, Save NCHW => Next layer interprets as NHWC
    if ((permutation[1] == 3) && (permutation[2] == 1) && (permutation[3] == 2))
    {
        auto reformatPart = std::make_unique<ReformatPart>(
            m_GraphOfParts.GeneratePartId(), transpose.GetInput(0).GetTensorInfo().m_Dimensions, BufferFormat::NHWC,
            BufferFormat::NHWC, transpose.GetOutput(0).GetTensorInfo().m_Dimensions, BufferFormat::NHWC,
            BufferFormat::NCHW, transpose.GetOutput(0).GetTensorInfo().m_QuantizationInfo,
            transpose.GetOutput(0).GetTensorInfo().m_DataType, std::set<uint32_t>{ transpose.GetId() },
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
        parts.push_back(reformatPart.get());
        m_GraphOfParts.AddPart(std::move(reformatPart));
    }
    // Transpose to 0 2 3 1 can be performed via converting between NHWC and NCHW formats.
    // 0 2 3 1 => Data in NHWC in DRAM => Load pretending it is NCHW => NWCHB, Save NHWC
    // (which will actually save as NWCH) => Next layer interprets as NHWC
    else if ((permutation[1] == 2) && (permutation[2] == 3) && (permutation[3] == 1))
    {
        auto reformatPart = std::make_unique<ReformatPart>(
            m_GraphOfParts.GeneratePartId(), transpose.GetOutput(0).GetTensorInfo().m_Dimensions, BufferFormat::NHWC,
            BufferFormat::NCHW, transpose.GetOutput(0).GetTensorInfo().m_Dimensions, BufferFormat::NHWC,
            BufferFormat::NHWC, transpose.GetOutput(0).GetTensorInfo().m_QuantizationInfo,
            transpose.GetOutput(0).GetTensorInfo().m_DataType, std::set<uint32_t>{ transpose.GetId() },
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
        parts.push_back(reformatPart.get());
        m_GraphOfParts.AddPart(std::move(reformatPart));
    }
    // Transpose to 0 2 1 3 can be performed via H and W swapping PLE kernel
    else if ((permutation[1] == 2) && (permutation[2] == 1) && (permutation[3] == 3))
    {
        const uint32_t numIfm   = inputTensorInfo.m_Dimensions[3];
        const float weightScale = 0.5f;
        const float biasScale   = weightScale * inputTensorInfo.m_QuantizationInfo.GetScale();

        std::vector<uint8_t> weightsData(1 * 1 * 1 * numIfm, 2);
        std::vector<int32_t> biasData(numIfm, 0);

        TensorInfo weightInfo{ { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale } };
        TensorInfo biasInfo{ { 1, 1, 1, numIfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };

        ShapeMultiplier shapeMultiplier = {
            Fraction{ inputTensorInfo.m_Dimensions[2], inputTensorInfo.m_Dimensions[1] },
            Fraction{ inputTensorInfo.m_Dimensions[1], inputTensorInfo.m_Dimensions[2] }, Fraction{ 1, 1 }
        };

        // Add fuse only ple operation with transpose kernel
        auto fusedPlePart = std::make_unique<FusedPlePart>(
            m_GraphOfParts.GeneratePartId(), inputTensorInfo.m_Dimensions, outputTensorInfo.m_Dimensions,
            inputTensorInfo.m_QuantizationInfo, outputTensorInfo.m_QuantizationInfo, PleOperation::TRANSPOSE_XY,
            shapeMultiplier, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
            std::set<uint32_t>{ transpose.GetId() }, inputTensorInfo.m_DataType, outputTensorInfo.m_DataType,
            m_DebuggingContext, m_ThreadPool, std::map<std::string, std::string>{}, std::map<std::string, int>{},
            std::map<std::string, int>{});
        parts.push_back(fusedPlePart.get());
        m_GraphOfParts.AddPart(std::move(fusedPlePart));
    }
    // Transpose to 0 1 3 2 utilizes converting between NHWC, NCHW formats and H & W swap ple kernel
    // Load pretending it is NCHW => NWCHB, PLE swap HW (which is actually WC) => NCWHB, Save NCHW
    // (which will actually save as NHCW) => Next layer interprets as NHWC
    else if ((permutation[1] == 1) && (permutation[2] == 3) && (permutation[3] == 2))
    {
        TensorShape intermediateShape1 = { inputTensorInfo.m_Dimensions[0], inputTensorInfo.m_Dimensions[2],
                                           inputTensorInfo.m_Dimensions[3], inputTensorInfo.m_Dimensions[1] };

        TensorShape intermediateShape2 = { inputTensorInfo.m_Dimensions[0], inputTensorInfo.m_Dimensions[3],
                                           inputTensorInfo.m_Dimensions[2], inputTensorInfo.m_Dimensions[1] };

        auto reformatPart1 = std::make_unique<ReformatPart>(
            m_GraphOfParts.GeneratePartId(), intermediateShape1, BufferFormat::NHWC, BufferFormat::NCHW,
            intermediateShape1, BufferFormat::NHWC, BufferFormat::NHWC,
            transpose.GetOutput(0).GetTensorInfo().m_QuantizationInfo,
            transpose.GetOutput(0).GetTensorInfo().m_DataType, std::set<uint32_t>{ transpose.GetId() },
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
        parts.push_back(reformatPart1.get());
        m_GraphOfParts.AddPart(std::move(reformatPart1));

        ShapeMultiplier shapeMultiplier = { Fraction{ intermediateShape1[2], intermediateShape1[1] },
                                            Fraction{ intermediateShape1[1], intermediateShape1[2] },
                                            Fraction{ 1, 1 } };

        auto fusedPlePart = std::make_unique<FusedPlePart>(
            m_GraphOfParts.GeneratePartId(), intermediateShape1, intermediateShape2, inputTensorInfo.m_QuantizationInfo,
            outputTensorInfo.m_QuantizationInfo, PleOperation::TRANSPOSE_XY, shapeMultiplier,
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities, std::set<uint32_t>{ transpose.GetId() },
            inputTensorInfo.m_DataType, outputTensorInfo.m_DataType, m_DebuggingContext, m_ThreadPool,
            std::map<std::string, std::string>{}, std::map<std::string, int>{}, std::map<std::string, int>{});
        parts.push_back(fusedPlePart.get());
        m_GraphOfParts.AddPart(std::move(fusedPlePart));

        auto reformatPart2 = std::make_unique<ReformatPart>(
            m_GraphOfParts.GeneratePartId(), intermediateShape2, BufferFormat::NHWC, BufferFormat::NHWC,
            transpose.GetOutput(0).GetTensorInfo().m_Dimensions, BufferFormat::NHWC, BufferFormat::NCHW,
            transpose.GetOutput(0).GetTensorInfo().m_QuantizationInfo,
            transpose.GetOutput(0).GetTensorInfo().m_DataType, std::set<uint32_t>{ transpose.GetId() },
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
        parts.push_back(reformatPart2.get());
        m_GraphOfParts.AddPart(std::move(reformatPart2));
    }

    // Transpose to 0 3 2 1 utilizes converting between NHWC, NCHW formats and H & W swap ple kernel
    // 0 3 2 1  => Data in NHWC in DRAM => Load pretending it is NCHW => NWCHB, PLE swap HW
    // (which is actually WC) => NCWHB, Save NHWC (which will actually save as NCWH) => Next layer
    // interprets as NHWC
    else if ((permutation[1] == 3) && (permutation[2] == 2) && (permutation[3] == 1))
    {
        TensorShape intermediateShape1 = { inputTensorInfo.m_Dimensions[0], inputTensorInfo.m_Dimensions[2],
                                           inputTensorInfo.m_Dimensions[3], inputTensorInfo.m_Dimensions[1] };

        auto reformatPart = std::make_unique<ReformatPart>(
            m_GraphOfParts.GeneratePartId(), intermediateShape1, BufferFormat::NHWC, BufferFormat::NCHW,
            intermediateShape1, BufferFormat::NHWC, BufferFormat::NHWC,
            transpose.GetOutput(0).GetTensorInfo().m_QuantizationInfo,
            transpose.GetOutput(0).GetTensorInfo().m_DataType, std::set<uint32_t>{ transpose.GetId() },
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
        parts.push_back(reformatPart.get());
        m_GraphOfParts.AddPart(std::move(reformatPart));

        ShapeMultiplier shapeMultiplier = { Fraction{ intermediateShape1[2], intermediateShape1[1] },
                                            Fraction{ intermediateShape1[1], intermediateShape1[2] },
                                            Fraction{ 1, 1 } };

        auto fusedPlePart = std::make_unique<FusedPlePart>(
            m_GraphOfParts.GeneratePartId(), intermediateShape1, outputTensorInfo.m_Dimensions,
            inputTensorInfo.m_QuantizationInfo, outputTensorInfo.m_QuantizationInfo, PleOperation::TRANSPOSE_XY,
            shapeMultiplier, m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
            std::set<uint32_t>{ transpose.GetId() }, inputTensorInfo.m_DataType, outputTensorInfo.m_DataType,
            m_DebuggingContext, m_ThreadPool, std::map<std::string, std::string>{}, std::map<std::string, int>{},
            std::map<std::string, int>{});
        parts.push_back(fusedPlePart.get());
        m_GraphOfParts.AddPart(std::move(fusedPlePart));
    }
    else if ((permutation[1] == 1) && (permutation[2] == 2) && (permutation[3] == 3))
    {
        // 0, 1, 2, 3 is equivalent to no-op.
        ConnectNoOp(transpose);
        return;
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
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
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
    const TensorInfo& inputInfo  = spaceToDepth.GetInput(0).GetTensorInfo();
    const TensorInfo& outputInfo = spaceToDepth.GetOutput(0).GetTensorInfo();

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
            std::set<uint32_t>{ spaceToDepth.GetId() }, m_EstimationOptions.value(), m_CompilationOptions,
            m_Capabilities);

        parts.push_back(estimateOnlyPart.get());
        m_GraphOfParts.AddPart(std::move(estimateOnlyPart));
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

void NetworkToGraphOfPartsConverter::ConnectParts(Operation& operation, std::vector<BasePart*>& parts)
{
    // This function currently supports Operations with no/single Output.
    // cppcheck-suppress assertWithSideEffect
    assert(operation.GetOutputs().size() <= 1);

    // Loop through all parts in the vector of BaseParts and connect them together.
    for (uint32_t i = 0; i < static_cast<uint32_t>(parts.size()) - 1; i++)
    {
        m_GraphOfParts.AddConnection({ parts[i + 1]->GetPartId(), 0 }, { parts[i]->GetPartId(), 0 });
    }

    uint32_t i = 0;
    // Loop through all input Operands of current Operation and connect first Part in vector of BaseParts with
    // the preceding Part that has the same Operand as output.
    for (const Operand* op : operation.GetInputs())
    {
        m_GraphOfParts.AddConnection({ parts.front()->GetPartId(), i },
                                     { m_OperandToPart.at(op)->GetPartId(), op->GetProducerOutputIndex() });
        i += 1;
    }

    // Check if current operation has outputs and if so mark them for connection with the subsequent operation.
    if (operation.GetOutputs().size() > 0)
    {
        m_OperandToPart[&operation.GetOutput(0)] = parts.back();
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
