//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "NetworkToGraphOfPartsConverter.hpp"

#include "GraphNodes.hpp"
#include "Utils.hpp"
#include "cascading/MceEstimationUtils.hpp"
#include <fstream>

using namespace ethosn::support_library::utils;

namespace ethosn
{
namespace support_library
{
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
    auto inputPart                        = std::make_unique<InputPart>(
        this->graphOfParts.GeneratePartId(), input.GetTensorInfo().m_Dimensions, compilerDataFormat,
        input.GetTensorInfo().m_QuantizationInfo, std::set<uint32_t>{ input.GetId() }, m_EstimationOptions.value(),
        m_CompilationOptions, m_Capabilities);
    parts.push_back(std::move(inputPart.get()));
    this->graphOfParts.m_Parts.push_back(std::move(inputPart));
    ConnectParts(input, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Output& output)
{
    std::vector<BasePart*> parts;
    CompilerDataFormat compilerDataFormat = ConvertExternalToCompilerDataFormat(output.GetTensorInfo().m_DataFormat);
    auto outputPart                       = std::make_unique<OutputPart>(
        this->graphOfParts.GeneratePartId(), output.GetTensorInfo().m_Dimensions, compilerDataFormat,
        output.GetTensorInfo().m_QuantizationInfo, std::set<uint32_t>{ output.GetId() }, m_EstimationOptions.value(),
        m_CompilationOptions, m_Capabilities);
    parts.push_back(std::move(outputPart.get()));
    this->graphOfParts.m_Parts.push_back(std::move(outputPart));
    ConnectParts(output, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Convolution& convolution)
{
    std::vector<BasePart*> parts;
    auto convInfo = convolution.GetConvolutionInfo();

    // Check if it is a strided convolution and add a FusedPlePart.
    if (convInfo.m_Stride.m_X > 1 || convInfo.m_Stride.m_Y > 1)
    {
        auto fusedPlePart = std::make_unique<FusedPlePart>(
            this->graphOfParts.GeneratePartId(), convolution.GetInput(0).GetTensorInfo().m_Dimensions,
            convolution.GetOutput(0).GetTensorInfo().m_Dimensions,
            convolution.GetInput(0).GetTensorInfo().m_QuantizationInfo,
            convolution.GetOutput(0).GetTensorInfo().m_QuantizationInfo,
            command_stream::PleOperation::INTERLEAVE_2X2_2_2,
            utils::ShapeMultiplier{ { 1, convInfo.m_Stride.m_Y },
                                    { 1, convInfo.m_Stride.m_X },
                                    { convInfo.m_Stride.m_X * convInfo.m_Stride.m_Y } },
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
            std::set<uint32_t>{ convolution.GetId(), convolution.GetBias().GetId(), convolution.GetWeights().GetId() });
        parts.push_back(std::move(fusedPlePart.get()));
        this->graphOfParts.m_Parts.push_back(std::move(fusedPlePart));
    }

    auto mcePart = std::make_unique<McePart>(
        this->graphOfParts.GeneratePartId(), convolution.GetInput(0).GetTensorInfo().m_Dimensions,
        convolution.GetOutput(0).GetTensorInfo().m_Dimensions,
        convolution.GetInput(0).GetTensorInfo().m_QuantizationInfo,
        convolution.GetOutput(0).GetTensorInfo().m_QuantizationInfo, convolution.GetWeights().GetTensorInfo(),
        OverrideWeights(convolution.GetWeights().GetDataVector(), convolution.GetWeights().GetTensorInfo()),
        convolution.GetBias().GetTensorInfo(), GetDataVectorAs<int32_t, uint8_t>(convolution.GetBias().GetDataVector()),
        convolution.GetConvolutionInfo().m_Stride, convolution.GetConvolutionInfo().m_Padding.m_Top,
        convolution.GetConvolutionInfo().m_Padding.m_Left, command_stream::MceOperation::CONVOLUTION,
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities,
        std::set<uint32_t>{ convolution.GetId(), convolution.GetBias().GetId(), convolution.GetWeights().GetId() });
    parts.push_back(std::move(mcePart.get()));
    this->graphOfParts.m_Parts.push_back(std::move(mcePart));
    ConnectParts(convolution, parts);
}

void NetworkToGraphOfPartsConverter::Visit(Pooling& pooling)
{
    std::vector<BasePart*> parts;
    const PoolingInfo& poolingInfo = pooling.GetPoolingInfo();
    if (poolingInfo == PoolingInfo{ 2, 2, 2, 2, poolingInfo.m_Padding, PoolingType::MAX })
    {
        auto poolingFusedPlePart = std::make_unique<FusedPlePart>(
            this->graphOfParts.GeneratePartId(), pooling.GetInput(0).GetTensorInfo().m_Dimensions,
            pooling.GetOutput(0).GetTensorInfo().m_Dimensions, pooling.GetInput(0).GetTensorInfo().m_QuantizationInfo,
            pooling.GetOutput(0).GetTensorInfo().m_QuantizationInfo, command_stream::PleOperation::MAXPOOL_2X2_2_2,
            utils::ShapeMultiplier{
                { 1, pooling.GetPoolingInfo().m_PoolingStrideY }, { 1, pooling.GetPoolingInfo().m_PoolingStrideX }, 1 },
            m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities, std::set<uint32_t>{ pooling.GetId() });
        parts.push_back(std::move(poolingFusedPlePart.get()));
        this->graphOfParts.m_Parts.push_back(std::move(poolingFusedPlePart));
        ConnectParts(pooling, parts);
    }
    else
    {
        throw std::invalid_argument("Only PoolingType::MAX is supported at the moment");
    }
}

void NetworkToGraphOfPartsConverter::Visit(Reshape& reshape)
{
    std::vector<BasePart*> parts;
    auto reshapePart = std::make_unique<ReshapePart>(
        this->graphOfParts.GeneratePartId(), reshape.GetInput(0).GetTensorInfo().m_Dimensions,
        reshape.GetOutput(0).GetTensorInfo().m_Dimensions, CompilerDataFormat::NHWC,
        reshape.GetOutput(0).GetTensorInfo().m_QuantizationInfo, std::set<uint32_t>{ reshape.GetId() },
        m_EstimationOptions.value(), m_CompilationOptions, m_Capabilities);
    parts.push_back(std::move(reshapePart.get()));
    this->graphOfParts.m_Parts.push_back(std::move(reshapePart));
    ConnectParts(reshape, parts);
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
    return std::move(graphOfParts);
}

void NetworkToGraphOfPartsConverter::ConnectParts(Operation& operation, std::vector<BasePart*>& m_Part)
{
    // Loop through all parts in the vector of BaseParts and connect them together.
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_Part.size()) - 1; i++)
    {
        this->graphOfParts.AddConnection({ m_Part[i + 1]->GetPartId(), 0 }, { m_Part[i]->GetPartId(), 0 });
    }

    // Loop through all input operands of current operation and connect first part in vector of BaseParts with
    // the preceding part that has the same operand as output.
    for (const Operand* op : operation.GetInputs())
    {
        // REVISIT (konkar01): Currently we connect by default to slot0. Need to add support for multiple input/output slots.
        this->graphOfParts.AddConnection({ m_Part.front()->GetPartId(), 0 },
                                         { m_OperandToPart.at(op)->GetPartId(), 0 });
    }

    // Check if current operation has outputs and if so mark them for connection with the subsequent operation.
    if (operation.GetOutputs().size() > 0)
    {
        m_OperandToPart[&operation.GetOutput(0)] = m_Part.back();
    }
}

}    // namespace support_library
}    // namespace ethosn
