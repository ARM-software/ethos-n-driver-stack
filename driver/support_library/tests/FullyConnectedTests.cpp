//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "../src/Compiler.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

namespace
{

void PopulateWeights(std::vector<uint8_t>& weight, uint32_t numOutputs)
{
    size_t numWeights = weight.size();

    for (uint32_t i = 0; i < numWeights; ++i)
    {
        uint32_t ifmIndex = i / numOutputs;
        uint32_t ofmIndex = i % numOutputs;
        uint8_t w         = static_cast<uint8_t>(ofmIndex * 4 + (ifmIndex % 7));

        weight[ofmIndex + ifmIndex * numOutputs] = w;
    }
}

}    // namespace

TEST_CASE("FullyConnectedSupported")
{
    char reason[1024];

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N57));

    SECTION("InputInfo is not UINT8_QUANTIZED.")
    {
        TensorInfo inputNotUint8Quant({ 1, 1, 1, 4096 }, DataType::INT32_QUANTIZED, DataFormat::NHWC,
                                      QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsFullyConnectedSupported(TensorInfo(), TensorInfo(), FullyConnectedInfo(), inputNotUint8Quant,
                                                  nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "UINT8_QUANTIZED"));
    }

    SECTION("Invalid input data format")
    {
        TensorInfo inputInvalidFormat({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO,
                                      QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsFullyConnectedSupported(TensorInfo(), TensorInfo(), FullyConnectedInfo(), inputInvalidFormat,
                                                  nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Only NHWC and NHWCB"));
    }

    SECTION("Invalid weights data type")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weightsInvalidType{ { 1, 1, 4096, 1000 }, DataType::INT32_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias;
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weightsInvalidType, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Weights for fully connected must be UINT8_QUANTIZED"));
    }

    SECTION("Invalid weights data format")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weightsInvalidFormat{
            { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.0f }
        };
        TensorInfo bias;
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weightsInvalidFormat, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Weights for fully connected must be HWIO"));
    }

    SECTION("Weights invalid W")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weightsInvalidW{ { 1, 2, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias;
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weightsInvalidW, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Weights tensor must have H and W set to 1 as these dimensions are not needed."));
    }

    SECTION("Weights invalid H")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weightsInvalidH{ { 2, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias;
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weightsInvalidH, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Weights tensor must have H and W set to 1 as these dimensions are not needed."));
    }

    SECTION("Weights invalid I")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weightsInvalidI{ { 1, 1, 4097, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias;
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weightsInvalidI, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason,
                         "Weights tensor must have I dimension equal to the number of channels of the input tensor."));
    }

    SECTION("Invalid bias data type")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo biasInvalidDataType{ { 1, 1, 1, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.0f } };
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(biasInvalidDataType, weights, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Bias for fully connected must be INT32_QUANTIZED"));
    }

    SECTION("Invalid bias data format")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo biasInvalidDataFormat{ { 1, 1, 1, 1000 }, DataType::INT32_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(biasInvalidDataFormat, weights, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Bias for fully connected must be NHWC"));
    }

    SECTION("Invalid bias shape")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias{ { 1, 2, 3, 4 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.0f } };
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid bias tensor dimensions"));
    }

    SECTION("Output info incorrect")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias{ { 1, 1, 1, 1000 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.0f } };
        FullyConnectedInfo fcInfo;
        TensorInfo output = TensorInfo({ 1, 2, 3, 4 });
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, &output, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
    }

    SECTION("EstimateOnly for implicit reshape on input")
    {
        TensorInfo input({ 1, 8, 8, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weights{ { 1, 1, 320, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias{ { 1, 1, 1, 1000 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.0f } };
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "one dimensional"));
    }

    SECTION("Estimate only for bias quant scale mismatch")
    {
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo bias{ { 1, 1, 1, 1000 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 99.0f } };
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Bias for fully connected must have quantization parameters"));
    }

    SECTION("Estimate only for overall multiplier out of range")
    {
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo bias{ { 1, 1, 1, 1000 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.0f } };
        FullyConnectedInfo fcInfo({ 0, 1.0f });
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Overall scale"));
    }

    SECTION("Successful case")
    {
        const auto inputDataType = GENERATE(DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED);

        float weightScale = 1.0f / (16 * 16 * 16 * 8);
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, weightScale } };
        TensorInfo input({ 1, 1, 1, 4096 }, inputDataType, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo bias{ { 1, 1, 1, 1000 },
                         DataType::INT32_QUANTIZED,
                         DataFormat::NHWC,
                         { 0, weightScale * input.m_QuantizationInfo.GetScale() } };
        TensorInfo output = TensorInfo();
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, &output) == SupportedLevel::Supported);
    }
}

/// Checks the CompiledNetwork that the support_library produces when given fully connected layer.
/// This is a simple test. It assumes the input data is one dimensional, i.e. reshaping already done.
TEST_CASE("Fully Connected")
{

    constexpr uint32_t ifmWidth       = 1;
    constexpr uint32_t ifmHeight      = 1;
    constexpr uint32_t ifmDepth       = 1536;
    constexpr uint32_t totalNumInputs = (ifmWidth * ifmHeight * ifmDepth);
    // Ifm will be rounded up to the nearest multiple of 1024.
    constexpr uint32_t roundedUpIfms = totalNumInputs + (totalNumInputs % 1024);
    constexpr uint32_t numOutputs    = 16;
    constexpr uint32_t numFcWeights  = (totalNumInputs * numOutputs);
    constexpr uint8_t blockWidth     = 8;
    constexpr uint8_t blockHeight    = 8;

    std::vector<uint8_t> weightsData(numFcWeights);
    std::vector<uint32_t> biasData(numOutputs, 0);

    TensorInfo inputInfo{
        { { 1, ifmWidth, ifmHeight, ifmDepth } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 0.007795065175741911f },
    };

    TensorInfo weightsInfo{
        { { 1, 1, totalNumInputs, numOutputs } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 127, 0.00779726692274505f },
    };

    TensorInfo biasInfo{
        { { 1, 1, 1, numOutputs } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, inputInfo.m_QuantizationInfo.GetScale() * weightsInfo.m_QuantizationInfo.GetScale() },
    };

    FullyConnectedInfo fcInfo{
        { 118, 0.037594329609590416f },
    };

    // populate the weight vector
    PopulateWeights(weightsData, numFcWeights);

    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());

    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;

    // Build up the network
    std::shared_ptr<Operand> input          = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> fullyConnected = AddFullyConnected(network, *input, *bias, *weights, fcInfo).tensor;
    std::shared_ptr<Output> output          = AddOutput(network, *fullyConnected).tensor;

    options.m_DebugInfo.m_DumpRam = true;

    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork = Compile(*network, options);

    using namespace ethosn::command_stream;

    McePle fully_connected;

    fully_connected.m_InputInfo().m_DataType()         = ethosn::command_stream::DataType::U8;
    fully_connected.m_InputInfo().m_DataFormat()       = ethosn::command_stream::DataFormat::NHWCB;
    fully_connected.m_InputInfo().m_TensorShape()      = { 1, 8, 8, 32 };
    fully_connected.m_InputInfo().m_SupertensorShape() = { 1, 8, 8, 32 };
    fully_connected.m_InputInfo().m_StripeShape()      = { 1, 8, 8, 32 };
    fully_connected.m_InputInfo().m_TileSize()         = 1 * 8 * 8 * 32;
    fully_connected.m_InputInfo().m_DramBufferId()     = 1;
    fully_connected.m_InputInfo().m_SramOffset()       = 0x1000;
    fully_connected.m_InputInfo().m_ZeroPoint()        = 0;
    fully_connected.m_InputInfo().m_DataLocation()     = ethosn::command_stream::DataLocation::DRAM;

    fully_connected.m_WeightInfo().m_DataType()         = ethosn::command_stream::DataType::U8;
    fully_connected.m_WeightInfo().m_DataFormat()       = ethosn::command_stream::DataFormat::WEIGHT_STREAM;
    fully_connected.m_WeightInfo().m_TensorShape()      = { 1, 1, roundedUpIfms, numOutputs };
    fully_connected.m_WeightInfo().m_SupertensorShape() = { 1, 1, roundedUpIfms, numOutputs };
    fully_connected.m_WeightInfo().m_StripeShape()      = { 1, 1, roundedUpIfms, 16 };
    fully_connected.m_WeightInfo().m_TileSize()         = 0x10200;
    fully_connected.m_WeightInfo().m_DramBufferId()     = 2;
    fully_connected.m_WeightInfo().m_SramOffset()       = 0x80 + 0x1000;
    fully_connected.m_WeightInfo().m_ZeroPoint()        = 127;
    fully_connected.m_WeightInfo().m_DataLocation()     = ethosn::command_stream::DataLocation::DRAM;

    fully_connected.m_WeightMetadataBufferId() = 3;

    fully_connected.m_OutputInfo().m_DataType()          = ethosn::command_stream::DataType::U8;
    fully_connected.m_OutputInfo().m_DataFormat()        = ethosn::command_stream::DataFormat::NHWC;
    fully_connected.m_OutputInfo().m_TensorShape()       = { 1, 1, 1, numOutputs };
    fully_connected.m_OutputInfo().m_SupertensorShape()  = { 1, 1, 1, numOutputs };
    fully_connected.m_OutputInfo().m_SupertensorOffset() = { 0, 0, 0, 0 };
    fully_connected.m_OutputInfo().m_StripeShape()       = { 1, 8, 8, 16 };
    fully_connected.m_OutputInfo().m_TileSize()          = 1 * 8 * 8 * 16;
    fully_connected.m_OutputInfo().m_DramBufferId()      = 4;
    fully_connected.m_OutputInfo().m_SramOffset()        = 0xffc0;

    fully_connected.m_OutputInfo().m_ZeroPoint() = static_cast<int16_t>(fcInfo.m_OutputQuantizationInfo.GetZeroPoint());
    fully_connected.m_OutputInfo().m_DataLocation()       = ethosn::command_stream::DataLocation::DRAM;
    fully_connected.m_SramConfig().m_AllocationStrategy() = SramAllocationStrategy::STRATEGY_X;

    fully_connected.m_BlockConfig().m_BlockWidth()  = blockWidth;
    fully_connected.m_BlockConfig().m_BlockHeight() = blockHeight;

    fully_connected.m_MceData().m_Stride().m_Y()            = 1;
    fully_connected.m_MceData().m_Stride().m_X()            = 1;
    fully_connected.m_MceData().m_PadTop()                  = 0;
    fully_connected.m_MceData().m_PadLeft()                 = 0;
    fully_connected.m_MceData().m_UninterleavedInputShape() = { 1, 1, 1, 1536 };
    fully_connected.m_MceData().m_OutputShape()             = { 1, 1, 1, numOutputs };
    fully_connected.m_MceData().m_OutputStripeShape()       = { 1, 8, 8, numOutputs };
    fully_connected.m_MceData().m_Operation()               = ethosn::command_stream::MceOperation::FULLY_CONNECTED;
    fully_connected.m_MceData().m_Algorithm()               = MceAlgorithm::DIRECT;
    fully_connected.m_MceData().m_ActivationMin()           = 0;
    fully_connected.m_MceData().m_ActivationMax()           = 255;
    fully_connected.m_MceData().m_UpsampleType()            = UpsampleType::OFF;
    fully_connected.m_MceData().m_OutputZeroPoint() =
        static_cast<int16_t>(fcInfo.m_OutputQuantizationInfo.GetZeroPoint());

    fully_connected.m_PleData().m_CeSram()    = 0x0;
    fully_connected.m_PleData().m_PleSram()   = 0x0;
    fully_connected.m_PleData().m_Operation() = ethosn::command_stream::PleOperation::PASSTHROUGH;

    ethosn::command_stream::CommandStreamBuffer expectedCmdStream;
    expectedCmdStream.EmplaceBack(fully_connected);

    DumpDram cmdStrDumpDram;
    cmdStrDumpDram.m_DramBufferId() = 4;
    cmdStrDumpDram.m_Filename()     = Filename{ "EthosNIntermediateBuffer_4_UINT8_QUANTIZED_NHWC_1_1_1_16.hex" };
    expectedCmdStream.EmplaceBack(cmdStrDumpDram);

    DumpSram cmdStrDumpSram;
    cmdStrDumpSram.m_Filename() = Filename{ "output_ce_0" };
    expectedCmdStream.EmplaceBack(cmdStrDumpSram);

    std::vector<uint8_t> expectedCmdStreamData = GetCommandStreamData(expectedCmdStream);
    std::vector<uint8_t> expectedConstantControlUnitData(expectedCmdStreamData.begin(), expectedCmdStreamData.end());
    std::vector<uint8_t> padding(utils::RoundUpToNearestMultiple(expectedConstantControlUnitData.size(), 64) -
                                     expectedConstantControlUnitData.size(),
                                 0);
    expectedConstantControlUnitData.insert(expectedConstantControlUnitData.end(), padding.begin(), padding.end());
    std::vector<uint8_t> metaDataBuffer = {
        0x0, 0x0, 0x0, 0x0, 0x0, 0x32, 0x0, 0x0,
    };
    expectedConstantControlUnitData.insert(expectedConstantControlUnitData.end(), metaDataBuffer.begin(),
                                           metaDataBuffer.end());
    constexpr size_t compiledWeightSize = 12800;

    REQUIRE(compiledNetwork.size() == 1);
    const CompiledNetworkImpl* cnImpl = static_cast<const CompiledNetworkImpl*>(compiledNetwork[0].get());
    REQUIRE(cnImpl->GetConstantControlUnitData() == expectedConstantControlUnitData);
    REQUIRE(cnImpl->GetConstantControlUnitDataBufferInfos() ==
            std::vector<CompiledNetworkImpl::BufferInfoInternal>{
                { 0u, 0u, static_cast<uint32_t>(expectedCmdStreamData.size()) },
                { 3u, static_cast<uint32_t>(expectedCmdStreamData.size() + 63) & ~63, 0x8 },
            });
    REQUIRE(cnImpl->GetConstantDmaData().size() == compiledWeightSize);
    REQUIRE(cnImpl->GetConstantDmaDataBufferInfos() ==
            std::vector<CompiledNetworkImpl::BufferInfoInternal>{ { 2, 0, compiledWeightSize } });
    REQUIRE(cnImpl->GetInputBufferInfosInternal() ==
            std::vector<CompiledNetworkImpl::BufferInfoInternal>{ { 1, 0, 0x800, 2, 0 } });
    REQUIRE(cnImpl->GetOutputBufferInfosInternal() ==
            std::vector<CompiledNetworkImpl::BufferInfoInternal>{ { 4, 0, 0x10, 3, 0 } });
    REQUIRE(cnImpl->GetIntermediateDataBufferInfos() == std::vector<CompiledNetworkImpl::BufferInfoInternal>{});
}
