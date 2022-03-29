//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/CapabilitiesInternal.hpp"
#include "../src/Graph.hpp"
#include "../src/GraphNodes.hpp"
#include "../src/Network.hpp"
#include "../src/NetworkToGraphConverter.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("ReinterpretQuantization Supported")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("Output size same as input")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo output({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsReinterpretQuantizationSupported(ReinterpretQuantizationInfo(QuantizationInfo(0, 1.0f)),
                                                           input, &output) == SupportedLevel::Supported);
    }

    SECTION("Output info filled in")
    {
        TensorInfo output;
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsReinterpretQuantizationSupported(ReinterpretQuantizationInfo(QuantizationInfo(1, 1.0f)),
                                                           input, &output) == SupportedLevel::Supported);
        REQUIRE(output ==
                TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 1.0f)));
    }
}

TEST_CASE("ReinterpretQuantization Unsupported")
{
    char reason[1024];
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("Incorrect batch size")
    {
        TensorInfo input({ 2, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsReinterpretQuantizationSupported(ReinterpretQuantizationInfo(QuantizationInfo(0, 1.0f)),
                                                           input, nullptr, reason,
                                                           sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Batch size must be 1"));
    }

    SECTION("Incorrect tensor depth")
    {
        //Max available SRAM is 1024x1024 for ETHOS_N78_4TOPS_4PLE_RATIO
        //Max supported tensor depth is 8x8xC where C is number of channel and 8x8xC <= 1024x1024
        //Therefore, when C > 1024*1024/64 => C > 16384, this test fails
        TensorInfo input({ 1, 16, 16, 16385 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsReinterpretQuantizationSupported(ReinterpretQuantizationInfo(QuantizationInfo(0, 1.0f)),
                                                           input, nullptr, reason,
                                                           sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Input to reinterpret quantization: Tensor max depth cannot fit in SRAM"));
    }

    SECTION("Incorrect input data type")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsReinterpretQuantizationSupported(ReinterpretQuantizationInfo(QuantizationInfo(0, 1.0f)),
                                                           input, nullptr, reason,
                                                           sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Input to reinterpret quantization must be UINT8_QUANTIZED or INT8_QUANTIZED"));
    }

    SECTION("Incorrect input data format")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NCHW, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsReinterpretQuantizationSupported(ReinterpretQuantizationInfo(QuantizationInfo(0, 1.0f)),
                                                           input, nullptr, reason,
                                                           sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Input to reinterpret quantization must be NHWC or NHWCB"));
    }

    SECTION("Incorrect output shape")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo output({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsReinterpretQuantizationSupported(ReinterpretQuantizationInfo(QuantizationInfo(0, 1.0f)),
                                                           input, &output, reason,
                                                           sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
    }

    SECTION("Invalid zero point in input")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(-5, 1.0f));
        REQUIRE(queries.IsReinterpretQuantizationSupported(ReinterpretQuantizationInfo(QuantizationInfo(0, 1.0f)),
                                                           input, nullptr, reason,
                                                           sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Zero point out of range for input info"));
    }

    SECTION("Invalid zero point in output")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsReinterpretQuantizationSupported(ReinterpretQuantizationInfo(QuantizationInfo(-10, 1.0f)),
                                                           input, nullptr, reason,
                                                           sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Zero point out of range for reinterpretQuantizationInfo"));
    }
}

namespace
{
const QuantizationInfo expectedQuantizationInfo(1, 1.1f);

std::shared_ptr<Network> GetNetworkToTest()
{
    uint32_t autoDetectSram = 0;
    std::vector<char> hardwareCapabilitiesVect =
        GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, autoDetectSram);
    std::shared_ptr<Network> networkToTest = CreateNetwork(hardwareCapabilitiesVect);

    TensorInfo inputInfo{
        { { 1, 128, 128, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 0.5f },
    };

    std::shared_ptr<Operand> input = AddInput(networkToTest, inputInfo).tensor;
    AddReinterpretQuantization(networkToTest, *input, ReinterpretQuantizationInfo{ expectedQuantizationInfo });

    return networkToTest;
}

void AddOperationsToNetwork(std::shared_ptr<Network>& network, bool addReinterpret)
{
    std::shared_ptr<Operand> input = AddInput(network, TensorInfo({ 1, 16, 16, 16 })).tensor;

    Padding padding;
    padding.m_Top    = 0;
    padding.m_Bottom = 0;
    padding.m_Left   = 0;
    padding.m_Right  = 0;

    std::shared_ptr<Constant> biasConv1 =
        AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.f }),
                    std::vector<uint8_t>(16, 0).data())
            .tensor;
    std::shared_ptr<Constant> weightsConv1 =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    std::shared_ptr<Operand> conv1 =
        AddConvolution(network, *input, *biasConv1, *weightsConv1,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 2.f)))
            .tensor;

    std::shared_ptr<Operand> maxPool =
        AddPooling(network, *conv1, PoolingInfo(2, 2, 2, 2, padding, PoolingType::MAX)).tensor;

    std::shared_ptr<Operand> inputConv2;
    if (addReinterpret)
    {
        inputConv2 =
            AddReinterpretQuantization(network, *maxPool, ReinterpretQuantizationInfo{ QuantizationInfo(0, 2.f) })
                .tensor;
    }
    else
    {
        inputConv2 = maxPool;
    }

    std::shared_ptr<Constant> biasConv2 =
        AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 2.f }),
                    std::vector<uint8_t>(16, 0).data())
            .tensor;
    std::shared_ptr<Constant> weightsConv2 =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    std::shared_ptr<Operand> conv2 =
        AddConvolution(network, *inputConv2, *biasConv2, *weightsConv2,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 4.f)))
            .tensor;

    std::shared_ptr<Output> output = AddOutput(network, *conv2).tensor;
}
}    // namespace

SCENARIO("Add ReinterpretQuantization Operation")
{
    GIVEN("There is a network with an input operation and a ReinterpretQuantization operation")
    {
        std::shared_ptr<Network> networkToTest = GetNetworkToTest();

        THEN("The first operation has the correct quantization info")
        {
            // GetNetworkToTest only creates 2 operations. Input and ReinterpretQuantization
            // so we test the second operation i.e. operationIdxToCheck = 1
            constexpr unsigned int operationIdxToCheck = 1;
            unsigned int operationIdx                  = 0;
            for (const auto& operation : *networkToTest)
            {
                if (operationIdx == operationIdxToCheck)
                {
                    const TensorInfo& tensorInfo                   = operation->GetOutput(0).GetTensorInfo();
                    const QuantizationInfo& actualQuantizationInfo = tensorInfo.m_QuantizationInfo;
                    CAPTURE(actualQuantizationInfo.GetZeroPoint(), actualQuantizationInfo.GetScale(),
                            expectedQuantizationInfo.GetZeroPoint(), actualQuantizationInfo.GetScale());
                    REQUIRE(actualQuantizationInfo == expectedQuantizationInfo);
                    break;
                }
                ++operationIdx;
            }
        }
    }
}

SCENARIO("Visit ReinterpretQuantization Operation")
{
    GIVEN("There is a network with an input node and a ReinterpretQuantization node")
    {
        std::shared_ptr<Network> networkToTest = GetNetworkToTest();
        WHEN("The network is transformed into a graph")
        {
            uint32_t autoDetectSram = 0;
            FirmwareAndHardwareCapabilities hardwareCapabilities =
                GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, autoDetectSram);
            bool strictPrecision = false;

            Graph graph(*networkToTest, hardwareCapabilities, EstimationOptions{}, strictPrecision);

            THEN("The second node has the correct quantization info")
            {
                // When the network is converted to graph, the following nodes are generated
                // Input --> FormatConversion --> ReinterpretQuantization
                // i.e. nodeIdxToCheck = 2;
                constexpr unsigned int nodeIdxToCheck                = 2;
                const std::vector<std::unique_ptr<Node>>& graphNodes = graph.GetNodes();
                const std::unique_ptr<Node>& outputNode              = graphNodes[nodeIdxToCheck];

                //Check that the ReinterpretNode is correctly created
                auto reinterpretNode = outputNode.get();
                REQUIRE(reinterpretNode->GetNodeType() == NodeType::ReinterpretNode);

                const QuantizationInfo actualQuantizationInfo = reinterpretNode->GetQuantizationInfo();
                CAPTURE(actualQuantizationInfo.GetZeroPoint(), actualQuantizationInfo.GetScale(),
                        expectedQuantizationInfo.GetZeroPoint(), actualQuantizationInfo.GetScale());
                REQUIRE(actualQuantizationInfo == expectedQuantizationInfo);
            }
        }
    }
}

SCENARIO("ReinterpretQuantization doesn't have any side effect on the command stream")
{
    GIVEN("There is a network with the following operations Input -> Conv -> MaxPool -> Convolution -> Output")
    {
        std::shared_ptr<Network> networkWithoutReinterpret = CreateNetwork(GetRawDefaultCapabilities());
        AddOperationsToNetwork(networkWithoutReinterpret, false);

        AND_GIVEN(
            "There is a network with the following operations Input -> Conv -> MaxPool -> ReinterpretQuantization -> "
            "Convolution -> Output")
        {
            std::shared_ptr<Network> networkWithReinterpret = CreateNetwork(GetRawDefaultCapabilities());
            AddOperationsToNetwork(networkWithReinterpret, true);

            WHEN("Both network are succesfully compiled")
            {
                CompilationOptions compilationOption;
                std::vector<std::unique_ptr<CompiledNetwork>> compiledNetworkWithoutReinterpret =
                    ethosn::support_library::Compile(*networkWithoutReinterpret, compilationOption);
                std::vector<std::unique_ptr<CompiledNetwork>> compiledNetworkWithReinterpret =
                    ethosn::support_library::Compile(*networkWithReinterpret, compilationOption);

                THEN("The command stream of both compiled networks is the same")
                {
                    using namespace ethosn::command_stream;
                    CommandStream commandStreamWithoutReinterpret =
                        GetCommandStream(compiledNetworkWithoutReinterpret[0].get());
                    CommandStream commandStreamWithReinterpret =
                        GetCommandStream(compiledNetworkWithReinterpret[0].get());

                    REQUIRE(std::equal(commandStreamWithoutReinterpret.begin(), commandStreamWithoutReinterpret.end(),
                                       commandStreamWithReinterpret.begin(), commandStreamWithReinterpret.end(),
                                       [](auto& lhs, auto& rhs) { return AreCommandsEqual(lhs, rhs); }));
                }
            }
        }
    }
}
