//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "../src/Compiler.hpp"
#include "../src/cascading/ConstantPart.hpp"
#include "../src/cascading/NetworkToGraphOfPartsConverter.hpp"
#include "../src/cascading/OutputPart.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

TEST_CASE("ConstantSupported", "[Constant]")
{
    char reason[1024];
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("Supported")
    {
        TensorInfo info({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsConstantSupported(info) == SupportedLevel::Supported);
    }

    SECTION("Invalid zero point")
    {
        TensorInfo info({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(-10, 1.0f));
        REQUIRE(queries.IsConstantSupported(info, reason, sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Zero point out of range"));
    }
}

TEST_CASE("Constant used as input to operation compiles succesfully", "[Constant]")
{
    // Create the network
    CompilationOptions options;
    const std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());

    std::shared_ptr<Constant> constant =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.f }),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    auto constantOperand           = GetOperand(constant);
    std::shared_ptr<Output> output = AddOutput(network, *constantOperand).tensor;

    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    // Check that it contains a copy command, from input to output
    REQUIRE(graph.GetNumParts() == 2);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    const ConstantPart* constantPart = dynamic_cast<const ConstantPart*>(&graph.GetPart(0));
    REQUIRE(constantPart != nullptr);

    REQUIRE(graph.GetPartInputs(1).size() == 1);
    REQUIRE(graph.GetPartOutputs(1).size() == 0);
    const OutputPart* outputPart1 = dynamic_cast<const OutputPart*>(&graph.GetPart(1));
    REQUIRE(outputPart1 != nullptr);

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetworks =
        ethosn::support_library::Compile(*network, options);
    REQUIRE(compiledNetworks.size() == 1);

    // Check that the constant data is included in the compiled network
    const CompiledNetworkImpl* cnImpl = static_cast<const CompiledNetworkImpl*>(compiledNetworks[0].get());
    REQUIRE(cnImpl->GetConstantDmaDataBufferInfos().size() == 1);
    REQUIRE(cnImpl->GetConstantDmaDataBufferInfos()[0].m_Size == 1 * 1 * 16 * 16);
}
