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

TEST_CASE("ConstantSupported", "[Constant]")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N57));

    TensorInfo info({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsConstantSupported(info) == SupportedLevel::Supported);
}

TEST_CASE("Constant used as input to operation compiles succesfully", "[Constant]")
{
    // Create the network
    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());

    std::shared_ptr<Constant> constant =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.f }),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    auto constantOperand           = GetOperand(constant);
    std::shared_ptr<Output> output = AddOutput(network, *constantOperand).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetworks =
        ethosn::support_library::Compile(*network, options);
    REQUIRE(compiledNetworks.size() == 1);

    // Check that it contains a single copy command, from input to output
    using namespace ethosn::command_stream;
    CommandStream cs = GetCommandStream(compiledNetworks[0].get());
    auto it          = cs.begin();
    REQUIRE(it->m_Opcode() == Opcode::OPERATION_CONVERT);
    ++it;
    REQUIRE(it == cs.end());

    // Check that the constant data is included in the compiled network
    const CompiledNetworkImpl* cnImpl = static_cast<const CompiledNetworkImpl*>(compiledNetworks[0].get());
    REQUIRE(cnImpl->GetConstantDmaDataBufferInfos().size() == 1);
    REQUIRE(cnImpl->GetConstantDmaDataBufferInfos()[0].m_Size == 1 * 1 * 16 * 16);
}
