//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

namespace
{

struct Xy
{
    uint32_t x;
    uint32_t y;
};

SupportedLevel IsPoolingSupportedImpl(SupportQueries& queries,
                                      const Xy& inSize,
                                      const Xy& kSize,
                                      const Xy& stride,
                                      const Padding& padding,
                                      const PoolingType poolingType)
{
    const PoolingInfo poolingInfo(kSize.x, kSize.y, stride.x, stride.y, padding, poolingType);

    const TensorInfo input({ 1, inSize.y, inSize.x, 16 });

    const uint32_t outputSizeY = ((inSize.y - kSize.y + padding.m_Top + padding.m_Bottom) / stride.y) + 1;
    const uint32_t outputSizeX = ((inSize.x - kSize.x + padding.m_Left + padding.m_Right) / stride.x) + 1;

    TensorInfo output({ 1, outputSizeY, outputSizeX, 16 });

    return queries.IsPoolingSupported(poolingInfo, input, &output);
}
}    // namespace

TEST_CASE("PoolingSupported")
{
    char reason[1024];

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N57));

    const Padding noPad    = { 0, 0, 0, 0 };
    const Padding padAfter = { 0, 1, 0, 1 };
    const Padding padAll   = { 1, 1, 1, 1 };

    // Invalid pooling size
    {
        TensorInfo input({ 1, 10, 10, 16 });
        PoolingInfo poolingInfo(0, 0, 2, 2, { 0, 0, 0, 0 }, PoolingType::MAX);
        REQUIRE(queries.IsPoolingSupported(poolingInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid pooling size/stride"));
    }

    // Invalid pooling stride
    {
        TensorInfo input({ 1, 10, 10, 16 });
        PoolingInfo poolingInfo(2, 2, 0, 0, { 0, 0, 0, 0 }, PoolingType::MAX);
        REQUIRE(queries.IsPoolingSupported(poolingInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid pooling size/stride"));
    }

    // Incorrect output info
    {
        TensorInfo input({ 1, 10, 10, 16 });
        PoolingInfo poolingInfo(5, 5, 2, 2, { 0, 0, 0, 0 }, PoolingType::MAX);
        TensorInfo output({ 1, 2, 3, 4 });
        REQUIRE(queries.IsPoolingSupported(poolingInfo, input, &output, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
    }

    // EstimateOnly
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 2, 2 }, { 1, 1 }, noPad, PoolingType::MAX) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 3, 3 }, { 1, 1 }, noPad, PoolingType::MAX) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 3, 3 }, { 1, 1 }, noPad, PoolingType::AVG) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 5, 5 }, { 3, 3 }, noPad, PoolingType::AVG) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 7, 7 }, { 1, 1 }, noPad, PoolingType::AVG) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 7, 7 }, { 2, 2 }, noPad, PoolingType::AVG) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 8, 8 }, { 1, 1 }, noPad, PoolingType::AVG) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 8, 8 }, { 2, 2 }, noPad, PoolingType::AVG) ==
            SupportedLevel::EstimateOnly);

    // Supported
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 1, 1 }, { 2, 2 }, noPad, PoolingType::MAX) ==
            SupportedLevel::Supported);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 2, 2 }, { 2, 2 }, noPad, PoolingType::MAX) ==
            SupportedLevel::Supported);
    REQUIRE(IsPoolingSupportedImpl(queries, { 17, 17 }, { 2, 2 }, { 2, 2 }, padAfter, PoolingType::MAX) ==
            SupportedLevel::Supported);
    REQUIRE(IsPoolingSupportedImpl(queries, { 17, 17 }, { 3, 3 }, { 2, 2 }, noPad, PoolingType::MAX) ==
            SupportedLevel::Supported);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 3, 3 }, { 2, 2 }, padAfter, PoolingType::MAX) ==
            SupportedLevel::Supported);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 3, 3 }, { 1, 1 }, padAll, PoolingType::AVG) ==
            SupportedLevel::Supported);
    REQUIRE(IsPoolingSupportedImpl(queries, { 7, 7 }, { 7, 7 }, { 1, 1 }, noPad,
                                   PoolingType::AVG) == SupportedLevel::Supported);    // mean cases
    REQUIRE(IsPoolingSupportedImpl(queries, { 8, 8 }, { 8, 8 }, { 1, 1 }, noPad,
                                   PoolingType::AVG) == SupportedLevel::Supported);    // mean cases
}

/// Tests that a network comprising a single pooling creates an identity depthwise convolution beforehand.
TEST_CASE("SinglePool")
{
    // Create the network
    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input   = AddInput(network, TensorInfo({ 1, 16, 16, 16 })).tensor;

    Padding padding;
    padding.m_Top    = 0;
    padding.m_Bottom = 0;
    padding.m_Left   = 0;
    padding.m_Right  = 0;

    std::shared_ptr<Operand> relu =
        AddPooling(network, *input, PoolingInfo(2, 2, 2, 2, padding, PoolingType::MAX)).tensor;
    std::shared_ptr<Output> output = AddOutput(network, *relu).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, GetDefaultCompilationOptions());

    // Extract all the conv commands
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());
    std::vector<McePle> convCmds;
    for (const auto& cmdHeader : cmdStream)
    {
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_MCE_PLE)
        {
            convCmds.push_back(cmdHeader.GetCommand<Opcode::OPERATION_MCE_PLE>()->m_Data());
        }
    }

    // Check that the conv commands are as expected. There should be one which has a pooling afterwards.
    REQUIRE(convCmds.size() == 1);
    REQUIRE(convCmds[0].m_PleData().m_Operation() == ethosn::command_stream::PleOperation::MAXPOOL_2X2_2_2);
}

/// Tests that a network comprising a Avg pooling with large input tensor can't compile.
TEST_CASE("Large input tensor Avg Pool")
{
    // Create the network
    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input   = AddInput(network, TensorInfo({ 1, 480, 128, 64 })).tensor;

    const Padding padAll = { 1, 1, 1, 1 };

    std::shared_ptr<Operand> relu =
        AddPooling(network, *input, PoolingInfo(3, 3, 1, 1, padAll, PoolingType::AVG)).tensor;
    std::shared_ptr<Output> output = AddOutput(network, *relu).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, GetDefaultCompilationOptions());

    REQUIRE(compiledNetwork.size() == 0);
}
