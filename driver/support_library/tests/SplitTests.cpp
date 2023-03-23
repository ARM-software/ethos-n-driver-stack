//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "../src/Compiler.hpp"
#include "../src/Utils.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;
using namespace ethosn::support_library::utils;

TEST_CASE("SplitSupported")
{
    char reason[1024];

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Not enough splits
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                SplitInfo(3, {}), nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Must have at least 1 output"));

    // Unsupported datatype
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                SplitInfo(3, { 32, 32 }), nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Input tensor must be UINT8_QUANTIZED or INT8_QUANTIZED"));

    // Unsupported data format
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(1, 2)),
                SplitInfo(3, { 32, 32 }), nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Input tensor must be NHWC or NHWCB"));

    // Invalid axis
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                SplitInfo(7, { 32, 32 }), nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Axis must refer to a valid dimension"));

    // Invalid sum of sizes
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                SplitInfo(3, { 32, 16 }), nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Sizes must sum to the total size of the input tensor along the split axis"));

    // Invalid number of outputInfos provided
    {
        std::vector<TensorInfo> outputInfos(3);
        REQUIRE(queries.IsSplitSupported(
                    TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                    SplitInfo(3, { 32, 32 }), &outputInfos, reason, sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfos array has incorrect size"));
    }

    // Invalid outputInfo provided
    {
        std::vector<TensorInfo> outputInfos{
            TensorInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
            TensorInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
        };
        REQUIRE(queries.IsSplitSupported(
                    TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                    SplitInfo(3, { 32, 32 }), &outputInfos, reason, sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfo at index 0 is incorrect"));
    }

    // Unsupported axis
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                SplitInfo(0, { 0, 1 }), nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Split cannot be performed along batch axis"));

    // Non-multiple of 16 along channels axis
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                SplitInfo(3, { 30, 34 }), nullptr, reason, sizeof(reason)) == SupportedLevel::EstimateOnly);
    REQUIRE(Contains(reason, "Split along the channels dimension (axis 3) requires all output sizes (specified in "
                             "splitInfo.m_Sizes) to be multiples of 16"));

    // Zero point outside of valid range
    {
        REQUIRE(queries.IsSplitSupported(TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                                                    QuantizationInfo(-10, 2)),
                                         SplitInfo(3, { 30, 34 }), nullptr, reason,
                                         sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Zero point out of range for input info"));
    }

    // Successful case (output info provided)
    {
        std::vector<TensorInfo> outputInfos{
            TensorInfo({ 1, 16, 16, 32 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
            TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
            TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2))
        };
        REQUIRE(queries.IsSplitSupported(
                    TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                    SplitInfo(3, { 32, 16, 16 }), &outputInfos) == SupportedLevel::Supported);
    }

    // Successful case (output infos filled in)
    {
        std::vector<TensorInfo> outputInfos(3);
        REQUIRE(queries.IsSplitSupported(
                    TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                    SplitInfo(3, { 32, 16, 16 }), &outputInfos) == SupportedLevel::Supported);
        REQUIRE(outputInfos.size() == 3);
        REQUIRE(outputInfos[0] ==
                TensorInfo({ 1, 16, 16, 32 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)));
        REQUIRE(outputInfos[1] ==
                TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)));
        REQUIRE(outputInfos[2] ==
                TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)));
    }
}

// Tests that a split that can be performed using NHWCB does so,
// rather than falling back to NHWC.
TEST_CASE("Split NHWCB", "[.]")
{
    const auto inputDataType         = GENERATE(DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED);
    const auto expectedInputDataType = utils::GetCommandDataType(inputDataType);

    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());

    std::shared_ptr<Operand> input =
        AddInput(network, TensorInfo({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWCB)).tensor;

    std::vector<std::shared_ptr<Operand>> split = AddSplit(network, *input, SplitInfo(1, { 8, 8 })).tensors;

    AddOutput(network, *split[0]);
    AddOutput(network, *split[1]);

    // Compile the network
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork = ethosn::support_library::Compile(*network, options);

    // Figure out which output is which
    size_t firstOutputIdx = FindIndexIf(compiledNetwork[0]->GetOutputBufferInfos(),
                                        [&](const auto& b) { return b.m_SourceOperationOutputIndex == 0; })
                                .second;
    const CompiledNetworkImpl* cnImpl = static_cast<const CompiledNetworkImpl*>(compiledNetwork[0].get());
    uint32_t firstOutputBufferId      = cnImpl->GetOutputBufferInfosInternal()[firstOutputIdx].m_Id;

    // Extract the McePle operations
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

    // There should be two of them, each extracting NHWCB from the input buffer at different supertensor offsets.
    REQUIRE(convCmds.size() == 2);
    size_t firstOutputCmdIdx =
        FindIndexIf(convCmds, [&](const auto& c) { return c.m_OutputInfo().m_DramBufferId() == firstOutputBufferId; })
            .second;
    size_t secondOutputCmdIdx = 1 - firstOutputCmdIdx;

    REQUIRE(convCmds[firstOutputCmdIdx].m_InputInfo().m_DataType() == expectedInputDataType);
    REQUIRE(convCmds[firstOutputCmdIdx].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWCB);
    REQUIRE(convCmds[firstOutputCmdIdx].m_InputInfo().m_SupertensorShape() == TensorShape{ 1, 16, 16, 16 });
    REQUIRE(convCmds[firstOutputCmdIdx].m_InputInfo().m_SupertensorOffset() == TensorShape{ 0, 0, 0, 0 });
    REQUIRE(convCmds[firstOutputCmdIdx].m_InputInfo().m_TensorShape() == TensorShape{ 1, 8, 16, 16 });

    REQUIRE(convCmds[secondOutputCmdIdx].m_InputInfo().m_DataType() == expectedInputDataType);
    REQUIRE(convCmds[secondOutputCmdIdx].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWCB);
    REQUIRE(convCmds[secondOutputCmdIdx].m_InputInfo().m_SupertensorShape() == TensorShape{ 1, 16, 16, 16 });
    REQUIRE(convCmds[secondOutputCmdIdx].m_InputInfo().m_SupertensorOffset() == TensorShape{ 0, 8, 0, 0 });
    REQUIRE(convCmds[secondOutputCmdIdx].m_InputInfo().m_TensorShape() == TensorShape{ 1, 8, 16, 16 });

    REQUIRE(convCmds[firstOutputCmdIdx].m_InputInfo().m_DramBufferId() ==
            convCmds[secondOutputCmdIdx].m_InputInfo().m_DramBufferId());
}

// Tests that a split that must be performed using NHWC does so,
// rather than trying to use to NHWCB which can't work.
TEST_CASE("Split NHWC", "[.]")
{
    const auto inputDataType         = GENERATE(DataType::UINT8_QUANTIZED, DataType::INT8_QUANTIZED);
    const auto expectedOtputDataType = utils::GetCommandDataType(inputDataType);

    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());

    std::shared_ptr<Operand> input =
        AddInput(network, TensorInfo({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWCB)).tensor;

    std::vector<std::shared_ptr<Operand>> split = AddSplit(network, *input, SplitInfo(1, { 9, 7 })).tensors;

    AddOutput(network, *split[0]);
    AddOutput(network, *split[1]);

    // Compile the network
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork = ethosn::support_library::Compile(*network, options);

    // Figure out which output is which
    size_t firstOutputIdx = FindIndexIf(compiledNetwork[0]->GetOutputBufferInfos(),
                                        [&](const auto& b) { return b.m_SourceOperationOutputIndex == 0; })
                                .second;
    const CompiledNetworkImpl* cnImpl = static_cast<const CompiledNetworkImpl*>(compiledNetwork[0].get());
    uint32_t firstOutputBufferId      = cnImpl->GetOutputBufferInfosInternal()[firstOutputIdx].m_Id;

    // Extract the McePle operations
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

    // There should be two of them, each extracting NHWC from the input buffer at different supertensor offsets.
    REQUIRE(convCmds.size() == 2);
    size_t firstOutputCmdIdx =
        FindIndexIf(convCmds, [&](const auto& c) { return c.m_OutputInfo().m_DramBufferId() == firstOutputBufferId; })
            .second;
    size_t secondOutputCmdIdx = 1 - firstOutputCmdIdx;

    REQUIRE(convCmds[firstOutputCmdIdx].m_InputInfo().m_DataType() == expectedOtputDataType);
    REQUIRE(convCmds[firstOutputCmdIdx].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[firstOutputCmdIdx].m_InputInfo().m_SupertensorShape() == TensorShape{ 1, 16, 16, 16 });
    REQUIRE(convCmds[firstOutputCmdIdx].m_InputInfo().m_SupertensorOffset() == TensorShape{ 0, 0, 0, 0 });
    REQUIRE(convCmds[firstOutputCmdIdx].m_InputInfo().m_TensorShape() == TensorShape{ 1, 9, 16, 16 });

    REQUIRE(convCmds[secondOutputCmdIdx].m_InputInfo().m_DataType() == expectedOtputDataType);
    REQUIRE(convCmds[secondOutputCmdIdx].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[secondOutputCmdIdx].m_InputInfo().m_SupertensorShape() == TensorShape{ 1, 16, 16, 16 });
    REQUIRE(convCmds[secondOutputCmdIdx].m_InputInfo().m_SupertensorOffset() == TensorShape{ 0, 9, 0, 0 });
    REQUIRE(convCmds[secondOutputCmdIdx].m_InputInfo().m_TensorShape() == TensorShape{ 1, 7, 16, 16 });

    REQUIRE(convCmds[firstOutputCmdIdx].m_InputInfo().m_DramBufferId() ==
            convCmds[secondOutputCmdIdx].m_InputInfo().m_DramBufferId());
}
