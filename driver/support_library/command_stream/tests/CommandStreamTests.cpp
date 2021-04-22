//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_command_stream/CommandStreamBuffer.hpp"

#include <catch.hpp>

#include <iostream>

using namespace ethosn::command_stream;

TEST_CASE("CommandStream")
{
    constexpr McePle conv1x1 = {
        /* InputInfo = */
        TensorInfo{
            /* DataType = */ DataType::U8,
            /* DataFormat = */ DataFormat::NHWCB,
            /* TensorShape = */ TensorShape{ { 2U, 25U, 50U, 32U } },
            /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* StripeShape = */ TensorShape{ { 1U, 25U, 50U, 32U } },
            /* TileSize = */ 40000U,
            /* DramBufferId = */ 77U,
            /* SramOffset = */ 0U,
            /* ZeroPoint = */ int16_t{ 123 },
        },
        /* WeightInfo = */
        TensorInfo{
            /* DataType = */ DataType::U8,
            /* DataFormat = */ DataFormat::WEIGHT_STREAM,
            /* TensorShape = */ TensorShape{ { 3U, 3U, 32U, 96U } },
            /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* StripeShape = */ TensorShape{ { 3U, 3U, 32U, 96U } },
            /* TileSize = */ 27648U,
            /* DramBufferId = */ 78U,
            /* SramOffset = */ 0U,
            /* ZeroPoint = */ int16_t{ 0 },
        },
        /* WeightMetadataBufferId = */ 22U,
        /* OutputInfo = */
        TensorInfo{
            /* DataType = */ DataType::U8,
            /* DataFormat = */ DataFormat::NHWC,
            /* TensorShape = */ TensorShape{ { 2U, 23U, 48U, 96U } },
            /* SupertensorShape = */ TensorShape{ { 2U, 23U, 48U, 96U } },
            /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* StripeShape = */ TensorShape{ { 1U, 23U, 48U, 96U } },
            /* TileSize = */ 105984U,
            /* DramBufferId = */ 79U,
            /* SramOffset = */ 0U,
            /* ZeroPoint = */ int16_t{ 0 },
        },
        /* SramConfig = */
        SramConfig{
            /* AllocationStrategy = */ SramAllocationStrategy::STRATEGY_0,
        },
        /* BlockConfig = */
        BlockConfig{
            /* BlockWidth = */ 16U,
            /* BlockHeight = */ 16U,
        },
        /* MceData = */
        MceData{
            /* Stride = */ MceStrideConfig{ 1U, 1U },
            /* PadTop = */ 1U,
            /* PadLeft = */ 1U,
            /* UninterleavedInputShape = */ TensorShape{ 1U, 23U, 48U, 96U },
            /* OutputShape = */ TensorShape{ 1U, 23U, 48U, 96U },
            /* OutputStripeShape = */ TensorShape{ 1U, 23U, 48U, 96U },
            /* OutputZeroPoint = */ int16_t{ 0 },
            /* UpsampleType = */ UpsampleType::OFF,
            /* UpsampleEdgeModeRow = */ UpsampleEdgeMode::GENERATE,
            /* UpsampleEdgeModeCol = */ UpsampleEdgeMode::GENERATE,
            /* Operation = */ MceOperation::CONVOLUTION,
            /* Algorithm = */ MceAlgorithm::DIRECT,
            /* ActivationMin = */ uint8_t{ 3 },
            /* ActivationMax = */ uint8_t{ 253 },
        },
    };

    constexpr McePle conv1x1B = {
        /* InputInfo = */
        TensorInfo{
            /* DataType = */ DataType::U8,
            /* DataFormat = */ DataFormat::NHWCB,
            /* TensorShape = */ TensorShape{ { 2U, 25U, 50U, 32U } },
            /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* StripeShape = */ TensorShape{ { 2U, 25U, 50U, 32U } },
            /* TileSize = */ 80000U,
            /* DramBufferId = */ 77U,
            /* SramOffset = */ 0U,
        },
    };

    constexpr Convert convert = {
        /* InputInfo = */
        TensorInfo{
            /* DataType = */ DataType::U8,
            /* DataFormat = */ DataFormat::NHWC,
            /* TensorShape = */ TensorShape{ { 1U, 224, 224, 3U } },
            /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* StripeShape = */ TensorShape{ { 1U, 224U, 224U, 3U } },
            /* TileSize = */ 11239424U,
            /* DramBufferId = */ 0U,
            /* SramOffset = */ 0U,
            /* ZeroPoint = */ int16_t{ 123 },
        },
        /* OutputInfo = */
        TensorInfo{
            /* DataType = */ DataType::U8,
            /* DataFormat = */ DataFormat::NHWC,
            /* TensorShape = */ TensorShape{ { 1U, 12U, 112U, 224U } },
            /* SupertensorShape = */ TensorShape{ { 1U, 12U, 112U, 224U } },
            /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* StripeShape = */ TensorShape{ { 1U, 12U, 112U, 224U } },
            /* TileSize = */ 301056U,
            /* DramBufferId = */ 0U,
            /* SramOffset = */ 0x25000U,
            /* ZeroPoint = */ int16_t{ 0 },
        },
    };

    constexpr SpaceToDepth spaceToDepth = {
        /* InputInfo = */
        TensorInfo{
            /* DataType = */ DataType::U8,
            /* DataFormat = */ DataFormat::NHWC,
            /* TensorShape = */ TensorShape{ { 1U, 224, 224, 3U } },
            /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* StripeShape = */ TensorShape{ { 1U, 224U, 224U, 3U } },
            /* TileSize = */ 11239424U,
            /* DramBufferId = */ 0U,
            /* SramOffset = */ 0U,
            /* ZeroPoint = */ int16_t{ 0 },
        },
        /* OutputInfo = */
        TensorInfo{
            /* DataType = */ DataType::U8,
            /* DataFormat = */ DataFormat::NHWC,
            /* TensorShape = */ TensorShape{ { 1U, 112U, 112U, 12U } },
            /* SupertensorShape = */ TensorShape{ { 1U, 112U, 112U, 12U } },
            /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
            /* StripeShape = */ TensorShape{ { 1U, 112U, 112U, 12U } },
            /* TileSize = */ 301056U,
            /* DramBufferId = */ 0U,
            /* SramOffset = */ 0x25000U,
            /* ZeroPoint = */ int16_t{ 0 },
        },
    };

    constexpr TensorShape shape1 = conv1x1.m_InputInfo().m_TensorShape();

    CommandStreamBuffer csbuffer;

    csbuffer.EmplaceBack(conv1x1);
    csbuffer.EmplaceBack(Fence{});
    csbuffer.EmplaceBack(conv1x1B);
    csbuffer.EmplaceBack(convert);
    csbuffer.EmplaceBack(spaceToDepth);

    CommandStream cstream(&*csbuffer.begin(), &*csbuffer.end());
    REQUIRE(cstream.IsValid());

    auto it = cstream.begin();

    {
        const auto command = it->GetCommand<Opcode::OPERATION_MCE_PLE>();

        REQUIRE(command != nullptr);

        if (command != nullptr)
        {
            REQUIRE(command->m_Data() == conv1x1);
            REQUIRE(command->m_Data() != conv1x1B);
            REQUIRE(command->m_Data().m_InputInfo().m_TensorShape() == shape1);
        }

        REQUIRE(it->GetCommand<Opcode::FENCE>() == nullptr);
    }

    ++it;

    {
        const auto command = it->GetCommand<Opcode::FENCE>();

        REQUIRE(command != nullptr);

        if (command != nullptr)
        {
            REQUIRE(command->m_Data() == Fence{});
        }

        REQUIRE(it->GetCommand<Opcode::OPERATION_MCE_PLE>() == nullptr);
    }

    ++it;

    {
        const auto command = it->GetCommand<Opcode::OPERATION_MCE_PLE>();

        REQUIRE(command != nullptr);

        if (command != nullptr)
        {
            REQUIRE(command->m_Data() == conv1x1B);
            REQUIRE(command->m_Data() != conv1x1);
        }

        REQUIRE(it->GetCommand<Opcode::FENCE>() == nullptr);
    }

    ++it;

    {
        const auto command = it->GetCommand<Opcode::OPERATION_CONVERT>();

        REQUIRE(command != nullptr);

        if (command != nullptr)
        {
            REQUIRE(command->m_Data() == convert);
        }

        REQUIRE(it->GetCommand<Opcode::FENCE>() == nullptr);
    }

    ++it;

    {
        const auto command = it->GetCommand<Opcode::OPERATION_SPACE_TO_DEPTH>();

        REQUIRE(command != nullptr);

        if (command != nullptr)
        {
            REQUIRE(command->m_Data() == spaceToDepth);
        }

        REQUIRE(it->GetCommand<Opcode::FENCE>() == nullptr);
    }
}

TEST_CASE("CommandStreamBuffer Version Header")
{
    GIVEN("An empty command stream")
    {
        CommandStreamBuffer cs;
        WHEN("The raw data is inspected")
        {
            const std::vector<uint32_t>& data = cs.GetData();

            THEN("There is a header with the version information")
            {
                const std::vector<uint32_t> expectedData = {
                    static_cast<uint32_t>('E') | (static_cast<uint32_t>('N') << 8) |
                        (static_cast<uint32_t>('C') << 16) | (static_cast<uint32_t>('S') << 24),
                    ETHOSN_COMMAND_STREAM_VERSION_MAJOR, ETHOSN_COMMAND_STREAM_VERSION_MINOR,
                    ETHOSN_COMMAND_STREAM_VERSION_PATCH
                };

                REQUIRE(data == expectedData);
            }
        }
    }
}

TEST_CASE("CommandStream Version Header")
{
    std::vector<uint32_t> validCmdStreamData = {
        static_cast<uint32_t>('E') | (static_cast<uint32_t>('N') << 8) | (static_cast<uint32_t>('C') << 16) |
            (static_cast<uint32_t>('S') << 24),
        ETHOSN_COMMAND_STREAM_VERSION_MAJOR, ETHOSN_COMMAND_STREAM_VERSION_MINOR, ETHOSN_COMMAND_STREAM_VERSION_PATCH
    };

    GIVEN("A valid command stream")
    {
        std::vector<uint32_t> data = validCmdStreamData;

        WHEN("Constructing a CommandStream object around this data")
        {
            CommandStream cs(data.data(), data.data() + data.size());

            THEN("The CommandStream is valid and reports the correct version")
            {
                REQUIRE(cs.IsValid());
                REQUIRE(cs.GetVersionMajor() == ETHOSN_COMMAND_STREAM_VERSION_MAJOR);
                REQUIRE(cs.GetVersionMinor() == ETHOSN_COMMAND_STREAM_VERSION_MINOR);
                REQUIRE(cs.GetVersionPatch() == ETHOSN_COMMAND_STREAM_VERSION_PATCH);
            }
        }
    }

    GIVEN("A command stream that is too short")
    {
        std::vector<uint32_t> data = { 0, 1, 2 };

        WHEN("Constructing a CommandStream object around this data")
        {
            CommandStream cs(data.data(), data.data() + data.size());

            THEN("The CommandStream is invalid and has no version information")
            {
                REQUIRE(!cs.IsValid());
                REQUIRE(cs.GetVersionMajor() == 0);
                REQUIRE(cs.GetVersionMinor() == 0);
                REQUIRE(cs.GetVersionPatch() == 0);
            }
        }
    }

    GIVEN("A command stream that has the wrong fourcc code")
    {
        std::vector<uint32_t> data = validCmdStreamData;
        data[0]                    = 1234;

        WHEN("Constructing a CommandStream object around this data")
        {
            CommandStream cs(data.data(), data.data() + data.size());

            THEN("The CommandStream is invalid and has no version information")
            {
                REQUIRE(!cs.IsValid());
                REQUIRE(cs.GetVersionMajor() == 0);
                REQUIRE(cs.GetVersionMinor() == 0);
                REQUIRE(cs.GetVersionPatch() == 0);
            }
        }
    }

    GIVEN("A command stream that has the wrong version")
    {
        std::vector<uint32_t> data = validCmdStreamData;
        data[1]                    = ETHOSN_COMMAND_STREAM_VERSION_MAJOR + 1;

        WHEN("Constructing a CommandStream object around this data")
        {
            CommandStream cs(data.data(), data.data() + data.size());

            THEN("The CommandStream is invalid but the version is reported correctly")
            {
                REQUIRE(!cs.IsValid());
                REQUIRE(cs.GetVersionMajor() == ETHOSN_COMMAND_STREAM_VERSION_MAJOR + 1);
                REQUIRE(cs.GetVersionMinor() == ETHOSN_COMMAND_STREAM_VERSION_MINOR);
                REQUIRE(cs.GetVersionPatch() == ETHOSN_COMMAND_STREAM_VERSION_PATCH);
            }
        }
    }
}
