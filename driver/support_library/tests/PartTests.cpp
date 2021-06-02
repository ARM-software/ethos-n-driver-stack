//
// Copyright © 2020-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/Graph.hpp"
#include "../src/GraphNodes.hpp"
#include "../src/cascading/Part.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("Validate Tile Size")
{
    Graph graph;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

    SECTION("Check the tile calculation when the stripe is streamed in width and height")
    {
        const TensorShape& inputTensorShape  = TensorShape{ 1, 112, 112, 32 };
        const TensorShape& inputStripeShape  = TensorShape{ 1, 16, 16, 32 };
        const TensorShape& outputStripeShape = TensorShape{ 1, 16, 16, 32 };
        uint32_t nonBoundaryStripes          = 3;

        MceOperationNode* node = graph.CreateAndAddNode<MceOperationNode>(
            inputTensorShape, inputTensorShape, DataType::UINT8_QUANTIZED, QuantizationInfo(),
            ethosn::support_library::TensorInfo({ 3, 3, 32, 1 }, ethosn::support_library::DataType::UINT8_QUANTIZED,
                                                ethosn::support_library::DataFormat::HWIO, QuantizationInfo(0, 0.9f)),
            std::vector<uint8_t>({ 1 }), ethosn::support_library::TensorInfo({ 1, 1, 32, 1 }),
            std::vector<int32_t>{ 0 }, Stride(), 0, 0, ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION,
            CompilerDataFormat::NHWCB, std::set<uint32_t>{ 1 });

        uint32_t tileSize =
            CalculateTileSize(node, hwCaps, inputTensorShape, inputStripeShape, outputStripeShape, nonBoundaryStripes);

        // The upper boundary size = 8 (ie brickGroupHeight) * (16 * 32) (ie inputStripeXZ) which is 4096
        // The lower boundary size is also the same as the upper boundary size ie 4096
        // The stripe size is 16 * 16 * 32 ie 8192
        // The tileSize is (8192 + 4096 + 4096) * 3 = 49152
        REQUIRE(tileSize == 49152U);
    }

    SECTION("Check the tile calculation when the stripe is streamed in weight")
    {
        const TensorShape& inputTensorShape  = TensorShape{ 1, 112, 112, 32 };
        const TensorShape& inputStripeShape  = TensorShape{ 1, 112, 16, 32 };
        const TensorShape& outputStripeShape = TensorShape{ 1, 112, 16, 32 };
        uint32_t nonBoundaryStripes          = 3;

        MceOperationNode* node = graph.CreateAndAddNode<MceOperationNode>(
            inputTensorShape, inputTensorShape, DataType::UINT8_QUANTIZED, QuantizationInfo(),
            ethosn::support_library::TensorInfo({ 3, 3, 32, 1 }, ethosn::support_library::DataType::UINT8_QUANTIZED,
                                                ethosn::support_library::DataFormat::HWIO, QuantizationInfo(0, 0.9f)),
            std::vector<uint8_t>({ 1 }), ethosn::support_library::TensorInfo({ 1, 1, 32, 1 }),
            std::vector<int32_t>{ 0 }, Stride(), 0, 0, ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION,
            CompilerDataFormat::NHWCB, std::set<uint32_t>{ 1 });

        uint32_t tileSize =
            CalculateTileSize(node, hwCaps, inputTensorShape, inputStripeShape, outputStripeShape, nonBoundaryStripes);

        // We do not need to stream in width and height so the boundary tiles are not needed.
        // The stripe size is 116 * 16 * 32 ie 57344
        // The tileSize is (57344) * 3 = 172032
        REQUIRE(tileSize == 172032U);
    }

    SECTION("Check the tile calculation when the stripe is streamed in height")
    {
        const TensorShape& inputTensorShape  = TensorShape{ 1, 112, 112, 32 };
        const TensorShape& inputStripeShape  = TensorShape{ 1, 16, 112, 32 };
        const TensorShape& outputStripeShape = TensorShape{ 1, 16, 112, 32 };
        uint32_t nonBoundaryStripes          = 3;

        MceOperationNode* node = graph.CreateAndAddNode<MceOperationNode>(
            inputTensorShape, inputTensorShape, DataType::UINT8_QUANTIZED, QuantizationInfo(),
            ethosn::support_library::TensorInfo({ 3, 3, 32, 1 }, ethosn::support_library::DataType::UINT8_QUANTIZED,
                                                ethosn::support_library::DataFormat::HWIO, QuantizationInfo(0, 0.9f)),
            std::vector<uint8_t>({ 1 }), ethosn::support_library::TensorInfo({ 1, 1, 32, 1 }),
            std::vector<int32_t>{ 0 }, Stride(), 0, 0, ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION,
            CompilerDataFormat::NHWCB, std::set<uint32_t>{ 1 });

        uint32_t tileSize =
            CalculateTileSize(node, hwCaps, inputTensorShape, inputStripeShape, outputStripeShape, nonBoundaryStripes);

        // We do not need to stream in width and height so the boundary tiles are not needed.
        // The stripe size is 116 * 16 * 32 ie 57344
        // The tileSize is (57344) * 3 = 172032
        REQUIRE(tileSize == 172032U);
    }
}

TEST_CASE("GetNumInvalidPlans")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    GraphOfParts gOfParts;
    Parts& parts = gOfParts.m_Parts;

    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_NumInvalidPlans = 1UL;
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_NumInvalidPlans = 2UL;
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_NumInvalidPlans = 3UL;

    REQUIRE(gOfParts.GetNumInvalidPlans() == 6UL);
}
