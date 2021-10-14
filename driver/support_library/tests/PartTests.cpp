//
// Copyright © 2020-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/Graph.hpp"
#include "../src/GraphNodes.hpp"
#include "../src/cascading/Cascading.hpp"
#include "../src/cascading/PartUtils.hpp"
#include "../src/cascading/PartV1.hpp"
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

        uint32_t tileSize = impl::CalculateTileSize(node, hwCaps, inputTensorShape, inputStripeShape, outputStripeShape,
                                                    nonBoundaryStripes);

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

        uint32_t tileSize = impl::CalculateTileSize(node, hwCaps, inputTensorShape, inputStripeShape, outputStripeShape,
                                                    nonBoundaryStripes);

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

        uint32_t tileSize = impl::CalculateTileSize(node, hwCaps, inputTensorShape, inputStripeShape, outputStripeShape,
                                                    nonBoundaryStripes);

        // We do not need to stream in width and height so the boundary tiles are not needed.
        // The stripe size is 116 * 16 * 32 ie 57344
        // The tileSize is (57344) * 3 = 172032
        REQUIRE(tileSize == 172032U);
    }
}

TEST_CASE("GraphOfParts simple linear")
{
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    // p1 -> p2 -> p3

    auto p1 = std::make_unique<MockPart>(1);
    auto p2 = std::make_unique<MockPart>(2);
    auto p3 = std::make_unique<MockPart>(3);
    parts.push_back(std::move(p1));
    parts.push_back(std::move(p2));
    parts.push_back(std::move(p3));

    // connect up the parts
    connections.insert({ PartInputSlot{ 2, 0 }, PartOutputSlot{ 1, 0 } });
    connections.insert({ PartInputSlot{ 3, 0 }, PartOutputSlot{ 2, 0 } });

    auto sourcep1 = graph.GetSourceParts(1);
    REQUIRE(sourcep1.size() == 0);
    auto sourcep2 = graph.GetSourceParts(2);
    REQUIRE(sourcep2.size() == 1);
    REQUIRE(sourcep2.at(0) == PartOutputSlot{ 1, 0 });
    auto sourcep3 = graph.GetSourceParts(3);
    REQUIRE(sourcep3.size() == 1);
    REQUIRE(sourcep3.at(0) == PartOutputSlot{ 2, 0 });

    auto destp1 = graph.GetDestinationParts(1);
    REQUIRE(destp1.size() == 1);
    REQUIRE(destp1.at(0) == PartInputSlot{ 2, 0 });
    auto destp2 = graph.GetDestinationParts(2);
    REQUIRE(destp2.size() == 1);
    REQUIRE(destp2.at(0) == PartInputSlot{ 3, 0 });
    auto destp3 = graph.GetDestinationParts(3);
    REQUIRE(destp3.size() == 0);
}

TEST_CASE("GraphOfParts multiple input slots for one output slot")
{
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    // p1 "0th" output connects to p2 and p3.
    //
    // p1 0->0 p2
    //    0->0 p3

    auto p1   = std::make_unique<MockPart>(1);
    auto p2   = std::make_unique<MockPart>(2);
    auto p3   = std::make_unique<MockPart>(3);
    auto p1Id = p1->GetPartId();
    auto p2Id = p2->GetPartId();
    auto p3Id = p3->GetPartId();
    parts.push_back(std::move(p1));
    parts.push_back(std::move(p2));
    parts.push_back(std::move(p3));

    PartOutputSlot p1OutputSlot = PartOutputSlot{ p1Id, 0 };
    PartInputSlot p2InputSlot   = PartInputSlot{ p2Id, 0 };
    PartInputSlot p3InputSlot   = PartInputSlot{ p3Id, 0 };

    // connect up the parts
    connections.insert({ p2InputSlot, p1OutputSlot });
    connections.insert({ p3InputSlot, p1OutputSlot });

    auto inputSlots = graph.GetConnectedInputSlots(p1OutputSlot);
    REQUIRE(inputSlots.size() == 2);
    REQUIRE(utils::Find(inputSlots, p2InputSlot).first);
    REQUIRE(utils::Find(inputSlots, p3InputSlot).first);

    {
        auto outputSlot = graph.GetConnectedOutputSlot(p2InputSlot);
        REQUIRE(outputSlot.has_value());
        REQUIRE(outputSlot.value() == p1OutputSlot);
    }
    {
        auto outputSlot = graph.GetConnectedOutputSlot(p3InputSlot);
        REQUIRE(outputSlot.has_value());
        REQUIRE(outputSlot.value() == p1OutputSlot);
    }
}

TEST_CASE("GraphOfParts GetPartInputs/Outputs")
{
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    // p1 "0th" output connects to p2 and p3
    // p1 "1st" output connects to p3's 0th and 1st input
    //
    // p1 0->0 p2
    //    0->0 p3
    //    1->1 p3

    auto p1   = std::make_unique<MockPart>(1);
    auto p2   = std::make_unique<MockPart>(2);
    auto p3   = std::make_unique<MockPart>(3);
    auto p1Id = p1->GetPartId();
    auto p2Id = p2->GetPartId();
    auto p3Id = p3->GetPartId();
    parts.push_back(std::move(p1));
    parts.push_back(std::move(p2));
    parts.push_back(std::move(p3));

    PartOutputSlot p1OutputSlot0 = PartOutputSlot{ p1Id, 0 };
    PartOutputSlot p1OutputSlot1 = PartOutputSlot{ p1Id, 1 };
    PartInputSlot p2InputSlot    = PartInputSlot{ p2Id, 0 };
    PartInputSlot p3InputSlot0   = PartInputSlot{ p3Id, 0 };
    PartInputSlot p3InputSlot1   = PartInputSlot{ p3Id, 1 };

    // connect up the parts
    connections.insert({ p2InputSlot, p1OutputSlot0 });
    connections.insert({ p3InputSlot0, p1OutputSlot0 });
    connections.insert({ p3InputSlot1, p1OutputSlot1 });

    {
        auto inputSlots = graph.GetPartInputs(p1Id);
        REQUIRE(inputSlots.size() == 0);
        auto outputSlots = graph.GetPartOutputs(p1Id);
        REQUIRE(outputSlots.size() == 3);
    }
    {
        auto inputSlots = graph.GetPartInputs(p2Id);
        REQUIRE(inputSlots.size() == 1);
        REQUIRE(utils::Find(inputSlots, p2InputSlot).first);

        auto outputSlots = graph.GetPartOutputs(p2Id);
        REQUIRE(outputSlots.size() == 0);
    }
    {
        auto inputSlots = graph.GetPartInputs(p3Id);
        REQUIRE(inputSlots.size() == 2);
        REQUIRE(utils::Find(inputSlots, p3InputSlot0).first);
        REQUIRE(utils::Find(inputSlots, p3InputSlot1).first);

        auto outputSlots = graph.GetPartOutputs(p3Id);
        REQUIRE(outputSlots.size() == 0);
    }
}

/// Test case to create a graph of parts with PartV1 parts
/// make sure the parts are connected correctly
TEST_CASE("CreateGraphOfParts")
{
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    // Create simple graph
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    graph.Connect(nodeA, nodeB);
    graph.Connect(nodeC, nodeB);

    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

    GraphOfParts graphOfParts = CreateGraphOfParts(graph, estOpt, compOpt, caps);
    const auto& parts         = graphOfParts.m_Parts;

    auto node = static_cast<NameOnlyNode*>(static_cast<PartV1*>(parts.at(2).get())->m_SubGraph.back());
    REQUIRE(node->m_Name == "b");

    REQUIRE(graphOfParts.m_Connections.size() == 2);
    REQUIRE(graphOfParts.GetSourceParts(parts.at(2)->GetPartId()).size() == 2);
}
