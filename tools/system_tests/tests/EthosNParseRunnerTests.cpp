//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../EthosNParseRunner.hpp"

#include <ConcreteOperations.hpp>
#include <Network.hpp>
#include <Operation.hpp>
#include <catch.hpp>

using namespace ethosn;
using namespace ethosn::support_library;

namespace ethosn
{
namespace system_tests
{

/// Checks that the EthosNParseRunner correctly parses and adds a convolution layer to its internal Network.
TEST_CASE("EthosNParseRunner Parse Convolution")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 16, 16, 16
conv layer, name conv1, bottom data, top conv1, num output 16, kernel size 3, stride 1, pad 1
)");
    LayerData layerData;
    EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
    const Network* network = parser.GetNetwork();

    struct EthosNLayerVisitor : public NetworkVisitor
    {
        using NetworkVisitor::Visit;

        void Visit(Convolution& convolution) override
        {
            REQUIRE(!m_Found);
            REQUIRE(convolution.GetWeights().GetTensorInfo().m_Dimensions == TensorShape{ 3, 3, 16, 16 });
            m_Found = true;
        }

        bool m_Found = false;
    } visitor;

    network->Accept(visitor);
    REQUIRE(visitor.m_Found);
}

TEST_CASE("EthosNParseRunner Parse Convolution with signed weights")
{
    std::stringstream ggfContents(R"(
# Weight_Precision: i8
input layer, name data, top data, shape 1, 16, 16, 16
conv layer, name conv1, bottom data, top conv1, num output 16, kernel size 3, stride 1, pad 1
)");
    LayerData layerData;
    EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
    const Network* network = parser.GetNetwork();

    struct EthosNLayerVisitor : public NetworkVisitor
    {
        using NetworkVisitor::Visit;

        void Visit(Convolution& convolution) override
        {
            REQUIRE(!m_Found);
            REQUIRE(convolution.GetWeights().GetTensorInfo().m_Dimensions == TensorShape{ 3, 3, 16, 16 });
            REQUIRE(convolution.GetWeights().GetTensorInfo().m_DataType ==
                    ethosn::support_library::DataType::INT8_QUANTIZED);
            m_Found = true;
        }

        bool m_Found = false;
    } visitor;

    network->Accept(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the EthosNParseRunner correctly parses and adds a leakyRelu layer to its internal Network.
TEST_CASE("EthosNParseRunner Parse Estimated LeakyRelu")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 16, 16, 16
conv layer, name conv1, bottom data, top conv1, num output 16, kernel size 3, stride 1, pad 1
leakyrelu layer, name leakyrelu1, bottom conv1, top conv1, alpha 0.1
)");
    LayerData layerData;
    EthosNParseRunner::CreationOptions creationOptions(ggfContents, layerData);
    creationOptions.m_EstimationMode = true;
    EthosNParseRunner parser(creationOptions);
    const Network* network = parser.GetNetwork();

    struct EthosNLayerVisitor : public NetworkVisitor
    {
        using NetworkVisitor::Visit;

        void Visit(LeakyRelu& leakyRelu) override
        {
            REQUIRE(!m_Found);
            REQUIRE(leakyRelu.GetLeakyReluInfo().m_Alpha == 0.1f);
            m_Found = true;
        }

        bool m_Found = false;
    } visitor;

    network->Accept(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the EthosNParseRunner correctly parses and adds a requantize layer to its internal Network.
TEST_CASE("EthosNParseRunner Parse Requantize")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 16, 16, 16
conv layer, name conv1, bottom data, top conv1, num output 16, kernel size 3, stride 1, pad 1
requantize layer, name requantize1, bottom conv1, top requantize1, zeroPoint 1, scale 0.5f
)");
    SECTION("Estimation mode")
    {
        LayerData layerData;
        EthosNParseRunner::CreationOptions creationOptions(ggfContents, layerData);
        creationOptions.m_EstimationMode = true;
        EthosNParseRunner parser(creationOptions);
        const Network* network = parser.GetNetwork();

        struct EthosNLayerVisitor : public NetworkVisitor
        {
            using NetworkVisitor::Visit;

            void Visit(Requantize& requantize)
            {
                REQUIRE(!m_Found);
                REQUIRE(requantize.GetRequantizeInfo().m_OutputQuantizationInfo.GetZeroPoint() == 1);
                REQUIRE(requantize.GetRequantizeInfo().m_OutputQuantizationInfo.GetScale() == 0.5f);
                m_Found = true;
            }

            bool m_Found = false;
        } visitor;

        network->Accept(visitor);
        REQUIRE(visitor.m_Found);
    }
    SECTION("Compilation mode")
    {
        LayerData layerData;
        EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
        const Network* network = parser.GetNetwork();

        struct EthosNLayerVisitor : public NetworkVisitor
        {
            using NetworkVisitor::Visit;

            void Visit(Requantize& requantize)
            {
                REQUIRE(!m_Found);
                REQUIRE(requantize.GetRequantizeInfo().m_OutputQuantizationInfo.GetZeroPoint() == 1);
                REQUIRE(requantize.GetRequantizeInfo().m_OutputQuantizationInfo.GetScale() == 0.5f);
                m_Found = true;
            }

            bool m_Found = false;
        } visitor;

        network->Accept(visitor);
        REQUIRE(visitor.m_Found);
    }
}

/// Checks that the EthosNParseRunner correctly parses and adds a split layer to its internal Network.
TEST_CASE("EthosNParseRunner Parse Split")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 16, 16, 48
split layer, name split1, bottom data, top split1, axis 3, sizes 16, 32
)");
    LayerData layerData;
    EthosNParseRunner::CreationOptions creationOptions(ggfContents, layerData);
    creationOptions.m_EstimationMode = true;
    EthosNParseRunner parser(creationOptions);
    const Network* network = parser.GetNetwork();

    struct EthosNLayerVisitor : public NetworkVisitor
    {
        using NetworkVisitor::Visit;

        void Visit(support_library::Split& split) override
        {
            REQUIRE(!m_Found);
            // Check the split configuration
            REQUIRE(split.GetSplitInfo().m_Axis == 3);
            REQUIRE(split.GetSplitInfo().m_Sizes == std::vector<uint32_t>{ 16, 32 });
            // Check the outputs are correctly connected and have the correct size
            REQUIRE(split.GetOutput(0).GetTensorInfo().m_Dimensions == TensorShape{ 1, 16, 16, 16 });
            REQUIRE(split.GetOutput(0).GetConsumers().size() == 1);
            REQUIRE(split.GetOutput(1).GetTensorInfo().m_Dimensions == TensorShape{ 1, 16, 16, 32 });
            REQUIRE(split.GetOutput(1).GetConsumers().size() == 1);
            m_Found = true;
        }

        bool m_Found = false;
    } visitor;

    network->Accept(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the EthosNParseRunner correctly parses and adds a depth-to-space layer to its internal Network.
TEST_CASE("EthosNParseRunner Parse DepthToSpace")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 5, 5, 4
depthtospace layer, name depthy, bottom data, top depthy, block_size 2
)");
    LayerData layerData;
    EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
    const Network* network = parser.GetNetwork();

    struct EthosNLayerVisitor : public NetworkVisitor
    {
        using NetworkVisitor::Visit;

        void Visit(DepthToSpace& depthToSpace) override
        {
            REQUIRE(!m_Found);
            REQUIRE(depthToSpace.GetDepthToSpaceInfo().m_BlockSize == 2);
            m_Found = true;
        }

        bool m_Found = false;
    } visitor;

    network->Accept(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the EthosNParseRunner correctly parses and adds a MeanXy layer to its internal Network.
TEST_CASE("EthosNParseRunner Parse Mean")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 7, 7, 16
mean layer, name mean1, top mean1, bottom data, keep_dims 1, dimension 2_3
)");
    LayerData layerData;
    EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
    const Network* network = parser.GetNetwork();

    struct EthosNLayerVisitor : public NetworkVisitor
    {
        using NetworkVisitor::Visit;

        void Visit(MeanXy&) override
        {
            REQUIRE(!m_Found);
            m_Found = true;
        }

        bool m_Found = false;
    } visitor;

    network->Accept(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the EthosNParseRunner correctly parses and adds a constant to its internal Network.
TEST_CASE("EthosNParseRunner Parse Constant")
{
    std::stringstream ggfContents(R"(
input layer, name data0, top data0, shape 1, 8, 8, 1
const layer, name data1, top data1, shape 1, 1, 1, 64
)");
    LayerData layerData;
    EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
    const Network* network = parser.GetNetwork();

    struct EthosNLayerVisitor : public NetworkVisitor
    {
        using NetworkVisitor::Visit;
        void Visit(support_library::Constant& constant) override
        {
            REQUIRE(!m_Found);
            // Check the outputs are correctly connected and have the correct size
            REQUIRE(constant.GetOutput(0).GetTensorInfo().m_Dimensions == TensorShape{ 1, 1, 1, 64 });
            REQUIRE(constant.GetOutput(0).GetConsumers().size() == 1);
            m_Found = true;
        }

        bool m_Found = false;
    } visitor;

    network->Accept(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the EthosNParseRunner correctly parses and adds an upsample (resize) to its internal Network.
TEST_CASE("EthosNParseRunner Parse Upsample")
{
    SECTION("Scale algorithm: bilinear using ratio")
    {
        std::stringstream ggfContents(R"(
input layer, name data0, top data0, shape 1, 16, 16, 16
upsample layer, name upsample1, bottom data0, top upsample1, upsample scale height ratio 2.0, upsample scale width ratio 2.0, upsample mode height 1, upsample mode width 1, scale_algo 2
)");
        LayerData layerData;
        EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
        const Network* network = parser.GetNetwork();

        struct EthosNLayerVisitor : public NetworkVisitor
        {
            using NetworkVisitor::Visit;
            void Visit(support_library::Resize& resize) override
            {
                REQUIRE(!m_Found);
                REQUIRE(resize.GetResizeInfo().m_Algo == ResizeAlgorithm::BILINEAR);
                REQUIRE(resize.GetResizeInfo().m_NewHeight == 32U);
                REQUIRE(resize.GetResizeInfo().m_NewWidth == 32U);
                m_Found = true;
            }

            bool m_Found = false;
        } visitor;

        network->Accept(visitor);
        REQUIRE(visitor.m_Found);
    }
    SECTION("Scale algorithm: nearest neighbour using ratio")
    {
        std::stringstream ggfContents(R"(
input layer, name data0, top data0, shape 1, 16, 16, 16
upsample layer, name upsample1, bottom data0, top upsample1, upsample scale height ratio 2.0, upsample scale width ratio 2.0, upsample mode height 1, upsample mode width 1, scale_algo 0
)");
        LayerData layerData;
        EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
        const Network* network = parser.GetNetwork();

        struct EthosNLayerVisitor : public NetworkVisitor
        {
            using NetworkVisitor::Visit;
            void Visit(support_library::Resize& resize) override
            {
                REQUIRE(!m_Found);
                REQUIRE(resize.GetResizeInfo().m_Algo == ResizeAlgorithm::NEAREST_NEIGHBOUR);
                REQUIRE(resize.GetResizeInfo().m_NewHeight == 32U);
                REQUIRE(resize.GetResizeInfo().m_NewWidth == 32U);
                m_Found = true;
            }

            bool m_Found = false;
        } visitor;

        network->Accept(visitor);
        REQUIRE(visitor.m_Found);
    }
    SECTION("Using new size")
    {
        std::stringstream ggfContents(R"(
input layer, name data0, top data0, shape 1, 16, 16, 16
upsample layer, name upsample1, bottom data0, top upsample1, new height 31, new width 31, scale_algo 2
)");
        LayerData layerData;
        EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
        const Network* network = parser.GetNetwork();

        struct EthosNLayerVisitor : public NetworkVisitor
        {
            using NetworkVisitor::Visit;
            void Visit(support_library::Resize& resize) override
            {
                REQUIRE(!m_Found);
                REQUIRE(resize.GetResizeInfo().m_Algo == ResizeAlgorithm::BILINEAR);
                REQUIRE(resize.GetResizeInfo().m_NewHeight == 31U);
                REQUIRE(resize.GetResizeInfo().m_NewWidth == 31U);
                m_Found = true;
            }

            bool m_Found = false;
        } visitor;

        network->Accept(visitor);
        REQUIRE(visitor.m_Found);
    }
}

TEST_CASE("EthosNParseRunner_PerchannelQuantization")
{
    std::stringstream ggfContents(R"(
# Weight_Precision: i8
# EnablePerChannelQuantization : true
input layer, name data, top data, shape 1, 16, 16, 16
conv layer, name conv1, bottom data, top conv1, num output 16, kernel size 3, stride 1, pad 1
)");
    LayerData layerData;
    EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
    const Network* network = parser.GetNetwork();

    REQUIRE(layerData.GetPerChannelQuantization() == true);

    struct EthosNLayerVisitor : public NetworkVisitor
    {
        using NetworkVisitor::Visit;

        void Visit(Convolution& convolution) override
        {
            REQUIRE(!m_Found);
            REQUIRE(convolution.GetWeights().GetTensorInfo().m_Dimensions == TensorShape{ 3, 3, 16, 16 });
            QuantizationInfo convQuantInfo = convolution.GetWeights().GetTensorInfo().m_QuantizationInfo;
            REQUIRE(convQuantInfo.GetScales().size() == 16);
            m_Found = true;
        }

        bool m_Found = false;
    } visitor;

    network->Accept(visitor);
    REQUIRE(visitor.m_Found);
}

TEST_CASE("EthosNParseRunner_PerchannelQuantization_False")
{
    std::stringstream ggfContents(R"(
# EnablePerChannelQuantization : false
input layer, name data, top data, shape 1, 16, 16, 16
conv layer, name conv1, bottom data, top conv1, num output 16, kernel size 3, stride 1, pad 1
)");
    LayerData layerData;
    EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
    const Network* network = parser.GetNetwork();

    REQUIRE(!layerData.GetPerChannelQuantization());

    struct EthosNLayerVisitor : public NetworkVisitor
    {
        using NetworkVisitor::Visit;

        void Visit(Convolution& convolution) override
        {
            REQUIRE(!m_Found);
            REQUIRE(convolution.GetWeights().GetTensorInfo().m_Dimensions == TensorShape{ 3, 3, 16, 16 });
            QuantizationInfo convQuantInfo = convolution.GetWeights().GetTensorInfo().m_QuantizationInfo;
            REQUIRE(convQuantInfo.GetScales().size() == 1);
            m_Found = true;
        }

        bool m_Found = false;
    } visitor;

    network->Accept(visitor);
    REQUIRE(visitor.m_Found);
}

}    // namespace system_tests
}    // namespace ethosn
