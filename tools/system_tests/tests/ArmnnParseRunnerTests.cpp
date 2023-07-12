//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../ArmnnParseRunner.hpp"

#include <armnn/IStrategy.hpp>
#include <catch.hpp>

using namespace armnn;

namespace ethosn
{
namespace system_tests
{

/// Checks that the ArmnnParseRunner correctly parses and adds a convolution layer to its internal INetwork.
TEST_CASE("ArmnnParseRunner Parse Convolution")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 16, 16, 16
conv layer, name conv1, bottom data, top conv1, num output 16, kernel size 3, stride 1, pad 1
)");
    LayerData layerData;
    ArmnnParseRunner parser(ggfContents, layerData);
    const INetwork* network = parser.GetNetwork();

    struct ArmnnLayerVisitor : public armnn::IStrategy
    {
        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor&,
                             const std::vector<armnn::ConstTensor>&,
                             const char*,
                             const armnn::LayerBindingId) override
        {
            if (layer->GetType() == LayerType::Convolution2d)
            {
                REQUIRE(!m_Found);
                REQUIRE(layer->GetNumInputSlots() >= 2);
                const IConnectableLayer& weightInput =
                    layer->GetInputSlot(1).GetConnection()->GetOwningIConnectableLayer();
                REQUIRE(weightInput.GetType() == LayerType::Constant);
                CHECK(weightInput.GetOutputSlot(0).GetTensorInfo().GetShape() == TensorShape{ 16, 3, 3, 16 });
                m_Found = true;
            }
        }

        bool m_Found = false;
    } visitor;

    network->ExecuteStrategy(visitor);
    REQUIRE(visitor.m_Found);
}

TEST_CASE("ArmnnParseRunner Parse Convolution with signed weights")
{
    std::stringstream ggfContents(R"(
# Weight_Precision: i8
input layer, name data, top data, shape 1, 16, 16, 16
conv layer, name conv1, bottom data, top conv1, num output 16, kernel size 3, stride 1, pad 1
)");
    LayerData layerData;
    ArmnnParseRunner parser(ggfContents, layerData);
    const INetwork* network = parser.GetNetwork();

    struct ArmnnLayerVisitor : public armnn::IStrategy
    {
        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor&,
                             const std::vector<armnn::ConstTensor>&,
                             const char*,
                             const armnn::LayerBindingId) override
        {
            if (layer->GetType() == LayerType::Convolution2d)
            {
                REQUIRE(!m_Found);
                REQUIRE(layer->GetNumInputSlots() >= 2);
                const IConnectableLayer& weightInput =
                    layer->GetInputSlot(1).GetConnection()->GetOwningIConnectableLayer();
                REQUIRE(weightInput.GetType() == LayerType::Constant);
                CHECK(weightInput.GetOutputSlot(0).GetTensorInfo().GetShape() == TensorShape{ 16, 3, 3, 16 });
                CHECK(weightInput.GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::QAsymmS8);
                m_Found = true;
            }
        }

        bool m_Found = false;
    } visitor;

    network->ExecuteStrategy(visitor);
    REQUIRE(visitor.m_Found);
}

TEST_CASE("ArmnnParseRunner Parse Convolution with different strides")
{
    std::stringstream ggfContents(R"(
input layer, name data0, top data0, shape 1, 10, 49, 1
conv layer, name conv0, bottom data0, top conv0, num output 276, kernel h 4, kernel w 10, stride h 2, stride w 1, pad 1, bias_enable 0
)");
    LayerData layerData;
    ArmnnParseRunner parser(ggfContents, layerData);
    const INetwork* network = parser.GetNetwork();

    struct ArmnnLayerVisitor : public armnn::IStrategy
    {
        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor& desc,
                             const std::vector<armnn::ConstTensor>&,
                             const char*,
                             const armnn::LayerBindingId) override
        {
            if (layer->GetType() == LayerType::Convolution2d)
            {
                REQUIRE(!m_Found);
                REQUIRE(static_cast<const Convolution2dDescriptor&>(desc).m_StrideX == 1);
                REQUIRE(static_cast<const Convolution2dDescriptor&>(desc).m_StrideY == 2);
                m_Found = true;
            }
        }

        bool m_Found = false;
    } visitor;

    network->ExecuteStrategy(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the ArmnnParseRunner correctly parses and adds a leakyRelu layer to its internal INetwork.
TEST_CASE("ArmnnParseRunner Parse LeakyRelu")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 16, 16, 16
conv layer, name conv1, bottom data, top conv1, num output 16, kernel size 3, stride 1, pad 1
leakyrelu layer, name leakyrelu1, bottom conv1, top conv1, alpha 0.1
)");
    LayerData layerData;
    ArmnnParseRunner parser(ggfContents, layerData);
    const INetwork* network = parser.GetNetwork();

    struct ArmnnLayerVisitor : public armnn::IStrategy
    {
        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor& desc,
                             const std::vector<armnn::ConstTensor>&,
                             const char*,
                             const armnn::LayerBindingId) override
        {
            if (layer->GetType() == LayerType::Activation)
            {
                REQUIRE(!m_Found);
                REQUIRE(static_cast<const ActivationDescriptor&>(desc).m_A == 0.1f);
                REQUIRE(static_cast<const ActivationDescriptor&>(desc).m_Function == ActivationFunction::LeakyReLu);
                m_Found = true;
            }
        }

        bool m_Found = false;
    } visitor;

    network->ExecuteStrategy(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the ArmnnParseRunner correctly parses and adds a requantize layer to its internal INetwork.
TEST_CASE("ArmnnParseRunner Parse Requantize")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 16, 16, 16
conv layer, name conv1, bottom data, top conv1, num output 16, kernel size 3, stride 1, pad 1
requantize layer, name requantize1, bottom conv1, top conv1, zeroPoint 1, scale 0.5
)");
    LayerData layerData;
    ArmnnParseRunner parser(ggfContents, layerData);
    const INetwork* network = parser.GetNetwork();

    struct ArmnnLayerVisitor : public armnn::IStrategy
    {
        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor&,
                             const std::vector<armnn::ConstTensor>&,
                             const char*,
                             const armnn::LayerBindingId) override
        {
            if (layer->GetType() == LayerType::Quantize)
            {
                REQUIRE(!m_Found);
                REQUIRE(layer->GetOutputSlot(0).GetTensorInfo().GetQuantizationOffset() == 1);
                REQUIRE(layer->GetOutputSlot(0).GetTensorInfo().GetQuantizationScale() == 0.5f);
                m_Found = true;
            }
        }

        bool m_Found = false;
    } visitor;

    network->ExecuteStrategy(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the ArmnnParseRunner correctly parses and adds a split layer to its internal INetwork.
TEST_CASE("ArmnnParseRunner Parse Split")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 16, 16, 48
split layer, name split1, bottom data, top split1, axis 3, sizes 16, 32
)");
    LayerData layerData;
    ArmnnParseRunner parser(ggfContents, layerData);
    const INetwork* network = parser.GetNetwork();

    struct ArmnnLayerVisitor : public armnn::IStrategy
    {
        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor& descriptor,
                             const std::vector<armnn::ConstTensor>&,
                             const char*,
                             const armnn::LayerBindingId) override
        {
            if (layer->GetType() == LayerType::Splitter)
            {
                REQUIRE(!m_Found);

                const SplitterDescriptor& desc = static_cast<const SplitterDescriptor&>(descriptor);

                // Check the splitter configuration
                REQUIRE(desc.GetNumViews() == 2);
                REQUIRE(desc.GetViewOrigin(0)[0] == 0);
                REQUIRE(desc.GetViewOrigin(0)[1] == 0);
                REQUIRE(desc.GetViewOrigin(0)[2] == 0);
                REQUIRE(desc.GetViewOrigin(0)[3] == 0);
                REQUIRE(desc.GetViewSizes(0)[0] == 1);
                REQUIRE(desc.GetViewSizes(0)[1] == 16);
                REQUIRE(desc.GetViewSizes(0)[2] == 16);
                REQUIRE(desc.GetViewSizes(0)[3] == 16);
                REQUIRE(desc.GetViewOrigin(1)[0] == 0);
                REQUIRE(desc.GetViewOrigin(1)[1] == 0);
                REQUIRE(desc.GetViewOrigin(1)[2] == 0);
                REQUIRE(desc.GetViewOrigin(1)[3] == 16);
                REQUIRE(desc.GetViewSizes(1)[0] == 1);
                REQUIRE(desc.GetViewSizes(1)[1] == 16);
                REQUIRE(desc.GetViewSizes(1)[2] == 16);
                REQUIRE(desc.GetViewSizes(1)[3] == 32);

                // Check the inputs and outputs are correctly connected
                REQUIRE(layer->GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() ==
                        TensorShape({ 1, 16, 16, 48 }));
                REQUIRE(layer->GetOutputSlot(0).GetNumConnections() == 1);
                REQUIRE(layer->GetOutputSlot(0).GetTensorInfo().GetShape() == TensorShape({ 1, 16, 16, 16 }));
                REQUIRE(layer->GetOutputSlot(1).GetNumConnections() == 1);
                REQUIRE(layer->GetOutputSlot(1).GetTensorInfo().GetShape() == TensorShape({ 1, 16, 16, 32 }));

                m_Found = true;
            }
        }

        bool m_Found = false;
    } visitor;

    network->ExecuteStrategy(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the ArmnnParseRunner correctly parses and adds a depth-to-space layer to its internal INetwork.
TEST_CASE("ArmnnParseRunner Parse DepthToSpace")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 5, 5, 4
depthtospace layer, name depthy, bottom data, top depthy, block_size 2
)");
    LayerData layerData;
    ArmnnParseRunner parser(ggfContents, layerData);
    const INetwork* network = parser.GetNetwork();

    struct ArmnnLayerVisitor : public armnn::IStrategy
    {
        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor& desc,
                             const std::vector<armnn::ConstTensor>&,
                             const char*,
                             const armnn::LayerBindingId) override
        {
            if (layer->GetType() == LayerType::DepthToSpace)
            {
                REQUIRE(!m_Found);
                REQUIRE(static_cast<const DepthToSpaceDescriptor&>(desc).m_BlockSize == 2);
                m_Found = true;
            }
        }

        bool m_Found = false;
    } visitor;

    network->ExecuteStrategy(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the ArmnnParseRunner correctly parses and adds a Mean layer to its internal INetwork.
TEST_CASE("ArmnnParseRunner Parse Mean")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 7, 7, 16
mean layer, name mean1, top mean1, bottom data, keep_dims 1, dimension 2_3
)");
    LayerData layerData;
    ArmnnParseRunner parser(ggfContents, layerData);
    const INetwork* network = parser.GetNetwork();

    struct ArmnnLayerVisitor : public armnn::IStrategy
    {
        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor& desc,
                             const std::vector<armnn::ConstTensor>&,
                             const char*,
                             const armnn::LayerBindingId) override
        {
            if (layer->GetType() == LayerType::Mean)
            {
                REQUIRE(!m_Found);
                REQUIRE(static_cast<const MeanDescriptor&>(desc).m_KeepDims);
                REQUIRE((static_cast<const MeanDescriptor&>(desc).m_Axis == std::vector<unsigned int>{ 1, 2 }));
                m_Found = true;
            }
        }

        bool m_Found = false;
    } visitor;

    network->ExecuteStrategy(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the ArmnnParseRunner correctly parses and adds a constant to its internal Network.
TEST_CASE("ArmnnParseRunner Parse Constant")
{
    std::stringstream ggfContents(R"(
input layer, name data0, top data0, shape 1, 8, 8, 1
const layer, name data1, top data1, shape 1, 1, 1, 64
)");

    LayerData layerData;
    ArmnnParseRunner parser(ggfContents, layerData);
    const INetwork* network = parser.GetNetwork();

    struct ArmnnLayerVisitor : public armnn::IStrategy
    {
        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor&,
                             const std::vector<armnn::ConstTensor>&,
                             const char*,
                             const armnn::LayerBindingId) override
        {
            if (layer->GetType() == LayerType::Constant)
            {
                REQUIRE(!m_Found);
                // Check the outputs are correctly connected and have the correct size.
                REQUIRE(layer->GetOutputSlot(0).GetTensorInfo().GetShape() == TensorShape{ 1, 1, 1, 64 });
                REQUIRE(layer->GetOutputSlot(0).GetNumConnections() == 1);
                m_Found = true;
            }
        }

        bool m_Found = false;
    } visitor;

    network->ExecuteStrategy(visitor);
    REQUIRE(visitor.m_Found);
}

/// Checks that the ArmnnParseRunner correctly parses and adds a upsample (resize) layer to its internal INetwork.
TEST_CASE("ArmnnParseRunner Parse Upsample")
{
    SECTION("Scale algorithm: bilinear using ratio")
    {
        std::stringstream ggfContents(R"(
input layer, name data0, top data0, shape 1, 16, 16, 16
upsample layer, name upsample1, bottom data0, top upsample1, upsample scale height ratio 2.0, upsample scale width ratio 2.0, upsample mode height 1, upsample mode width 1, scale_algo 2
)");
        LayerData layerData;
        ArmnnParseRunner parser(ggfContents, layerData);
        const INetwork* network = parser.GetNetwork();

        struct ArmnnLayerVisitor : public armnn::IStrategy
        {
            void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                                 const armnn::BaseDescriptor& descriptor,
                                 const std::vector<armnn::ConstTensor>&,
                                 const char*,
                                 const armnn::LayerBindingId) override
            {
                if (layer->GetType() == LayerType::Resize)
                {
                    REQUIRE(!m_Found);
                    const ResizeDescriptor& desc = static_cast<const ResizeDescriptor&>(descriptor);
                    REQUIRE(desc.m_Method == ResizeMethod::Bilinear);
                    REQUIRE(desc.m_TargetHeight == 32U);
                    REQUIRE(desc.m_TargetWidth == 32U);
                    m_Found = true;
                }
            }

            bool m_Found = false;
        } visitor;

        network->ExecuteStrategy(visitor);
        REQUIRE(visitor.m_Found);
    }
    SECTION("Scale algorithm: nearest neighbour using ratio")
    {
        std::stringstream ggfContents(R"(
input layer, name data0, top data0, shape 1, 16, 16, 16
upsample layer, name upsample1, bottom data0, top upsample1, upsample scale height ratio 2.0, upsample scale width ratio 2.0, upsample mode height 0, upsample mode width 0, scale_algo 0
)");
        LayerData layerData;
        ArmnnParseRunner parser(ggfContents, layerData);
        const INetwork* network = parser.GetNetwork();

        struct ArmnnLayerVisitor : public armnn::IStrategy
        {
            void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                                 const armnn::BaseDescriptor& descriptor,
                                 const std::vector<armnn::ConstTensor>&,
                                 const char*,
                                 const armnn::LayerBindingId) override
            {
                if (layer->GetType() == LayerType::Resize)
                {
                    REQUIRE(!m_Found);
                    const ResizeDescriptor& desc = static_cast<const ResizeDescriptor&>(descriptor);
                    REQUIRE(desc.m_Method == ResizeMethod::NearestNeighbor);
                    REQUIRE(desc.m_TargetHeight == 31U);
                    REQUIRE(desc.m_TargetWidth == 31U);
                    m_Found = true;
                }
            }

            bool m_Found = false;
        } visitor;

        network->ExecuteStrategy(visitor);
        REQUIRE(visitor.m_Found);
    }
    SECTION("Using new size")
    {
        std::stringstream ggfContents(R"(
input layer, name data0, top data0, shape 1, 16, 16, 16
upsample layer, name upsample1, bottom data0, top upsample1, new height 31, new width 31, scale_algo 2
)");
        LayerData layerData;
        ArmnnParseRunner parser(ggfContents, layerData);
        const INetwork* network = parser.GetNetwork();

        struct ArmnnLayerVisitor : public armnn::IStrategy
        {
            void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                                 const armnn::BaseDescriptor& descriptor,
                                 const std::vector<armnn::ConstTensor>&,
                                 const char*,
                                 const armnn::LayerBindingId) override
            {
                if (layer->GetType() == LayerType::Resize)
                {
                    REQUIRE(!m_Found);
                    const ResizeDescriptor& desc = static_cast<const ResizeDescriptor&>(descriptor);
                    REQUIRE(desc.m_Method == ResizeMethod::Bilinear);
                    REQUIRE(desc.m_TargetHeight == 31U);
                    REQUIRE(desc.m_TargetWidth == 31U);
                    m_Found = true;
                }
            }

            bool m_Found = false;
        } visitor;

        network->ExecuteStrategy(visitor);
        REQUIRE(visitor.m_Found);
    }
}

}    // namespace system_tests
}    // namespace ethosn
