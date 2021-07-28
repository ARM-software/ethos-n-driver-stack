//
// Copyright © 2020-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNLayerSupport.hpp"
#include "EthosNReplaceUnsupported.hpp"

#include "EthosNConfig.hpp"

#include <armnn/INetwork.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>
#include <test/GraphUtils.hpp>

#include <doctest/doctest.h>

#include <numeric>

using namespace armnn;
using namespace armnn::ethosnbackend;

// By default, specific unsupported layer patterns are substituted for patterns
// that can be optimized on the backend.
TEST_SUITE("EthosNReplaceUnsupported")
{

    // Multiplication operations that take as input a Constant tensor in the shape
    // { 1, 1, 1, C } can be substituted for DepthwiseConvolution2d.
    //
    // Original pattern:
    // Input    ->
    //              Multiplication -> Output
    // Constant ->
    //
    // Expected modified pattern:
    // Input -> DepthwiseConvolution2d -> Output
    TEST_CASE("ConstMulToDepthwiseReplacement")
    {
        auto net = std::make_unique<NetworkImpl>();

        TensorInfo inputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);
        TensorInfo constInfo({ 1, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0);
        TensorInfo outputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);

        std::vector<uint8_t> constData(constInfo.GetNumElements(), 0);
        std::iota(constData.begin(), constData.end(), 0);
        ConstTensor constTensor(constInfo, constData);

        // Add the original pattern
        IConnectableLayer* const input    = net->AddInputLayer(0, "input");
        IConnectableLayer* const constant = net->AddConstantLayer(constTensor, "const");
        IConnectableLayer* const mul      = net->AddMultiplicationLayer("mul");
        IConnectableLayer* const output   = net->AddOutputLayer(0, "output");

        // Create connections between layers
        input->GetOutputSlot(0).SetTensorInfo(inputInfo);
        constant->GetOutputSlot(0).SetTensorInfo(constInfo);
        mul->GetOutputSlot(0).SetTensorInfo(outputInfo);

        input->GetOutputSlot(0).Connect(mul->GetInputSlot(0));
        constant->GetOutputSlot(0).Connect(mul->GetInputSlot(1));
        mul->GetOutputSlot(0).Connect(output->GetInputSlot(0));

        // Substitute the subgraph and check for expected pattern and connections
        Graph pattern = net->GetGraph();
        ethosnbackend::ReplaceUnsupportedLayers(pattern, EthosNConfig(), EthosNMappings(),
                                                EthosNConfig().QueryCapabilities());

        CHECK(pattern.GetNumLayers() == 3);

        const std::vector<Layer*> vecPattern(pattern.begin(), pattern.end());

        Layer* inputLayer     = vecPattern[0];
        Layer* depthwiseLayer = vecPattern[1];
        Layer* outputLayer    = vecPattern[2];

        CHECK(inputLayer->GetType() == LayerType::Input);
        CHECK(depthwiseLayer->GetType() == LayerType::DepthwiseConvolution2d);
        CHECK(outputLayer->GetType() == LayerType::Output);

        Layer* depthwiseInput  = &depthwiseLayer->GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayer();
        Layer* depthwiseOutput = &depthwiseLayer->GetOutputSlots()[0].GetConnections()[0]->GetOwningLayer();
        CHECK(depthwiseInput == inputLayer);
        CHECK(depthwiseOutput == outputLayer);

        Layer* inputNextLayer  = &inputLayer->GetOutputSlots()[0].GetConnections()[0]->GetOwningLayer();
        Layer* outputPrevLayer = &outputLayer->GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayer();
        CHECK(inputNextLayer == depthwiseLayer);
        CHECK(outputPrevLayer == depthwiseLayer);

        // Depthwise weights should be exact with the Constant data
        const uint8_t* dwWeightData = PolymorphicPointerDowncast<DepthwiseConvolution2dLayer>(depthwiseLayer)
                                          ->m_Weight->GetConstTensor<uint8_t>();
        std::vector<uint8_t> depthwiseWeights(dwWeightData, dwWeightData + constData.size());
        CHECK(depthwiseWeights == constData);
    }

    // Multiplication operations that take as input a Constant tensor in the shape
    // { 1, 1, 1, 1 } can be substituted for ReinterpretQuantize.
    //
    // Original pattern:
    // Input    ->
    //              Multiplication -> Output
    // Constant ->
    //
    // Expected modified pattern:
    // Input -> ReinterpretQuantize -> Output
    TEST_CASE("ScalarMulToReinterpretQuantizeReplacement")
    {
        auto net = std::make_unique<NetworkImpl>();

        // Quantization scale is calculated for floating range [0,2]
        float providedConstantQuantisation = (static_cast<float>(2)) / (static_cast<float>(255));
        // Floating point constant data is 2.0
        float providedConstantValue = 255;
        TensorInfo inputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 0.5f, 0);
        TensorInfo constInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, providedConstantQuantisation, 0);
        TensorInfo outputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);

        std::vector<uint8_t> constData(constInfo.GetNumElements(), 0);
        std::iota(constData.begin(), constData.end(), providedConstantValue);
        ConstTensor constTensor(constInfo, constData);

        // Add the original pattern
        IConnectableLayer* const input    = net->AddInputLayer(0, "input");
        IConnectableLayer* const constant = net->AddConstantLayer(constTensor, "const");
        IConnectableLayer* const mul      = net->AddMultiplicationLayer("mul");
        IConnectableLayer* const output   = net->AddOutputLayer(0, "output");

        // Create connections between layers
        input->GetOutputSlot(0).SetTensorInfo(inputInfo);
        constant->GetOutputSlot(0).SetTensorInfo(constInfo);
        mul->GetOutputSlot(0).SetTensorInfo(outputInfo);

        input->GetOutputSlot(0).Connect(mul->GetInputSlot(0));
        constant->GetOutputSlot(0).Connect(mul->GetInputSlot(1));
        mul->GetOutputSlot(0).Connect(output->GetInputSlot(0));

        // Substitute the subgraph and check for expected pattern and connections
        Graph pattern = net->GetGraph();
        ethosnbackend::ReplaceUnsupportedLayers(pattern, EthosNConfig(), EthosNMappings(),
                                                EthosNConfig().QueryCapabilities());

        CHECK(pattern.GetNumLayers() == 3);

        const std::vector<Layer*> vecPattern(pattern.begin(), pattern.end());

        Layer* inputLayer   = vecPattern[0];
        Layer* standInLayer = vecPattern[1];
        Layer* outputLayer  = vecPattern[2];

        CHECK(inputLayer->GetType() == LayerType::Input);
        CHECK(standInLayer->GetType() == LayerType::StandIn);
        CHECK(standInLayer->GetNameStr() == "EthosNBackend:ReplaceScalarMulWithReinterpretQuantization");
        CHECK(outputLayer->GetType() == LayerType::Output);

        Layer* standInLayerInput  = &standInLayer->GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayer();
        Layer* standInLayerOutput = &standInLayer->GetOutputSlots()[0].GetConnections()[0]->GetOwningLayer();
        CHECK(standInLayerInput == inputLayer);
        CHECK(standInLayerOutput == outputLayer);

        Layer* inputNextLayer  = &inputLayer->GetOutputSlots()[0].GetConnections()[0]->GetOwningLayer();
        Layer* outputPrevLayer = &outputLayer->GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayer();
        CHECK(inputNextLayer == standInLayer);
        CHECK(outputPrevLayer == standInLayer);
    }

    TEST_CASE("CalcConstantAddToDepthwiseReplacementConfigTest")
    {
        auto ExpectFail = [](const TensorInfo& inputInfo, const TensorInfo& constantInfo, const TensorInfo& outputInfo,
                             const char* expectedFailureReason) {
            std::string failureReason;
            Optional<ConstantAddToDepthwiseReplacementConfig> result =
                CalcConstantAddToDepthwiseReplacementConfig(inputInfo, constantInfo, outputInfo, failureReason);
            CHECK((!result.has_value() && failureReason == expectedFailureReason));
        };

        // Valid inputs
        TensorInfo validInput(TensorShape{ 1, 16, 16, 3 }, DataType::QAsymmU8, 1.0f, 0);
        TensorInfo validConstant(TensorShape{ 1, 1, 1, 3 }, DataType::QAsymmU8, 2.0f, 0);
        TensorInfo validOutput(TensorShape{ 1, 16, 16, 3 }, DataType::QAsymmU8, 4.0f, 0);

        // Error case - input has unsupported datatype
        {
            TensorInfo invalidInput = validInput;
            invalidInput.SetDataType(DataType::Float32);
            ExpectFail(invalidInput, validConstant, validOutput, "Unsupported datatype");
        }
        // Error case - constant has unsupported datatype
        {
            TensorInfo invalidConstant = validConstant;
            invalidConstant.SetDataType(DataType::Float32);
            ExpectFail(validInput, invalidConstant, validOutput, "Unsupported datatype");
        }
        // Error case - output has unsupported datatype
        {
            TensorInfo invalidOutput = validOutput;
            invalidOutput.SetDataType(DataType::Float32);
            ExpectFail(validInput, validConstant, invalidOutput, "Unsupported datatype");
        }

        // Error case - input has wrong number of dims
        {
            TensorInfo invalidInput = validInput;
            invalidInput.SetShape(TensorShape{ 1, 16, 16, 3, 16 });
            ExpectFail(invalidInput, validConstant, validOutput, "Shapes not compatible");
        }
        // Error case - constant has wrong number of dims
        {
            TensorInfo invalidConstant = validConstant;
            invalidConstant.SetShape(TensorShape{ 3, 5 });
            ExpectFail(validInput, invalidConstant, validOutput, "Shapes not compatible");
        }
        // Error case - constant has wrong shape
        {
            TensorInfo invalidConstant = validConstant;
            invalidConstant.SetShape(TensorShape{ 1, 1, 1, 4 });
            ExpectFail(validInput, invalidConstant, validOutput, "Shapes not compatible");
        }

        // Error case where no valid weight scale is possible
        {
            TensorInfo invalidInput = validInput;
            invalidInput.SetQuantizationScale(100000);
            ExpectFail(invalidInput, validConstant, validOutput, "Couldn't find valid weight scale");
        }

        // Valid case
        {
            std::string failureReason;
            Optional<ConstantAddToDepthwiseReplacementConfig> result =
                CalcConstantAddToDepthwiseReplacementConfig(validInput, validConstant, validOutput, failureReason);
            CHECK((result.has_value() && failureReason.empty()));
            ConstantAddToDepthwiseReplacementConfig config = result.value();
            CHECK(config.m_Desc.m_BiasEnabled == true);
            CHECK(config.m_Desc.m_DataLayout == DataLayout::NHWC);
            CHECK(config.m_WeightsInfo == TensorInfo(TensorShape{ 1, 1, 1, 3 }, DataType::QAsymmU8, 0.5f, 0));
            CHECK(config.m_WeightsQuantizedValue == 2);
            CHECK(config.m_BiasInfo == TensorInfo(TensorShape{ 1, 1, 1, 3 }, DataType::Signed32, 0.5f, 0));
        }
    }

    namespace
    {

    /// Creates a graph comprising an Addition of two other layers, which are either Inputs or Constants, depending
    /// on the flags provided. For any layers which are Constants, dummy constant data is generated.
    Graph CreateAdditionGraph(const TensorInfo& input0Info,
                              bool isInput0Constant,
                              const TensorInfo& input1Info,
                              bool isInput1Constant,
                              const TensorInfo& outputInfo)
    {
        auto net = std::make_unique<NetworkImpl>();

        auto AddConstLayer = [&net](const TensorInfo& info, const char* name) -> IConnectableLayer* {
            switch (info.GetDataType())
            {
                case DataType::QAsymmU8:
                {
                    std::vector<uint8_t> data(info.GetNumElements(), 0);
                    std::iota(data.begin(), data.end(), 0);
                    ConstTensor tensor(info, data);
                    return net->AddConstantLayer(tensor, name);
                }
                case DataType::QAsymmS8:    // Deliberate fallthrough
                case DataType::QSymmS8:
                {
                    std::vector<int8_t> data(info.GetNumElements(), 0);
                    std::iota(data.begin(), data.end(), -3);    // Include negative numbers for better test coverage
                    ConstTensor tensor(info, data);
                    return net->AddConstantLayer(tensor, name);
                }
                default:
                {
                    ARMNN_ASSERT(!"Not implemented");
                    return nullptr;
                }
            }
        };

        IConnectableLayer* const input0 =
            isInput0Constant ? AddConstLayer(input0Info, "input0") : net->AddInputLayer(0, "input0");
        IConnectableLayer* const input1 =
            isInput1Constant ? AddConstLayer(input1Info, "input1") : net->AddInputLayer(1, "input1");
        IConnectableLayer* const add    = net->AddAdditionLayer("add");
        IConnectableLayer* const output = net->AddOutputLayer(0, "output");

        input0->GetOutputSlot(0).SetTensorInfo(input0Info);
        input1->GetOutputSlot(0).SetTensorInfo(input1Info);
        add->GetOutputSlot(0).SetTensorInfo(outputInfo);

        input0->GetOutputSlot(0).Connect(add->GetInputSlot(0));
        input1->GetOutputSlot(0).Connect(add->GetInputSlot(1));
        add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

        return net->GetGraph();
    }

    }    // namespace

    TEST_CASE("ReplaceConstantAdditionWithDepthwiseTest")
    {
        // Failure case - not an Addition layer
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0), true,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            CHECK(ReplaceConstantAdditionWithDepthwise(g, *g.begin()) == false);
        }

        // Failure case - addition that doesn't need replacing (as it is supported natively - not a broadcast)
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), true,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            AdditionLayer* addLayer = PolymorphicPointerDowncast<AdditionLayer>(GetFirstLayerWithName(g, "add"));
            CHECK(ReplaceConstantAdditionWithDepthwise(g, addLayer) == false);
        }

        // Error case - neither input is a constant - Depthwise
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            AdditionLayer* addLayer = PolymorphicPointerDowncast<AdditionLayer>(GetFirstLayerWithName(g, "add"));
            CHECK(ReplaceConstantAdditionWithDepthwise(g, addLayer) == false);
        }

        // Valid cases
        auto ValidTest = [](bool isInput0Constant, bool isInput1Constant, DataType constantDataType) {
            // Note we use non-trivial quant params for the constant to better test the requantization that takes place
            TensorInfo constantInfo({ 1, 1, 1, 4 }, constantDataType, 10.0f, 2);
            TensorInfo inputInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0);

            Graph g                 = CreateAdditionGraph(isInput0Constant ? constantInfo : inputInfo, isInput0Constant,
                                          isInput1Constant ? constantInfo : inputInfo, isInput1Constant,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            AdditionLayer* addLayer = PolymorphicPointerDowncast<AdditionLayer>(GetFirstLayerWithName(g, "add"));
            CHECK(ReplaceConstantAdditionWithDepthwise(g, addLayer) == true);

            // Original pattern:
            // Input    ->
            //              Multiplication -> Output
            // Constant ->
            //
            // Expected modified pattern:
            // Input -> DepthwiseConvolution2d -> Output
            const std::vector<Layer*> outLayers(g.begin(), g.end());
            CHECK(outLayers.size() == 3);

            Layer* inputLayer  = outLayers[0];
            Layer* layer1      = outLayers[1];
            Layer* outputLayer = outLayers[2];

            CHECK(inputLayer->GetType() == LayerType::Input);
            CHECK(layer1->GetType() == LayerType::DepthwiseConvolution2d);
            CHECK(outputLayer->GetType() == LayerType::Output);

            const DepthwiseConvolution2dLayer* depthwiseLayer =
                PolymorphicPointerDowncast<DepthwiseConvolution2dLayer>(layer1);

            Layer* depthwiseInput  = &depthwiseLayer->GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayer();
            Layer* depthwiseOutput = &depthwiseLayer->GetOutputSlots()[0].GetConnections()[0]->GetOwningLayer();
            CHECK(depthwiseInput == inputLayer);
            CHECK(depthwiseOutput == outputLayer);

            Layer* inputNextLayer  = &inputLayer->GetOutputSlots()[0].GetConnections()[0]->GetOwningLayer();
            Layer* outputPrevLayer = &outputLayer->GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayer();
            CHECK(inputNextLayer == depthwiseLayer);
            CHECK(outputPrevLayer == depthwiseLayer);

            // Check weights tensor info and data
            CHECK(depthwiseLayer->m_Weight->GetTensorInfo() ==
                  TensorInfo(TensorShape{ 1, 1, 1, 4 }, DataType::QAsymmU8, 0.5f, 0, true));
            const uint8_t* dwWeightData = depthwiseLayer->m_Weight->GetConstTensor<uint8_t>();
            CHECK(std::all_of(dwWeightData, dwWeightData + depthwiseLayer->m_Weight->GetShape().GetNumElements(),
                              [](auto x) { return x == 2; }));

            // Check bias tensor info and data
            CHECK(depthwiseLayer->m_Bias->GetTensorInfo() ==
                  TensorInfo(TensorShape{ 1, 1, 1, 4 }, DataType::Signed32, 0.5f, 0, true));
            const int32_t* dwBiasData = depthwiseLayer->m_Bias->GetConstTensor<int32_t>();
            std::vector<int32_t> expectedBiasData;
            switch (constantDataType)
            {
                case DataType::QAsymmU8:
                    expectedBiasData = { -40, -20, 0, 20 };
                    break;
                case DataType::QAsymmS8:
                    expectedBiasData = { -100, -80, -60, -40 };
                    break;
                case DataType::QSymmS8:
                    expectedBiasData = { -60, -40, -20, 0 };
                    break;
                default:
                    ARMNN_ASSERT(!"Not implemented");
            }
            CHECK(
                (std::vector<int32_t>(dwBiasData, dwBiasData + depthwiseLayer->m_Weight->GetShape().GetNumElements()) ==
                 expectedBiasData));
        };
        // Try both combinations of input/const as first/second input. The resulting graph should be identical
        // no matter the order of the inputs.
        ValidTest(true, false, DataType::QAsymmU8);
        ValidTest(false, true, DataType::QAsymmU8);
        // Test signed data types for the constant input
        ValidTest(true, false, DataType::QAsymmS8);
        ValidTest(true, false, DataType::QSymmS8);
    }

    namespace
    {

    /// Creates a graph comprising an Multiplication of two other layers, which are either Inputs or Constants, depending
    /// on the flags provided. For any layers which are Constants, dummy constant data is generated.
    Graph CreateMultiplicationGraph(const TensorInfo& input0Info,
                                    bool isInput0Constant,
                                    const TensorInfo& input1Info,
                                    bool isInput1Constant,
                                    const TensorInfo& outputInfo,
                                    int startData = 0)
    {
        auto net = std::make_unique<NetworkImpl>();

        auto AddConstLayer = [&net](const TensorInfo& info, const char* name, int startData) -> IConnectableLayer* {
            switch (info.GetDataType())
            {
                case DataType::QAsymmU8:
                {
                    std::vector<uint8_t> data(info.GetNumElements(), 0);
                    std::iota(data.begin(), data.end(), startData);
                    ConstTensor tensor(info, data);
                    return net->AddConstantLayer(tensor, name);
                }
                case DataType::QAsymmS8:    // Deliberate fallthrough
                case DataType::QSymmS8:
                {
                    std::vector<int8_t> data(info.GetNumElements(), 0);
                    std::iota(data.begin(), data.end(), startData);
                    ConstTensor tensor(info, data);
                    return net->AddConstantLayer(tensor, name);
                }
                case DataType::Signed64:
                {
                    std::vector<int64_t> data(info.GetNumElements(), 0);
                    std::iota(data.begin(), data.end(), startData);
                    ConstTensor tensor(info, data);
                    return net->AddConstantLayer(tensor, name);
                }
                default:
                {
                    ARMNN_ASSERT(!"Not implemented");
                    return nullptr;
                }
            }
        };

        IConnectableLayer* const input0 =
            isInput0Constant ? AddConstLayer(input0Info, "input0", startData) : net->AddInputLayer(0, "input0");
        IConnectableLayer* const input1 =
            isInput1Constant ? AddConstLayer(input1Info, "input1", startData) : net->AddInputLayer(1, "input1");
        IConnectableLayer* const mul    = net->AddMultiplicationLayer("mul");
        IConnectableLayer* const output = net->AddOutputLayer(0, "output");

        input0->GetOutputSlot(0).SetTensorInfo(input0Info);
        input1->GetOutputSlot(0).SetTensorInfo(input1Info);
        mul->GetOutputSlot(0).SetTensorInfo(outputInfo);

        input0->GetOutputSlot(0).Connect(mul->GetInputSlot(0));
        input1->GetOutputSlot(0).Connect(mul->GetInputSlot(1));
        mul->GetOutputSlot(0).Connect(output->GetInputSlot(0));

        return net->GetGraph();
    }

    }    // namespace

    TEST_CASE("ScalarMulToReinterpretQuantizeReplacementTest")
    {
        std::string failureReason;

        // Failure case - not a Multiplication layer
        {
            Graph g = CreateMultiplicationGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 1.0f, 0), true,
                                                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(
                      g, *g.begin(), EthosNConfig(), EthosNMappings(), EthosNConfig().QueryCapabilities(),
                      failureReason) == false);
        }

        // Failure case - multiplication that doesn't need replacing with ReinterpretQuantization as it needs
        // to be replaced with Depthwise instead
        {
            Graph g = CreateMultiplicationGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0), true,
                                                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            MultiplicationLayer* mulLayer =
                PolymorphicPointerDowncast<MultiplicationLayer>(GetFirstLayerWithName(g, "mul"));
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(g, mulLayer, EthosNConfig(), EthosNMappings(),
                                                                         EthosNConfig().QueryCapabilities(),
                                                                         failureReason) == false);
        }

        // Error case - neither input is a constant
        {
            Graph g = CreateMultiplicationGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            MultiplicationLayer* mulLayer =
                PolymorphicPointerDowncast<MultiplicationLayer>(GetFirstLayerWithName(g, "mul"));
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(g, mulLayer, EthosNConfig(), EthosNMappings(),
                                                                         EthosNConfig().QueryCapabilities(),
                                                                         failureReason) == false);
        }

        // Error case - Incorrect data-type for constant
        {
            Graph g = CreateMultiplicationGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 1, 1, 1 }, DataType::Signed64, 1.0f, 0), true,
                                                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), 0);
            MultiplicationLayer* mulLayer =
                PolymorphicPointerDowncast<MultiplicationLayer>(GetFirstLayerWithName(g, "mul"));
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(g, mulLayer, EthosNConfig(), EthosNMappings(),
                                                                         EthosNConfig().QueryCapabilities(),
                                                                         failureReason) == false);
            CHECK(failureReason == "Data type is not supported");
        }

        // Error case - constant is negative
        {
            Graph g = CreateMultiplicationGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 0.007f, 127), true,
                                                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            MultiplicationLayer* mulLayer =
                PolymorphicPointerDowncast<MultiplicationLayer>(GetFirstLayerWithName(g, "mul"));
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(g, mulLayer, EthosNConfig(), EthosNMappings(),
                                                                         EthosNConfig().QueryCapabilities(),
                                                                         failureReason) == false);
            CHECK(failureReason == "Quantization info for input, scalar and output are not coherent");
        }

        // Error case - constant is zero
        {
            Graph g = CreateMultiplicationGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 0.007f, 127), true,
                                                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), 127);
            MultiplicationLayer* mulLayer =
                PolymorphicPointerDowncast<MultiplicationLayer>(GetFirstLayerWithName(g, "mul"));
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(g, mulLayer, EthosNConfig(), EthosNMappings(),
                                                                         EthosNConfig().QueryCapabilities(),
                                                                         failureReason) == false);
            CHECK(failureReason == "Quantization info for input, scalar and output are not coherent");
        }

        // Error case - Quantization info is not coherent
        {
            float providedConstantQuantisation = (static_cast<float>(2)) / (static_cast<float>(255));
            int providedConstantValue          = static_cast<int>(10 / providedConstantQuantisation);

            Graph g = CreateMultiplicationGraph(
                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 0.5f, 0), false,
                TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, providedConstantQuantisation, 0), true,
                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), providedConstantValue);
            MultiplicationLayer* mulLayer =
                PolymorphicPointerDowncast<MultiplicationLayer>(GetFirstLayerWithName(g, "mul"));
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(g, mulLayer, EthosNConfig(), EthosNMappings(),
                                                                         EthosNConfig().QueryCapabilities(),
                                                                         failureReason) == false);
            CHECK(failureReason == "Quantization info for input, scalar and output are not coherent");
        }

        // Error case - Constant shape is not supported
        {
            // Floating point range of the constant is [0,2.0]
            float providedConstantQuantisation = (static_cast<float>(2)) / (static_cast<float>(255));

            EthosNLayerSupport layerSupport(EthosNConfig(), EthosNMappings(), EthosNConfig().QueryCapabilities());

            const TensorInfo input0 = TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmS8, 0.5f, 0);
            const TensorInfo input1 = TensorInfo({ 1, 2, 2, 1 }, DataType::QAsymmU8, providedConstantQuantisation, 0);
            const TensorInfo output = TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0);

            Graph g = CreateMultiplicationGraph(input0, false, input1, true, output, 255);
            MultiplicationLayer* mulLayer =
                PolymorphicPointerDowncast<MultiplicationLayer>(GetFirstLayerWithName(g, "mul"));

            CHECK(layerSupport.GetMultiplicationSupportedMode(input0, input1, output) ==
                  EthosNLayerSupport::MultiplicationSupportedMode::None);
            CHECK(ReplaceMultiplication(g, mulLayer, EthosNConfig(), EthosNMappings(),
                                        EthosNConfig().QueryCapabilities()) == false);
        }

        // Error case - Constant shape is supported as an EstimateOnly operation in PerfOnly mode
        {
            // Floating point range of the constant is [0,2.0]
            float providedConstantQuantisation = (static_cast<float>(2)) / (static_cast<float>(255));

            EthosNConfig config;
            config.m_PerfOnly = true;

            EthosNLayerSupport layerSupport(config, EthosNMappings(), config.QueryCapabilities());

            const TensorInfo input0 = TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmS8, 0.5f, 0);
            const TensorInfo input1 = TensorInfo({ 1, 2, 2, 1 }, DataType::QAsymmU8, providedConstantQuantisation, 0);
            const TensorInfo output = TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0);

            Graph g = CreateMultiplicationGraph(input0, false, input1, true, output, 255);
            MultiplicationLayer* mulLayer =
                PolymorphicPointerDowncast<MultiplicationLayer>(GetFirstLayerWithName(g, "mul"));

            CHECK(layerSupport.GetMultiplicationSupportedMode(input0, input1, output) ==
                  EthosNLayerSupport::MultiplicationSupportedMode::EstimateOnly);
            CHECK(ReplaceMultiplication(g, mulLayer, config, EthosNMappings(), config.QueryCapabilities()) == false);
        }
    }

    TEST_CASE("ReplaceScalarAdditionWithReinterpretQuantizationTest")
    {
        std::string reason;

        // Failure case - not an Addition layer
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0), true,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            CHECK(ReplaceConstantAdditionWithReinterpretQuantization(g, *g.begin(), reason) == false);
        }

        // Failure case - addition that doesn't need replacing (as it is supported natively)
        // Fails as it does not need to be replaced by Reinterpret Quantization
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), true,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            AdditionLayer* addLayer = PolymorphicPointerDowncast<AdditionLayer>(GetFirstLayerWithName(g, "add"));
            CHECK(ReplaceConstantAdditionWithReinterpretQuantization(g, addLayer, reason) == false);
        }

        // Error case - neither input is a constant which is a requirement for Reinterpret Quantization
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            AdditionLayer* addLayer = PolymorphicPointerDowncast<AdditionLayer>(GetFirstLayerWithName(g, "add"));
            CHECK(ReplaceConstantAdditionWithReinterpretQuantization(g, addLayer, reason) == false);
        }

        // Error case - Quantization info is not coherent (Output Offset different than expected)
        // Positive constant means the output offset should be lower than input offset
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 5), false,
                                          TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 1.0f, 0), true,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 10));
            AdditionLayer* addLayer = PolymorphicPointerDowncast<AdditionLayer>(GetFirstLayerWithName(g, "add"));
            CHECK(ReplaceConstantAdditionWithReinterpretQuantization(g, addLayer, reason) == false);
            CHECK(reason == "Quantization info for input, scalar and output are not coherent");
        }
    }
}
