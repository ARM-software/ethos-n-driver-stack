//
// Copyright © 2020-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNLayerSupport.hpp"
#include "EthosNReplaceUnsupported.hpp"

#include "EthosNConfig.hpp"

#include <CommonTestUtils.hpp>
#include <GraphUtils.hpp>
#include <armnn/INetwork.hpp>

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
        TensorInfo constInfo({ 1, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0, true);
        TensorInfo outputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);

        std::vector<uint8_t> constData(constInfo.GetNumElements(), 0);
        std::iota(constData.begin(), constData.end(), 0);
        ConstTensor constTensor(constInfo, constData);

        // Add the original pattern
        IConnectableLayer* const input    = net->AddInputLayer(0, "input");
        IConnectableLayer* const constant = net->AddConstantLayer(constTensor, "const");
        IConnectableLayer* const mul =
            net->AddElementwiseBinaryLayer(ElementwiseBinaryDescriptor(BinaryOperation::Mul), "mul");
        IConnectableLayer* const output = net->AddOutputLayer(0, "output");

        // Create connections between layers
        input->GetOutputSlot(0).SetTensorInfo(inputInfo);
        constant->GetOutputSlot(0).SetTensorInfo(constInfo);
        mul->GetOutputSlot(0).SetTensorInfo(outputInfo);

        input->GetOutputSlot(0).Connect(mul->GetInputSlot(0));
        constant->GetOutputSlot(0).Connect(mul->GetInputSlot(1));
        mul->GetOutputSlot(0).Connect(output->GetInputSlot(0));

        // Substitute the subgraph and check for expected pattern and connections
        SubgraphView::SubgraphViewPtr pattern(new SubgraphView({ input, constant, mul, output }, {}, {}));
        SubgraphView workingCopy = pattern->GetWorkingCopy();
        INetworkPtr network      = INetwork::Create();
        ethosnbackend::ReplaceUnsupportedLayers(workingCopy, *network, EthosNConfig(),
                                                EthosNConfig().QueryCapabilities());

        CHECK(workingCopy.GetIConnectableLayers().size() == 4);

        const std::vector<IConnectableLayer*> vecPattern(workingCopy.beginIConnectable(),
                                                         workingCopy.endIConnectable());

        IConnectableLayer* inputLayer     = vecPattern[0];
        IConnectableLayer* weightsLayer   = vecPattern[1];
        IConnectableLayer* depthwiseLayer = vecPattern[2];
        IConnectableLayer* outputLayer    = vecPattern[3];

        CHECK(inputLayer->GetType() == LayerType::Input);
        CHECK(weightsLayer->GetType() == LayerType::Constant);
        CHECK(depthwiseLayer->GetType() == LayerType::DepthwiseConvolution2d);
        CHECK(outputLayer->GetType() == LayerType::Output);

        IConnectableLayer* depthwiseInput =
            &depthwiseLayer->GetInputSlot(0).GetConnection()->GetOwningIConnectableLayer();
        IConnectableLayer* depthwiseInput1 =
            &depthwiseLayer->GetInputSlot(1).GetConnection()->GetOwningIConnectableLayer();
        IConnectableLayer* depthwiseOutput =
            &depthwiseLayer->GetOutputSlot(0).GetConnection(0)->GetOwningIConnectableLayer();
        CHECK(depthwiseInput == inputLayer);
        CHECK(depthwiseInput1 == weightsLayer);
        CHECK(depthwiseOutput == outputLayer);

        IConnectableLayer* inputNextLayer =
            &inputLayer->GetOutputSlot(0).GetConnection(0)->GetOwningIConnectableLayer();
        IConnectableLayer* outputPrevLayer =
            &outputLayer->GetInputSlot(0).GetConnection()->GetOwningIConnectableLayer();
        CHECK(inputNextLayer == depthwiseLayer);
        CHECK(outputPrevLayer == depthwiseLayer);

        // Depthwise weights should be exact with the Constant data
        const uint8_t* dwWeightData =
            PolymorphicPointerDowncast<ConstantLayer>(weightsLayer)->m_LayerOutput->GetConstTensor<uint8_t>();
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
        uint8_t providedConstantValue = 255;
        TensorInfo inputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 0.5f, 0);
        TensorInfo constInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, providedConstantQuantisation, 0, true);
        TensorInfo outputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);

        std::vector<uint8_t> constData(constInfo.GetNumElements(), 0);
        std::iota(constData.begin(), constData.end(), providedConstantValue);
        ConstTensor constTensor(constInfo, constData);

        // Add the original pattern
        IConnectableLayer* const input    = net->AddInputLayer(0, "input");
        IConnectableLayer* const constant = net->AddConstantLayer(constTensor, "const");
        IConnectableLayer* const mul =
            net->AddElementwiseBinaryLayer(ElementwiseBinaryDescriptor(BinaryOperation::Mul), "mul");
        IConnectableLayer* const output = net->AddOutputLayer(0, "output");

        // Create connections between layers
        input->GetOutputSlot(0).SetTensorInfo(inputInfo);
        constant->GetOutputSlot(0).SetTensorInfo(constInfo);
        mul->GetOutputSlot(0).SetTensorInfo(outputInfo);

        input->GetOutputSlot(0).Connect(mul->GetInputSlot(0));
        constant->GetOutputSlot(0).Connect(mul->GetInputSlot(1));
        mul->GetOutputSlot(0).Connect(output->GetInputSlot(0));

        // Substitute the subgraph and check for expected pattern and connections
        SubgraphView::SubgraphViewPtr pattern(new SubgraphView({ input, constant, mul, output }, {}, {}));
        SubgraphView workingCopy = pattern->GetWorkingCopy();
        INetworkPtr network      = INetwork::Create();
        ethosnbackend::ReplaceUnsupportedLayers(workingCopy, *network, EthosNConfig(),
                                                EthosNConfig().QueryCapabilities());

        CHECK(workingCopy.GetIConnectableLayers().size() == 3);

        const std::vector<IConnectableLayer*> vecPattern(workingCopy.beginIConnectable(),
                                                         workingCopy.endIConnectable());

        IConnectableLayer* inputLayer   = vecPattern[0];
        IConnectableLayer* standInLayer = vecPattern[1];
        IConnectableLayer* outputLayer  = vecPattern[2];

        CHECK(inputLayer->GetType() == LayerType::Input);
        CHECK(standInLayer->GetType() == LayerType::StandIn);
        CHECK(std::string(standInLayer->GetName()) == "EthosNBackend:ReplaceScalarMulWithReinterpretQuantization");
        CHECK(outputLayer->GetType() == LayerType::Output);

        IConnectableLayer* standInLayerInput =
            &standInLayer->GetInputSlot(0).GetConnection()->GetOwningIConnectableLayer();
        IConnectableLayer* standInLayerOutput =
            &standInLayer->GetOutputSlot(0).GetConnection(0)->GetOwningIConnectableLayer();
        CHECK(standInLayerInput == inputLayer);
        CHECK(standInLayerOutput == outputLayer);

        IConnectableLayer* inputNextLayer =
            &inputLayer->GetOutputSlot(0).GetConnection(0)->GetOwningIConnectableLayer();
        IConnectableLayer* outputPrevLayer =
            &outputLayer->GetInputSlot(0).GetConnection()->GetOwningIConnectableLayer();
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
            CHECK(config.m_WeightsInfo == TensorInfo(TensorShape{ 1, 1, 1, 3 }, DataType::QAsymmU8, 0.5f, 0, true));
            CHECK(config.m_WeightsQuantizedValue == 2);
            CHECK(config.m_BiasInfo == TensorInfo(TensorShape{ 1, 1, 1, 3 }, DataType::Signed32, 0.5f, 0, true));
        }
    }

    namespace
    {

    armnn::IConnectableLayer* GetFirstLayerWithName(const armnn::SubgraphView& graph, const std::string& name)
    {
        for (auto it = graph.beginIConnectable(); it != graph.endIConnectable(); it++)
        {
            if (std::string((*it)->GetName()) == name)
            {
                return *it;
            }
        }
        return nullptr;
    }

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
                    ARMNN_ASSERT_MSG(false, "Not implemented");
                    return nullptr;
                }
            }
        };

        IConnectableLayer* const input0 =
            isInput0Constant ? AddConstLayer(input0Info, "input0") : net->AddInputLayer(0, "input0");
        IConnectableLayer* const input1 =
            isInput1Constant ? AddConstLayer(input1Info, "input1") : net->AddInputLayer(1, "input1");
        IConnectableLayer* const add =
            net->AddElementwiseBinaryLayer(ElementwiseBinaryDescriptor(BinaryOperation::Add), "add");
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
                                          TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0, true), true,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceConstantAdditionWithDepthwise(workingCopy, *g.begin(), *network) == false);
        }

        // Failure case - addition that doesn't need replacing (as it is supported natively - not a broadcast)
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0, true), true,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* addLayer            = GetFirstLayerWithName(workingCopy, "add");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceConstantAdditionWithDepthwise(workingCopy, addLayer, *network) == false);
        }

        // Error case - neither input is a constant - Depthwise
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* addLayer            = GetFirstLayerWithName(workingCopy, "add");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceConstantAdditionWithDepthwise(workingCopy, addLayer, *network) == false);
        }

        // Valid cases
        auto ValidTest = [](bool isInput0Constant, bool isInput1Constant, DataType constantDataType) {
            // Note we use non-trivial quant params for the constant to better test the requantization that takes place
            TensorInfo constantInfo({ 1, 1, 1, 4 }, constantDataType, 10.0f, 2, true);
            TensorInfo inputInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0);

            Graph g = CreateAdditionGraph(isInput0Constant ? constantInfo : inputInfo, isInput0Constant,
                                          isInput1Constant ? constantInfo : inputInfo, isInput1Constant,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* addLayer            = GetFirstLayerWithName(workingCopy, "add");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceConstantAdditionWithDepthwise(workingCopy, addLayer, *network) == true);

            // Original pattern:
            // Input    ->
            //              Multiplication -> Output
            // Constant ->
            //
            // Expected modified pattern:
            // Input -> DepthwiseConvolution2d -> Output
            const std::vector<IConnectableLayer*> outLayers(workingCopy.beginIConnectable(),
                                                            workingCopy.endIConnectable());
            CHECK(outLayers.size() == 5);

            IConnectableLayer* inputLayer     = outLayers[0];
            IConnectableLayer* weightsLayer   = outLayers[1];
            IConnectableLayer* biasLayer      = outLayers[2];
            IConnectableLayer* depthwiseLayer = outLayers[3];
            IConnectableLayer* outputLayer    = outLayers[4];

            CHECK(inputLayer->GetType() == LayerType::Input);
            CHECK(weightsLayer->GetType() == LayerType::Constant);
            CHECK(biasLayer->GetType() == LayerType::Constant);
            CHECK(depthwiseLayer->GetType() == LayerType::DepthwiseConvolution2d);
            CHECK(outputLayer->GetType() == LayerType::Output);

            IConnectableLayer* depthwiseInput =
                &depthwiseLayer->GetInputSlot(0).GetConnection()->GetOwningIConnectableLayer();
            IConnectableLayer* depthwiseInput1 =
                &depthwiseLayer->GetInputSlot(1).GetConnection()->GetOwningIConnectableLayer();
            IConnectableLayer* depthwiseInput2 =
                &depthwiseLayer->GetInputSlot(2).GetConnection()->GetOwningIConnectableLayer();
            IConnectableLayer* depthwiseOutput =
                &depthwiseLayer->GetOutputSlot(0).GetConnection(0)->GetOwningIConnectableLayer();
            CHECK(depthwiseInput == inputLayer);
            CHECK(depthwiseInput1 == weightsLayer);
            CHECK(depthwiseInput2 == biasLayer);
            CHECK(depthwiseOutput == outputLayer);

            IConnectableLayer* inputNextLayer =
                &inputLayer->GetOutputSlot(0).GetConnection(0)->GetOwningIConnectableLayer();
            IConnectableLayer* outputPrevLayer =
                &outputLayer->GetInputSlot(0).GetConnection()->GetOwningIConnectableLayer();
            CHECK(inputNextLayer == depthwiseLayer);
            CHECK(outputPrevLayer == depthwiseLayer);

            // Check weights tensor info and data
            CHECK(weightsLayer->GetOutputSlot(0).GetTensorInfo() ==
                  TensorInfo(TensorShape{ 1, 1, 1, 4 }, DataType::QAsymmU8, 0.5f, 0, true));
            const uint8_t* dwWeightData =
                PolymorphicPointerDowncast<ConstantLayer>(weightsLayer)->m_LayerOutput->GetConstTensor<uint8_t>();
            CHECK(std::all_of(dwWeightData,
                              dwWeightData + weightsLayer->GetOutputSlot(0).GetTensorInfo().GetShape().GetNumElements(),
                              [](auto x) { return x == 2; }));

            // Check bias tensor info and data
            CHECK(biasLayer->GetOutputSlot(0).GetTensorInfo() ==
                  TensorInfo(TensorShape{ 1, 1, 1, 4 }, DataType::Signed32, 0.5f, 0, true));
            const int32_t* dwBiasData =
                PolymorphicPointerDowncast<ConstantLayer>(biasLayer)->m_LayerOutput->GetConstTensor<int32_t>();
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
                    ARMNN_ASSERT_MSG(false, "Not implemented");
            }
            CHECK((std::vector<int32_t>(
                       dwBiasData,
                       dwBiasData + weightsLayer->GetOutputSlot(0).GetTensorInfo().GetShape().GetNumElements()) ==
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
                    ARMNN_ASSERT_MSG(false, "Not implemented");
                    return nullptr;
                }
            }
        };

        IConnectableLayer* const input0 =
            isInput0Constant ? AddConstLayer(input0Info, "input0", startData) : net->AddInputLayer(0, "input0");
        IConnectableLayer* const input1 =
            isInput1Constant ? AddConstLayer(input1Info, "input1", startData) : net->AddInputLayer(1, "input1");
        IConnectableLayer* const mul =
            net->AddElementwiseBinaryLayer(ElementwiseBinaryDescriptor(BinaryOperation::Mul), "mul");
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
                                                TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 1.0f, 0, true), true,
                                                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(
                      workingCopy, *g.begin(), *network, EthosNConfig(), EthosNConfig().QueryCapabilities(),
                      failureReason) == false);
        }

        // Failure case - multiplication that doesn't need replacing with ReinterpretQuantization as it needs
        // to be replaced with Depthwise instead
        {
            Graph g = CreateMultiplicationGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0, true), true,
                                                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* mulLayer            = GetFirstLayerWithName(workingCopy, "mul");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(
                      workingCopy, mulLayer, *network, EthosNConfig(), EthosNConfig().QueryCapabilities(),
                      failureReason) == false);
        }

        // Error case - neither input is a constant
        {
            Graph g = CreateMultiplicationGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* mulLayer            = GetFirstLayerWithName(workingCopy, "mul");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(
                      workingCopy, mulLayer, *network, EthosNConfig(), EthosNConfig().QueryCapabilities(),
                      failureReason) == false);
        }

        // Error case - Incorrect data-type for constant
        {
            Graph g = CreateMultiplicationGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 1, 1, 1 }, DataType::Signed64, 1.0f, 0, true), true,
                                                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), 0);
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* mulLayer            = GetFirstLayerWithName(workingCopy, "mul");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(
                      workingCopy, mulLayer, *network, EthosNConfig(), EthosNConfig().QueryCapabilities(),
                      failureReason) == false);
            CHECK(failureReason == "Data type is not supported");
        }

        // Error case - constant is negative
        {
            Graph g = CreateMultiplicationGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 0.007f, 127, true), true,
                                                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* mulLayer            = GetFirstLayerWithName(workingCopy, "mul");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(
                      workingCopy, mulLayer, *network, EthosNConfig(), EthosNConfig().QueryCapabilities(),
                      failureReason) == false);
            CHECK(failureReason == "Quantization info for input, scalar and output are not coherent");
        }

        // Error case - constant is zero
        {
            Graph g = CreateMultiplicationGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                                TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 0.007f, 127, true), true,
                                                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), 127);
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* mulLayer            = GetFirstLayerWithName(workingCopy, "mul");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(
                      workingCopy, mulLayer, *network, EthosNConfig(), EthosNConfig().QueryCapabilities(),
                      failureReason) == false);
            CHECK(failureReason == "Quantization info for input, scalar and output are not coherent");
        }

        // Error case - Quantization info is not coherent
        {
            float providedConstantQuantisation = (static_cast<float>(2)) / (static_cast<float>(255));
            int providedConstantValue          = static_cast<int>(10 / providedConstantQuantisation);

            Graph g = CreateMultiplicationGraph(
                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 0.5f, 0), false,
                TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, providedConstantQuantisation, 0, true), true,
                TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), providedConstantValue);
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* mulLayer            = GetFirstLayerWithName(workingCopy, "mul");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceScalarMultiplicationWithReinterpretQuantization(
                      workingCopy, mulLayer, *network, EthosNConfig(), EthosNConfig().QueryCapabilities(),
                      failureReason) == false);
            CHECK(failureReason == "Quantization info for input, scalar and output are not coherent");
        }

        // Error case - Constant shape is not supported
        {
            // Floating point range of the constant is [0,2.0]
            float providedConstantQuantisation = (static_cast<float>(2)) / (static_cast<float>(255));

            EthosNLayerSupport layerSupport(EthosNConfig(), EthosNConfig().QueryCapabilities());

            const TensorInfo input0 = TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmS8, 0.5f, 0);
            const TensorInfo input1 =
                TensorInfo({ 1, 2, 2, 1 }, DataType::QAsymmU8, providedConstantQuantisation, 0, true);
            const TensorInfo output = TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0);

            Graph g = CreateMultiplicationGraph(input0, false, input1, true, output, 255);

            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* mulLayer            = GetFirstLayerWithName(workingCopy, "mul");
            INetworkPtr network                    = INetwork::Create();
            CHECK(layerSupport.GetMultiplicationSupportedMode(input0, input1, output) ==
                  EthosNLayerSupport::MultiplicationSupportedMode::None);
            CHECK(ReplaceMultiplication(workingCopy, mulLayer, *network, EthosNConfig(),
                                        EthosNConfig().QueryCapabilities()) == false);
        }

        // Error case - Constant shape is supported as an EstimateOnly operation in PerfOnly mode
        {
            // Floating point range of the constant is [0,2.0]
            float providedConstantQuantisation = (static_cast<float>(2)) / (static_cast<float>(255));

            EthosNConfig config;
            config.m_PerfOnly = true;

            EthosNLayerSupport layerSupport(config, config.QueryCapabilities());

            const TensorInfo input0 = TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmS8, 0.5f, 0);
            const TensorInfo input1 =
                TensorInfo({ 1, 2, 2, 1 }, DataType::QAsymmU8, providedConstantQuantisation, 0, true);
            const TensorInfo output = TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0);

            Graph g = CreateMultiplicationGraph(input0, false, input1, true, output, 255);

            CHECK(layerSupport.GetMultiplicationSupportedMode(input0, input1, output) ==
                  EthosNLayerSupport::MultiplicationSupportedMode::EstimateOnly);
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* mulLayer            = GetFirstLayerWithName(workingCopy, "mul");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceMultiplication(workingCopy, mulLayer, *network, config, config.QueryCapabilities()) == false);
        }
    }

    TEST_CASE("ReplaceScalarAdditionWithReinterpretQuantizationTest")
    {
        std::string reason;

        // Failure case - not an Addition layer
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0, true), true,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceConstantAdditionWithReinterpretQuantization(workingCopy, *workingCopy.beginIConnectable(),
                                                                     *network, reason) == false);
        }

        // Failure case - addition that doesn't need replacing (as it is supported natively)
        // Fails as it does not need to be replaced by Reinterpret Quantization
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0, true), true,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* addLayer            = GetFirstLayerWithName(workingCopy, "add");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceConstantAdditionWithReinterpretQuantization(workingCopy, addLayer, *network, reason) == false);
        }

        // Error case - neither input is a constant which is a requirement for Reinterpret Quantization
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0), false,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 0));
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* addLayer            = GetFirstLayerWithName(workingCopy, "add");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceConstantAdditionWithReinterpretQuantization(workingCopy, addLayer, *network, reason) == false);
        }

        // Error case - Quantization info is not coherent (Output Offset different than expected)
        // Positive constant means the output offset should be lower than input offset
        {
            Graph g = CreateAdditionGraph(TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 5), false,
                                          TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 1.0f, 0, true), true,
                                          TensorInfo({ 1, 8, 8, 4 }, DataType::QAsymmU8, 1.0f, 10));
            SubgraphView::SubgraphViewPtr subgraph = std::make_shared<SubgraphView>(g);
            SubgraphView workingCopy               = subgraph->GetWorkingCopy();
            IConnectableLayer* addLayer            = GetFirstLayerWithName(workingCopy, "add");
            INetworkPtr network                    = INetwork::Create();
            CHECK(ReplaceConstantAdditionWithReinterpretQuantization(workingCopy, addLayer, *network, reason) == false);
            CHECK(reason == "Quantization info for input, scalar and output are not coherent");
        }
    }
}
