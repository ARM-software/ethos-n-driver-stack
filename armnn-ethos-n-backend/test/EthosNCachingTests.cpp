//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNBackend.hpp"
#include "EthosNCaching.hpp"
#include "EthosNSubgraphViewConverter.hpp"
#include "EthosNTestUtils.hpp"

#include <CommonTestUtils.hpp>
#include <doctest/doctest.h>

using namespace armnn;

namespace
{

// Create a simple network with one input, relu and output layers.
armnn::INetworkPtr CreateSimpleNetwork(TensorInfo& inputTensorInfo, TensorInfo& outputTensorInfo)
{
    armnn::INetworkPtr net = armnn::INetwork::Create();

    armnn::IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input layer");
    CHECK(inputLayer);

    ActivationDescriptor reluDesc;
    reluDesc.m_A                              = 100;
    reluDesc.m_B                              = 0;
    reluDesc.m_Function                       = ActivationFunction::BoundedReLu;
    armnn::IConnectableLayer* const reluLayer = net->AddActivationLayer(reluDesc, "relu layer");
    CHECK(reluLayer);

    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output layer");
    CHECK(outputLayer);

    inputLayer->GetOutputSlot(0).Connect(reluLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    reluLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    reluLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    return net;
}

// Creates a simple subgraph with one input, convolution and output layers.
SubgraphView::SubgraphViewPtr CreateSimpleSubgraph(Graph& graph)
{
    const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0, true);
    const TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0, true);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input layer");
    CHECK(inputLayer);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    Convolution2dLayer* const convLayer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv layer");
    CHECK(convLayer);

    // Arm NN is transitioning from having weights/bias as intrinsic properties of the layer to having them
    // as separate layers with connections. For now, we need to do both.
    auto weights           = graph.AddLayer<ConstantLayer>("Weights");
    weights->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightInfo);
    weights->m_LayerOutput->Allocate();
    weights->GetOutputSlot().SetTensorInfo(weightInfo);
    weights->GetOutputSlot().Connect(convLayer->GetInputSlot(1));
    auto bias           = graph.AddLayer<ConstantLayer>("Bias");
    bias->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasInfo);
    bias->m_LayerOutput->Allocate();
    bias->GetOutputSlot().SetTensorInfo(biasInfo);
    bias->GetOutputSlot().Connect(convLayer->GetInputSlot(2));

    SetWeightAndBias(convLayer, weightInfo, biasInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output layer");
    CHECK(outputLayer);

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network

    armnn::SubgraphView view({ convLayer, weights, bias }, { &convLayer->GetInputSlot(0) },
                             { &convLayer->GetOutputSlot(0) });
    return std::make_unique<SubgraphView>(view);
}

}    // anonymous namespace

TEST_SUITE("EthosNCaching")
{

    /// Checks that GetEthosNCachingOptions correctly handles user-provided ModelOptions.
    TEST_CASE("TestGetEthosNCachingOptionsFromModelOptions")
    {
        // Default with no caching
        CHECK(GetEthosNCachingOptionsFromModelOptions({}).m_SaveCachedNetwork == false);
        CHECK(GetEthosNCachingOptionsFromModelOptions({}).m_CachedNetworkFilePath == "");

        // Create temp file.
        const testing_utils::TempDir tmpDir;
        std::string filePath = tmpDir.Str() + "/EthosN-CachingOptions-TempFile1.bin";
        std::ofstream file{ filePath };

        // Enable caching and set file path correctly E.g. file exists.
        BackendOptions backendOptions(EthosNBackend::GetIdStatic(),
                                      { { "SaveCachedNetwork", true }, { "CachedNetworkFilePath", filePath } });
        CHECK(GetEthosNCachingOptionsFromModelOptions({ backendOptions }).m_SaveCachedNetwork == true);
        CHECK(GetEthosNCachingOptionsFromModelOptions({ backendOptions }).m_CachedNetworkFilePath == filePath);

        // Other backend options are ignored
        BackendOptions optOtherBackend("OtherBackend",
                                       { { "SaveCachedNetwork", true }, { "CachedNetworkFilePath", filePath } });
        CHECK(GetEthosNCachingOptionsFromModelOptions({ optOtherBackend }).m_SaveCachedNetwork == false);
        CHECK(GetEthosNCachingOptionsFromModelOptions({ optOtherBackend }).m_CachedNetworkFilePath == "");

        // Invalid option (Wrong SaveCachedNetwork type)
        BackendOptions optInvalidTypeSaveCache(EthosNBackend::GetIdStatic(), { { "SaveCachedNetwork", "test" } });
        CHECK_THROWS_WITH_AS(GetEthosNCachingOptionsFromModelOptions({ optInvalidTypeSaveCache }),
                             "Invalid option type for SaveCachedNetwork - must be bool.", InvalidArgumentException);

        // Invalid option (Wrong CachedNetworkFilePath type)
        BackendOptions optInvalidTypeFilePath(EthosNBackend::GetIdStatic(), { { "CachedNetworkFilePath", true } });
        CHECK_THROWS_WITH_AS(GetEthosNCachingOptionsFromModelOptions({ optInvalidTypeFilePath }),
                             "Invalid option type for CachedNetworkFilePath - must be string.",
                             InvalidArgumentException);

        // Invalid option (File doesn't exist)
        BackendOptions optInvalidFilePath(EthosNBackend::GetIdStatic(), { { "CachedNetworkFilePath", "test" } });
        CHECK_THROWS_WITH_AS(GetEthosNCachingOptionsFromModelOptions({ optInvalidFilePath }),
                             "The file used to write cached networks to is invalid or doesn't exist.",
                             InvalidArgumentException);
    }

    TEST_CASE("TestCachingEndToEnd")
    {
        // Reset the caching pointer
        EthosNCachingService& resetService = EthosNCachingService::GetInstance();
        resetService.SetEthosNCachingPtr(std::make_shared<EthosNCaching>());

        // Create a temp directory and empty binary file to write to.
        const testing_utils::TempDir tmpDir;
        std::string filePath = tmpDir.Str() + "/EthosN-CachingEndToEnd-TempFile.bin";
        std::ofstream file{ filePath };

        // Create two networks, one first will be used for saving a cached network.
        // The second will load the previously saved network from the file.
        TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8, 0.9f, 0, true);
        TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8, 0.9f, 0, true);

        const std::string EthosnBackendId{ "EthosNAcc" };

        // Creates structures for inputs and outputs.
        const std::vector<uint8_t> inputData(inputTensorInfo.GetNumElements(), 1);
        std::vector<uint8_t> outputDataSave(outputTensorInfo.GetNumElements(), 0);
        std::vector<uint8_t> outputDataLoad(outputTensorInfo.GetNumElements(), 0);

        {
            // Save network run.
            armnn::INetworkPtr networkSave = CreateSimpleNetwork(inputTensorInfo, outputTensorInfo);

            // Create default Arm NN runtime
            IRuntime::CreationOptions options;
            IRuntimePtr runtimeSave = IRuntime::Create(options);

            armnn::OptimizerOptions saveOptions;
            BackendOptions saveBackendOptions(EthosnBackendId,
                                              { { "SaveCachedNetwork", true }, { "CachedNetworkFilePath", filePath } });
            saveOptions.m_ModelOptions.push_back(saveBackendOptions);

            armnn::IOptimizedNetworkPtr optSaveNetwork =
                armnn::Optimize(*networkSave, { EthosnBackendId }, runtimeSave->GetDeviceSpec(), saveOptions);

            // Cached file should be empty until first network is loaded.
            CHECK(fs::is_empty(filePath));

            // Load first graph into runtime.
            armnn::NetworkId networkIdentifierSave;
            runtimeSave->LoadNetwork(networkIdentifierSave, std::move(optSaveNetwork));

            // File should now exist and shouldn't be empty.
            CHECK(fs::exists(filePath));
            std::string fileContents = testing_utils::ReadFile(filePath);
            CHECK(!fileContents.empty());

            armnn::InputTensors inputTensor{ { 0, armnn::ConstTensor(inputTensorInfo, inputData.data()) } };
            armnn::OutputTensors outputTensorSave{ { 0, armnn::Tensor(outputTensorInfo, outputDataSave.data()) } };

            // Execute networks
            runtimeSave->EnqueueWorkload(networkIdentifierSave, inputTensor, outputTensorSave);
        }

        {
            // Load network run
            armnn::INetworkPtr networkLoad = CreateSimpleNetwork(inputTensorInfo, outputTensorInfo);

            IRuntime::CreationOptions options;
            IRuntimePtr runtimeLoad = IRuntime::Create(options);

            armnn::OptimizerOptions loadOptions;
            BackendOptions loadBackendOptions(
                EthosnBackendId, { { "SaveCachedNetwork", false }, { "CachedNetworkFilePath", filePath } });
            loadOptions.m_ModelOptions.push_back(loadBackendOptions);

            armnn::IOptimizedNetworkPtr optLoadNetwork =
                armnn::Optimize(*networkLoad, { EthosnBackendId }, runtimeLoad->GetDeviceSpec(), loadOptions);

            armnn::NetworkId networkIdentifierLoad;
            runtimeLoad->LoadNetwork(networkIdentifierLoad, std::move(optLoadNetwork));

            armnn::InputTensors inputTensor{ { 0, armnn::ConstTensor(inputTensorInfo, inputData.data()) } };
            armnn::OutputTensors outputTensorLoad{ { 0, armnn::Tensor(outputTensorInfo, outputDataLoad.data()) } };

            runtimeLoad->EnqueueWorkload(networkIdentifierLoad, inputTensor, outputTensorLoad);
        }

        // Compare outputs from both networks.
        CHECK(std::equal(outputDataSave.begin(), outputDataSave.end(), outputDataLoad.begin(), outputDataLoad.end()));
    }

    // Test that emulates an example where there are two subgraphs.
    TEST_CASE("TestCachingWithMultipleSubgraphs")
    {
        // Reset the caching pointer
        EthosNCachingService& resetService = EthosNCachingService::GetInstance();
        resetService.SetEthosNCachingPtr(std::make_shared<EthosNCaching>());

        // Create temp file.
        const testing_utils::TempDir tmpDir;
        std::string filePath = tmpDir.Str() + "/EthosN-MultipleSubgraphs-TempFile.bin";
        std::ofstream file{ filePath };

        // It's hard to create this in an end to end test
        // so instead we will emulate saving by invoking OptimizeSubgraphView twice.
        {
            Graph graph1;
            Graph graph2;

            // Create a fully optimizable subgraph
            SubgraphViewSelector::SubgraphViewPtr subgraphPtr1 = CreateSimpleSubgraph(graph1);
            CHECK((subgraphPtr1 != nullptr));

            SubgraphViewSelector::SubgraphViewPtr subgraphPtr2 = CreateSimpleSubgraph(graph2);
            CHECK((subgraphPtr2 != nullptr));

            // Create a backend object
            auto backendObjPtr = CreateBackendObject(EthosNBackend::GetIdStatic());
            CHECK((backendObjPtr != nullptr));

            BackendOptions backendOptions(EthosNBackend::GetIdStatic(),
                                          { { "SaveCachedNetwork", true }, { "CachedNetworkFilePath", filePath } });

            // Optimize the subgraphs (Saving - Adds compiled networks to vector)
            CHECK_NOTHROW(backendObjPtr->OptimizeSubgraphView(*subgraphPtr1, { backendOptions }));
            CHECK_NOTHROW(backendObjPtr->OptimizeSubgraphView(*subgraphPtr2, { backendOptions }));

            auto caching                = EthosNCachingService::GetInstance().GetEthosNCachingPtr();
            auto cachedCompiledNetworks = caching->GetCompiledNetworks();
            CHECK(cachedCompiledNetworks.size() == 2);

            // Cached file should be empty until save is invoked.
            CHECK(fs::is_empty(filePath));

            // Save the vector to a file and reset object.
            caching->Save();

            // File should now exist and shouldn't be empty.
            CHECK(fs::exists(filePath));
            std::string fileContents = testing_utils::ReadFile(filePath);
            CHECK(!fileContents.empty());
        }

        // Loading is a little harder to emulate two subgraphs due to how it's designed.
        // However, a simple call of the load function is enough to check if it's loaded correctly into the EthosNCaching object.
        {
            Graph graph1;
            Graph graph2;

            // Create a fully optimizable subgraph
            SubgraphViewSelector::SubgraphViewPtr subgraphPtr1 = CreateSimpleSubgraph(graph1);
            CHECK((subgraphPtr1 != nullptr));

            SubgraphViewSelector::SubgraphViewPtr subgraphPtr2 = CreateSimpleSubgraph(graph2);
            CHECK((subgraphPtr2 != nullptr));

            // Create a backend object
            auto backendObjPtr = CreateBackendObject(EthosNBackend::GetIdStatic());
            CHECK((backendObjPtr != nullptr));

            BackendOptions backendOptions(EthosNBackend::GetIdStatic(),
                                          { { "SaveCachedNetwork", false }, { "CachedNetworkFilePath", filePath } });

            // Reload the vector from the file created above.
            auto caching = EthosNCachingService::GetInstance().GetEthosNCachingPtr();
            caching->SetEthosNCachingOptions({ backendOptions });
            caching->Load();

            // The compiled networks should have been added to the vector from the file.
            auto cachedCompiledNetworks = caching->GetCompiledNetworks();
            CHECK(cachedCompiledNetworks.size() == 2);

            // Optimize the subgraphs (Loading - Uses compiled networks from vector)
            CHECK_NOTHROW(backendObjPtr->OptimizeSubgraphView(*subgraphPtr1, { backendOptions }));
            CHECK_NOTHROW(backendObjPtr->OptimizeSubgraphView(*subgraphPtr2, { backendOptions }));
        }
        // Save the compiled networks so we reset the caching  pointer so we don't break other tests.
        auto service = EthosNCachingService::GetInstance();
        service.GetEthosNCachingPtr()->Save();
    }
}
