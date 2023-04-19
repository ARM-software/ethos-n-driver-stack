//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNBackendId.hpp"
#include "EthosNConfig.hpp"
#include "EthosNSubgraphViewConverter.hpp"
#include "EthosNTensorHandle.hpp"
#include "EthosNTestUtils.hpp"
#include "EthosNWorkloadFactory.hpp"
#include "EthosNWorkloads.hpp"

#include <armnnUtils/Filesystem.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("EthosNCreateEstimationWorkload")
{

    // Tests that the NPU config file is parsed correctly
    TEST_CASE("ParseEthosNConfig")
    {
        // Note we don't use any helper function to write the file here,
        // because we want this test to fail if the format of or names in the
        // config file change, as this would be a change to the public API and should be explicitly acknowledged
        // by updating this test case.
        std::stringstream os;
        os << armnn::EthosNConfig::PERF_ONLY_VAR << " = 1\n";
        os << armnn::EthosNConfig::PERF_VARIANT_VAR << " = Ethos-N78_1TOPS_2PLE_RATIO\n";
        os << armnn::EthosNConfig::PERF_SRAM_SIZE_BYTES_OVERRIDE_VAR << " = 12\n";
        os << armnn::EthosNConfig::PERF_OUT_DIR_VAR << " = test\n";
        os << armnn::EthosNConfig::DUMP_DEBUG_FILES_VAR << " = 1\n";
        os << armnn::EthosNConfig::DUMP_RAM_VAR << " = 1\n";
        os << armnn::EthosNConfig::PERF_WEIGHT_COMPRESSION_SAVING << " = 0.5\n";
        os << armnn::EthosNConfig::PERF_ACTIVATION_COMPRESSION_SAVING << " = 0.5\n";
        os << armnn::EthosNConfig::PERF_CURRENT << " = 0\n";
        os << armnn::EthosNConfig::INTERMEDIATE_COMPRESSION << " = 1\n";
        os << armnn::EthosNConfig::INFERENCE_TIMEOUT << " = 25\n";

        os.seekg(0);
        armnn::EthosNConfig config;
        os >> config;
        CHECK(config.m_PerfOnly == true);
        CHECK(config.m_PerfVariant == ethosn::support_library::EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO);
        CHECK(config.m_PerfSramSizeBytesOverride == 12);
        CHECK(config.m_PerfOutDir == "test");
        CHECK(config.m_DumpDebugFiles == ethosn::support_library::CompilationOptions::DebugLevel::High);
        CHECK(config.m_DumpRam == true);
        CHECK(config.m_PerfActivationCompressionSaving == 0.5f);
        CHECK(config.m_PerfWeightCompressionSaving == 0.5f);
        CHECK(config.m_PerfCurrent == false);
        CHECK(config.m_IntermediateCompression == true);
        CHECK(config.m_InferenceTimeout == 25);
    }

    // A test which estimates the performance of a supported (relu) operation
    // and an operation which doesn't exist yet on the Ethos-N (abs).
    // it should return a proper estimate for the relu and all zeroes for the abs.
    TEST_CASE("EstimationOnlyWorkload")
    {
        using namespace testing_utils;

        const TempDir tmpDir;

        armnn::EthosNConfig config{};
        config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
        config.m_PerfOnly    = true;
        config.m_PerfOutDir  = tmpDir.Str();
        config.m_PerfCurrent = true;

        BackendGlobalConfigSetter configSetter(config, config.QueryCapabilities());

        // Register and get allocators to ensure the allocators exist
        EthosNBackendAllocatorService::GetInstance().RegisterAllocator(config, {});

        armnn::EthosNWorkloadFactory factory(config);
        // To create a PreCompiled layer, create a network and Optimize it.
        armnn::INetworkPtr net = armnn::INetwork::Create();

        armnn::IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input layer");
        CHECK(inputLayer);

        ActivationDescriptor reluDesc;
        reluDesc.m_A                              = 100;
        reluDesc.m_B                              = 0;
        reluDesc.m_Function                       = ActivationFunction::BoundedReLu;
        armnn::IConnectableLayer* const reluLayer = net->AddActivationLayer(reluDesc, "relu layer");
        CHECK(reluLayer);

        ElementwiseUnaryDescriptor unaryDesc;
        unaryDesc.m_Operation                    = UnaryOperation::Abs;
        armnn::IConnectableLayer* const absLayer = net->AddElementwiseUnaryLayer(unaryDesc, "abs layer");
        CHECK(absLayer);

        armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output layer");
        CHECK(outputLayer);

        TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
        inputTensorInfo.SetQuantizationOffset(0);
        inputTensorInfo.SetQuantizationScale(0.9f);
        inputTensorInfo.SetConstant();

        TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
        outputTensorInfo.SetQuantizationOffset(0);
        outputTensorInfo.SetQuantizationScale(0.9f);

        inputLayer->GetOutputSlot(0).Connect(reluLayer->GetInputSlot(0));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

        reluLayer->GetOutputSlot(0).Connect(absLayer->GetInputSlot(0));
        reluLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

        absLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
        absLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        std::vector<armnn::BackendId> backends = { factory.GetBackendId() };
        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
        armnn::OptimizerOptionsOpaque optimizerOptions;
        armnn::IOptimizedNetworkPtr optimizedNet =
            armnn::Optimize(*net, backends, runtime->GetDeviceSpec(), optimizerOptions);
        CHECK(optimizedNet != nullptr);

        // Load graph into runtime
        armnn::NetworkId networkIdentifier;
        runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet));

        //Creates structures for inputs and outputs.
        const std::vector<uint8_t> inputData(inputTensorInfo.GetNumElements());
        std::vector<uint8_t> outputData(outputTensorInfo.GetNumElements());

        armnn::InputTensors inputTensors{ { 0, armnn::ConstTensor(inputTensorInfo, inputData.data()) } };
        armnn::OutputTensors outputTensors{ { 0, armnn::Tensor(outputTensorInfo, outputData.data()) } };

        // Execute network
        runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

        const std::string reportFile = config.m_PerfOutDir + "/subgraph_0/report.json";
        const std::string result     = ReadFile(reportFile);

        const std::string golden = R"({
	"Config":
	{
		"Variant": "Ethos-N78_4TOPS_4PLE_RATIO",
		"SramSizeBytesOverride": 0,
		"ActivationCompressionSavings": 0,
		"WeightCompressionSavings": "Not Specified",
		"Current": 1
	},
	"OperationNames":
	{
		"0": "Input from input layer",
		"1": "relu layer",
		"2": "abs layer",
		"3": "Output from abs layer"
	},
	"Results":
	{
		"Stream":
		[
			{
				"OperationIds": [ 0, 1 ],
				"ParentIds": [ [] ],
				"Input":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 4096,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0
				},
				"Output":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 0,
					"SramBytes": 4096,
					"NumCentralStripes": 0,
					"NumBoundaryStripes": 0,
					"NumReloads": 0
				},
				"Weights":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 256,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0,
					"CompressionSavings": 0
				},
				"Mce":
				{
					"Operations": 8192,
					"CycleCount": 32
				},
				"Ple":
				{
					"NumOfPatches": 16,
					"Operation": 10
				}
			}
		],
		"Issues":
		{
			"2": "Could not be estimated and has zero performance impact. Reason: abs layer is not currently supported."
		}
	}
}
)";

        CHECK(result == golden);
    }

    // A test which estimates the performance of a supported (relu) operation
    // and an operation which can only be in the performance estimator (avg pooling stride 1 size 1).
    // it should return a proper estimate for the relu and all zeroes for the pooling.
    TEST_CASE("EstimationOnlyExistingWorkload")
    {
        using namespace testing_utils;

        const TempDir tmpDir;

        armnn::EthosNConfig config{};
        config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
        config.m_PerfOnly    = true;
        config.m_PerfOutDir  = tmpDir.Str();
        config.m_PerfCurrent = true;

        BackendGlobalConfigSetter configSetter(config, config.QueryCapabilities());

        // Register allocators
        EthosNBackendAllocatorService::GetInstance().RegisterAllocator(config, {});

        armnn::EthosNWorkloadFactory factory(config);
        // To create a PreCompiled layer, create a network and Optimize it.
        armnn::INetworkPtr net = armnn::INetwork::Create();

        armnn::IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input layer");
        CHECK(inputLayer);

        ActivationDescriptor reluDesc;
        reluDesc.m_A                              = 100;
        reluDesc.m_B                              = 0;
        reluDesc.m_Function                       = ActivationFunction::BoundedReLu;
        armnn::IConnectableLayer* const reluLayer = net->AddActivationLayer(reluDesc, "relu layer");
        CHECK(reluLayer);

        Pooling2dDescriptor poolDesc;
        poolDesc.m_DataLayout                     = DataLayout::NHWC;
        poolDesc.m_StrideX                        = 1;
        poolDesc.m_StrideY                        = 1;
        poolDesc.m_PadLeft                        = 0;
        poolDesc.m_PadRight                       = 0;
        poolDesc.m_PadBottom                      = 0;
        poolDesc.m_PadTop                         = 0;
        poolDesc.m_PoolWidth                      = 1;
        poolDesc.m_PoolHeight                     = 1;
        poolDesc.m_PoolType                       = PoolingAlgorithm::Average;
        armnn::IConnectableLayer* const poolLayer = net->AddPooling2dLayer(poolDesc, "pool layer");
        CHECK(poolLayer);

        armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output layer");
        CHECK(outputLayer);

        TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
        inputTensorInfo.SetQuantizationOffset(0);
        inputTensorInfo.SetQuantizationScale(0.9f);
        inputTensorInfo.SetConstant();

        TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
        outputTensorInfo.SetQuantizationOffset(0);
        outputTensorInfo.SetQuantizationScale(0.9f);

        inputLayer->GetOutputSlot(0).Connect(reluLayer->GetInputSlot(0));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

        reluLayer->GetOutputSlot(0).Connect(poolLayer->GetInputSlot(0));
        reluLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

        poolLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
        poolLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        std::vector<armnn::BackendId> backends = { EthosNBackend::GetIdStatic() };
        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
        armnn::OptimizerOptionsOpaque optimizerOptions;
        armnn::IOptimizedNetworkPtr optimizedNet =
            armnn::Optimize(*net, backends, runtime->GetDeviceSpec(), optimizerOptions);
        CHECK(optimizedNet != nullptr);

        // Load graph into runtime
        armnn::NetworkId networkIdentifier;
        runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet));

        //Creates structures for inputs and outputs.
        const std::vector<uint8_t> inputData(inputTensorInfo.GetNumElements());
        std::vector<uint8_t> outputData(outputTensorInfo.GetNumElements());

        armnn::InputTensors inputTensors{ { 0, armnn::ConstTensor(inputTensorInfo, inputData.data()) } };
        armnn::OutputTensors outputTensors{ { 0, armnn::Tensor(outputTensorInfo, outputData.data()) } };

        // Execute network
        runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

        const std::string reportFile = config.m_PerfOutDir + "/subgraph_0/report.json";
        const std::string result     = ReadFile(reportFile);

        const std::string golden = R"({
	"Config":
	{
		"Variant": "Ethos-N78_4TOPS_4PLE_RATIO",
		"SramSizeBytesOverride": 0,
		"ActivationCompressionSavings": 0,
		"WeightCompressionSavings": "Not Specified",
		"Current": 1
	},
	"OperationNames":
	{
		"0": "Input from input layer",
		"1": "relu layer",
		"2": "pool layer",
		"3": "Output from pool layer"
	},
	"Results":
	{
		"Stream":
		[
			{
				"OperationIds": [ 0, 1 ],
				"ParentIds": [ [] ],
				"Input":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 4096,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0
				},
				"Output":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 0,
					"SramBytes": 4096,
					"NumCentralStripes": 0,
					"NumBoundaryStripes": 0,
					"NumReloads": 0
				},
				"Weights":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 256,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0,
					"CompressionSavings": 0
				},
				"Mce":
				{
					"Operations": 8192,
					"CycleCount": 32
				},
				"Ple":
				{
					"NumOfPatches": 16,
					"Operation": 10
				}
			}
		],
		"Issues":
		{
			"2": "Could not be estimated and has zero performance impact. Reason: Unsupported configuration in AVG pooling."
		}
	}
}
)";

        CHECK(result == golden);
    }

    TEST_CASE("CreateEstimationWorkload")
    {
        using namespace testing_utils;

        const TempDir tmpDir;

        armnn::EthosNConfig config{};
        config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
        config.m_PerfOnly    = true;
        config.m_PerfOutDir  = tmpDir.Str();
        config.m_PerfCurrent = true;

        BackendGlobalConfigSetter configSetter(config, config.QueryCapabilities());

        CreateEthosNPrecompiledWorkloadTest();

        const std::string reportFile = config.m_PerfOutDir + "/subgraph_0/report.json";
        const std::string result     = ReadFile(reportFile);

        const std::string golden = R"({
	"Config":
	{
		"Variant": "Ethos-N78_4TOPS_4PLE_RATIO",
		"SramSizeBytesOverride": 0,
		"ActivationCompressionSavings": 0,
		"WeightCompressionSavings": "Not Specified",
		"Current": 1
	},
	"OperationNames":
	{
		"0": "Input from input layer",
		"3": "conv layer",
		"4": "Output from conv layer"
	},
	"Results":
	{
		"Stream":
		[
			{
				"OperationIds": [ 0, 1, 2, 3 ],
				"ParentIds": [ [] ],
				"Input":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 4096,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0
				},
				"Output":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 4096,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0
				},
				"Weights":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 768,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0,
					"CompressionSavings": 0
				},
				"Mce":
				{
					"Operations": 131072,
					"CycleCount": 32
				},
				"Ple":
				{
					"NumOfPatches": 16,
					"Operation": 10
				}
			}
		],
		"Issues":
		{
		}
	}
}
)";

        CHECK(result == golden);
    }

    TEST_CASE("EstimationCompressionOverride")
    {
        using namespace testing_utils;

        const TempDir tmpDir;

        armnn::EthosNConfig config{};
        config.m_PerfVariant                      = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
        config.m_PerfOnly                         = true;
        config.m_PerfOutDir                       = tmpDir.Str();
        config.m_PerfActivationCompressionSaving  = 0.6f;
        config.m_PerfUseWeightCompressionOverride = true;
        config.m_PerfWeightCompressionSaving      = 0.8f;
        config.m_PerfCurrent                      = false;

        BackendGlobalConfigSetter configSetter(config, config.QueryCapabilities());

        CreateEthosNPrecompiledWorkloadTest();

        const std::string reportFile = config.m_PerfOutDir + "/subgraph_0/report.json";
        const std::string result     = ReadFile(reportFile);

        const std::string golden = R"({
	"Config":
	{
		"Variant": "Ethos-N78_4TOPS_4PLE_RATIO",
		"SramSizeBytesOverride": 0,
		"ActivationCompressionSavings": 0.6,
		"WeightCompressionSavings": 0.8,
		"Current": 0
	},
)";
        CHECK(result.find(golden) != std::string::npos);
    }

    namespace
    {
    void ExecuteEstimationNetworkSplit()
    {
        // Construct Arm NN network
        const INetworkPtr myNetwork = INetwork::Create();

        // Arm NN weights tensor shape is OHWI (out channels, height, width, in channels) for NHWC
        const TensorInfo supportedWeightsInfo(TensorShape({ 16, 1, 1, 16 }), armnn::DataType::QAsymmU8, 0.9f, 0, true);
        // Arm NN weights tensor shape is OIHW (out channels, in channels, height, width) for NCHW
        const TensorInfo unsupportedWeightsInfo(TensorShape({ 16, 16, 1, 1 }), armnn::DataType::QAsymmU8, 0.9f, 0,
                                                true);

        CHECK(supportedWeightsInfo.GetNumElements() == unsupportedWeightsInfo.GetNumElements());

        const std::vector<uint8_t> weightsData(supportedWeightsInfo.GetNumElements());

        const armnn::ConstTensor supportedWeights(supportedWeightsInfo, weightsData);
        const armnn::ConstTensor unsupportedWeights(unsupportedWeightsInfo, weightsData);

        armnn::Convolution2dDescriptor supportedConvDesc{};
        supportedConvDesc.m_StrideX    = 1;
        supportedConvDesc.m_StrideY    = 1;
        supportedConvDesc.m_DataLayout = armnn::DataLayout::NHWC;

        armnn::Convolution2dDescriptor unsupportedConvDesc = supportedConvDesc;
        unsupportedConvDesc.m_DataLayout                   = armnn::DataLayout::NCHW;

        const TensorInfo tensorInfo(TensorShape({ 1, 16, 16, 16 }), DataType::QAsymmU8, 0.9f, 0);

        armnn::IConnectableLayer& inputLayer = *myNetwork->AddInputLayer(0, "input layer");

        inputLayer.GetOutputSlot(0).SetTensorInfo(tensorInfo);

        IConnectableLayer& supportedLayer1 = *myNetwork->AddConvolution2dLayer(supportedConvDesc, "supported layer 1");
        armnn::IConnectableLayer* supportedWeightsLayer =
            myNetwork->AddConstantLayer(supportedWeights, "Conv2dWeights");
        supportedWeightsLayer->GetOutputSlot(0).SetTensorInfo(supportedWeightsInfo);
        supportedWeightsLayer->GetOutputSlot(0).Connect(supportedLayer1.GetInputSlot(1));

        supportedLayer1.GetOutputSlot(0).SetTensorInfo(tensorInfo);
        inputLayer.GetOutputSlot(0).Connect(supportedLayer1.GetInputSlot(0));

        IConnectableLayer& unsupportedLayer =
            *myNetwork->AddConvolution2dLayer(unsupportedConvDesc, "unsupported layer");
        armnn::IConnectableLayer* unsupportedWeightsLayer =
            myNetwork->AddConstantLayer(unsupportedWeights, "Conv2dWeights");
        unsupportedWeightsLayer->GetOutputSlot(0).SetTensorInfo(unsupportedWeightsInfo);
        unsupportedWeightsLayer->GetOutputSlot(0).Connect(unsupportedLayer.GetInputSlot(1));

        unsupportedLayer.GetOutputSlot(0).SetTensorInfo(tensorInfo);
        supportedLayer1.GetOutputSlot(0).Connect(unsupportedLayer.GetInputSlot(0));

        IConnectableLayer& supportedLayer2 = *myNetwork->AddConvolution2dLayer(supportedConvDesc, "supported layer 2");
        armnn::IConnectableLayer* supportedWeightsLayer2 =
            myNetwork->AddConstantLayer(supportedWeights, "Conv2dWeights");
        supportedWeightsLayer2->GetOutputSlot(0).SetTensorInfo(supportedWeightsInfo);
        supportedWeightsLayer2->GetOutputSlot(0).Connect(supportedLayer2.GetInputSlot(1));

        supportedLayer2.GetOutputSlot(0).SetTensorInfo(tensorInfo);
        unsupportedLayer.GetOutputSlot(0).Connect(supportedLayer2.GetInputSlot(0));

        armnn::IConnectableLayer& outputLayer = *myNetwork->AddOutputLayer(0, "output layer");

        supportedLayer2.GetOutputSlot(0).Connect(outputLayer.GetInputSlot(0));

        // Create Arm NN runtime
        IRuntime::CreationOptions options;    // default options
        IRuntimePtr run = IRuntime::Create(options);

        // Optimise Arm NN network
        armnn::IOptimizedNetworkPtr optNet = Optimize(
            *myNetwork, std::vector<BackendId>{ armnn::EthosNBackendId(), Compute::CpuRef }, run->GetDeviceSpec());

        // Load graph into runtime
        armnn::NetworkId networkIdentifier;
        run->LoadNetwork(networkIdentifier, std::move(optNet));

        //Creates structures for inputs and outputs.
        const std::vector<uint8_t> inputData(tensorInfo.GetNumElements());
        std::vector<uint8_t> outputData(tensorInfo.GetNumElements());

        TensorInfo inputTensorInfo = tensorInfo;
        inputTensorInfo.SetConstant();
        armnn::InputTensors inputTensors{ { 0, armnn::ConstTensor(inputTensorInfo, inputData.data()) } };
        armnn::OutputTensors outputTensors{ { 0, armnn::Tensor(tensorInfo, outputData.data()) } };

        // Execute network
        run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
    }
    }    // namespace

    TEST_CASE("CreateEstimationWorkloadSplit")
    {
        using namespace testing_utils;

        const TempDir tmpDir;

        armnn::EthosNConfig config{};
        config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
        config.m_PerfOnly    = true;
        config.m_PerfOutDir  = tmpDir.Str();
        config.m_PerfCurrent = true;

        BackendGlobalConfigSetter configSetter(config, config.QueryCapabilities());

        ExecuteEstimationNetworkSplit();

        const std::string reportFile1 = config.m_PerfOutDir + "/subgraph_0/report.json";
        const std::string reportFile2 = config.m_PerfOutDir + "/subgraph_1/report.json";
        const std::string reportFile3 = config.m_PerfOutDir + "/subgraph_2/report.json";

        const std::string result1 = ReadFile(reportFile1);
        const std::string result2 = ReadFile(reportFile2);
        const std::string result3 = ReadFile(reportFile3);

        const std::string golden1 = R"({
	"Config":
	{
		"Variant": "Ethos-N78_4TOPS_4PLE_RATIO",
		"SramSizeBytesOverride": 0,
		"ActivationCompressionSavings": 0,
		"WeightCompressionSavings": "Not Specified",
		"Current": 1
	},
	"OperationNames":
	{
		"0": "Input from input layer",
		"3": "supported layer 1",
		"4": "Output from supported layer 1"
	},
	"Results":
	{
		"Stream":
		[
			{
				"OperationIds": [ 0, 1, 2, 3 ],
				"ParentIds": [ [] ],
				"Input":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 4096,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0
				},
				"Output":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 4096,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0
				},
				"Weights":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 512,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0,
					"CompressionSavings": 0
				},
				"Mce":
				{
					"Operations": 131072,
					"CycleCount": 32
				},
				"Ple":
				{
					"NumOfPatches": 16,
					"Operation": 10
				}
			}
		],
		"Issues":
		{
		}
	}
}
)";

        const std::string golden2 = R"({
	"Config":
	{
		"Variant": "Ethos-N78_4TOPS_4PLE_RATIO",
		"SramSizeBytesOverride": 0,
		"ActivationCompressionSavings": 0,
		"WeightCompressionSavings": "Not Specified",
		"Current": 1
	},
	"OperationNames":
	{
		"0": "Input from unsupported layer",
		"3": "supported layer 2",
		"4": "Output from supported layer 2"
	},
	"Results":
	{
		"Stream":
		[
			{
				"OperationIds": [ 0, 1, 2, 3 ],
				"ParentIds": [ [] ],
				"Input":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 4096,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0
				},
				"Output":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 4096,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0
				},
				"Weights":
				{
					"DramParallelBytes": 0,
					"DramNonParallelBytes": 512,
					"SramBytes": 0,
					"NumCentralStripes": 1,
					"NumBoundaryStripes": 0,
					"NumReloads": 0,
					"CompressionSavings": 0
				},
				"Mce":
				{
					"Operations": 131072,
					"CycleCount": 32
				},
				"Ple":
				{
					"NumOfPatches": 16,
					"Operation": 10
				}
			}
		],
		"Issues":
		{
		}
	}
}
)";

        // The order of the subgraphs is not deterministic due to the way Arm NN constructs them
        // map the subgraphs to the results
        std::string subgraph1;
        if (result1.find("Input from input layer") != std::string::npos)
        {
            subgraph1 = result1;
        }
        else if (result2.find("Input from input layer") != std::string::npos)
        {
            subgraph1 = result2;
        }
        else if (result3.find("Input from input layer") != std::string::npos)
        {
            subgraph1 = result3;
        }
        else
        {
            CHECK(false);
        }

        std::string subgraph2;
        if (result1.find("Input from unsupported layer") != std::string::npos)
        {
            subgraph2 = result1;
        }
        else if (result2.find("Input from unsupported layer") != std::string::npos)
        {
            subgraph2 = result2;
        }
        else if (result3.find("Input from unsupported layer") != std::string::npos)
        {
            subgraph2 = result3;
        }
        else
        {
            CHECK(false);
        }

        CHECK(subgraph1 == golden1);
        CHECK(subgraph2 == golden2);
    }
}
