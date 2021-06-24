//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNBackendId.hpp"
#include "EthosNConfig.hpp"
#include "EthosNSubgraphViewConverter.hpp"
#include "EthosNTensorHandle.hpp"
#include "EthosNTestUtils.hpp"
#include "EthosNWorkloadFactory.hpp"
#include "EthosNWorkloads.hpp"

#include <Filesystem.hpp>
#include <test/CreateWorkload.hpp>

BOOST_AUTO_TEST_SUITE(CreateEstimationWorkloadEthosN)

// Tests that the NPU config file is parsed correctly
BOOST_AUTO_TEST_CASE(ParseEthosNConfig)
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
    os << armnn::EthosNConfig::COMPILER_ALGORITHM << " = Auto\n";
    os << armnn::EthosNConfig::INTERMEDIATE_COMPRESSION << " = 1\n";

    os.seekg(0);
    armnn::EthosNConfig config;
    os >> config;
    BOOST_CHECK(config.m_PerfOnly == true);
    BOOST_CHECK(config.m_PerfVariant == ethosn::support_library::EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO);
    BOOST_CHECK(config.m_PerfSramSizeBytesOverride == 12);
    BOOST_CHECK(config.m_PerfOutDir == "test");
    BOOST_CHECK(config.m_DumpDebugFiles == ethosn::support_library::CompilationOptions::DebugLevel::High);
    BOOST_CHECK(config.m_DumpRam == true);
    BOOST_CHECK(config.m_PerfActivationCompressionSaving == 0.5f);
    BOOST_CHECK(config.m_PerfWeightCompressionSaving == 0.5f);
    BOOST_CHECK(config.m_PerfCurrent == false);
    BOOST_CHECK(config.m_CompilerAlgorithm == ethosn::support_library::CompilerAlgorithm::Auto);
    BOOST_CHECK(config.m_IntermediateCompression == true);
}

BOOST_AUTO_TEST_CASE(ParseEthosNConfigCascadingOk)
{
    std::stringstream os;
    os << armnn::EthosNConfig::COMPILER_ALGORITHM << " = CascadingOnly\n";

    os.seekg(0);
    armnn::EthosNConfig config;
    os >> config;

    BOOST_CHECK(config.m_CompilerAlgorithm == ethosn::support_library::CompilerAlgorithm::CascadingOnly);
}

BOOST_AUTO_TEST_CASE(ParseEthosNConfigCascadingNOk)
{
    std::stringstream os;
    os << armnn::EthosNConfig::COMPILER_ALGORITHM << " = foo\n";

    os.seekg(0);
    armnn::EthosNConfig config;

    bool exceptionCaught = false;
    try
    {
        os >> config;
    }
    catch (...)
    {
        exceptionCaught = true;
    }
    BOOST_CHECK(exceptionCaught == true);
}

// A test which estimates the performance of a supported (relu) operation
// and an operation which doesn't exist yet on the Ethos-N (abs).
// it should return a proper estimate for the relu and all zeroes for the abs.
BOOST_AUTO_TEST_CASE(EstimationOnlyWorkload)
{
    // Reset backend-internal subgraph converter instance id
    armnn::EthosNSubgraphViewConverter::ResetNextInstanceId();

    using namespace testing_utils;

    const TempDir tmpDir;

    armnn::EthosNConfig config{};
    config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
    config.m_PerfOnly    = true;
    config.m_PerfOutDir  = tmpDir.Str();
    config.m_PerfCurrent = true;

    BackendGlobalConfigSetter configSetter(config, EthosNMappings(), config.QueryCapabilities());

    armnn::EthosNWorkloadFactory factory(config);
    // To create a PreCompiled layer, create a network and Optimize it.
    armnn::INetworkPtr net = armnn::INetwork::Create();

    armnn::IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input layer");
    BOOST_TEST(inputLayer);

    ActivationDescriptor reluDesc;
    reluDesc.m_A                              = 100;
    reluDesc.m_B                              = 0;
    reluDesc.m_Function                       = ActivationFunction::BoundedReLu;
    armnn::IConnectableLayer* const reluLayer = net->AddActivationLayer(reluDesc, "relu layer");
    BOOST_TEST(reluLayer);

    ElementwiseUnaryDescriptor unaryDesc;
    unaryDesc.m_Operation                    = UnaryOperation::Abs;
    armnn::IConnectableLayer* const absLayer = net->AddElementwiseUnaryLayer(unaryDesc, "abs layer");
    BOOST_TEST(absLayer);

    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output layer");
    BOOST_TEST(outputLayer);

    TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    inputTensorInfo.SetQuantizationOffset(0);
    inputTensorInfo.SetQuantizationScale(0.9f);

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
    armnn::OptimizerOptions optimizerOptions;
    armnn::IOptimizedNetworkPtr optimizedNet =
        armnn::Optimize(*net, backends, runtime->GetDeviceSpec(), optimizerOptions);
    BOOST_CHECK(optimizedNet != nullptr);

    armnn::Graph& optimisedGraph = GetGraphForTesting(optimizedNet.get());
    Layer* preCompiledLayer      = nullptr;
    for (auto& layer : optimisedGraph)
    {
        if (layer->GetType() == LayerType::PreCompiled)
        {
            preCompiledLayer = layer;
        }
    }
    BOOST_CHECK(preCompiledLayer != nullptr);

    CreateTensorHandles(optimisedGraph, factory);

    auto workload = MakeAndCheckWorkload<armnn::EthosNPreCompiledWorkload>(*preCompiledLayer, factory);

    PreCompiledQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    // Execute outputs the performance estimation to a file.
    // read it back so it can be compared.
    workload->Execute();

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
			"2": "Could not be estimated: Please provide a mapping file entry for this operation"
		}
	}
}
)";

    BOOST_TEST(result == golden);
}

// A test which estimates the performance of a supported (relu) operation
// and an operation which can only be in the performance estimator (avg pooling stride 1 size 1).
// it should return a proper estimate for the relu and all zeroes for the pooling.
BOOST_AUTO_TEST_CASE(EstimationOnlyExistingWorkload)
{
    // Reset backend-internal subgraph converter instance id
    armnn::EthosNSubgraphViewConverter::ResetNextInstanceId();

    using namespace testing_utils;

    const TempDir tmpDir;

    armnn::EthosNConfig config{};
    config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
    config.m_PerfOnly    = true;
    config.m_PerfOutDir  = tmpDir.Str();
    config.m_PerfCurrent = true;

    BackendGlobalConfigSetter configSetter(config, EthosNMappings(), config.QueryCapabilities());

    armnn::EthosNWorkloadFactory factory(config);
    // To create a PreCompiled layer, create a network and Optimize it.
    armnn::INetworkPtr net = armnn::INetwork::Create();

    armnn::IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input layer");
    BOOST_TEST(inputLayer);

    ActivationDescriptor reluDesc;
    reluDesc.m_A                              = 100;
    reluDesc.m_B                              = 0;
    reluDesc.m_Function                       = ActivationFunction::BoundedReLu;
    armnn::IConnectableLayer* const reluLayer = net->AddActivationLayer(reluDesc, "relu layer");
    BOOST_TEST(reluLayer);

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
    BOOST_TEST(poolLayer);

    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output layer");
    BOOST_TEST(outputLayer);

    TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    inputTensorInfo.SetQuantizationOffset(0);
    inputTensorInfo.SetQuantizationScale(0.9f);

    TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    outputTensorInfo.SetQuantizationOffset(0);
    outputTensorInfo.SetQuantizationScale(0.9f);

    inputLayer->GetOutputSlot(0).Connect(reluLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    reluLayer->GetOutputSlot(0).Connect(poolLayer->GetInputSlot(0));
    reluLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    poolLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    poolLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    std::vector<armnn::BackendId> backends = { factory.GetBackendId() };
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    armnn::OptimizerOptions optimizerOptions;
    armnn::IOptimizedNetworkPtr optimizedNet =
        armnn::Optimize(*net, backends, runtime->GetDeviceSpec(), optimizerOptions);
    BOOST_CHECK(optimizedNet != nullptr);

    armnn::Graph& optimisedGraph = GetGraphForTesting(optimizedNet.get());
    Layer* preCompiledLayer      = nullptr;
    for (auto& layer : optimisedGraph)
    {
        if (layer->GetType() == LayerType::PreCompiled)
        {
            preCompiledLayer = layer;
        }
    }
    BOOST_CHECK(preCompiledLayer != nullptr);

    CreateTensorHandles(optimisedGraph, factory);

    auto workload = MakeAndCheckWorkload<armnn::EthosNPreCompiledWorkload>(*preCompiledLayer, factory);

    PreCompiledQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    // Execute outputs the performance estimation to a file.
    // read it back so it can be compared.
    workload->Execute();

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
			"2": "Could not be estimated: Please provide a mapping file entry for this operation"
		}
	}
}
)";

    BOOST_TEST(result == golden);
}

// A test which estimates the performance of an unsupported (sqrt) operation
// it should return a proper estimate for the sqrt using the mapping
BOOST_AUTO_TEST_CASE(EstimationOnlyUnsupportedWithMapping)
{
    // Reset backend-internal subgraph converter instance id
    armnn::EthosNSubgraphViewConverter::ResetNextInstanceId();

    using namespace testing_utils;

    const TempDir tmpDir;

    armnn::EthosNConfig config{};
    config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
    config.m_PerfOnly    = true;
    config.m_PerfOutDir  = tmpDir.Str();
    config.m_PerfCurrent = true;

    std::stringstream os;
    os << "pattern:\n";
    os << "input firstInput, 1x_x_x_\n";
    os << "output firstOutput, 1x_x_x_\n";
    os << "Activation, (firstInput), (firstOutput), ((function=Sqrt))\n";
    os << "graph-replacement:\n";
    os << "Activation, (firstInput), (firstOutput), ((function=Sigmoid), (name=SigmoidFunc))";
    os.seekg(0);
    EthosNMappings mappings = ParseMappings(os);

    BackendGlobalConfigSetter configSetter(config, mappings, config.QueryCapabilities());

    armnn::EthosNWorkloadFactory factory(config);
    // To create a PreCompiled layer, create a network and Optimize it.
    armnn::INetworkPtr net = armnn::INetwork::Create();

    armnn::IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input layer");
    BOOST_TEST(inputLayer);

    ActivationDescriptor tanDesc;
    tanDesc.m_A                               = 1;
    tanDesc.m_B                               = 1;
    tanDesc.m_Function                        = ActivationFunction::Sqrt;
    armnn::IConnectableLayer* const tanhLayer = net->AddActivationLayer(tanDesc, "Sqrt layer");
    BOOST_TEST(tanhLayer);

    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output layer");
    BOOST_TEST(outputLayer);

    TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    inputTensorInfo.SetQuantizationOffset(0);
    inputTensorInfo.SetQuantizationScale(0.9f);

    TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    outputTensorInfo.SetQuantizationOffset(0);
    outputTensorInfo.SetQuantizationScale(0.9f);

    inputLayer->GetOutputSlot(0).Connect(tanhLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    tanhLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    tanhLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    std::vector<armnn::BackendId> backends = { factory.GetBackendId() };
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    armnn::OptimizerOptions optimizerOptions;
    armnn::IOptimizedNetworkPtr optimizedNet =
        armnn::Optimize(*net, backends, runtime->GetDeviceSpec(), optimizerOptions);
    BOOST_CHECK(optimizedNet != nullptr);

    armnn::Graph& optimisedGraph = GetGraphForTesting(optimizedNet.get());
    Layer* preCompiledLayer      = nullptr;
    for (auto& layer : optimisedGraph)
    {
        if (layer->GetType() == LayerType::PreCompiled)
        {
            preCompiledLayer = layer;
        }
    }
    BOOST_CHECK(preCompiledLayer != nullptr);

    CreateTensorHandles(optimisedGraph, factory);

    auto workload = MakeAndCheckWorkload<armnn::EthosNPreCompiledWorkload>(*preCompiledLayer, factory);

    PreCompiledQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    // Execute outputs the performance estimation to a file.
    // read it back so it can be compared.
    workload->Execute();

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
		"1": "SigmoidFunc",
		"2": "Output from SigmoidFunc"
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
					"DramNonParallelBytes": 4096,
					"SramBytes": 0,
					"NumCentralStripes": 1,
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
					"Operation": 11
				}
			}
		],
		"Issues":
		{
		}
	}
}
)";

    BOOST_TEST(result == golden);
}

// A test which estimates the performance of a standin layer
// which has been replaced with sigmoid via the mapping file
BOOST_AUTO_TEST_CASE(EstimationOnlyStandInMapping)
{
    // Reset backend-internal subgraph converter instance id
    armnn::EthosNSubgraphViewConverter::ResetNextInstanceId();

    using namespace testing_utils;

    const TempDir tmpDir;

    armnn::EthosNConfig config{};
    config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
    config.m_PerfOnly    = true;
    config.m_PerfOutDir  = tmpDir.Str();
    config.m_PerfCurrent = true;

    std::stringstream os;
    os << "pattern:\n";
    os << "input firstInput, 1x_x_x_\n";
    os << "output firstOutput, 1x_x_x_\n";
    os << "StandIn, (firstInput), (firstOutput), ((name=StandInTest))\n";
    os << "graph-replacement:\n";
    os << "Activation, (firstInput), (firstOutput), ((function=Sigmoid), (name=SigmoidFunc))";
    os.seekg(0);
    EthosNMappings mappings = ParseMappings(os);

    BackendGlobalConfigSetter configSetter(config, mappings, config.QueryCapabilities());

    armnn::EthosNWorkloadFactory factory(config);
    // To create a PreCompiled layer, create a network and Optimize it.
    armnn::INetworkPtr net = armnn::INetwork::Create();

    armnn::IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input layer");
    BOOST_TEST(inputLayer);

    StandInDescriptor standInDesc;
    standInDesc.m_NumInputs                      = 1;
    standInDesc.m_NumOutputs                     = 1;
    armnn::IConnectableLayer* const standInLayer = net->AddStandInLayer(standInDesc, "StandInTest");
    BOOST_TEST(standInLayer);

    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output layer");
    BOOST_TEST(standInLayer);

    TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    inputTensorInfo.SetQuantizationOffset(0);
    inputTensorInfo.SetQuantizationScale(0.9f);

    TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    outputTensorInfo.SetQuantizationOffset(0);
    outputTensorInfo.SetQuantizationScale(0.9f);

    inputLayer->GetOutputSlot(0).Connect(standInLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    standInLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    standInLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    std::vector<armnn::BackendId> backends = { factory.GetBackendId() };
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    armnn::OptimizerOptions optimizerOptions;
    armnn::IOptimizedNetworkPtr optimizedNet =
        armnn::Optimize(*net, backends, runtime->GetDeviceSpec(), optimizerOptions);
    BOOST_CHECK(optimizedNet != nullptr);

    armnn::Graph& optimisedGraph = GetGraphForTesting(optimizedNet.get());
    Layer* preCompiledLayer      = nullptr;
    for (auto& layer : optimisedGraph)
    {
        if (layer->GetType() == LayerType::PreCompiled)
        {
            preCompiledLayer = layer;
        }
    }
    BOOST_CHECK(preCompiledLayer != nullptr);

    CreateTensorHandles(optimisedGraph, factory);

    auto workload = MakeAndCheckWorkload<armnn::EthosNPreCompiledWorkload>(*preCompiledLayer, factory);

    PreCompiledQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 1);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    // Execute outputs the performance estimation to a file.
    // read it back so it can be compared.
    workload->Execute();

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
		"1": "SigmoidFunc",
		"2": "Output from SigmoidFunc"
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
					"DramNonParallelBytes": 4096,
					"SramBytes": 0,
					"NumCentralStripes": 1,
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
					"Operation": 11
				}
			}
		],
		"Issues":
		{
		}
	}
}
)";

    BOOST_TEST(result == golden);
}

BOOST_AUTO_TEST_CASE(CreateEstimationWorkload)
{
    // Reset backend-internal subgraph converter instance id
    armnn::EthosNSubgraphViewConverter::ResetNextInstanceId();

    using namespace testing_utils;

    const TempDir tmpDir;

    armnn::EthosNConfig config{};
    config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
    config.m_PerfOnly    = true;
    config.m_PerfOutDir  = tmpDir.Str();
    config.m_PerfCurrent = true;

    BackendGlobalConfigSetter configSetter(config, EthosNMappings(), config.QueryCapabilities());

    armnn::Graph graph;
    armnn::EthosNWorkloadFactory factory(config);
    auto workload =
        CreatePreCompiledWorkloadTest<armnn::EthosNPreCompiledWorkload, armnn::DataType::QAsymmU8>(factory, graph);

    // Checks that inputs/outputs are as we expect them (see definition of CreatePreCompiledWorkloadTest).
    workload.second->Execute();

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

    BOOST_TEST(result == golden);
}

BOOST_AUTO_TEST_CASE(EstimationCompressionOverride)
{
    armnn::EthosNSubgraphViewConverter::ResetNextInstanceId();

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

    BackendGlobalConfigSetter configSetter(config, EthosNMappings(), config.QueryCapabilities());

    armnn::Graph graph;
    armnn::EthosNWorkloadFactory factory(config);
    auto workload =
        CreatePreCompiledWorkloadTest<armnn::EthosNPreCompiledWorkload, armnn::DataType::QAsymmU8>(factory, graph);

    workload.second->Execute();

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
    BOOST_TEST(result.find(golden) != std::string::npos);
}

namespace
{
void ExecuteEstimationNetworkSplit()
{
    // Construct Arm NN network
    const INetworkPtr myNetwork = INetwork::Create();

    // Arm NN weights tensor shape is OHWI (out channels, height, width, in channels) for NHWC
    const TensorInfo supportedWeightsInfo(TensorShape({ 16, 1, 1, 16 }), armnn::DataType::QAsymmU8, 0.9f, 0);
    // Arm NN weights tensor shape is OIHW (out channels, in channels, height, width) for NCHW
    const TensorInfo unsupportedWeightsInfo(TensorShape({ 16, 16, 1, 1 }), armnn::DataType::QAsymmU8, 0.9f, 0);

    BOOST_TEST(supportedWeightsInfo.GetNumElements() == unsupportedWeightsInfo.GetNumElements());

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

    armnn::IConnectableLayer& supportedLayer1 =
        *myNetwork->AddConvolution2dLayer(supportedConvDesc, supportedWeights, EmptyOptional(), "supported layer 1");

    supportedLayer1.GetOutputSlot(0).SetTensorInfo(tensorInfo);
    inputLayer.GetOutputSlot(0).Connect(supportedLayer1.GetInputSlot(0));

    armnn::IConnectableLayer& unsupportedLayer = *myNetwork->AddConvolution2dLayer(
        unsupportedConvDesc, unsupportedWeights, EmptyOptional(), "unsupported layer");

    unsupportedLayer.GetOutputSlot(0).SetTensorInfo(tensorInfo);
    supportedLayer1.GetOutputSlot(0).Connect(unsupportedLayer.GetInputSlot(0));

    armnn::IConnectableLayer& supportedLayer2 =
        *myNetwork->AddConvolution2dLayer(supportedConvDesc, supportedWeights, EmptyOptional(), "supported layer 2");

    supportedLayer2.GetOutputSlot(0).SetTensorInfo(tensorInfo);
    unsupportedLayer.GetOutputSlot(0).Connect(supportedLayer2.GetInputSlot(0));

    armnn::IConnectableLayer& outputLayer = *myNetwork->AddOutputLayer(0, "output layer");

    supportedLayer2.GetOutputSlot(0).Connect(outputLayer.GetInputSlot(0));

    // Create Arm NN runtime
    IRuntime::CreationOptions options;    // default options
    IRuntimePtr run = IRuntime::Create(options);

    // Optimise Arm NN network
    armnn::IOptimizedNetworkPtr optNet =
        Optimize(*myNetwork, std::vector<BackendId>{ armnn::EthosNBackendId(), Compute::CpuRef }, run->GetDeviceSpec());

    // Load graph into runtime
    armnn::NetworkId networkIdentifier;
    run->LoadNetwork(networkIdentifier, std::move(optNet));

    //Creates structures for inputs and outputs.
    const std::vector<uint8_t> inputData(tensorInfo.GetNumElements());
    std::vector<uint8_t> outputData(tensorInfo.GetNumElements());

    armnn::InputTensors inputTensors{ { 0, armnn::ConstTensor(tensorInfo, inputData.data()) } };
    armnn::OutputTensors outputTensors{ { 0, armnn::Tensor(tensorInfo, outputData.data()) } };

    // Execute network
    run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
}
}    // namespace

BOOST_AUTO_TEST_CASE(CreateEstimationWorkloadSplit)
{
    // Reset backend-internal subgraph converter instance id
    armnn::EthosNSubgraphViewConverter::ResetNextInstanceId();

    using namespace testing_utils;

    const TempDir tmpDir;

    armnn::EthosNConfig config{};
    config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
    config.m_PerfOnly    = true;
    config.m_PerfOutDir  = tmpDir.Str();
    config.m_PerfCurrent = true;

    BackendGlobalConfigSetter configSetter(config, EthosNMappings(), config.QueryCapabilities());

    ExecuteEstimationNetworkSplit();

    const std::string reportFile1 = config.m_PerfOutDir + "/subgraph_0/report.json";
    const std::string reportFile2 = config.m_PerfOutDir + "/subgraph_1/report.json";

    const std::string result1 = ReadFile(reportFile1);
    const std::string result2 = ReadFile(reportFile2);

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
    BOOST_TEST((((result1 == golden1) && (result2 == golden2)) || ((result1 == golden2) && (result2 == golden1))));
}

BOOST_AUTO_TEST_SUITE_END()
