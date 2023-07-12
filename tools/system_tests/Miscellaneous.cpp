//
// Copyright © 2020-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "GgfRunner.hpp"

#include "../../../driver/support_library/src/Capabilities.hpp"

#include <ethosn_driver_library/Device.hpp>
#include <ethosn_driver_library/Network.hpp>
#include <ethosn_driver_library/ProcMemAllocator.hpp>
#include <ethosn_driver_library/Profiling.hpp>
#include <ethosn_support_library/Support.hpp>

#include <ethosn_utils/VectorStream.hpp>

#include <armnn/INetwork.hpp>

#include <catch.hpp>

#include <functional>
#include <sstream>
#include <thread>

namespace ethosn
{
namespace system_tests
{
namespace
{
using ethosn::support_library::FirmwareAndHardwareCapabilities;

template <typename T>
std::string ToString(const T& x)
{
    return std::to_string(x);
}

std::string ToString(const ethosn::support_library::TensorShape& shape)
{
    std::string s = "{ ";
    s += ToString(shape[0]) + ", ";
    s += ToString(shape[1]) + ", ";
    s += ToString(shape[2]) + ", ";
    s += ToString(shape[3]);
    s += " }";
    return s;
}

template <typename T>
std::string ComparisionString(const std::string& prefixIfNotEqual, const T& a, const T& b)
{
    if (a != b)
    {
        return prefixIfNotEqual + ToString(a) + " != " + ToString(b) + "\n";
    }
    return "";
}

std::string ComparisionString(const FirmwareAndHardwareCapabilities& caps, const FirmwareAndHardwareCapabilities& caps2)
{
    std::stringstream ss;

    ss << ComparisionString("  Version = ", caps.m_Header.m_Version, caps2.m_Header.m_Version);
    ss << ComparisionString("  Size = ", caps.m_Header.m_Size, caps2.m_Header.m_Size);

    // Command stream version range
    ss << ComparisionString("  CommandStreamBeginRangeMajor: ", caps.m_CommandStreamBeginRangeMajor,
                            caps2.m_CommandStreamBeginRangeMajor);
    ss << ComparisionString("  CommandStreamBeginRangeMinor: ", caps.m_CommandStreamBeginRangeMinor,
                            caps2.m_CommandStreamBeginRangeMinor);
    ss << ComparisionString("  CommandStreamEndRangeMajor: ", caps.m_CommandStreamEndRangeMajor,
                            caps2.m_CommandStreamEndRangeMajor);
    ss << ComparisionString("  CommandStreamEndRangeMinor: ", caps.m_CommandStreamEndRangeMinor,
                            caps2.m_CommandStreamEndRangeMinor);

    // Hardware capabilities
    ss << ComparisionString("  TotalSramSize: ", caps.m_TotalSramSize, caps2.m_TotalSramSize);
    ss << ComparisionString("  NumberOfEngines: ", caps.m_NumberOfEngines, caps2.m_NumberOfEngines);
    ss << ComparisionString("  OgsPerEngine: ", caps.m_OgsPerEngine, caps2.m_OgsPerEngine);
    ss << ComparisionString("  IgsPerEngine: ", caps.m_IgsPerEngine, caps2.m_IgsPerEngine);
    ss << ComparisionString("  EmcPerEngine: ", caps.m_EmcPerEngine, caps2.m_EmcPerEngine);
    ss << ComparisionString("  MaxPleSize: ", caps.m_MaxPleSize, caps2.m_MaxPleSize);
    ss << ComparisionString("  BoundaryStripeHeight: ", caps.m_BoundaryStripeHeight, caps2.m_BoundaryStripeHeight);
    ss << ComparisionString("  NumBoundarySlots: ", caps.m_NumBoundarySlots, caps2.m_NumBoundarySlots);
    ss << ComparisionString("  NumCentralSlots: ", caps.m_NumCentralSlots, caps2.m_NumCentralSlots);
    ss << ComparisionString("  BrickGroupShape: ", caps.m_BrickGroupShape, caps2.m_BrickGroupShape);
    ss << ComparisionString("  PatchShape: ", caps.m_PatchShape, caps2.m_PatchShape);
    ss << ComparisionString("  MacUnitsPerOg: ", caps.m_MacUnitsPerOg, caps2.m_MacUnitsPerOg);
    ss << ComparisionString("  AccumulatorsPerMacUnit: ", caps.m_AccumulatorsPerMacUnit,
                            caps2.m_AccumulatorsPerMacUnit);
    ss << ComparisionString("  TotalAccumulatorsPerOg: ", caps.m_TotalAccumulatorsPerOg,
                            caps2.m_TotalAccumulatorsPerOg);
    ss << ComparisionString("  NumPleLanes: ", caps.m_NumPleLanes, caps2.m_NumPleLanes);
    ss << ComparisionString("  WeightCompressionVersion: ", caps.m_WeightCompressionVersion,
                            caps2.m_WeightCompressionVersion);
    ss << ComparisionString("  ActivationCompressionVersion: ", caps.m_ActivationCompressionVersion,
                            caps2.m_ActivationCompressionVersion);
    ss << ComparisionString("  IsNchwSupported: ", caps.m_IsNchwSupported, caps2.m_IsNchwSupported);

    return ss.str();
}

void MatchCapabilities(ethosn::support_library::EthosNVariant variant, uint32_t sramSizeKb)
{
    if (!ethosn::driver_library::VerifyKernel())
    {
        throw std::runtime_error("Kernel version is not supported");
    }

    std::vector<char> capabilities  = ethosn::driver_library::GetFirmwareAndHardwareCapabilities();
    std::vector<char> capabilities2 = ethosn::support_library::GetFwAndHwCapabilities(variant, (sramSizeKb * 1024));

    const auto caps  = reinterpret_cast<const FirmwareAndHardwareCapabilities*>(capabilities.data());
    const auto caps2 = reinterpret_cast<const FirmwareAndHardwareCapabilities*>(capabilities2.data());

    INFO(ComparisionString(*caps, *caps2));

    REQUIRE(std::equal(capabilities.begin(), capabilities.end(), capabilities2.begin(), capabilities2.end()));
}

uint64_t WaitForDeviceSupended(uint64_t counter, const std::string device)
{
    std::chrono::milliseconds delay(10);

    uint64_t result = ethosn::driver_library::profiling::GetCounterValue(
        ethosn::driver_library::profiling::PollCounterName::KernelDriverNumRuntimePowerSuspend, device);

    // Wait that the device is suspended
    while (result <= counter)
    {
        result = ethosn::driver_library::profiling::GetCounterValue(
            ethosn::driver_library::profiling::PollCounterName::KernelDriverNumRuntimePowerSuspend, device);

        std::this_thread::sleep_for(delay);
    }
    return result;
}

void WaitForDeviceConfigured(ethosn::driver_library::profiling::Configuration config, const std::string device)
{
    std::chrono::milliseconds delay(10);
    while (!ethosn::driver_library::profiling::Configure(config, device))
    {
        std::this_thread::sleep_for(delay);
    }
}

void TestSecondParentDevice(const std::function<void()>& f)
{
    // It requires two devices
    REQUIRE(ethosn::driver_library::GetNumberOfDevices() == 2U);

    const std::string dev0 = "/dev/ethosn0";
    const std::string dev1 = "/dev/ethosn1";
    ethosn::driver_library::profiling::Configuration config;
    config.m_EnableProfiling = true;
    WaitForDeviceConfigured(config, dev0);
    WaitForDeviceConfigured(config, dev1);

    uint64_t dev0RpmSuspendCount = 0;
    dev0RpmSuspendCount          = WaitForDeviceSupended(dev0RpmSuspendCount, dev0);
    uint64_t dev1RpmSuspendCount = 0;
    dev1RpmSuspendCount          = WaitForDeviceSupended(dev1RpmSuspendCount, dev1);

    // First device goes in runtime suspend after the profiling has been configured
    CHECK(dev0RpmSuspendCount == 1);
    // Second device goes in runtime suspend after the profiling has been configured.
    // Later on we are going to check that runtime counters increment
    // as expected on the second device only
    CHECK(dev1RpmSuspendCount == 1);

    // Call the test
    f();

    // Wait for the device to be suspended
    dev1RpmSuspendCount = WaitForDeviceSupended(dev1RpmSuspendCount, dev1);

    uint64_t dev1RpmResumeCount = ethosn::driver_library::profiling::GetCounterValue(
        ethosn::driver_library::profiling::PollCounterName::KernelDriverNumRuntimePowerResume, dev1);

    // Check that second device woke up to execute the inference
    CHECK(dev1RpmResumeCount == 1);

    // Second device goes in runtime suspend after the inference has been completed. We can
    // assume that the inference has been executed on the second device
    CHECK(dev1RpmSuspendCount == 2);

    // Get counter from first device
    dev0RpmSuspendCount = ethosn::driver_library::profiling::GetCounterValue(
        ethosn::driver_library::profiling::PollCounterName::KernelDriverNumRuntimePowerSuspend);

    // Check that first device is still in suspend and it didn't change its runtime power state
    CHECK(dev0RpmSuspendCount == 1);

    uint64_t dev0RpmResumeCount = ethosn::driver_library::profiling::GetCounterValue(
        ethosn::driver_library::profiling::PollCounterName::KernelDriverNumRuntimePowerResume);

    // Check that first device never resumed after the profiling has been enabled
    CHECK(dev0RpmResumeCount == 0);

    // Disable profiling
    config.m_EnableProfiling = false;
    WaitForDeviceConfigured(config, dev0);
    WaitForDeviceConfigured(config, dev1);
}

}    // namespace

TEST_CASE("MatchCapabilitiesN78 1TOPS 2Ple 384KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, 384);
}

TEST_CASE("MatchCapabilitiesN78 1TOPS 2Ple 448KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, 448);
}

TEST_CASE("MatchCapabilitiesN78 1TOPS 4Ple 448KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO, 448);
}

TEST_CASE("MatchCapabilitiesN78 1TOPS 4Ple 448KB: second parent device", "[Parent device selection]")
{
    if (!ethosn::driver_library::VerifyKernel())
    {
        throw std::runtime_error("Kernel version is not supported");
    }

    // It requires two devices
    REQUIRE(ethosn::driver_library::GetNumberOfDevices() == 2U);

    const std::string secondDevice                 = "/dev/ethosn1";
    ethosn::support_library::EthosNVariant variant = ethosn::support_library::EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO;
    const uint32_t sramSizeKb                      = 448U;

    std::vector<char> capabilities  = ethosn::driver_library::GetFirmwareAndHardwareCapabilities(secondDevice);
    std::vector<char> capabilities2 = ethosn::support_library::GetFwAndHwCapabilities(variant, (sramSizeKb * 1024));

    const auto caps  = reinterpret_cast<const FirmwareAndHardwareCapabilities*>(capabilities.data());
    const auto caps2 = reinterpret_cast<const FirmwareAndHardwareCapabilities*>(capabilities2.data());

    INFO(ComparisionString(*caps, *caps2));

    REQUIRE(std::equal(capabilities.begin(), capabilities.end(), capabilities2.begin(), capabilities2.end()));
}

TEST_CASE("MatchCapabilitiesN78 2TOPS 2Ple 2048KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO, 2048);
}

TEST_CASE("MatchCapabilitiesN78 2TOPS 4Ple 768KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO, 768);
}

TEST_CASE("MatchCapabilitiesN78 4TOPS 2Ple 512KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO, 512);
}

TEST_CASE("MatchCapabilitiesN78 4TOPS 2Ple 1024KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO, 1024);
}

TEST_CASE("MatchCapabilitiesN78 4TOPS 2Ple 1792KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO, 1792);
}

TEST_CASE("MatchCapabilitiesN78 4TOPS 4Ple 1024KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 1024);
}

TEST_CASE("MatchCapabilitiesN78 8TOPS 2Ple 512KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO, 512);
}

TEST_CASE("MatchCapabilitiesN78 8TOPS 2Ple 1024KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO, 1024);
}

TEST_CASE("MatchCapabilitiesN78 8TOPS 2Ple 2048KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO, 2048);
}

TEST_CASE("MatchCapabilitiesN78 8TOPS 2Ple 4096KB")
{
    MatchCapabilities(ethosn::support_library::EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO, 4096);
}

// This test relies on the profiling counters. Please note that things can
// become trickier as soon as another new test uses counters too since tests
// are executed in parallel.
// It is suggested that the backend test for the parent device selection is
// integrated in the one below for simplicity
TEST_CASE("Run inference: second parent device", "[Parent device selection]")
{
    auto testFun = []() {
        using namespace std;
        using namespace ethosn::support_library;

        const std::string dev1 = "/dev/ethosn1";

        // Input and output
        constexpr uint32_t height    = 24;
        constexpr uint32_t width     = 24;
        constexpr uint32_t depth     = 16;
        constexpr uint8_t inVal0     = 0x33;
        constexpr uint8_t inVal1     = 0x51;
        constexpr uint8_t outVal     = inVal0 + inVal1;
        constexpr uint32_t totalSize = height * width * depth;
        const TensorInfo tensorInfo({ 1, height, width, depth });
        InputTensor inputData0  = MakeTensor(std::vector<uint8_t>(totalSize, inVal0));
        InputTensor inputData1  = MakeTensor(std::vector<uint8_t>(totalSize, inVal1));
        OutputTensor outputData = MakeTensor(std::vector<uint8_t>(totalSize, static_cast<uint8_t>(~outVal)));

        ethosn::support_library::CompilationOptions options;

        // Create network
        shared_ptr<Network> network     = CreateNetwork(ethosn::driver_library::GetFirmwareAndHardwareCapabilities());
        shared_ptr<Operand> inputLayer0 = AddInput(network, tensorInfo).tensor;
        shared_ptr<Operand> inputLayer1 = AddInput(network, tensorInfo).tensor;
        shared_ptr<Operand> additionLayer =
            AddAddition(network, *inputLayer0, *inputLayer1, tensorInfo.m_QuantizationInfo).tensor;
        shared_ptr<Output> output = AddOutput(network, *additionLayer).tensor;

        // Compile network
        vector<unique_ptr<CompiledNetwork>> compiledNetworks0 = Compile(*network, options);

        std::vector<char> compiledNetworkData0;
        {
            ethosn::utils::VectorStream compiledNetworkStream0(compiledNetworkData0);
            compiledNetworks0[0]->Serialize(compiledNetworkStream0);
        }

        std::unique_ptr<ethosn::driver_library::ProcMemAllocator> processMemAllocator =
            std::make_unique<ethosn::driver_library::ProcMemAllocator>(dev1.c_str());
        ethosn::driver_library::Network netInst =
            processMemAllocator->CreateNetwork(compiledNetworkData0.data(), compiledNetworkData0.size());

        std::unique_ptr<ethosn::driver_library::Network> driverNetwork0 =
            std::make_unique<ethosn::driver_library::Network>(std::move(netInst));

        // Create input and output buffers and fetch pointers
        ethosn::driver_library::Buffer ifm0 = processMemAllocator->CreateBuffer(inputData0->GetByteData(), totalSize);
        ethosn::driver_library::Buffer ifm1 = processMemAllocator->CreateBuffer(inputData1->GetByteData(), totalSize);
        ethosn::driver_library::Buffer ofm  = processMemAllocator->CreateBuffer(totalSize);
        ethosn::driver_library::Buffer* ifmRaw[] = { &ifm0, &ifm1 };
        ethosn::driver_library::Buffer* ofmRaw[] = { &ofm };

        // Execute the inference. Second device is going to wake up and when finished it is
        // going to suspend again
        unique_ptr<ethosn::driver_library::Inference> result(driverNetwork0->ScheduleInference(
            ifmRaw, sizeof(ifmRaw) / sizeof(ifmRaw[0]), ofmRaw, sizeof(ofmRaw) / sizeof(ofmRaw[0])));
        driver_library::InferenceResult inferenceResult = result->Wait(60 * 1000);
        // Check that inference and output data are good
        REQUIRE(inferenceResult == driver_library::InferenceResult::Completed);
        CopyBuffers({ ofmRaw[0] }, { outputData->GetByteData() });

        REQUIRE(std::memcmp(ofm.Map(), outputData->GetByteData(), totalSize) == 0);
    };
    TestSecondParentDevice(testFun);
}

TEST_CASE("Run inference: second parent device using Arm NN", "[Parent device selection]")
{
    // Run the network through Arm NN on /dev/ethosn1
    auto testFun = []() {
        using namespace armnn;

        constexpr unsigned int input0BindingId = 0;
        constexpr unsigned int outputBindingId = 0;
        const std::string EthosnBackendId{ "EthosNAcc" };
        const std::string dev1 = "/dev/ethosn1";

        INetworkPtr net = INetwork::Create();
        TensorInfo tensorInfo0({ 1, 1, 1, 4 }, armnn::DataType::QAsymmU8, 0.0f, 0, true);
        tensorInfo0.SetQuantizationScale(0.9f);
        std::vector<uint8_t> data0(tensorInfo0.GetNumElements(), 0);
        std::iota(data0.begin(), data0.end(), 1);
        IConnectableLayer* const input0 = net->AddInputLayer(input0BindingId, "input0");

        TensorInfo tensorInfo1({ 1, 1, 1, 4 }, armnn::DataType::QAsymmU8, 0.0f, 0, true);
        tensorInfo1.SetQuantizationScale(0.9f);
        std::vector<uint8_t> data1(tensorInfo1.GetNumElements(), 0);
        std::iota(data1.begin(), data1.end(), 1);
        ConstTensor constTensor1(tensorInfo1, data1);
        IConnectableLayer* const input1 = net->AddConstantLayer(constTensor1, "input1");

        IConnectableLayer* const add =
            net->AddElementwiseBinaryLayer(ElementwiseBinaryDescriptor(BinaryOperation::Add), "add");

        TensorInfo outputInfo({ 1, 1, 1, 4 }, armnn::DataType::QAsymmU8, 0.0f, 0);
        outputInfo.SetQuantizationScale(0.9f);
        std::vector<uint8_t> ouputData(outputInfo.GetNumElements(), 0);
        std::iota(ouputData.begin(), ouputData.end(), 0);
        Tensor outputTensor(outputInfo, ouputData.data());
        IConnectableLayer* const output = net->AddOutputLayer(outputBindingId, "output");

        input0->GetOutputSlot(0).SetTensorInfo(tensorInfo0);
        input1->GetOutputSlot(0).SetTensorInfo(tensorInfo1);
        add->GetOutputSlot(0).SetTensorInfo(outputInfo);

        input0->GetOutputSlot(0).Connect(add->GetInputSlot(0));
        input1->GetOutputSlot(0).Connect(add->GetInputSlot(1));
        add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

        armnn::ConfigureLogging(true, true, LogSeverity::Debug);

        const armnn::BackendOptions ethosnDeviceOption(EthosnBackendId, { { "Device", dev1 } });
        IRuntime::CreationOptions creationOptions;

        IRuntimePtr runtime(IRuntime::Create(creationOptions));
        OptimizerOptionsOpaque optimizerOptions;
        optimizerOptions.AddModelOption(ethosnDeviceOption);
        IOptimizedNetworkPtr optimizedNet =
            Optimize(*net, { EthosnBackendId }, runtime->GetDeviceSpec(), optimizerOptions);

        NetworkId networkIdentifier;
        Status status = runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet));
        CHECK(status == Status::Success);

        InputTensors inputTensors;
        ConstTensor constTensor0(tensorInfo0, data0);
        inputTensors.emplace_back(input0BindingId, constTensor0);
        OutputTensors outputTensors;
        outputTensors.emplace_back(outputBindingId, outputTensor);
        status = runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
        CHECK(status == Status::Success);

        std::vector<uint8_t> expectedData = { 2, 4, 6, 8 };
        CHECK(std::equal(ouputData.begin(), ouputData.end(), expectedData.begin()));
    };
    TestSecondParentDevice(testFun);
}

TEST_CASE("Run inference: StrictPrecision using Arm NN")
{
    // Run the network through Arm NN
    using namespace armnn;

    const std::string EthosnBackendId{ "EthosNAcc" };
    INetworkPtr net = INetwork::Create();

    TensorInfo tensorInfo0({ 1, 1, 1, 16 }, armnn::DataType::QAsymmU8, 0.9f, 0, true);
    std::vector<uint8_t> data0(tensorInfo0.GetNumElements(), 0);
    std::iota(data0.begin(), data0.end(), 1);
    IConnectableLayer* const input0 = net->AddInputLayer(0, "input0");

    TensorInfo tensorInfo1({ 1, 1, 1, 16 }, armnn::DataType::QAsymmU8, 0.9f, 0, true);
    std::vector<uint8_t> data1(tensorInfo1.GetNumElements(), 0);
    std::iota(data1.begin(), data1.end(), 17);
    IConnectableLayer* const input1 = net->AddInputLayer(1, "input1");

    std::array<TensorShape, 2> concatInputShapes = { tensorInfo0.GetShape(), tensorInfo1.GetShape() };
    IConnectableLayer* const concat              = net->AddConcatLayer(
        CreateDescriptorForConcatenation(concatInputShapes.begin(), concatInputShapes.end(), 3), "concat");

    TensorInfo outputInfo({ 1, 1, 1, 32 }, armnn::DataType::QAsymmU8, 0.9f, 0);
    std::vector<uint8_t> ouputData(outputInfo.GetNumElements(), 0);
    std::iota(ouputData.begin(), ouputData.end(), 0);
    Tensor outputTensor(outputInfo, ouputData.data());
    IConnectableLayer* const output = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).SetTensorInfo(tensorInfo0);
    input1->GetOutputSlot(0).SetTensorInfo(tensorInfo1);
    concat->GetOutputSlot(0).SetTensorInfo(outputInfo);

    input0->GetOutputSlot(0).Connect(concat->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(concat->GetInputSlot(1));
    concat->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::ConfigureLogging(true, true, LogSeverity::Debug);

    const armnn::BackendOptions ethosnDeviceOption(EthosnBackendId, { { "StrictPrecision", true } });
    IRuntime::CreationOptions creationOptions;

    IRuntimePtr runtime(IRuntime::Create(creationOptions));
    OptimizerOptionsOpaque optimizerOptions;
    optimizerOptions.AddModelOption(ethosnDeviceOption);
    IOptimizedNetworkPtr optimizedNet = Optimize(*net, { EthosnBackendId }, runtime->GetDeviceSpec(), optimizerOptions);

    NetworkId networkIdentifier;
    Status status = runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet));
    CHECK(status == Status::Success);

    InputTensors inputTensors;
    ConstTensor constTensor0(tensorInfo0, data0);
    ConstTensor constTensor1(tensorInfo1, data1);
    inputTensors.emplace_back(0, constTensor0);
    inputTensors.emplace_back(1, constTensor1);

    OutputTensors outputTensors;
    outputTensors.emplace_back(0, outputTensor);

    status = runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
    CHECK(status == Status::Success);

    std::vector<uint8_t> expectedData(outputInfo.GetNumElements(), 0);
    std::iota(expectedData.begin(), expectedData.end(), 1);
    CHECK(std::equal(ouputData.begin(), ouputData.end(), expectedData.begin()));
}

}    // namespace system_tests
}    // namespace ethosn
