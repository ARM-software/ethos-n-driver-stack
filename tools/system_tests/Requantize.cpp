//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "SystemTestsUtils.hpp"

#include <ethosn_utils/VectorStream.hpp>

#include <ethosn_driver_library/ProcMemAllocator.hpp>

#include <catch.hpp>

namespace ethosn
{
namespace system_tests
{

template <typename InputType, typename OutputType>
void VerifyRequantizedOutput(const ethosn::support_library::Network& network,
                             InputType* const inputValues,
                             OutputType* const expectedOutputValues,
                             const uint8_t inputSize)
{
    using namespace ethosn::support_library;

    // Compile Network
    ethosn::support_library::CompilationOptions compilationOptions;
    compilationOptions.m_StrictPrecision                          = true;
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork = Compile(network, compilationOptions);

    REQUIRE(compiledNetwork.size() == 1);

    std::vector<char> compiledNetworkData0;
    {
        ethosn::utils::VectorStream compiledNetworkStream0(compiledNetworkData0);
        compiledNetwork[0]->Serialize(compiledNetworkStream0);
    }

    ethosn::driver_library::ProcMemAllocator processMemAllocator;
    ethosn::driver_library::Network driverNetwork0 =
        processMemAllocator.CreateNetwork(compiledNetworkData0.data(), compiledNetworkData0.size());

    uint32_t inBufSize  = compiledNetwork[0]->GetInputBufferInfos()[0].m_Size;
    uint32_t outBufSize = compiledNetwork[0]->GetOutputBufferInfos()[0].m_Size;

    // Create input and output buffers and fetch pointers.
    for (uint8_t index = 0; index < inputSize; index++)
    {
        InputTensor inputData = MakeTensor(std::vector<InputType>(inBufSize, inputValues[index]));
        ethosn::driver_library::Buffer inputBuffer =
            processMemAllocator.CreateBuffer(inputData->GetByteData(), inBufSize);
        ethosn::driver_library::Buffer* inputBufferRaw[] = { &inputBuffer };

        OutputTensor outputData = MakeTensor(std::vector<OutputType>(outBufSize));
        ethosn::driver_library::Buffer outputBuffer =
            processMemAllocator.CreateBuffer(outputData->GetByteData(), outBufSize);
        ethosn::driver_library::Buffer* outputBufferRaw[] = { &outputBuffer };

        // Execute the inference.
        std::unique_ptr<ethosn::driver_library::Inference> result(
            driverNetwork0.ScheduleInference(inputBufferRaw, sizeof(inputBufferRaw) / sizeof(inputBufferRaw[0]),
                                             outputBufferRaw, sizeof(outputBufferRaw) / sizeof(outputBufferRaw[0])));
        driver_library::InferenceResult inferenceResult = result->Wait(60 * 1000);
        REQUIRE(inferenceResult == driver_library::InferenceResult::Completed);
        CopyBuffers({ outputBufferRaw[0] }, { outputData->GetByteData() });

        REQUIRE(static_cast<OutputType>(*(outputData.get()->GetByteData())) == expectedOutputValues[index]);
    }
}

TEST_CASE("Check the requantized output tensor data when the requantize input/output are of different types")
{
    using namespace ethosn::support_library;

    const ethosn::support_library::DataType inputType =
        GENERATE(ethosn::support_library::DataType::UINT8_QUANTIZED, ethosn::support_library::DataType::INT8_QUANTIZED);
    const ethosn::support_library::DataType outputType =
        (inputType == ethosn::support_library::DataType::UINT8_QUANTIZED)
            ? ethosn::support_library::DataType::INT8_QUANTIZED
            : ethosn::support_library::DataType::UINT8_QUANTIZED;

    // Create Network
    auto network = CreateNetwork(ethosn::driver_library::GetFirmwareAndHardwareCapabilities());

    TensorInfo inputInfo{
        { { 1, 1, 1, 1 } },
        inputType,
        DataFormat::NHWCB,
        { 0, 1.0f },
    };

    int32_t zeroPoint = (inputType == ethosn::support_library::DataType::UINT8_QUANTIZED) ? -128 : 128;

    RequantizeInfo requantInfo({ zeroPoint, 1.0f });
    requantInfo.m_OutputDataType = outputType;

    auto input      = AddInput(network, inputInfo).tensor;
    auto requantize = AddRequantize(network, *input, requantInfo).tensor;
    auto output     = AddOutput(network, *requantize).tensor;

    if (inputType == ethosn::support_library::DataType::UINT8_QUANTIZED)
    {
        uint8_t inputSize             = 3;
        uint8_t inputValues[]         = { 0, 128, 255 };
        int8_t expectedOutputValues[] = { -128, 0, 127 };
        VerifyRequantizedOutput<uint8_t, int8_t>(*network, inputValues, expectedOutputValues, inputSize);
    }
    else
    {
        uint8_t inputSize              = 3;
        int8_t inputValues[]           = { -128, 0, 127 };
        uint8_t expectedOutputValues[] = { 0, 128, 255 };
        VerifyRequantizedOutput<int8_t, uint8_t>(*network, inputValues, expectedOutputValues, inputSize);
    }
}

}    // namespace system_tests
}    // namespace ethosn
