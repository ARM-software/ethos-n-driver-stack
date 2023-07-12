//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNParseRunner.hpp"

#include <catch.hpp>

#include <fstream>

namespace ethosn
{
namespace system_tests
{

using namespace ethosn::support_library;

// Trivial end-to-end Transpose Convolution test that puts in some simple fixed input data and checks
// that the result is as expected (and can be manually calculated).
TEST_CASE("TransposeConvolutionSimple")
{
    LayerData layerData;
    PaddingInfo info;
    layerData.SetInputTensorFormat(DataFormat::NHWC);
    layerData.SetOutputTensorFormat(DataFormat::NHWC);
    layerData.SetTensor("input - tensor", *MakeTensor(std::vector<uint8_t>{ 1, 2, 3, 4 }));
    layerData.SetQuantInfo("input - quantization parameters", QuantizationInfo(0, 1.0f / 256.0f));
    layerData.SetTensor("tconv - conv weights", *MakeTensor(std::vector<uint8_t>{ 1 }));
    layerData.SetQuantInfo("tconv - weight quantization parameters", QuantizationInfo(0, 1.0f));
    layerData.SetQuantInfo("tconv - output quantization parameters", QuantizationInfo(0, 1.00001f / 256.0f));

    std::ifstream dummy;
    EthosNParseRunner::CreationOptions creationOptions =
        EthosNParseRunner::CreationOptions::CreateWithGlobalOptions(dummy, layerData);
    EthosNParseRunner runner(creationOptions);
    info.alg = PaddingAlgorithm::VALID;
    runner.AddInput("input", TensorShape{ 1, 2, 2, 1 });
    runner.AddTransposeConvolution("tconv", "input", 1, 1, 2, 2, 1, false, WeightParams(), OutputParams(), info);
    runner.AddOutput("output", "tconv");

    InferenceOutputs result = runner.RunNetwork();

    std::vector<uint8_t> refOutput{ 1, 0, 2, 0, 0, 0, 3, 0, 4 };
    REQUIRE(result[0]->GetData<uint8_t>() == refOutput);
}
}    // namespace system_tests
}    // namespace ethosn
