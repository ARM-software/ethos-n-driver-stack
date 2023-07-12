//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ArmnnParseRunner.hpp"
#include "EthosNParseRunner.hpp"
#include "GgfRunner.hpp"
#include "SystemTestsUtils.hpp"

#include <catch.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string.h>

using namespace ethosn::support_library;

namespace ethosn
{
namespace system_tests
{

// Convolution test case that checks the exactness of multiple output tensors from a convolution layer.
TEST_CASE("ConvolutionIdenticalOutputs")
{
    LayerData layerData;
    layerData.SetInputTensorFormat(DataFormat::NHWC);
    layerData.SetOutputTensorFormat(DataFormat::NHWC);
    layerData.SetInputMin(0);
    layerData.SetInputMax(255);
    layerData.SetInputDataType(DataType::U8);
    layerData.SetQuantInfo("input - quantization parameters", QuantizationInfo(0, 1.0f));
    layerData.SetQuantInfo("conv0 - weight quantization parameters", QuantizationInfo(0, 1.4f));

    std::stringstream ggf;
    ggf << "input layer, name input, top input, shape 1, 16, 16, 16\n";
    ggf << "conv layer, name conv0, bottom input, top conv0, num output 16, kernel h 1, kernel w 1, stride h 1, stride "
           "w 1, pad 0, bias_enable 1\n";
    ggf << "output layer, name output1_0, bottom conv0\n";
    ggf << "output layer, name output1_1, bottom conv0\n";

    ArmnnParseRunner armnnRunner(ggf, layerData);
    InferenceOutputs armnnResult = armnnRunner.RunNetwork({ "CpuRef" });
    REQUIRE(CompareTensors(*armnnResult[0], *armnnResult[1], 0));

    {
        ggf.clear();
        ggf.seekg(0);
        EthosNParseRunner::CreationOptions creationOptions =
            EthosNParseRunner::CreationOptions::CreateWithGlobalOptions(ggf, layerData);
        creationOptions.m_StrictPrecision = true;
        EthosNParseRunner ethosnRunner(creationOptions);
        InferenceOutputs ethosnResult = ethosnRunner.RunNetwork();

        // The two outputs are expected to be exact
        REQUIRE(CompareTensors(*ethosnResult[0], *ethosnResult[1], 0));

        // Check that EthosN outputs are identical to reference ArmNN outputs
        REQUIRE(CompareTensors(*armnnResult[0], *ethosnResult[0], 0));
        REQUIRE(CompareTensors(*armnnResult[1], *ethosnResult[1], 0));
    }
}

}    // namespace system_tests
}    // namespace ethosn
