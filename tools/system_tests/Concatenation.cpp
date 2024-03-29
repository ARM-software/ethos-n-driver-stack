//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ArmnnParseRunner.hpp"
#include "EthosNParseRunner.hpp"
#include "GgfRunner.hpp"
#include "SystemTestsUtils.hpp"

#include "../../../driver/support_library/src/Compiler.hpp"
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

TEST_CASE("Concatenation different quantization")
{
    LayerData layerData;
    layerData.SetInputMin(0);
    layerData.SetInputMax(255);
    layerData.SetQuantInfo("input - quantization parameters", QuantizationInfo(0, 1.0f));
    layerData.SetQuantInfo("conv0 - output quantization parameters", QuantizationInfo(0, 1.1f));
    layerData.SetQuantInfo("conv1 - output quantization parameters", QuantizationInfo(0, 1.2f));
    layerData.SetQuantInfo("concat - output quantization parameters", QuantizationInfo(0, 1.3f));

    std::stringstream ggf;
    ggf << "input layer, name input, top input, shape 1, 17, 17, 16\n";
    ggf << "conv layer, name conv0, bottom input, top conv0, num output 16, kernel h 3, kernel w 3, stride h 2, stride "
           "w 2, pad 0, bias_enable 1\n";
    ggf << "conv layer, name conv1, bottom input, top conv1, num output 16, kernel h 3, kernel w 3, stride h 2, stride "
           "w 2, pad 0, bias_enable 1\n";
    ggf << "pooling layer, name pool0, bottom input, top pool0, pool max, kernel size 3, stride 2, pad 0\n";
    ggf << "concat layer, name concat, bottom conv0_conv1_pool0, top concat0, axis 3\n";

    CompareArmnnAndEthosNOutput(ggf, layerData, true, { { "*", 1.0f } });
}

}    // namespace system_tests
}    // namespace ethosn
