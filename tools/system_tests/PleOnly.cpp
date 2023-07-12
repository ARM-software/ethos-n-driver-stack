//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "GgfRunner.hpp"

#include <ethosn_utils/VectorStream.hpp>

#include <ethosn_driver_library/ProcMemAllocator.hpp>

#include <catch.hpp>

#include <fstream>

namespace ethosn
{
namespace system_tests
{

static const std::string g_AddTwoInputsGgfBuffer("input layer, name data0, top data0, shape 1, 16, 16, 16\n"
                                                 "input layer, name data1, top data1, shape 1, 16, 16, 16\n"
                                                 "add layer, name add, bottom data0_data1, top add\n");

TEST_CASE("AdditionRescale")
{
    LayerData layerData;

    using QuantizationInfo = ethosn::support_library::QuantizationInfo;

    struct TestDataSet
    {
        struct
        {
            uint8_t data;
            QuantizationInfo quantInfo;
        } input0;
        struct
        {
            uint8_t data;
            QuantizationInfo quantInfo;
        } input1;
        QuantizationInfo outputQuantInfo;
    };

    std::vector<TestDataSet> testDataSets;

    // test clamping of output to min value
    testDataSets.push_back({ { 0x14, QuantizationInfo{ 0, 0.5f } },
                             { 0x1c, QuantizationInfo{ 0, 0.25f } },
                             QuantizationInfo{ -30, 0.75f } });

    // test clamping of output to max value
    testDataSets.push_back({ { 0x14, QuantizationInfo{ 0, 0.5f } },
                             { 0x1c, QuantizationInfo{ 0, 0.25f } },
                             QuantizationInfo{ 236, 0.75f } });

    // test in-range values
    testDataSets.push_back({ { 0x10, QuantizationInfo{ 0, 0.5f } },
                             { 0x2c, QuantizationInfo{ 0, 0.25f } },
                             QuantizationInfo{ 0, 0.75f } });

    // Test sets that require +-2 diff
    testDataSets.push_back({ { 0x85, QuantizationInfo{ 0, 0.11f } },
                             { 0x7e, QuantizationInfo{ 0, 0.38f } },
                             QuantizationInfo{ 0, 0.49f } });
    testDataSets.push_back({ { 0x6d, QuantizationInfo{ 16, 0.5f } },
                             { 0x98, QuantizationInfo{ -76, 0.5f } },
                             QuantizationInfo{ -36, 0.81640625f } });

    g_Logger.Debug("AdditionInputOutputRescale: input0={0x%x,%d,%.2f}", testDataSets[0].input0.data,
                   testDataSets[0].input0.quantInfo.GetZeroPoint(), testDataSets[0].input0.quantInfo.GetScale());

    // Dimensions must match the input layer in the ggf file
    const int ifmHeight   = 16;
    const int ifmWidth    = 16;
    const int ifmChannels = 16;

    OwnedTensor inputData0 = MakeTensor(std::vector<uint8_t>(ifmHeight * ifmWidth * ifmChannels));
    OwnedTensor inputData1 = MakeTensor(std::vector<uint8_t>(ifmHeight * ifmWidth * ifmChannels));

    for (size_t i = 0; i < testDataSets.size(); ++i)
    {
        std::fill(inputData0->GetData<uint8_t>().begin(), inputData0->GetData<uint8_t>().end(),
                  testDataSets[i].input0.data);
        std::fill(inputData1->GetData<uint8_t>().begin(), inputData1->GetData<uint8_t>().end(),
                  testDataSets[i].input1.data);

        layerData.SetTensor("layer 0 input - tensor", *inputData0);
        layerData.SetTensor("layer 1 input - tensor", *inputData1);

        ethosn::support_library::QuantizationInfo input0 = testDataSets[i].input0.quantInfo;
        ethosn::support_library::QuantizationInfo input1 = testDataSets[i].input1.quantInfo;
        ethosn::support_library::QuantizationInfo output = testDataSets[i].outputQuantInfo;

        const float input0Max = input0.GetScale() * static_cast<float>(255 - input0.GetZeroPoint());
        const float input0Min = input0.GetScale() * static_cast<float>(0 - input0.GetZeroPoint());
        const float input1Max = input1.GetScale() * static_cast<float>(255 - input0.GetZeroPoint());
        const float input1Min = input1.GetScale() * static_cast<float>(0 - input1.GetZeroPoint());

        g_Logger.Debug("AdditionInputOutputRescale: input0={%d,%.2f} -> max=%.2f min=%.2f", input0.GetZeroPoint(),
                       input0.GetScale(), input0Max, input0Min);

        g_Logger.Debug("AdditionInputOutputRescale: input1={%d,%.2f} -> max=%.2f min=%.2f", input1.GetZeroPoint(),
                       input1.GetScale(), input1Max, input1Min);

        layerData.SetInputMax(std::min(input0Max, input1Max));
        layerData.SetInputMin(std::max(input0Min, input1Min));

        const float outputMax = output.GetScale() * static_cast<float>(255 - output.GetZeroPoint());
        const float outputMin = output.GetScale() * static_cast<float>(0 - output.GetZeroPoint());

        g_Logger.Debug("AdditionInputOutputRescale i=%zu output max=%.2f min=%.2f -> output={%d, %.2f}", i, outputMax,
                       outputMin, output.GetZeroPoint(), output.GetScale());

        layerData.SetQuantInfo("layer 0 input - quantization parameters", input0);
        layerData.SetQuantInfo("layer 1 input - quantization parameters", input1);
        layerData.SetQuantInfo("layer 2 add - quantization parameters", output);

        std::stringstream tmpGgfStream;
        tmpGgfStream << g_AddTwoInputsGgfBuffer;

        // The precision of the rescaling is currently limited causing the diff
        // compared to Arm NN to be +-2 for some combinations of qantization parameters.
        CompareArmnnAndEthosNOutput(tmpGgfStream, layerData, false, { { "*", 2 } });
    }
}

}    // namespace system_tests
}    // namespace ethosn
