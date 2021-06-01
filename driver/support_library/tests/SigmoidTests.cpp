//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/SupportQueries.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("SigmoidSupported")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("Supported configuration")
    {
        const auto inputDataType = GENERATE(DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED);
        const QuantizationInfo outQuantization((inputDataType == DataType::INT8_QUANTIZED) ? -128 : 0, 1.0f / 256);
        TensorInfo input({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        TensorInfo output({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWC, outQuantization);
        REQUIRE(queries.IsSigmoidSupported(input, &output) == SupportedLevel::Supported);
    }

    SECTION("Wrong quantization")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        TensorInfo output({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsSigmoidSupported(input, &output) == SupportedLevel::Unsupported);
    }

    SECTION("Wrong size")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        TensorInfo output({ 1, 8, 8, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                          QuantizationInfo(0, 1.0f / 256));
        REQUIRE(queries.IsSigmoidSupported(input, &output) == SupportedLevel::Unsupported);
    }
}
