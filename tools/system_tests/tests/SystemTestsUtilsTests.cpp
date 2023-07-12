//
// Copyright © 2018-2020,2023 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../SystemTestsUtils.hpp"

#include <catch.hpp>

namespace ethosn
{
namespace system_tests
{

TEST_CASE("CreateCacheHeader")
{
    InferenceOutputs outputs;
    outputs.resize(2);

    outputs[0] = MakeTensor(DataType::S8, 512);
    outputs[1] = MakeTensor(DataType::S32, 1024);

    std::vector<char> header = CreateCacheHeader(outputs);
    // The header contains the following bytes in little endian format:
    // 02 00 00 00 00 00 00 00 (2 outputs encoded in 64 bits)
    // 00 02 00 00 00 00 00 00 (512 bytes size encoded in 64 bits)
    // 01                      (U8 type encoded as 1 in 8 bits)
    // 00 10 00 00 00 00 00 00 (4096 byte size encoded in 64 bits)
    // 02                      (S32 type encoded as 2 in 8 bits)
    REQUIRE(header.size() == 26);
    REQUIRE(header[0] == 0x02);
    REQUIRE(header[9] == 0x02);
    REQUIRE(header[16] == 0x01);
    REQUIRE(header[18] == 0x10);
    REQUIRE(header[25] == 0x2);
}

TEST_CASE("GetOutputTensorsFromCache")
{
    std::vector<char> header{ 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 16, 0, 0, 0, 0, 0, 0, 2 };

    InferenceOutputs outputs = GetOutputTensorsFromCache(header);
    REQUIRE(outputs.size() == 2);
    REQUIRE(outputs[0]->GetNumElements() == 512);
    REQUIRE(outputs[0]->GetDataType() == DataType::S8);
    REQUIRE(outputs[0]->GetNumBytes() == 512);
    REQUIRE(outputs[1]->GetNumElements() == 1024);
    REQUIRE(outputs[1]->GetDataType() == DataType::S32);
    REQUIRE(outputs[1]->GetNumBytes() == 4096);
}

}    // namespace system_tests
}    // namespace ethosn
