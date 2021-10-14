//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../BinaryParser.hpp"
#include "../CMMParser.hpp"
#include "../XmlParser.hpp"

#include <ethosn_utils/Strings.hpp>

#include <catch.hpp>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace
{

const std::string g_XmlStr =
    R"(<?xml version="1.0" encoding="utf-8"?>
<STREAM VERSION_MAJOR="%VERSION_MAJOR%" VERSION_MINOR="%VERSION_MINOR%" VERSION_PATCH="%VERSION_PATCH%"><!--Command0-->
    <SECTION>
        <TYPE>SISO</TYPE>
    </SECTION>
    <!--Command1-->
    <OPERATION_MCE_PLE>
        <INPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWCB</DATA_FORMAT>
            <TENSOR_SHAPE>1 32 32 96</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>0 0 0 0</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 32 32 96</STRIPE_SHAPE>
            <TILE_SHAPE>1 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>0</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x0</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </INPUT_INFO>
        <WEIGHT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>WEIGHT_STREAM</DATA_FORMAT>
            <TENSOR_SHAPE>1 1 96 32</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>0 0 0 0</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 1 96 32</STRIPE_SHAPE>
            <TILE_SHAPE>1 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>1</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x1800</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </WEIGHT_INFO>
        <WEIGHTS_METADATA_BUFFER_ID>10</WEIGHTS_METADATA_BUFFER_ID>
        <OUTPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWCB</DATA_FORMAT>
            <TENSOR_SHAPE>1 32 32 32</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>1 32 32 32</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 32 32 32</STRIPE_SHAPE>
            <TILE_SHAPE>1 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>2</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x1900</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </OUTPUT_INFO>
        <SRAM_CONFIG>
            <ALLOCATION_STRATEGY>STRATEGY_0</ALLOCATION_STRATEGY>
        </SRAM_CONFIG>
        <BLOCK_CONFIG>
            <BLOCK_WIDTH>16</BLOCK_WIDTH>
            <BLOCK_HEIGHT>16</BLOCK_HEIGHT>
        </BLOCK_CONFIG>
        <MCE_OP_INFO>
            <STRIDE_X>1</STRIDE_X>
            <STRIDE_Y>1</STRIDE_Y>
            <PAD_TOP>0</PAD_TOP>
            <PAD_LEFT>0</PAD_LEFT>
            <UNINTERLEAVED_INPUT_SHAPE>1 16 16 16</UNINTERLEAVED_INPUT_SHAPE>
            <OUTPUT_SHAPE>1 16 16 16</OUTPUT_SHAPE>
            <OUTPUT_STRIPE_SHAPE>1 16 16 16</OUTPUT_STRIPE_SHAPE>
            <OPERATION>CONVOLUTION</OPERATION>
            <ALGO>DIRECT</ALGO>
            <ACTIVATION_MIN>118</ACTIVATION_MIN>
            <ACTIVATION_MAX>255</ACTIVATION_MAX>
            <UPSAMPLE_TYPE>OFF</UPSAMPLE_TYPE>
        </MCE_OP_INFO>
        <PLE_OP_INFO>
            <CE_SRAM>0x0</CE_SRAM>
            <PLE_SRAM>0x0</PLE_SRAM>
            <OPERATION>LEAKY_RELU</OPERATION>
            <RESCALE_MULTIPLIER0>0</RESCALE_MULTIPLIER0>
            <RESCALE_SHIFT0>0</RESCALE_SHIFT0>
            <RESCALE_MULTIPLIER1>0</RESCALE_MULTIPLIER1>
            <RESCALE_SHIFT1>0</RESCALE_SHIFT1>
        </PLE_OP_INFO>
    </OPERATION_MCE_PLE>
    <!--Command2-->
    <DELAY>
        <VALUE>3</VALUE>
    </DELAY>
    <!--Command3-->
    <FENCE/>
    <!--Command4-->
    <OPERATION_MCE_PLE>
        <INPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWCB</DATA_FORMAT>
            <TENSOR_SHAPE>1 8 8 512</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>0 0 0 0</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 8 8 128</STRIPE_SHAPE>
            <TILE_SHAPE>1 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>2</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x0</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </INPUT_INFO>
        <WEIGHT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>WEIGHT_STREAM</DATA_FORMAT>
            <TENSOR_SHAPE>1 1 1 32768</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>0 0 0 0</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 1 1 1024</STRIPE_SHAPE>
            <TILE_SHAPE>1000 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>3</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x8800</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </WEIGHT_INFO>
        <WEIGHTS_METADATA_BUFFER_ID>15</WEIGHTS_METADATA_BUFFER_ID>
        <OUTPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWCB</DATA_FORMAT>
            <TENSOR_SHAPE>1 8 8 32</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>1 8 8 32</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 8 8 8</STRIPE_SHAPE>
            <TILE_SHAPE>1 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>4</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x8000</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </OUTPUT_INFO>
        <SRAM_CONFIG>
            <ALLOCATION_STRATEGY>STRATEGY_1</ALLOCATION_STRATEGY>
        </SRAM_CONFIG>
        <BLOCK_CONFIG>
            <BLOCK_WIDTH>8</BLOCK_WIDTH>
            <BLOCK_HEIGHT>8</BLOCK_HEIGHT>
        </BLOCK_CONFIG>
        <MCE_OP_INFO>
            <STRIDE_X>1</STRIDE_X>
            <STRIDE_Y>1</STRIDE_Y>
            <PAD_TOP>0</PAD_TOP>
            <PAD_LEFT>0</PAD_LEFT>
            <UNINTERLEAVED_INPUT_SHAPE>1 32 32 2</UNINTERLEAVED_INPUT_SHAPE>
            <OUTPUT_SHAPE>1 32 32 2</OUTPUT_SHAPE>
            <OUTPUT_STRIPE_SHAPE>1 32 32 2</OUTPUT_STRIPE_SHAPE>
            <OPERATION>FULLY_CONNECTED</OPERATION>
            <ALGO>DIRECT</ALGO>
            <ACTIVATION_MIN>0</ACTIVATION_MIN>
            <ACTIVATION_MAX>255</ACTIVATION_MAX>
            <UPSAMPLE_TYPE>OFF</UPSAMPLE_TYPE>
        </MCE_OP_INFO>
        <PLE_OP_INFO>
            <CE_SRAM>0x0</CE_SRAM>
            <PLE_SRAM>0x0</PLE_SRAM>
            <OPERATION>PASSTHROUGH</OPERATION>
            <RESCALE_MULTIPLIER0>0</RESCALE_MULTIPLIER0>
            <RESCALE_SHIFT0>0</RESCALE_SHIFT0>
            <RESCALE_MULTIPLIER1>0</RESCALE_MULTIPLIER1>
            <RESCALE_SHIFT1>0</RESCALE_SHIFT1>
        </PLE_OP_INFO>
    </OPERATION_MCE_PLE>
    <!--Command5-->
    <FENCE/>
    <!--Command6-->
    <OPERATION_SOFTMAX>
        <INPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWCB</DATA_FORMAT>
            <TENSOR_SHAPE>1 32 32 96</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>0 0 0 0</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>0 0 0 0</STRIPE_SHAPE>
            <TILE_SHAPE>1 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>0</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x0</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </INPUT_INFO>
        <OUTPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWCB</DATA_FORMAT>
            <TENSOR_SHAPE>1 32 32 96</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>1 32 32 96</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>0 0 0 0</STRIPE_SHAPE>
            <TILE_SHAPE>1 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>1</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x0</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </OUTPUT_INFO>
        <SCALED_DIFF>0</SCALED_DIFF>
        <EXP_ACCUMULATION>1</EXP_ACCUMULATION>
        <INPUT_BETA_MULTIPLIER>2</INPUT_BETA_MULTIPLIER>
        <INPUT_BETA_LEFT_SHIFT>3</INPUT_BETA_LEFT_SHIFT>
        <DIFF_MIN>-1</DIFF_MIN>
    </OPERATION_SOFTMAX>
    <!--Command7-->
    <DUMP_DRAM>
        <DRAM_BUFFER_ID>2</DRAM_BUFFER_ID>
        <FILENAME>OutputModel_NHWCB.hex</FILENAME>
    </DUMP_DRAM>
    <!--Command8-->
    <DUMP_SRAM>
        <PREFIX>output_ce</PREFIX>
    </DUMP_SRAM>
    <!--Command9-->
    <OPERATION_PLE>
        <INPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWCB</DATA_FORMAT>
            <TENSOR_SHAPE>1 16 16 16</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>0 0 0 0</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 16 16 16</STRIPE_SHAPE>
            <TILE_SHAPE>1000 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>0</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x0</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </INPUT_INFO>
        <INPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWCB</DATA_FORMAT>
            <TENSOR_SHAPE>1 16 16 16</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>0 0 0 0</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 16 16 16</STRIPE_SHAPE>
            <TILE_SHAPE>1000 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>1</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x1000</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </INPUT_INFO>
        <OUTPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWCB</DATA_FORMAT>
            <TENSOR_SHAPE>1 16 16 16</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>1 16 16 16</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 16 16 16</STRIPE_SHAPE>
            <TILE_SHAPE>1000 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>2</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x2000</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </OUTPUT_INFO>
        <SRAM_CONFIG>
            <ALLOCATION_STRATEGY>STRATEGY_0</ALLOCATION_STRATEGY>
        </SRAM_CONFIG>
        <PLE_OP_INFO>
            <CE_SRAM>0x200</CE_SRAM>
            <PLE_SRAM>0x0</PLE_SRAM>
            <OPERATION>ADDITION</OPERATION>
            <RESCALE_MULTIPLIER0>0</RESCALE_MULTIPLIER0>
            <RESCALE_SHIFT0>0</RESCALE_SHIFT0>
            <RESCALE_MULTIPLIER1>0</RESCALE_MULTIPLIER1>
            <RESCALE_SHIFT1>0</RESCALE_SHIFT1>
        </PLE_OP_INFO>
    </OPERATION_PLE>
    <!--Command10-->
    <OPERATION_PLE>
        <INPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWCB</DATA_FORMAT>
            <TENSOR_SHAPE>1 16 16 16</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>0 0 0 0</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 16 16 16</STRIPE_SHAPE>
            <TILE_SHAPE>1000 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>0</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x0</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </INPUT_INFO>
        <OUTPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWCB</DATA_FORMAT>
            <TENSOR_SHAPE>1 16 16 16</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>1 16 16 16</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 16 16 16</STRIPE_SHAPE>
            <TILE_SHAPE>1000 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>1</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x100</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </OUTPUT_INFO>
        <SRAM_CONFIG>
            <ALLOCATION_STRATEGY>STRATEGY_0</ALLOCATION_STRATEGY>
        </SRAM_CONFIG>
        <PLE_OP_INFO>
            <CE_SRAM>0x200</CE_SRAM>
            <PLE_SRAM>0x0</PLE_SRAM>
            <OPERATION>PASSTHROUGH</OPERATION>
            <RESCALE_MULTIPLIER0>0</RESCALE_MULTIPLIER0>
            <RESCALE_SHIFT0>0</RESCALE_SHIFT0>
            <RESCALE_MULTIPLIER1>0</RESCALE_MULTIPLIER1>
            <RESCALE_SHIFT1>0</RESCALE_SHIFT1>
        </PLE_OP_INFO>
    </OPERATION_PLE>
    <!--Command11-->
    <OPERATION_CONVERT>
        <INPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWCB</DATA_FORMAT>
            <TENSOR_SHAPE>1 32 32 32</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>0 0 0 0</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 16 32 32</STRIPE_SHAPE>
            <TILE_SHAPE>4000 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>0</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x0</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </INPUT_INFO>
        <OUTPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWC</DATA_FORMAT>
            <TENSOR_SHAPE>1 32 32 32</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>1 32 32 32</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 16 32 32</STRIPE_SHAPE>
            <TILE_SHAPE>4000 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>0</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x0</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </OUTPUT_INFO>
    </OPERATION_CONVERT>
    <!--Command12-->
    <OPERATION_SPACE_TO_DEPTH>
        <INPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWC</DATA_FORMAT>
            <TENSOR_SHAPE>1 32 32 16</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>1 32 32 16</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 16 32 32</STRIPE_SHAPE>
            <TILE_SHAPE>4000 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>0</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x0</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </INPUT_INFO>
        <OUTPUT_INFO>
            <DATA_TYPE>U8</DATA_TYPE>
            <DATA_FORMAT>NHWC</DATA_FORMAT>
            <TENSOR_SHAPE>1 16 16 64</TENSOR_SHAPE>
            <SUPERTENSOR_SHAPE>1 16 16 64</SUPERTENSOR_SHAPE>
            <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
            <STRIPE_SHAPE>1 16 32 32</STRIPE_SHAPE>
            <TILE_SHAPE>4000 1 1 1</TILE_SHAPE>
            <DRAM_BUFFER_ID>0</DRAM_BUFFER_ID>
            <SRAM_OFFSET>0x0</SRAM_OFFSET>
            <ZERO_POINT>0</ZERO_POINT>
            <DATA_LOCATION>DRAM</DATA_LOCATION>
        </OUTPUT_INFO>
        <USED_EMCS>8</USED_EMCS>
        <INTERMEDIATE_1_SIZE>1024</INTERMEDIATE_1_SIZE>
        <INTERMEDIATE_2_SIZE>2048</INTERMEDIATE_2_SIZE>
    </OPERATION_SPACE_TO_DEPTH>
    <!--Command13-->
    <CASCADE>
        <SIZE>0</SIZE>
    </CASCADE>
</STREAM>
)";

std::string ReplaceVersionNumbers(const std::string& templateXml,
                                  uint32_t major = ETHOSN_COMMAND_STREAM_VERSION_MAJOR,
                                  uint32_t minor = ETHOSN_COMMAND_STREAM_VERSION_MINOR,
                                  uint32_t patch = ETHOSN_COMMAND_STREAM_VERSION_PATCH)
{
    std::string s = templateXml;
    s             = ethosn::utils::ReplaceAll(s, "%VERSION_MAJOR%", std::to_string(major));
    s             = ethosn::utils::ReplaceAll(s, "%VERSION_MINOR%", std::to_string(minor));
    s             = ethosn::utils::ReplaceAll(s, "%VERSION_PATCH%", std::to_string(patch));

    return s;
}

}    // namespace

TEST_CASE("XmlToBinary-BinaryToXml")
{
    std::string xmlStr = ReplaceVersionNumbers(g_XmlStr);

    std::stringstream inputXml(xmlStr);
    XmlParser xmlParser(inputXml);

    std::stringstream intermediateBinary;
    xmlParser.WriteBinary(intermediateBinary);

    intermediateBinary.seekg(0);

    BinaryParser binaryParser(intermediateBinary);
    std::stringstream outputXml;
    binaryParser.WriteXml(outputXml, 75);

    // Remove spaces since they can be different
    std::string inputString = inputXml.str();
    inputString.erase(std::remove(inputString.begin(), inputString.end(), ' '), inputString.end());

    std::string outputString = outputXml.str();
    outputString.erase(std::remove(outputString.begin(), outputString.end(), ' '), outputString.end());

    // compare the strings with no white spaces
    REQUIRE(inputString == outputString);
}

std::string g_BindingTableXmlStr =
    R"(<?xml version="1.0" encoding="utf-8"?>
<BIND>
  <BUFFER>
    <ID>0</ID>
    <ADDRESS>0x60100000</ADDRESS>
    <SIZE>2560</SIZE>
  </BUFFER>
  <BUFFER>
    <ID>1</ID>
    <ADDRESS>0x60100a00</ADDRESS>
    <SIZE>1488</SIZE>
  </BUFFER>
  <BUFFER>
    <ID>2</ID>
    <ADDRESS>0x60101000</ADDRESS>
    <SIZE>4096</SIZE>
  </BUFFER>
  <BUFFER>
    <ID>3</ID>
    <ADDRESS>0x60102000</ADDRESS>
    <SIZE>4096</SIZE>
  </BUFFER>
</BIND>
)";

// Test that Binding Table is correctly extracted when inference address is 16B aligned
TEST_CASE("ExtractBindingTableFromCMMBufferCountWord1")
{
    const std::string cmmSnippet = "00003540: 00003554 00003554 00000000 00000000\n"
                                   "00003550: 00000000 00000000 00000000 00000000\n"
                                   "00003560: 00000000 00000000 00000000 00000000\n"
                                   "60000000: 60000010 00000001 00000000 00000000\n"
                                   "60000010: 00000004 60100000 00000000 00000a00\n"
                                   "60000020: 60100a00 00000000 000005d0 60101000\n"
                                   "60000030: 00000000 00001000 60102000 00000000\n"
                                   "60000040: 00001000 00000000 00000000 00000000\n";

    std::stringstream input;
    std::stringstream output;
    input << cmmSnippet;
    CMMParser(input).ExtractBTFromCMM(output);

    // Remove spaces since they can be different
    std::string outputString = output.str();
    outputString.erase(std::remove(outputString.begin(), outputString.end(), ' '), outputString.end());

    g_BindingTableXmlStr.erase(std::remove(g_BindingTableXmlStr.begin(), g_BindingTableXmlStr.end(), ' '),
                               g_BindingTableXmlStr.end());

    // Compare the strings with no white spaces
    REQUIRE(g_BindingTableXmlStr == outputString);
}

// Test that Binding Table is correctly extracted when inference address is second word on the line
TEST_CASE("ExtractBindingTableFromCMMBufferCountWord2")
{

    const std::string cmmSnippet = "00003540: 00003554 00003554 00000000 00000000\n"
                                   "00003550: 00000000 00000000 00000000 00000000\n"
                                   "00003560: 00000000 00000000 00000000 00000000\n"
                                   "60000000: 60000014 00000001 00000000 00000000\n"
                                   "60000010: 00000000 00000004 60100000 00000000\n"
                                   "60000020: 00000a00 60100a00 00000000 000005d0\n"
                                   "60000030: 60101000 00000000 00001000 60102000\n"
                                   "60000040: 00000000 00001000 00000000 00000000\n";

    std::stringstream input;
    std::stringstream output;
    input << cmmSnippet;
    CMMParser(input).ExtractBTFromCMM(output);

    // Remove spaces since they can be different
    std::string outputString = output.str();
    outputString.erase(std::remove(outputString.begin(), outputString.end(), ' '), outputString.end());

    g_BindingTableXmlStr.erase(std::remove(g_BindingTableXmlStr.begin(), g_BindingTableXmlStr.end(), ' '),
                               g_BindingTableXmlStr.end());

    // Compare the strings with no white spaces
    REQUIRE(g_BindingTableXmlStr == outputString);
}

// Test that Binding Table is correctly extracted when inference address is third word on the line
TEST_CASE("ExtractBindingTableFromCMMBufferCountWord3")
{

    const std::string cmmSnippet = "00003540: 00003554 00003554 00000000 00000000\n"
                                   "00003550: 00000000 00000000 00000000 00000000\n"
                                   "00003560: 00000000 00000000 00000000 00000000\n"
                                   "60000000: 60000018 00000001 00000000 00000000\n"
                                   "60000010: 00000000 00000000 00000004 60100000\n"
                                   "60000020: 00000000 00000a00 60100a00 00000000\n"
                                   "60000030: 000005d0 60101000 00000000 00001000\n"
                                   "60000040: 60102000 00000000 00001000 00000000\n";

    std::stringstream input;
    std::stringstream output;
    input << cmmSnippet;
    CMMParser(input).ExtractBTFromCMM(output);

    // Remove spaces since they can be different
    std::string outputString = output.str();
    outputString.erase(std::remove(outputString.begin(), outputString.end(), ' '), outputString.end());

    g_BindingTableXmlStr.erase(std::remove(g_BindingTableXmlStr.begin(), g_BindingTableXmlStr.end(), ' '),
                               g_BindingTableXmlStr.end());

    // Compare the strings with no white spaces
    REQUIRE(g_BindingTableXmlStr == outputString);
}

// Test that Binding Table is correctly extracted when inference address is last word on the line
TEST_CASE("ExtractBindingTableFromCMMBufferCountWord4")
{

    const std::string cmmSnippet = "00003540: 00003554 00003554 00000000 00000000\n"
                                   "00003550: 00000000 00000000 00000000 00000000\n"
                                   "00003560: 00000000 00000000 00000000 00000000\n"
                                   "60000000: 6000001C 00000001 00000000 00000000\n"
                                   "60000010: 00000000 00000000 00000000 00000004\n"
                                   "60000020: 60100000 00000000 00000a00 60100a00\n"
                                   "60000030: 00000000 000005d0 60101000 00000000\n"
                                   "60000040: 00001000 60102000 00000000 00001000\n";

    std::stringstream input;
    std::stringstream output;
    input << cmmSnippet;
    CMMParser(input).ExtractBTFromCMM(output);

    // Remove spaces since they can be different
    std::string outputString = output.str();
    outputString.erase(std::remove(outputString.begin(), outputString.end(), ' '), outputString.end());

    g_BindingTableXmlStr.erase(std::remove(g_BindingTableXmlStr.begin(), g_BindingTableXmlStr.end(), ' '),
                               g_BindingTableXmlStr.end());

    // Compare the strings with no white spaces
    REQUIRE(g_BindingTableXmlStr == outputString);
}

// Test that Command Stream is correctly extracted
TEST_CASE("ExtractCommandStreamFromCMM")
{
    std::string commandStreamXml =
        R"(<?xml version="1.0" encoding="utf-8"?>
           <STREAM VERSION_MAJOR="%VERSION_MAJOR%" VERSION_MINOR="%VERSION_MINOR%" VERSION_PATCH="%VERSION_PATCH%"><!--Command0-->
             <SECTION>
               <TYPE>SISO</TYPE>
             </SECTION>
             <!--Command1-->
             <OPERATION_MCE_PLE>
               <INPUT_INFO>
                 <DATA_TYPE>U8</DATA_TYPE>
                 <DATA_FORMAT>NHWCB</DATA_FORMAT>
                 <TENSOR_SHAPE>1 16 16 16</TENSOR_SHAPE>
                 <SUPERTENSOR_SHAPE>0 0 0 0</SUPERTENSOR_SHAPE>
                 <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
                 <STRIPE_SHAPE>1 16 16 16</STRIPE_SHAPE>
                 <TILE_SHAPE>1000 1 1 1</TILE_SHAPE>
                 <DRAM_BUFFER_ID>2</DRAM_BUFFER_ID>
                 <SRAM_OFFSET>0x0</SRAM_OFFSET>
                 <ZERO_POINT>0</ZERO_POINT>
                 <DATA_LOCATION>DRAM</DATA_LOCATION>
               </INPUT_INFO>
               <WEIGHT_INFO>
                 <DATA_TYPE>U8</DATA_TYPE>
                 <DATA_FORMAT>WEIGHT_STREAM</DATA_FORMAT>
                 <TENSOR_SHAPE>3 3 16 16</TENSOR_SHAPE>
                 <SUPERTENSOR_SHAPE>0 0 0 0</SUPERTENSOR_SHAPE>
                 <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
                 <STRIPE_SHAPE>3 3 16 16</STRIPE_SHAPE>
                 <TILE_SHAPE>1 1 1 1</TILE_SHAPE>
                 <DRAM_BUFFER_ID>0</DRAM_BUFFER_ID>
                 <SRAM_OFFSET>0x200</SRAM_OFFSET>
                 <ZERO_POINT>128</ZERO_POINT>
                 <DATA_LOCATION>DRAM</DATA_LOCATION>
               </WEIGHT_INFO>
               <WEIGHTS_METADATA_BUFFER_ID>10</WEIGHTS_METADATA_BUFFER_ID>
               <OUTPUT_INFO>
                 <DATA_TYPE>U8</DATA_TYPE>
                 <DATA_FORMAT>NHWCB</DATA_FORMAT>
                 <TENSOR_SHAPE>1 16 16 16</TENSOR_SHAPE>
                 <SUPERTENSOR_SHAPE>1 16 16 16</SUPERTENSOR_SHAPE>
                 <SUPERTENSOR_OFFSET>0 0 0 0</SUPERTENSOR_OFFSET>
                 <STRIPE_SHAPE>1 16 16 16</STRIPE_SHAPE>
                 <TILE_SHAPE>1000 1 1 1</TILE_SHAPE>
                 <DRAM_BUFFER_ID>3</DRAM_BUFFER_ID>
                 <SRAM_OFFSET>0x100</SRAM_OFFSET>
                 <ZERO_POINT>100</ZERO_POINT>
                 <DATA_LOCATION>DRAM</DATA_LOCATION>
               </OUTPUT_INFO>
               <SRAM_CONFIG>
                 <ALLOCATION_STRATEGY>STRATEGY_1</ALLOCATION_STRATEGY>
               </SRAM_CONFIG>
               <BLOCK_CONFIG>
                 <BLOCK_WIDTH>16</BLOCK_WIDTH>
                 <BLOCK_HEIGHT>16</BLOCK_HEIGHT>
               </BLOCK_CONFIG>
               <MCE_OP_INFO>
                 <STRIDE_X>1</STRIDE_X>
                 <STRIDE_Y>1</STRIDE_Y>
                 <PAD_TOP>1</PAD_TOP>
                 <PAD_LEFT>1</PAD_LEFT>
                 <UNINTERLEAVED_INPUT_SHAPE>1 16 16 16</UNINTERLEAVED_INPUT_SHAPE>
                 <OUTPUT_SHAPE>1 16 16 16</OUTPUT_SHAPE>
                 <OUTPUT_STRIPE_SHAPE>1 16 16 16</OUTPUT_STRIPE_SHAPE>
                 <OPERATION>CONVOLUTION</OPERATION>
                 <ALGO>DIRECT</ALGO>
                 <ACTIVATION_MIN>100</ACTIVATION_MIN>
                 <ACTIVATION_MAX>255</ACTIVATION_MAX>
                 <UPSAMPLE_TYPE>OFF</UPSAMPLE_TYPE>
               </MCE_OP_INFO>
               <PLE_OP_INFO>
                 <CE_SRAM>0x0</CE_SRAM>
                 <PLE_SRAM>0x0</PLE_SRAM>
                 <OPERATION>PASSTHROUGH</OPERATION>
                 <RESCALE_MULTIPLIER0>0</RESCALE_MULTIPLIER0>
                 <RESCALE_SHIFT0>0</RESCALE_SHIFT0>
                 <RESCALE_MULTIPLIER1>0</RESCALE_MULTIPLIER1>
                 <RESCALE_SHIFT1>0</RESCALE_SHIFT1>
               </PLE_OP_INFO>
             </OPERATION_MCE_PLE>
             <!--Command2-->
             <FENCE />
             <!--Command3-->
             <DUMP_DRAM>
               <DRAM_BUFFER_ID>3</DRAM_BUFFER_ID>
               <FILENAME>1_16_16_16_CommandStream_Operation_0_OutputModel_NHWCB.hex</FILENAME>
             </DUMP_DRAM>
             <!--Command4-->
             <DUMP_SRAM>
               <PREFIX>output_ce</PREFIX>
             </DUMP_SRAM>
           </STREAM>
           )";
    commandStreamXml = ReplaceVersionNumbers(commandStreamXml);

    std::stringstream inputXml(commandStreamXml);
    const XmlParser xmlParser(inputXml);
    const std::vector<uint32_t>& commandStreamBinary = xmlParser.GetCommandStreamBuffer().GetData();

    std::stringstream cmmSnippet;

    cmmSnippet << "00003540: 00003554 00003554 00000000 00000000\n"
                  "00003550: 00000000 00000000 00000000 00000000\n"
                  "00003560: 00000000 00000000 00000000 00000000\n"
                  "60000000: 60000010 00000001 00000000 00000000\n";

    cmmSnippet << std::hex << std::setfill('0');
    cmmSnippet << "60000010: 00000001 60001000 00000000" << commandStreamBinary.size() * sizeof(commandStreamBinary[0])
               << "\n";

    size_t addr = 0x60001000;
    for (uint32_t wordIdx = 0; wordIdx < commandStreamBinary.size(); wordIdx += 4)
    {
        cmmSnippet << std::setw(8) << addr << ":";
        for (uint32_t i = 0; i < 4; ++i)
        {
            cmmSnippet << " " << std::setw(8)
                       << ((wordIdx + i) < commandStreamBinary.size() ? commandStreamBinary[wordIdx + i] : 0);
        }
        cmmSnippet << std::endl;

        addr += 16;
    }

    cmmSnippet.seekg(0);

    std::stringstream output;
    CMMParser(cmmSnippet).ExtractCSFromCMM(output, false);

    // Remove spaces since they can be different
    std::string outputString = output.str();
    outputString.erase(std::remove(outputString.begin(), outputString.end(), ' '), outputString.end());

    commandStreamXml.erase(std::remove(commandStreamXml.begin(), commandStreamXml.end(), ' '), commandStreamXml.end());

    // Compare the strings with no white spaces
    REQUIRE(commandStreamXml == outputString);
}

TEST_CASE("XmlParser incorrect version")
{
    using Catch::Matchers::Message;

    GIVEN("An XML command stream with an unsupported version")
    {
        std::string commandStreamXml =
            R"(<?xml version="1.0" encoding="utf-8"?>
           <STREAM VERSION_MAJOR="%VERSION_MAJOR%" VERSION_MINOR="%VERSION_MINOR%" VERSION_PATCH="%VERSION_PATCH%">
           </STREAM>
           )";
        commandStreamXml = ReplaceVersionNumbers(commandStreamXml, ETHOSN_COMMAND_STREAM_VERSION_MAJOR + 1, 0, 0);

        WHEN("Attempting to parse the file")
        {
            THEN("An exception is thrown")
            {
                std::stringstream inputXml(commandStreamXml);
                REQUIRE_THROWS_MATCHES(XmlParser(inputXml), ParseException, Message("Unsupported version"));
            }
        }
    }
}
