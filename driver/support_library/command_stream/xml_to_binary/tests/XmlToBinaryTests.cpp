//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../BinaryParser.hpp"
#include "../CMMParser.hpp"

#include <ethosn_utils/Strings.hpp>

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>
#include <ethosn_command_stream/cascading/CommandStream.hpp>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
using namespace ethosn::command_stream;
namespace
{

const std::string g_XmlStr =
    R"(<?xml version="1.0" encoding="utf-8"?>
<STREAM VERSION_MAJOR="%VERSION_MAJOR%" VERSION_MINOR="%VERSION_MINOR%" VERSION_PATCH="%VERSION_PATCH%">
    <!-- Command 0 -->
    <SECTION>
        <TYPE>SISO</TYPE>
    </SECTION>
    <!-- Command 1 -->
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
            <UPSAMPLE_EDGE_MODE_ROW>DROP</UPSAMPLE_EDGE_MODE_ROW>
            <UPSAMPLE_EDGE_MODE_COL>GENERATE</UPSAMPLE_EDGE_MODE_COL>
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
    <!-- Command 2 -->
    <DELAY>
        <VALUE>3</VALUE>
    </DELAY>
    <!-- Command 3 -->
    <FENCE/>
    <!-- Command 4 -->
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
            <UPSAMPLE_EDGE_MODE_ROW>GENERATE</UPSAMPLE_EDGE_MODE_ROW>
            <UPSAMPLE_EDGE_MODE_COL>GENERATE</UPSAMPLE_EDGE_MODE_COL>
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
    <!-- Command 5 -->
    <FENCE/>
    <!-- Command 6 -->
    <DUMP_DRAM>
        <DRAM_BUFFER_ID>2</DRAM_BUFFER_ID>
        <FILENAME>OutputModel_NHWCB.hex</FILENAME>
    </DUMP_DRAM>
    <!-- Command 7 -->
    <DUMP_SRAM>
        <PREFIX>output_ce</PREFIX>
    </DUMP_SRAM>
    <!-- Command 8 -->
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
    <!-- Command 9 -->
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
    <!-- Command 10 -->
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
    </OPERATION_CONVERT>)"
    R"(
    <!-- Command 11 -->
    <CASCADE>
        <AGENTS>
            <!-- Agent 0 -->
            <AGENT>
                <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
                <WGT_STREAMER>
                    <BUFFER_ID>3</BUFFER_ID>
                    <METADATA_BUFFER_ID>128</METADATA_BUFFER_ID>
                    <TILE>
                        <BASE_ADDR>32</BASE_ADDR>
                        <NUM_SLOTS>2</NUM_SLOTS>
                        <SLOT_SIZE>1024</SLOT_SIZE>
                    </TILE>
                    <NUM_STRIPES>
                        <OFM_CHANNELS>4</OFM_CHANNELS>
                        <IFM_CHANNELS>2</IFM_CHANNELS>
                    </NUM_STRIPES>
                    <STRIPE_ID_STRIDES>
                        <OFM_CHANNELS>2</OFM_CHANNELS>
                        <IFM_CHANNELS>1</IFM_CHANNELS>
                    </STRIPE_ID_STRIDES>
                </WGT_STREAMER>
            </AGENT>
            <!-- Agent 1 -->
            <AGENT>
                <NUM_STRIPES_TOTAL>96</NUM_STRIPES_TOTAL>
                <IFM_STREAMER>
                    <DRAM_OFFSET>512</DRAM_OFFSET>
                    <BUFFER_ID>3</BUFFER_ID>
                    <DATA_TYPE>NHWC</DATA_TYPE>
                    <FCAF_INFO>
                        <ZERO_POINT>0</ZERO_POINT>
                        <SIGNED_ACTIVATION>0</SIGNED_ACTIVATION>
                    </FCAF_INFO>
                    <TILE>
                        <BASE_ADDR>512</BASE_ADDR>
                        <NUM_SLOTS>2</NUM_SLOTS>
                        <SLOT_SIZE>512</SLOT_SIZE>
                    </TILE>
                    <DFLT_STRIPE_SIZE>
                        <HEIGHT>8</HEIGHT>
                        <WIDTH>4</WIDTH>
                        <CHANNELS>1</CHANNELS>
                    </DFLT_STRIPE_SIZE>
                    <EDGE_STRIPE_SIZE>
                        <HEIGHT>4</HEIGHT>
                        <WIDTH>4</WIDTH>
                        <CHANNELS>1</CHANNELS>
                    </EDGE_STRIPE_SIZE>
                    <SUPERTENSOR_SIZE_IN_CELLS>
                        <WIDTH>1</WIDTH>
                        <CHANNELS>2</CHANNELS>
                    </SUPERTENSOR_SIZE_IN_CELLS>
                    <NUM_STRIPES>
                        <HEIGHT>512</HEIGHT>
                        <WIDTH>128</WIDTH>
                        <CHANNELS>8</CHANNELS>
                    </NUM_STRIPES>
                    <STRIPE_ID_STRIDES>
                        <HEIGHT>4</HEIGHT>
                        <WIDTH>1</WIDTH>
                        <CHANNELS>2</CHANNELS>
                    </STRIPE_ID_STRIDES>
                    <PACKED_BOUNDARY_THICKNESS>
                        <LEFT>5</LEFT>
                        <TOP>6</TOP>
                        <RIGHT>7</RIGHT>
                        <BOTTOM>8</BOTTOM>
                    </PACKED_BOUNDARY_THICKNESS>
                    <IS_EXTRA_PACKED_BOUNDARY_DATA_ON_RIGHT_EDGE>1</IS_EXTRA_PACKED_BOUNDARY_DATA_ON_RIGHT_EDGE>
                    <IS_EXTRA_PACKED_BOUNDARY_DATA_ON_BOTTOM_EDGE>0</IS_EXTRA_PACKED_BOUNDARY_DATA_ON_BOTTOM_EDGE>
                </IFM_STREAMER>
            </AGENT>
            <!-- Agent 2 -->
            <AGENT>
                <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
                <OFM_STREAMER>
                    <DRAM_OFFSET>512</DRAM_OFFSET>
                    <BUFFER_ID>0</BUFFER_ID>
                    <DATA_TYPE>NHWC</DATA_TYPE>
                    <FCAF_INFO>
                        <ZERO_POINT>0</ZERO_POINT>
                        <SIGNED_ACTIVATION>0</SIGNED_ACTIVATION>
                    </FCAF_INFO>
                    <TILE>
                        <BASE_ADDR>0</BASE_ADDR>
                        <NUM_SLOTS>0</NUM_SLOTS>
                        <SLOT_SIZE>0</SLOT_SIZE>
                    </TILE>
                    <DFLT_STRIPE_SIZE>
                        <HEIGHT>8</HEIGHT>
                        <WIDTH>8</WIDTH>
                        <CHANNELS>8</CHANNELS>
                    </DFLT_STRIPE_SIZE>
                    <EDGE_STRIPE_SIZE>
                        <HEIGHT>8</HEIGHT>
                        <WIDTH>8</WIDTH>
                        <CHANNELS>8</CHANNELS>
                    </EDGE_STRIPE_SIZE>
                    <SUPERTENSOR_SIZE_IN_CELLS>
                        <WIDTH>8</WIDTH>
                        <CHANNELS>8</CHANNELS>
                    </SUPERTENSOR_SIZE_IN_CELLS>
                    <NUM_STRIPES>
                        <HEIGHT>8</HEIGHT>
                        <WIDTH>8</WIDTH>
                        <CHANNELS>8</CHANNELS>
                    </NUM_STRIPES>
                    <STRIPE_ID_STRIDES>
                        <HEIGHT>8</HEIGHT>
                        <WIDTH>8</WIDTH>
                        <CHANNELS>8</CHANNELS>
                    </STRIPE_ID_STRIDES>
                </OFM_STREAMER>
            </AGENT>)"
    R"(
            <!-- Agent 3 -->
            <AGENT>
                <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
                <MCE_SCHEDULER>
                    <IFM_TILE>
                        <BASE_ADDR>0</BASE_ADDR>
                        <NUM_SLOTS>0</NUM_SLOTS>
                        <SLOT_SIZE>0</SLOT_SIZE>
                    </IFM_TILE>
                    <WGT_TILE>
                        <BASE_ADDR>0</BASE_ADDR>
                        <NUM_SLOTS>0</NUM_SLOTS>
                        <SLOT_SIZE>0</SLOT_SIZE>
                    </WGT_TILE>
                    <BLOCK_SIZE>
                        <HEIGHT>0</HEIGHT>
                        <WIDTH>0</WIDTH>
                    </BLOCK_SIZE>
                    <DFLT_STRIPE_SIZE>
                        <OFM_HEIGHT>8</OFM_HEIGHT>
                        <OFM_WIDTH>8</OFM_WIDTH>
                        <OFM_CHANNELS>8</OFM_CHANNELS>
                        <IFM_CHANNELS>8</IFM_CHANNELS>
                    </DFLT_STRIPE_SIZE>
                    <EDGE_STRIPE_SIZE>
                        <OFM_HEIGHT>8</OFM_HEIGHT>
                        <OFM_WIDTH>8</OFM_WIDTH>
                        <OFM_CHANNELS>8</OFM_CHANNELS>
                        <IFM_CHANNELS>8</IFM_CHANNELS>
                    </EDGE_STRIPE_SIZE>
                    <NUM_STRIPES>
                        <OFM_HEIGHT>8</OFM_HEIGHT>
                        <OFM_WIDTH>8</OFM_WIDTH>
                        <OFM_CHANNELS>8</OFM_CHANNELS>
                        <IFM_CHANNELS>8</IFM_CHANNELS>
                    </NUM_STRIPES>
                    <STRIPE_ID_STRIDES>
                        <OFM_HEIGHT>8</OFM_HEIGHT>
                        <OFM_WIDTH>8</OFM_WIDTH>
                        <OFM_CHANNELS>8</OFM_CHANNELS>
                        <IFM_CHANNELS>8</IFM_CHANNELS>
                    </STRIPE_ID_STRIDES>
                    <CONV_STRIDE_XY>
                        <X>2</X>
                        <Y>2</Y>
                    </CONV_STRIDE_XY>
                    <IFM_ZERO_POINT>-2</IFM_ZERO_POINT>
                    <IS_IFM_SIGNED>1</IS_IFM_SIGNED>
                    <IS_OFM_SIGNED>0</IS_OFM_SIGNED>
                    <UPSAMPLE_TYPE>TRANSPOSE</UPSAMPLE_TYPE>
                    <UPSAMPLE_EDGE_MODE>
                        <ROW>DROP</ROW>
                        <COL>GENERATE</COL>
                    </UPSAMPLE_EDGE_MODE>
                    <MCE_OP_MODE>DEPTHWISE_CONVOLUTION</MCE_OP_MODE>
                    <ALGORITHM>WINOGRAD</ALGORITHM>
                    <IS_WIDE_FILTER>1</IS_WIDE_FILTER>
                    <IS_EXTRA_IFM_STRIPE_AT_RIGHT_EDGE>1</IS_EXTRA_IFM_STRIPE_AT_RIGHT_EDGE>
                    <IS_EXTRA_IFM_STRIPE_AT_BOTTOM_EDGE>1</IS_EXTRA_IFM_STRIPE_AT_BOTTOM_EDGE>
                    <IS_PACKED_BOUNDARY_X>1</IS_PACKED_BOUNDARY_X>
                    <IS_PACKED_BOUNDARY_Y>1</IS_PACKED_BOUNDARY_Y>
                    <FILTER_SHAPE>
                        <VALUE_0>
                            <WIDTH>2</WIDTH>
                            <HEIGHT>2</HEIGHT>
                        </VALUE_0>
                        <VALUE_1>
                            <WIDTH>2</WIDTH>
                            <HEIGHT>1</HEIGHT>
                        </VALUE_1>
                        <VALUE_2>
                            <WIDTH>1</WIDTH>
                            <HEIGHT>2</HEIGHT>
                        </VALUE_2>
                        <VALUE_3>
                            <WIDTH>1</WIDTH>
                            <HEIGHT>1</HEIGHT>
                        </VALUE_3>
                    </FILTER_SHAPE>
                    <PADDING>
                        <VALUE_0>
                            <LEFT>12</LEFT>
                            <TOP>15</TOP>
                        </VALUE_0>
                        <VALUE_1>
                            <LEFT>15</LEFT>
                            <TOP>12</TOP>
                        </VALUE_1>
                        <VALUE_2>
                            <LEFT>0</LEFT>
                            <TOP>8</TOP>
                        </VALUE_2>
                        <VALUE_3>
                            <LEFT>8</LEFT>
                            <TOP>0</TOP>
                        </VALUE_3>
                    </PADDING>
                    <IFM_DELTA_DEFAULT>
                        <VALUE_0>
                            <WIDTH>3</WIDTH>
                            <HEIGHT>-3</HEIGHT>
                        </VALUE_0>
                        <VALUE_1>
                            <WIDTH>-3</WIDTH>
                            <HEIGHT>3</HEIGHT>
                        </VALUE_1>
                        <VALUE_2>
                            <WIDTH>2</WIDTH>
                            <HEIGHT>-2</HEIGHT>
                        </VALUE_2>
                        <VALUE_3>
                            <WIDTH>-2</WIDTH>
                            <HEIGHT>2</HEIGHT>
                        </VALUE_3>
                    </IFM_DELTA_DEFAULT>
                    <IFM_DELTA_ONE_FROM_EDGE>
                        <VALUE_0>
                            <WIDTH>4</WIDTH>
                            <HEIGHT>-4</HEIGHT>
                        </VALUE_0>
                        <VALUE_1>
                            <WIDTH>-4</WIDTH>
                            <HEIGHT>4</HEIGHT>
                        </VALUE_1>
                        <VALUE_2>
                            <WIDTH>5</WIDTH>
                            <HEIGHT>-5</HEIGHT>
                        </VALUE_2>
                        <VALUE_3>
                            <WIDTH>-5</WIDTH>
                            <HEIGHT>5</HEIGHT>
                        </VALUE_3>
                    </IFM_DELTA_ONE_FROM_EDGE>
                    <IFM_DELTA_EDGE>
                        <VALUE_0>
                            <WIDTH>1</WIDTH>
                            <HEIGHT>-2</HEIGHT>
                        </VALUE_0>
                        <VALUE_1>
                            <WIDTH>-2</WIDTH>
                            <HEIGHT>1</HEIGHT>
                        </VALUE_1>
                        <VALUE_2>
                            <WIDTH>1</WIDTH>
                            <HEIGHT>1</HEIGHT>
                        </VALUE_2>
                        <VALUE_3>
                            <WIDTH>-1</WIDTH>
                            <HEIGHT>-1</HEIGHT>
                        </VALUE_3>
                    </IFM_DELTA_EDGE>
                    <IFM_STRIPE_SHAPE_DEFAULT>
                        <WIDTH>10</WIDTH>
                        <HEIGHT>11</HEIGHT>
                    </IFM_STRIPE_SHAPE_DEFAULT>
                    <IFM_STRIPE_SHAPE_EDGE>
                        <WIDTH>5</WIDTH>
                        <HEIGHT>6</HEIGHT>
                    </IFM_STRIPE_SHAPE_EDGE>
                    <RELU_ACTIV>
                        <MIN>-3</MIN>
                        <MAX>2</MAX>
                    </RELU_ACTIV>
                    <PLE_KERNEL_ID>DOWNSAMPLE_2X2_16X16_1</PLE_KERNEL_ID>
                </MCE_SCHEDULER>
            </AGENT>
            <!-- Agent 4 -->
            <AGENT>
                <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
                <PLE_LOADER>
                    <PLE_KERNEL_ID>SIGMOID_16X8_1_S</PLE_KERNEL_ID>
                    <SRAM_ADDR>4096</SRAM_ADDR>
                </PLE_LOADER>
            </AGENT>)"
    R"(
            <!-- Agent 5 -->
            <AGENT>
                <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
                <PLE_SCHEDULER>
                    <OFM_TILE>
                        <BASE_ADDR>0</BASE_ADDR>
                        <NUM_SLOTS>0</NUM_SLOTS>
                        <SLOT_SIZE>0</SLOT_SIZE>
                    </OFM_TILE>
                    <OFM_ZERO_POINT>3</OFM_ZERO_POINT>
                    <DFLT_STRIPE_SIZE>
                        <HEIGHT>8</HEIGHT>
                        <WIDTH>8</WIDTH>
                        <CHANNELS>8</CHANNELS>
                    </DFLT_STRIPE_SIZE>
                    <EDGE_STRIPE_SIZE>
                        <HEIGHT>8</HEIGHT>
                        <WIDTH>8</WIDTH>
                        <CHANNELS>8</CHANNELS>
                    </EDGE_STRIPE_SIZE>
                    <NUM_STRIPES>
                        <HEIGHT>8</HEIGHT>
                        <WIDTH>8</WIDTH>
                        <CHANNELS>8</CHANNELS>
                    </NUM_STRIPES>
                    <STRIPE_ID_STRIDES>
                        <HEIGHT>8</HEIGHT>
                        <WIDTH>8</WIDTH>
                        <CHANNELS>8</CHANNELS>
                    </STRIPE_ID_STRIDES>
                    <INPUT_MODE>MCE_ONE_OG</INPUT_MODE>
                    <PLE_KERNEL_ID>DOWNSAMPLE_2X2_16X16_1</PLE_KERNEL_ID>
                    <PLE_KERNEL_SRAM_ADDR>4096</PLE_KERNEL_SRAM_ADDR>
                    <IFM_TILE_0>
                        <BASE_ADDR>0</BASE_ADDR>
                        <NUM_SLOTS>0</NUM_SLOTS>
                        <SLOT_SIZE>0</SLOT_SIZE>
                    </IFM_TILE_0>
                    <IFM_INFO_0>
                        <ZERO_POINT>0</ZERO_POINT>
                        <MULTIPLIER>1</MULTIPLIER>
                        <SHIFT>2</SHIFT>
                    </IFM_INFO_0>
                    <IFM_TILE_1>
                        <BASE_ADDR>0</BASE_ADDR>
                        <NUM_SLOTS>0</NUM_SLOTS>
                        <SLOT_SIZE>0</SLOT_SIZE>
                    </IFM_TILE_1>
                    <IFM_INFO_1>
                        <ZERO_POINT>0</ZERO_POINT>
                        <MULTIPLIER>1</MULTIPLIER>
                        <SHIFT>2</SHIFT>
                    </IFM_INFO_1>
                </PLE_SCHEDULER>
            </AGENT>
        </AGENTS>
        <DMA_RD_COMMANDS>
            <!-- DmaRd Command 0 -->
            <COMMAND>
                <TYPE>LoadIfmStripe</TYPE>
                <!-- Agent type is WGT_STREAMER -->
                <AGENT_ID>0</AGENT_ID>
                <STRIPE_ID>0</STRIPE_ID>
            </COMMAND>
        </DMA_RD_COMMANDS>
        <DMA_WR_COMMANDS>
            <!-- DmaWr Command 0 -->
            <COMMAND>
                <TYPE>StoreOfmStripe</TYPE>
                <!-- Agent type is OFM_STREAMER -->
                <AGENT_ID>2</AGENT_ID>
                <STRIPE_ID>3</STRIPE_ID>
            </COMMAND>
        </DMA_WR_COMMANDS>
        <MCE_COMMANDS>
            <!-- Mce Command 0 -->
            <COMMAND>
                <TYPE>StartMceStripe</TYPE>
                <!-- Agent type is WGT_STREAMER -->
                <AGENT_ID>0</AGENT_ID>
                <STRIPE_ID>0</STRIPE_ID>
            </COMMAND>
        </MCE_COMMANDS>
        <PLE_COMMANDS>
            <!-- Ple Command 0 -->
            <COMMAND>
                <TYPE>WaitForAgent</TYPE>
                <!-- Agent type is WGT_STREAMER -->
                <AGENT_ID>0</AGENT_ID>
                <STRIPE_ID>0</STRIPE_ID>
            </COMMAND>
        </PLE_COMMANDS>
    </CASCADE>
    <!-- Command 12 -->
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
</STREAM>
)";

std::array<char, 128> convertCharsToArray(int size, const char name[])
{
    std::array<char, 128> nameArr{};
    for (int i = 0; i < size; ++i)
    {
        nameArr[i] = name[i];
    }
    return nameArr;
}

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
    Section conv1x1comm0 = { /* SectionType = */ SectionType::SISO };
    McePle conv1x1comm1  = { /* InputInfo = */
                            TensorInfo{
                                /* DataType = */ DataType::U8,
                                /* DataFormat = */ DataFormat::NHWCB,
                                /* TensorShape = */ TensorShape{ { 1U, 32U, 32U, 96U } },
                                /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                /* StripeShape = */ TensorShape{ { 1U, 32U, 32U, 96U } },
                                /* TileSize = */ uint32_t{ 1U },
                                /* DramBufferId = */ 0U,
                                /* SramOffset = */ 0U,
                                /* ZeroPoint = */ int16_t{ 0 },
                                /* DataLocation = */ DataLocation::DRAM,
                            },
                            /* WeightInfo = */
                            TensorInfo{
                                /* DataType = */ DataType::U8,
                                /* DataFormat = */ DataFormat::WEIGHT_STREAM,
                                /* TensorShape = */ TensorShape{ { 1U, 1U, 96U, 32U } },
                                /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                /* StripeShape = */ TensorShape{ { 1U, 1U, 96U, 32U } },
                                /* TileSize = */ uint32_t{ 1 },
                                /* DramBufferId = */ 1U,
                                /* SramOffset = */ 6144U,
                                /* ZeroPoint = */ int16_t{ 0 },
                                /* DataLocation = */ DataLocation::DRAM,
                            },
                            /* WeightMetadataBufferId = */ 10U,
                            /* OutputInfo = */
                            TensorInfo{
                                /* DataType = */ DataType::U8,
                                /* DataFormat = */ DataFormat::NHWCB,
                                /* TensorShape = */ TensorShape{ { 1U, 32U, 32U, 32U } },
                                /* SupertensorShape = */ TensorShape{ { 1U, 32U, 32U, 32U } },
                                /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                /* StripeShape = */ TensorShape{ { 1U, 32U, 32U, 32U } },
                                /* TileSize = */ uint32_t{ 1U },
                                /* DramBufferId = */ 2U,
                                /* SramOffset = */ 6400U,
                                /* ZeroPoint = */ int16_t{ 0 },
                                /* DataLocation = */ DataLocation::DRAM,
                            },
                            /* SramConfig = */
                            SramConfig{
                                /* AllocationStrategy = */ SramAllocationStrategy::STRATEGY_0,
                            },
                            /* BlockConfig = */
                            BlockConfig{
                                /* BlockWidth = */ 16U,
                                /* BlockHeight = */ 16U,
                            },
                            /* MceData = */
                            MceData{
                                /* StrideX = */ MceStrideConfig{ 1U, 1U },
                                /* PadTop = */ 0U,
                                /* PadLeft = */ 0U,
                                /* UninterleavedInputShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                /* OutputShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                /* OutputStripeShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                /* OutputZeroPoint = */ int16_t{ 0 },
                                /* UpsampleType = */ UpsampleType::OFF,
                                /* UpsampleEdgeModeRow = */ UpsampleEdgeMode::DROP,
                                /* UpsampleEdgeModeCol = */ UpsampleEdgeMode::GENERATE,
                                /* Operation = */ MceOperation::CONVOLUTION,
                                /* Algo = */ MceAlgorithm::DIRECT,
                                /* ActivationMin = */ uint8_t{ 118 },
                                /* ActivationMax = */ uint8_t{ 255 },
                            },
                            /* PleData = */
                            PleData{
                                /* CeSram = */ uint32_t{ 0 },
                                /* PleSram = */ uint32_t{ 0 },
                                /* Operation = */ PleOperation::LEAKY_RELU,
                                /* RescaleMultiplier0 = */ uint16_t{ 0 },
                                /* RescaleShift0 = */ uint16_t{ 0 },
                                /* RescaleMultiplier1 = */ uint16_t{ 0 },
                                /* RescaleShift1 = */ uint16_t{ 0 },
                            }
    };

    Delay conv1x1comm2 = { /* Value = */ uint32_t{ 3 } };

    Fence conv1x1comm3 = {};

    McePle conv1x1comm4 = { /* InputInfo = */
                            TensorInfo{ /* DataType = */ DataType::U8,
                                        /* DataFormat = */ DataFormat::NHWCB,
                                        /* TensorShape = */ TensorShape{ { 1U, 8U, 8U, 512U } },
                                        /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                        /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                        /* StripeShape = */ TensorShape{ { 1U, 8U, 8U, 128U } },
                                        /* TileSize = */ uint32_t{ 1U },
                                        /* DramBufferId = */ 2U,
                                        /* SramOffset = */ 0U,
                                        /* ZeroPoint = */ int16_t{ 0 },
                                        /* DataLocation = */ DataLocation::DRAM },
                            /* WeightInfo = */
                            TensorInfo{ /* DataType = */ DataType::U8,
                                        /* DataFormat = */ DataFormat::WEIGHT_STREAM,
                                        /* TensorShape = */ TensorShape{ { 1U, 1U, 1U, 32768U } },
                                        /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                        /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                        /* StripeShape = */ TensorShape{ { 1U, 1U, 1U, 1024U } },
                                        /* TileSize = */ uint32_t{ 1000U },
                                        /* DramBufferId = */ 3U,
                                        /* SramOffset = */ 34816U,
                                        /* ZeroPoint = */ int16_t{ 0 },
                                        /* DataLocation = */ DataLocation::DRAM },
                            /* WeightMetadataBufferId = */ 15U,
                            /* OutputInfo = */
                            TensorInfo{ /* DataType = */ DataType::U8,
                                        /* DataFormat = */ DataFormat::NHWCB,
                                        /* TensorShape = */ TensorShape{ { 1U, 8U, 8U, 32U } },
                                        /* SupertensorShape = */ TensorShape{ { 1U, 8U, 8U, 32U } },
                                        /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                        /* StripeSize = */ TensorShape{ { 1U, 8U, 8U, 8U } },
                                        /* TileSize = */ uint32_t{ 1U },
                                        /* DramBufferId = */ 4U,
                                        /* SramOffset = */ 32768U,
                                        /* ZeroPoint = */ int16_t{ 0 },
                                        /* DataLocation = */ DataLocation::DRAM },
                            /* SramConfig = */
                            SramConfig{
                                /* AllocationStrategy = */ SramAllocationStrategy::STRATEGY_1,
                            },
                            /* BlockConfig = */
                            BlockConfig{
                                /* BlockWidth = */ 8U,
                                /* BlockHeight = */ 8U,
                            },
                            /* MceData = */
                            MceData{
                                /* StrideX = */ MceStrideConfig{ 1U, 1U },
                                /* PadTop = */ 0U,
                                /* PadLeft = */ 0U,
                                /* UninterleavedInputShape = */ TensorShape{ { 1U, 32U, 32U, 2U } },
                                /* OutputShape = */ TensorShape{ { 1U, 32U, 32U, 2U } },
                                /* OutputStripeShape = */ TensorShape{ { 1U, 32U, 32U, 2U } },
                                /* OutputZeroPoint = */ int16_t{ 0 },
                                /* UpsampleType = */ UpsampleType::OFF,
                                /* UpsampleEdgeModeRow = */ UpsampleEdgeMode::GENERATE,
                                /* UpsampleEdgeModeCol = */ UpsampleEdgeMode::GENERATE,
                                /* Operation = */ MceOperation::FULLY_CONNECTED,
                                /* Algo = */ MceAlgorithm::DIRECT,
                                /* ActivationMin = */ uint8_t{ 0 },
                                /* ActivationMax = */ uint8_t{ 255 },
                            },
                            /* PleData = */
                            PleData{ /* CeSram = */ uint32_t{ 0 },
                                     /* PleSram = */ uint32_t{ 0 },
                                     /* Operation = */ PleOperation::PASSTHROUGH,
                                     /* RescaleMultiplier0 = */ uint16_t{ 0 },
                                     /* RescaleShift0 = */ uint16_t{ 0 },
                                     /* RescaleMultiplier1 = */ uint16_t{ 0 },
                                     /* RescaleShift1 = */ uint16_t{ 0 } }
    };

    Fence conv1x1comm5 = {};

    DumpDram conv1x1comm6 = { /* DramBufferId = */ uint32_t{ 2 },
                              /* Filename = */ Filename(convertCharsToArray(22, "OutputModel_NHWCB.hex")) };

    DumpSram conv1x1comm7 = { /* Prefix */ Filename(convertCharsToArray(10, "output_ce")) };

    PleOnly conv1x1comm8 = { /* NumInputInfos */ int32_t{ 2 },
                             /* InputInfo = */
                             TensorInfo{ /* DataType = */ DataType::U8,
                                         /* DataFormat = */ DataFormat::NHWCB,
                                         /* TensorShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                         /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                         /* StripeShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* TileSize = */ uint32_t{ 1000U },
                                         /* DramBufferId = */ 0U,
                                         /* SramOffset = */ 0U,
                                         /* ZeroPoint = */ int16_t{ 0 },
                                         /* DataLocation = */ DataLocation::DRAM },
                             /* InputInfo2 = */
                             TensorInfo{ /* DataType = */ DataType::U8,
                                         /* DataFormat = */ DataFormat::NHWCB,
                                         /* TensorShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                         /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                         /* StripeShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* TileSize = */ uint32_t{ 1000U },
                                         /* DramBufferId = */ uint32_t{ 1U },
                                         /* SramOffset = */ uint32_t{ 4096U },
                                         /* ZeroPoint = */ int16_t{ 0 },
                                         /* DataLocation = */ DataLocation::DRAM },
                             /* OutputInfo = */
                             TensorInfo{ /* DataType = */ DataType::U8,
                                         /* DataFormat = */ DataFormat::NHWCB,
                                         /* TensorShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* SupertensorShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                         /* StripeShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* TileSize = */ uint32_t{ 1000U },
                                         /* DramBufferId = */ 2U,
                                         /* SramOffset = */ 8192U,
                                         /* ZeroPoint = */ int16_t{ 0 },
                                         /* DataLocation = */ DataLocation::DRAM

                             },
                             /* SramConfig = */
                             SramConfig{
                                 /* AllocationStrategy = */ SramAllocationStrategy::STRATEGY_0,
                             },
                             /* PleOpInfo = */
                             PleData{
                                 /* CeSram = */ uint32_t{ 512 },
                                 /* PleSram = */ uint32_t{ 0 },
                                 /* Operation = */ PleOperation::ADDITION,
                                 /* RescaleMultiplier0 = */ uint16_t{ 0 },
                                 /* RescaleShift0 = */ uint16_t{ 0 },
                                 /* RescaleMultiplier1 = */ uint16_t{ 0 },
                                 /* RescaleShift1 = */ uint16_t{ 0 },
                             } };

    PleOnly conv1x1comm9 = { /* NumInputInfos */ int32_t{ 1 },
                             /* InputInfo = */
                             TensorInfo{ /* DataType = */ DataType::U8,
                                         /* DataFormat = */ DataFormat::NHWCB,
                                         /* TensorShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                         /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                         /* StripeShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* TileSize = */ uint32_t{ 1000U },
                                         /* DramBufferId = */ 0U,
                                         /* SramOffset = */ 0U,
                                         /* ZeroPoint = */ int16_t{ 0 },
                                         /* DataLocation = */ DataLocation::DRAM },
                             /* InputInfo1 = */
                             TensorInfo{ /* DataType = */ DataType::U8,
                                         /* DataFormat = */ DataFormat::NHWCB,
                                         /* TensorShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                         /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                         /* StripeShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* TileSize = */ uint32_t{ 1000U },
                                         /* DramBufferId = */ 0U,
                                         /* SramOffset = */ 0U,
                                         /* ZeroPoint = */ int16_t{ 0 },
                                         /* DataLocation = */ DataLocation::DRAM },
                             /* OutputInfo = */
                             TensorInfo{ /* DataType = */ DataType::U8,
                                         /* DataFormat = */ DataFormat::NHWCB,
                                         /* TensorShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* SupertensorShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                         /* StripeShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                                         /* TileSize = */ uint32_t{ 1000U },
                                         /* DramBufferId = */ 1U,
                                         /* SramOffset = */ 256U,
                                         /* ZeroPoint = */ int16_t{ 0 },
                                         /* DataLocation = */ DataLocation::DRAM },
                             /* SramConfig = */
                             SramConfig{
                                 /* AllocationStrategy = */ SramAllocationStrategy::STRATEGY_0,
                             },
                             /* PleOpInfo = */
                             PleData{
                                 /* CeSram = */ uint32_t{ 512 },
                                 /* PleSram = */ uint32_t{ 0 },
                                 /* Operation = */ PleOperation::PASSTHROUGH,
                                 /* RescaleMultiplier0 = */ uint16_t{ 0 },
                                 /* RescaleShift0 = */ uint16_t{ 0 },
                                 /* RescaleMultiplier1 = */ uint16_t{ 0 },
                                 /* RescaleShift1 = */ uint16_t{ 0 },
                             } };

    Convert conv1x1comm10 = { /* InputInfo = */
                              TensorInfo{ /* DataType = */ DataType::U8,
                                          /* DataFormat = */ DataFormat::NHWCB,
                                          /* TensorShape = */ TensorShape{ { 1U, 32U, 32U, 32U } },
                                          /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                          /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                          /* StripeShape = */ TensorShape{ { 1U, 16U, 32U, 32U } },
                                          /* TileSize = */ uint32_t{ 4000U },
                                          /* DramBufferId = */ 0U,
                                          /* SramOffset = */ 0U,
                                          /* ZeroPoint = */ int16_t{ 0 },
                                          /* DataLocation = */ DataLocation::DRAM },
                              /* OutputInfo = */
                              TensorInfo{ /* DataType = */ DataType::U8,
                                          /* DataFormat = */ DataFormat::NHWC,
                                          /* TensorShape = */ TensorShape{ { 1U, 32U, 32U, 32U } },
                                          /* SupertensorShape = */ TensorShape{ { 1U, 32U, 32U, 32U } },
                                          /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                          /* StripeShape = */ TensorShape{ { 1U, 16U, 32U, 32U } },
                                          /* TileSize = */ uint32_t{ 4000U },
                                          /* DramBufferId = */ 0U,
                                          /* SramOffset = */ 0U,
                                          /* ZeroPoint = */ int16_t{ 0 },
                                          /* DataLocation = */ DataLocation::DRAM }
    };

    Cascade conv1x1comm11;
    conv1x1comm11.TotalSize           = sizeof(Cascade) + 6 * sizeof(cascading::Agent) + 4 * sizeof(cascading::Command);
    conv1x1comm11.AgentsOffset        = sizeof(Cascade);
    conv1x1comm11.NumAgents           = 6;
    conv1x1comm11.DmaRdCommandsOffset = sizeof(Cascade) + 6 * sizeof(cascading::Agent) + 0 * sizeof(cascading::Command);
    conv1x1comm11.NumDmaRdCommands    = 1;
    conv1x1comm11.DmaWrCommandsOffset = sizeof(Cascade) + 6 * sizeof(cascading::Agent) + 1 * sizeof(cascading::Command);
    conv1x1comm11.NumDmaWrCommands    = 1;
    conv1x1comm11.MceCommandsOffset   = sizeof(Cascade) + 6 * sizeof(cascading::Agent) + 2 * sizeof(cascading::Command);
    conv1x1comm11.NumMceCommands      = 1;
    conv1x1comm11.PleCommandsOffset   = sizeof(Cascade) + 6 * sizeof(cascading::Agent) + 3 * sizeof(cascading::Command);
    conv1x1comm11.NumPleCommands      = 1;

    SpaceToDepth conv1x1comm12 = { /* InputInfo */
                                   TensorInfo{ /* DataType = */ DataType::U8,
                                               /* DataFormat = */ DataFormat::NHWC,
                                               /* TensorShape = */ TensorShape{ { 1U, 32U, 32U, 16U } },
                                               /* SupertensorShape = */ TensorShape{ { 1U, 32U, 32U, 16U } },
                                               /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                               /* StripeShape = */ TensorShape{ { 1U, 16U, 32U, 32U } },
                                               /* TileSize = */ uint32_t{ 4000U },
                                               /* DramBufferId = */ 0U,
                                               /* SramOffset = */ 0U,
                                               /* ZeroPoint = */ int16_t{ 0 },
                                               /* DataLocation = */ DataLocation::DRAM },
                                   /* OutputInfo */
                                   TensorInfo{
                                       /* DataType = */ DataType::U8,
                                       /* DataFormat = */ DataFormat::NHWC,
                                       /* TensorShape = */ TensorShape{ { 1U, 16U, 16U, 64U } },
                                       /* SupertensorShape = */ TensorShape{ { 1U, 16U, 16U, 64U } },
                                       /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                                       /* StripeShape = */ TensorShape{ { 1U, 16U, 32U, 32U } },
                                       /* TileSize = */ uint32_t{ 4000U },
                                       /* DramBufferId = */ 0U,
                                       /* SramOffset = */ 0U,
                                       /* ZeroPoint = */ int16_t{ 0 },
                                       /* DataLocation = */ DataLocation::DRAM,
                                   },
                                   /* UsedEmcs = */ uint32_t{ 8 },
                                   /* Intermediate1Size = */ uint32_t{ 1024 },
                                   /* Intermediate2Size = */ uint32_t{ 2048 }
    };

    /* Agent 0 = */
    cascading::Agent agent0 = {
        /* NumStripesTotal = */ uint16_t{ 64 },
        /* AgentData = */
        cascading::AgentData{ cascading::WgtS{ /* BufferId = */ uint16_t{ 3 },
                                               /* MetadataBufferId = */ uint16_t{ 128 },
                                               /* Tile = */
                                               cascading::Tile{ /* BaseAddress = */ uint32_t{ 32 },
                                                                /* NumSlots = */ uint16_t{ 2 },
                                                                /* SlotSize = */ uint32_t{ 1024 } },
                                               /* NumStripes = */
                                               cascading::WgtS::WorkSize{ /* OfmChannels = */ uint16_t{ 4 },
                                                                          /* IfmChannels = */ uint16_t{ 2 } },
                                               /* StripeIdStrides = */
                                               cascading::WgtS::WorkSize{ /* OfmChannels = */ uint16_t{ 2 },
                                                                          /* IfmChannels = */ uint16_t{ 1 } } } },
    };

    /* Agent 1 = */
    cascading::Agent agent1 = {
        /* NumStripesTotal = */ uint16_t{ 96 },
        /* AgentData = */
        cascading::AgentData{
            cascading::IfmS{ /* FmsData = */
                             cascading::FmSData{ /* DramOffset = */ uint32_t{ 512 },
                                                 /* BufferId = */ uint16_t{ 3 },
                                                 /* DataType = */ cascading::FmsDataType::NHWC,
                                                 /* FcafInfo = */
                                                 cascading::FcafInfo{ /* ZeroPoint = */ int16_t{ 0 },
                                                                      /* SignedActivation = */ false },
                                                 /* Tile = */
                                                 cascading::Tile{ /* BaseAddr = */ uint32_t{ 512 },
                                                                  /* NumSlots = */ uint16_t{ 2 },
                                                                  /* SlotSize = */ uint32_t{ 512 } },
                                                 /* DfltStripeSize = */
                                                 cascading::TensorSize<uint16_t>{ /* Height = */ uint16_t{ 8 },
                                                                                  /* Width = */ uint16_t{ 4 },
                                                                                  /* Channels = */ uint16_t{ 1 } },
                                                 /* EdgeStripeSize = */
                                                 cascading::TensorSize<uint16_t>{ /* Height = */ uint16_t{ 4 },
                                                                                  /* Width = */ uint16_t{ 4 },
                                                                                  /* Channels = */ uint16_t{ 1 } },
                                                 /* SupertensorSizeInCells = */
                                                 cascading::SupertensorSize<uint16_t>{ /* Width = */ uint16_t{ 1 },
                                                                                       /* Channels = */ uint16_t{ 2 } },
                                                 /* NumStripes = */
                                                 cascading::TensorSize<uint16_t>{ /* Height = */ uint16_t{ 512 },
                                                                                  /* Width = */ uint16_t{ 128 },
                                                                                  /* Channels = */ uint16_t{ 8 } },
                                                 /* StripeIdStrides = */
                                                 cascading::TensorSize<uint16_t>{ /* Height = */ uint16_t{ 4 },
                                                                                  /* Width = */ uint16_t{ 1 },
                                                                                  /* Channels = */ uint16_t{ 2 } } },
                             /* PackedBoundaryThickness = */
                             cascading::PackedBoundaryThickness{ /* Left = */ uint8_t{ 5 },
                                                                 /* Top = */ uint8_t{ 6 },
                                                                 /* Right = */ uint8_t{ 7 },
                                                                 /* Bottom = */ uint8_t{ 8 } },
                             /* isExtraPackedBoundaryDataOnRightEdge = */
                             uint8_t{ 1 },
                             /* isExtraPackedBoundaryDataOnBottomEdge = */
                             uint8_t{ 0 } } },
    };

    /* Agent 2 = */
    cascading::Agent agent2 = {
        /* NumStripesTotal = */ uint16_t{ 64 },
        /* AgentData = */
        cascading::AgentData{ cascading::OfmS{
            /* FmsData = */
            cascading::FmSData{ /* DramOffset = */ uint32_t{ 512 },
                                /* BufferId = */ uint16_t{ 0 },
                                /* DataType = */ cascading::FmsDataType::NHWC,
                                /* FcafInfo = */
                                cascading::FcafInfo{ /* ZeroPoint = */ int16_t{ 0 },
                                                     /* SignedActivation = */ false },
                                /* Tile = */
                                cascading::Tile{ /* BaseAddr = */ uint32_t{ 0 },
                                                 /* NumSlots = */ uint16_t{ 0 },
                                                 /* SlotSize = */ uint32_t{ 0 } },
                                /* DfltStripeSize = */
                                cascading::TensorSize<uint16_t>{ /* Height = */ uint16_t{ 8 },
                                                                 /* Width = */ uint16_t{ 8 },
                                                                 /* Channels = */ uint16_t{ 8 } },
                                /* EdgeStripeSize = */
                                cascading::TensorSize<uint16_t>{ /* Height = */ uint16_t{ 8 },
                                                                 /* Width = */ uint16_t{ 8 },
                                                                 /* Channels = */ uint16_t{ 8 } },
                                /* SupertensorSizeInCells = */
                                cascading::SupertensorSize<uint16_t>{ /* Width = */ uint16_t{ 8 },
                                                                      /* Channels = */ uint16_t{ 8 } },
                                /* NumStripes = */
                                cascading::TensorSize<uint16_t>{ /* Height = */ uint16_t{ 8 },
                                                                 /* Width = */ uint16_t{ 8 },
                                                                 /* Channels = */ uint16_t{ 8 } },
                                /* StripeIdStrides = */
                                cascading::TensorSize<uint16_t>{ /* Height = */ uint16_t{ 8 },
                                                                 /* Width = */ uint16_t{ 8 },
                                                                 /* Channels = */ uint16_t{ 8 } } } } },
    };

    /* Agent 3 = */
    cascading::Agent agent3 = {
        /* NumStripesTotal = */ uint16_t{ 64 },
        /* AgentData = */
        cascading::AgentData{
            /* MceScheduler = */
            cascading::MceS{ /* IfmTile = */
                             cascading::Tile{ /* BaseAddr = */ uint32_t{ 0 },
                                              /* NumSlots = */ uint16_t{ 0 },
                                              /* SlotSize = */ uint32_t{ 0 } },
                             /* WgtTile = */
                             cascading::Tile{ /* BaseAddr = */ uint32_t{ 0 },
                                              /* NumSlots = */ uint16_t{ 0 },
                                              /* SlotSize = */ uint32_t{ 0 } },
                             /* BlockSize = */
                             cascading::BlockSize{ /* Height = */ uint8_t{ 0 },
                                                   /* Width = */ uint8_t{ 0 } },
                             /* DfltStripeSize = */
                             cascading::MceS::WorkSize{
                                 /* OfmHeight = */ uint16_t{ 8 },
                                 /* OfmWidth = */ uint16_t{ 8 },
                                 /* OfmChannels = */ uint16_t{ 8 },
                                 /* IfmChannels = */ uint16_t{ 8 },
                             },
                             /* EdgeStripeSize = */
                             cascading::MceS::WorkSize{
                                 /* OfmHeight = */ uint16_t{ 8 },
                                 /* OfmWidth = */ uint16_t{ 8 },
                                 /* OfmChannels = */ uint16_t{ 8 },
                                 /* IfmChannels = */ uint16_t{ 8 },
                             },
                             /* NumStripes = */
                             cascading::MceS::WorkSize{
                                 /* OfmHeight = */ uint16_t{ 8 },
                                 /* OfmWidth = */ uint16_t{ 8 },
                                 /* OfmChannels = */ uint16_t{ 8 },
                                 /* IfmChannels = */ uint16_t{ 8 },
                             },
                             /* StripeIdStrides = */
                             cascading::MceS::WorkSize{
                                 /* OfmHeight = */ uint16_t{ 8 },
                                 /* OfmWidth = */ uint16_t{ 8 },
                                 /* OfmChannels = */ uint16_t{ 8 },
                                 /* IfmChannels = */ uint16_t{ 8 },
                             },
                             /* ConvStrideXy = */
                             cascading::StrideXy<uint8_t>{ /* X = */ uint8_t{ 2 },
                                                           /* Y= */ uint8_t{ 2 } },
                             /* IfmZeroPoint = */ int16_t{ -2 },
                             /* IsIfmSigned = */ uint8_t{ 1 },
                             /* IsOfmSigned = */ uint8_t{ 0 },
                             /* UpsampleType = */ cascading::UpsampleType::TRANSPOSE,
                             /* UpsampleEdgeMode = */
                             cascading::UpsampleEdgeModeType{ /* Row = */ cascading::UpsampleEdgeMode::DROP,
                                                              /* Column = */ cascading::UpsampleEdgeMode::GENERATE },
                             /* MceOpMode = */ cascading::MceOperation::DEPTHWISE_CONVOLUTION,
                             /* Algo = */ cascading::MceAlgorithm::WINOGRAD,
                             /* IsWideFilter = */ uint8_t{ 1 },
                             /* IsExtraIfmStripeAtRightEdge = */ uint8_t{ 1 },
                             /* IsExtraIfmStripeAtBottomEdge = */ uint8_t{ 1 },
                             /* IsPackedBoundaryX = */ uint8_t{ 1 },
                             /* IsPackedBoundaryY = */ uint8_t{ 1 },
                             /* FilterShape = */
                             std::array<cascading::FilterShape, static_cast<uint8_t>(4U)>{
                                 cascading::FilterShape{ /* Width = */ uint8_t{ 2 },
                                                         /* Height = */ uint8_t{ 2 } },
                                 cascading::FilterShape{ /* Width = */ uint8_t{ 2 },
                                                         /* Height = */ uint8_t{ 1 } },
                                 cascading::FilterShape{ /* Width = */ uint8_t{ 1 },
                                                         /* Height = */ uint8_t{ 2 } },
                                 cascading::FilterShape{ /* Width = */ uint8_t{ 1 },
                                                         /* Height = */ uint8_t{ 1 } } },
                             /* Padding = */
                             std::array<cascading::Padding, static_cast<uint8_t>(4U)>{
                                 cascading::Padding{ /* Left= */ uint8_t{ 12 },
                                                     /* Top = */ uint8_t{ 15 } },
                                 cascading::Padding{ /* Left= */ uint8_t{ 15 },
                                                     /* Top = */ uint8_t{ 12 } },
                                 cascading::Padding{ /* Left= */ uint8_t{ 0 },
                                                     /* Top = */ uint8_t{ 8 } },
                                 cascading::Padding{ /* Left= */ uint8_t{ 8 },
                                                     /* Top = */ uint8_t{ 0 } } },
                             /* IfmDeltaDefault = */
                             std::array<cascading::IfmDelta, static_cast<uint8_t>(4U)>{
                                 cascading::IfmDelta{ /* Width = */ int8_t{ 3 },
                                                      /*Height = */ int8_t{ -3 } },
                                 cascading::IfmDelta{ /* Width = */ int8_t{ -3 },
                                                      /*Height = */ int8_t{ 3 } },
                                 cascading::IfmDelta{ /* Width = */ int8_t{ 2 },
                                                      /*Height = */ int8_t{ -2 } },
                                 cascading::IfmDelta{ /* Width = */ int8_t{ -2 },
                                                      /*Height = */ int8_t{ 2 } } },
                             /* IfmDeltaOneFromEdge = */
                             std::array<cascading::IfmDelta, static_cast<uint8_t>(4U)>{
                                 cascading::IfmDelta{ /* Width = */ int8_t{ 4 },
                                                      /*Height = */ int8_t{ -4 } },
                                 cascading::IfmDelta{ /* Width = */ int8_t{ -4 },
                                                      /*Height = */ int8_t{ 4 } },
                                 cascading::IfmDelta{ /* Width = */ int8_t{ 5 },
                                                      /*Height = */ int8_t{ -5 } },
                                 cascading::IfmDelta{ /* Width = */ int8_t{ -5 },
                                                      /*Height = */ int8_t{ 5 } } },
                             /* IfmDeltaEdge = */
                             std::array<cascading::IfmDelta, static_cast<uint8_t>(4U)>{
                                 cascading::IfmDelta{ /* Width = */ int8_t{ 1 },
                                                      /*Height = */ int8_t{ -2 } },
                                 cascading::IfmDelta{ /* Width = */ int8_t{ -2 },
                                                      /*Height = */ int8_t{ 1 } },
                                 cascading::IfmDelta{ /* Width = */ int8_t{ 1 },
                                                      /*Height = */ int8_t{ 1 } },
                                 cascading::IfmDelta{ /* Width = */ int8_t{ -1 },
                                                      /*Height = */ int8_t{ -1 } } },
                             /* IfmStripeShapeDefault = */
                             cascading::IfmStripeShape{ /* Width = */ uint16_t{ 10 },
                                                        /* Height = */ uint16_t{ 11 } },
                             /* IfmStripeShapeEdge = */
                             cascading::IfmStripeShape{ /* Width = */ uint16_t{ 5 },
                                                        /* Height = */ uint16_t{ 6 } },
                             /* ReluActiv = */
                             cascading::ReluActivation{ /* Min = */ int16_t{ -3 },
                                                        /* Max = */ int16_t{ 2 } },
                             /* PleKernelId = */ cascading::PleKernelId::DOWNSAMPLE_2X2_16X16_1 } },
    };

    /* Agent 4 = */
    cascading::Agent agent4 = {
        /* NumStripesTotal = */ uint16_t{ 64 },
        /* AgentData = */
        cascading::AgentData{ /* PleLoader = */
                              cascading::PleL{ /* PleKernelId = */ cascading::PleKernelId::SIGMOID_16X8_1_S,
                                               /* SramAddr = */ uint32_t{ 4096 } } },
    };

    /* Agent 5 = */
    cascading::Agent agent5 = {
        /* NumStripesTotal = */ uint16_t{ 64 },
        /* AgentData = */
        cascading::AgentData{                  /* PleScheduler = */
                              cascading::PleS{ /* OfmTile = */
                                               cascading::Tile{ /* BaseAddr = */ uint32_t{ 0 },
                                                                /* NumSlots = */ uint16_t{ 0 },
                                                                /* SlotSize = */ uint32_t{ 0 } },
                                               /* OfmZeroPoint = */ int16_t{ 3 },
                                               /* DfltStripeSize = */
                                               cascading::TensorSize<uint16_t>{ /* Height = */ uint16_t{ 8 },
                                                                                /* Width = */ uint16_t{ 8 },
                                                                                /* Channels = */ uint16_t{ 8 } },
                                               /* EdgeStripeSize = */
                                               cascading::TensorSize<uint16_t>{ /* Height = */ uint16_t{ 8 },
                                                                                /* Width = */ uint16_t{ 8 },
                                                                                /* Channels = */ uint16_t{ 8 } },
                                               /* NumStripes = */
                                               cascading::TensorSize<uint16_t>{ /* Height = */ uint16_t{ 8 },
                                                                                /* Width = */ uint16_t{ 8 },
                                                                                /* Channels = */ uint16_t{ 8 } },
                                               /* StripeIdStrides = */
                                               cascading::TensorSize<uint16_t>{ /* Height = */ uint16_t{ 8 },
                                                                                /* Width = */ uint16_t{ 8 },
                                                                                /* Channels = */ uint16_t{ 8 } },
                                               /* InputMode = */ cascading::PleInputMode::MCE_ONE_OG,
                                               /* PleKernelId = */ cascading::PleKernelId::DOWNSAMPLE_2X2_16X16_1,
                                               /* PleKernelSramAddress = */ uint32_t{ 4096 },
                                               /* IfmTile0 = */
                                               cascading::Tile{ /* BaseAddr = */ uint32_t{ 0 },
                                                                /* NumSlots = */ uint16_t{ 0 },
                                                                /* SlotSize = */ uint32_t{ 0 } },
                                               /* IfmInfo0 = */
                                               cascading::PleIfmInfo{ /* ZeroPoint = */ int16_t{ 0 },
                                                                      /* Multiplier = */ uint16_t{ 1 },
                                                                      /* Shift = */ uint16_t{ 2 } },
                                               /* IfmTile1 = */
                                               cascading::Tile{ /* BaseAddr = */ uint32_t{ 0 },
                                                                /* NumSlots = */ uint16_t{ 0 },
                                                                /* SlotSize = */ uint32_t{ 0 } },
                                               /* IfmInfo1 = */
                                               cascading::PleIfmInfo{ /* ZeroPoint = */ int16_t{ 0 },
                                                                      /* Multiplier = */ uint16_t{ 1 },
                                                                      /* Shift = */ uint16_t{ 2 } } } },
    };

    cascading::Command cascadingCommand1 = { cascading::CommandType::LoadIfmStripe, 0, 0 };
    cascading::Command cascadingCommand2 = { cascading::CommandType::StoreOfmStripe, 2, 3 };
    cascading::Command cascadingCommand3 = { cascading::CommandType::StartMceStripe, 0, 0 };
    cascading::Command cascadingCommand4 = { cascading::CommandType::WaitForAgent, 0, 0 };

    std::string xmlStr = ReplaceVersionNumbers(g_XmlStr);
    std::stringstream inputXml(xmlStr);

    CommandStreamBuffer buffer;
    buffer.EmplaceBack(conv1x1comm0);
    buffer.EmplaceBack(conv1x1comm1);
    buffer.EmplaceBack(conv1x1comm2);
    buffer.EmplaceBack(conv1x1comm3);
    buffer.EmplaceBack(conv1x1comm4);
    buffer.EmplaceBack(conv1x1comm5);
    buffer.EmplaceBack(conv1x1comm6);
    buffer.EmplaceBack(conv1x1comm7);
    buffer.EmplaceBack(conv1x1comm8);
    buffer.EmplaceBack(conv1x1comm9);
    buffer.EmplaceBack(conv1x1comm10);
    buffer.EmplaceBack(conv1x1comm11);
    buffer.EmplaceBack(agent0);
    buffer.EmplaceBack(agent1);
    buffer.EmplaceBack(agent2);
    buffer.EmplaceBack(agent3);
    buffer.EmplaceBack(agent4);
    buffer.EmplaceBack(agent5);
    buffer.EmplaceBack(cascadingCommand1);
    buffer.EmplaceBack(cascadingCommand2);
    buffer.EmplaceBack(cascadingCommand3);
    buffer.EmplaceBack(cascadingCommand4);
    buffer.EmplaceBack(conv1x1comm12);
    const std::vector<uint32_t> commandStreamBinary = buffer.GetData();

    BinaryParser binaryParser(commandStreamBinary);
    std::stringstream outputXml;
    binaryParser.WriteXml(outputXml);

    std::string inputString  = inputXml.str();
    std::string outputString = outputXml.str();

    if (inputString != outputString)
    {
        std::ofstream expected("expected.txt");
        expected << inputString;
        std::ofstream actual("actual.txt");
        actual << outputString;
        FAIL("Strings don't match - see files expected.txt and actual.txt");
    }
}

std::string g_BindingTableXmlStr =
    R"(<?xml version="1.0" encoding="utf-8"?>
<BIND>
  <BUFFER>
    <ID>0</ID>
    <ADDRESS>0x60100000</ADDRESS>
    <SIZE>2560</SIZE>
    <TYPE>INPUT</TYPE>
  </BUFFER>
  <BUFFER>
    <ID>1</ID>
    <ADDRESS>0x60100a00</ADDRESS>
    <SIZE>1488</SIZE>
    <TYPE>INTERMEDIATE</TYPE>
  </BUFFER>
  <BUFFER>
    <ID>2</ID>
    <ADDRESS>0x60101000</ADDRESS>
    <SIZE>4096</SIZE>
    <TYPE>OUTPUT</TYPE>
  </BUFFER>
  <BUFFER>
    <ID>3</ID>
    <ADDRESS>0x60102000</ADDRESS>
    <SIZE>4096</SIZE>
    <TYPE>CONSTANT</TYPE>
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
                                   "60000020: 00000000 60100a00 00000000 000005d0\n"
                                   "60000030: 00000001 60101000 00000000 00001000\n"
                                   "60000040: 00000002 60102000 00000000 00001000\n"
                                   "60000050: 00000003 00000000 00000000 00000000\n";

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
                                   "60000020: 00000a00 00000000 60100a00 00000000\n"
                                   "60000030: 000005d0 00000001 60101000 00000000\n"
                                   "60000040: 00001000 00000002 60102000 00000000\n"
                                   "60000050: 00001000 00000003 00000000 00000000\n";

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
                                   "60000020: 00000000 00000a00 00000000 60100a00\n"
                                   "60000030: 00000000 000005d0 00000001 60101000\n"
                                   "60000040: 00000000 00001000 00000002 60102000\n"
                                   "60000050: 00000000 00001000 00000003 00000000\n";

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
                                   "60000020: 60100000 00000000 00000a00 00000000\n"
                                   "60000030: 60100a00 00000000 000005d0 00000001\n"
                                   "60000040: 60101000 00000000 00001000 00000002\n"
                                   "60000050: 60102000 00000000 00001000 00000003\n";

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
    McePle comm1 = { /* InputInfo = */
                     TensorInfo{
                         /* DataType = */ DataType::U8,
                         /* DataFormat = */ DataFormat::NHWCB,
                         /* TensorShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                         /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                         /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                         /* StripeShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                         /* TileSize = */ uint32_t{ 1000U },
                         /* DramBufferId = */ 2U,
                         /* SramOffset = */ 0U,
                         /* ZeroPoint = */ int16_t{ 0 },
                         /* DataLocation= */ DataLocation::DRAM,
                     },
                     /* WeightInfo = */
                     TensorInfo{
                         /* DataType = */ DataType::U8,
                         /* DataFormat = */ DataFormat::WEIGHT_STREAM,
                         /* TensorShape = */ TensorShape{ { 3U, 3U, 16U, 16U } },
                         /* SupertensorShape = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                         /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                         /* StripeShape = */ TensorShape{ { 3U, 3U, 16U, 16U } },
                         /* TileSize = */ uint32_t{ 1U },
                         /* DramBufferId = */ 0U,
                         /* SramOffset = */ 512U,
                         /* ZeroPoint = */ int16_t{ 128 },
                         /* DataLocation= */ DataLocation::DRAM,
                     },
                     /* WeightMetadataBufferId = */ 10U,
                     /* OutputInfo = */
                     TensorInfo{
                         /* DataType = */ DataType::U8,
                         /* DataFormat = */ DataFormat::NHWCB,
                         /* TensorShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                         /* SupertensorShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                         /* SupertensorOffset = */ TensorShape{ { 0U, 0U, 0U, 0U } },
                         /* StripeShape = */ TensorShape{ { 1U, 16U, 16U, 16U } },
                         /* TileSize = */ uint32_t{ 1000U },
                         /* DramBufferId = */ 3U,
                         /* SramOffset = */ 512U,
                         /* ZeroPoint = */ int16_t{ 100 },
                         /* DataLocation= */ DataLocation::DRAM,
                     },
                     /* SramConfig = */
                     SramConfig{
                         /* AllocationStrategy = */ SramAllocationStrategy::STRATEGY_1,
                     },
                     /* BlockConfig = */
                     BlockConfig{
                         /* BlockWidth = */ 16U,
                         /* BlockHeight = */ 16U,
                     },
                     /* MceData = */
                     MceData{
                         /* Stride = */ MceStrideConfig{ 1U, 1U },
                         /* PadTop = */ 1U,
                         /* PadLeft = */ 1U,
                         /* UninterleavedInputShape = */ TensorShape{ 1U, 16U, 16U, 16U },
                         /* OutputShape = */ TensorShape{ 1U, 16U, 16U, 16U },
                         /* OutputStripeShape = */ TensorShape{ 1U, 16U, 16U, 16U },
                         /* OutputZeroPoint = */ int16_t{ 0 },
                         /* UpsampleType = */ UpsampleType::OFF,
                         /* UpsampleEdgeModeRow = */ UpsampleEdgeMode::GENERATE,
                         /* UpsampleEdgeModeCol = */ UpsampleEdgeMode::GENERATE,
                         /* Operation = */ MceOperation::CONVOLUTION,
                         /* Algorithm = */ MceAlgorithm::DIRECT,
                         /* ActivationMin = */ uint8_t{ 100 },
                         /* ActivationMax = */ uint8_t{ 255 },
                     },
                     /* PleData = */
                     PleData{
                         /* CeSram = */ uint32_t{ 0 },
                         /* PleSram = */ uint32_t{ 0 },
                         /* PleOperation = */ PleOperation::PASSTHROUGH,
                         /* RescaleMultiplier0 = */ uint16_t{ 0 },
                         /* RescaleShift0 = */ uint16_t{ 0 },
                         /* RescaleMultiplier1 = */ uint16_t{ 0 },
                         /* RescaleShift1 = */ uint16_t{ 0 },
                     }
    };

    Fence comm2 = {};

    DumpDram comm3 = { /* DramBufferId = */ uint32_t{ 0 },
                       /* Filename = */ Filename(
                           convertCharsToArray(59, "1_16_16_16_CommandStream_Operation_0_OutputModel_NHWCB.hex")) };

    DumpSram comm4 = { /* Prefix = */ Filename(convertCharsToArray(10, "output_ce")) };

    CommandStreamBuffer buffer;
    buffer.EmplaceBack(comm1);
    buffer.EmplaceBack(comm2);
    buffer.EmplaceBack(comm3);
    buffer.EmplaceBack(comm4);
    const std::vector<uint32_t> commandStreamBinary = buffer.GetData();

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

    BinaryParser binaryParser(commandStreamBinary);
    std::stringstream outputXml;
    binaryParser.WriteXml(outputXml);

    // Remove spaces since they can be different
    std::string outputString = output.str();
    outputString.erase(std::remove(outputString.begin(), outputString.end(), ' '), outputString.end());

    std::string commandStreamXml = outputXml.str();
    commandStreamXml.erase(std::remove(commandStreamXml.begin(), commandStreamXml.end(), ' '), commandStreamXml.end());

    // Compare the strings with no white spaces
    REQUIRE(commandStreamXml == outputString);
}

// END OF XMLTOBINARYTESTS
