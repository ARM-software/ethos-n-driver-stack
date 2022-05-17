//
// Copyright © 2018-2022 Arm Limited.
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
)"
    R"(<!--Command12-->
    <CASCADE>
        <NUM_AGENTS>6</NUM_AGENTS>
        <AGENT>
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
            <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
            <SCHEDULE_DEPENDENCY>
                <RELATIVE_AGENT_ID>150</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>1</OTHER>
                    <SELF>2</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </SCHEDULE_DEPENDENCY>
            <READ_DEPENDENCY>
                <RELATIVE_AGENT_ID>17</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>1</OTHER>
                    <SELF>2</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </READ_DEPENDENCY>
            <READ_DEPENDENCY>
                <RELATIVE_AGENT_ID>5</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>2</SELF>
                </INNER_RATIO>
                <BOUNDARY>8</BOUNDARY>
            </READ_DEPENDENCY>
            <WRITE_DEPENDENCY>
                <RELATIVE_AGENT_ID>5</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>1</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>2</BOUNDARY>
            </WRITE_DEPENDENCY>
        </AGENT>
        <AGENT>
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
            </IFM_STREAMER>
            <NUM_STRIPES_TOTAL>96</NUM_STRIPES_TOTAL>
            <SCHEDULE_DEPENDENCY>
                <RELATIVE_AGENT_ID>117</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>1</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>1</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>2</BOUNDARY>
            </SCHEDULE_DEPENDENCY>
            <READ_DEPENDENCY>
                <RELATIVE_AGENT_ID>5</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>2</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>2</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </READ_DEPENDENCY>
            <READ_DEPENDENCY>
                <RELATIVE_AGENT_ID>11</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </READ_DEPENDENCY>
            <WRITE_DEPENDENCY>
                <RELATIVE_AGENT_ID>12</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </WRITE_DEPENDENCY>
        </AGENT>
        <AGENT>
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
            <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
            <SCHEDULE_DEPENDENCY>
                <RELATIVE_AGENT_ID>5</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </SCHEDULE_DEPENDENCY>
            <READ_DEPENDENCY>
                <RELATIVE_AGENT_ID>5</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </READ_DEPENDENCY>
            <READ_DEPENDENCY>
                <RELATIVE_AGENT_ID>5</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </READ_DEPENDENCY>
            <WRITE_DEPENDENCY>
                <RELATIVE_AGENT_ID>5</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </WRITE_DEPENDENCY>
        </AGENT>
        <AGENT>
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
                    <0>
                        <WIDTH>2</WIDTH>
                        <HEIGHT>2</HEIGHT>
                    </0>
                    <1>
                        <WIDTH>2</WIDTH>
                        <HEIGHT>1</HEIGHT>
                    </1>
                    <2>
                        <WIDTH>1</WIDTH>
                        <HEIGHT>2</HEIGHT>
                    </2>
                    <3>
                        <WIDTH>1</WIDTH>
                        <HEIGHT>1</HEIGHT>
                    </3>
                </FILTER_SHAPE>
                <PADDING>
                    <0>
                        <LEFT>12</LEFT>
                        <TOP>15</TOP>
                    </0>
                    <1>
                        <LEFT>15</LEFT>
                        <TOP>12</TOP>
                    </1>
                    <2>
                        <LEFT>0</LEFT>
                        <TOP>8</TOP>
                    </2>
                    <3>
                        <LEFT>8</LEFT>
                        <TOP>0</TOP>
                    </3>
                </PADDING>
                <IFM_DELTA_DEFAULT>
                    <0>
                        <WIDTH>3</WIDTH>
                        <HEIGHT>-3</HEIGHT>
                    </0>
                    <1>
                        <WIDTH>-3</WIDTH>
                        <HEIGHT>3</HEIGHT>
                    </1>
                    <2>
                        <WIDTH>2</WIDTH>
                        <HEIGHT>-2</HEIGHT>
                    </2>
                    <3>
                        <WIDTH>-2</WIDTH>
                        <HEIGHT>2</HEIGHT>
                    </3>
                </IFM_DELTA_DEFAULT>
                <IFM_DELTA_EDGE>
                    <0>
                        <WIDTH>1</WIDTH>
                        <HEIGHT>-2</HEIGHT>
                    </0>
                    <1>
                        <WIDTH>-2</WIDTH>
                        <HEIGHT>1</HEIGHT>
                    </1>
                    <2>
                        <WIDTH>1</WIDTH>
                        <HEIGHT>1</HEIGHT>
                    </2>
                    <3>
                        <WIDTH>-1</WIDTH>
                        <HEIGHT>-1</HEIGHT>
                    </3>
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
            </MCE_SCHEDULER>
            <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
        </AGENT>
)"
    R"(<AGENT>
            <PLE_LOADER>
                <PLE_KERNEL_ID>SIGMOID_16X8_1_S</PLE_KERNEL_ID>
                <SRAM_ADDR>4096</SRAM_ADDR>
            </PLE_LOADER>
            <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
            <SCHEDULE_DEPENDENCY>
                <RELATIVE_AGENT_ID>5</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </SCHEDULE_DEPENDENCY>
            <READ_DEPENDENCY>
                <RELATIVE_AGENT_ID>5</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </READ_DEPENDENCY>
            <WRITE_DEPENDENCY>
                <RELATIVE_AGENT_ID>5</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </WRITE_DEPENDENCY>
        </AGENT>
        <AGENT>
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
            <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
            <SCHEDULE_DEPENDENCY>
                <RELATIVE_AGENT_ID>5</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </SCHEDULE_DEPENDENCY>
            <READ_DEPENDENCY>
                <RELATIVE_AGENT_ID>6</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </READ_DEPENDENCY>
            <READ_DEPENDENCY>
                <RELATIVE_AGENT_ID>7</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </READ_DEPENDENCY>
            <WRITE_DEPENDENCY>
                <RELATIVE_AGENT_ID>8</RELATIVE_AGENT_ID>
                <OUTER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </OUTER_RATIO>
                <INNER_RATIO>
                    <OTHER>2</OTHER>
                    <SELF>1</SELF>
                </INNER_RATIO>
                <BOUNDARY>4</BOUNDARY>
            </WRITE_DEPENDENCY>
        </AGENT>
    </CASCADE>
    <!--Command13-->
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
