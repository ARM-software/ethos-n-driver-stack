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
                </WGT_STREAMER>
            </AGENT>
            <!-- Agent 1 -->
            <AGENT>
                <NUM_STRIPES_TOTAL>96</NUM_STRIPES_TOTAL>
                <IFM_STREAMER>
                    <BUFFER_ID>3</BUFFER_ID>
                    <DMA_COMP_CONFIG0>0x3534265</DMA_COMP_CONFIG0>
                    <DMA_STRIDE1>0x23424</DMA_STRIDE1>
                    <DMA_STRIDE2>0x213426</DMA_STRIDE2>
                </IFM_STREAMER>
            </AGENT>
            <!-- Agent 2 -->
            <AGENT>
                <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
                <OFM_STREAMER>
                    <BUFFER_ID>0</BUFFER_ID>
                    <DMA_COMP_CONFIG0>0x89679</DMA_COMP_CONFIG0>
                    <DMA_STRIDE1>0x12346</DMA_STRIDE1>
                    <DMA_STRIDE2>0x209347f</DMA_STRIDE2>
                </OFM_STREAMER>
            </AGENT>
            <!-- Agent 3 -->
            <AGENT>
                <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
                <MCE_SCHEDULER>
                    <MCE_OP_MODE>DEPTHWISE_CONVOLUTION</MCE_OP_MODE>
                    <PLE_KERNEL_ID>DOWNSAMPLE_2X2_16X16_1</PLE_KERNEL_ID>
                    <ACTIVATION_CONFIG>0x12348235</ACTIVATION_CONFIG>
                    <WIDE_KERNEL_CONTROL>0x87978</WIDE_KERNEL_CONTROL>
                    <FILTER>0x1234675</FILTER>
                    <IFM_ZERO_POINT>0x234235</IFM_ZERO_POINT>
                    <IFM_DEFAULT_SLOT_SIZE>0x234</IFM_DEFAULT_SLOT_SIZE>
                    <IFM_SLOT_STRIDE>0x8679</IFM_SLOT_STRIDE>
                    <STRIPE_BLOCK_CONFIG>0x1845768</STRIPE_BLOCK_CONFIG>
                    <DEPTHWISE_CONTROL>0x11234</DEPTHWISE_CONTROL>
                    <IFM_SLOT_BASE_ADDRESS>0x32442335</IFM_SLOT_BASE_ADDRESS>
                    <PLE_MCEIF_CONFIG>0x10098957</PLE_MCEIF_CONFIG>
                </MCE_SCHEDULER>
            </AGENT>
            <!-- Agent 4 -->
            <AGENT>
                <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
                <PLE_LOADER>
                    <PLE_KERNEL_ID>SIGMOID_16X8_1_S</PLE_KERNEL_ID>
                </PLE_LOADER>
            </AGENT>
            <!-- Agent 5 -->
            <AGENT>
                <NUM_STRIPES_TOTAL>64</NUM_STRIPES_TOTAL>
                <PLE_SCHEDULER>
                    <INPUT_MODE>MCE_ONE_OG</INPUT_MODE>
                    <PLE_KERNEL_ID>DOWNSAMPLE_2X2_16X16_1</PLE_KERNEL_ID>
                    <PLE_KERNEL_SRAM_ADDR>4096</PLE_KERNEL_SRAM_ADDR>
                </PLE_SCHEDULER>
            </AGENT>
        </AGENTS>)"
    R"(
        <DMA_RD_COMMANDS>
            <!-- DmaRd Command 0 -->
            <COMMAND>
                <TYPE>LoadIfmStripe</TYPE>
                <!-- Agent type is WGT_STREAMER -->
                <AGENT_ID>0</AGENT_ID>
                <STRIPE_ID>0</STRIPE_ID>
                <DMA_EXTRA_DATA>
                    <DRAM_OFFSET>0x123412</DRAM_OFFSET>
                    <SRAM_ADDR>0x6543</SRAM_ADDR>
                    <DMA_SRAM_STRIDE>0x2345</DMA_SRAM_STRIDE>
                    <DMA_STRIDE0>0x7995</DMA_STRIDE0>
                    <DMA_STRIDE3>0x23245</DMA_STRIDE3>
                    <DMA_CHANNELS>0x12345</DMA_CHANNELS>
                    <DMA_EMCS>0x989</DMA_EMCS>
                    <DMA_TOTAL_BYTES>0xfea</DMA_TOTAL_BYTES>
                    <DMA_CMD>0xa</DMA_CMD>
                    <IS_LAST_CHUNK>1</IS_LAST_CHUNK>
                </DMA_EXTRA_DATA>
            </COMMAND>
        </DMA_RD_COMMANDS>
        <DMA_WR_COMMANDS>
            <!-- DmaWr Command 0 -->
            <COMMAND>
                <TYPE>StoreOfmStripe</TYPE>
                <!-- Agent type is OFM_STREAMER -->
                <AGENT_ID>2</AGENT_ID>
                <STRIPE_ID>3</STRIPE_ID>
                <DMA_EXTRA_DATA>
                    <DRAM_OFFSET>0xabe</DRAM_OFFSET>
                    <SRAM_ADDR>0x6ee</SRAM_ADDR>
                    <DMA_SRAM_STRIDE>0xebbb5</DMA_SRAM_STRIDE>
                    <DMA_STRIDE0>0x79aa</DMA_STRIDE0>
                    <DMA_STRIDE3>0xdef</DMA_STRIDE3>
                    <DMA_CHANNELS>0xffeed</DMA_CHANNELS>
                    <DMA_EMCS>0xdd2</DMA_EMCS>
                    <DMA_TOTAL_BYTES>0xfa12a</DMA_TOTAL_BYTES>
                    <DMA_CMD>0x11a</DMA_CMD>
                    <IS_LAST_CHUNK>0</IS_LAST_CHUNK>
                </DMA_EXTRA_DATA>
            </COMMAND>
        </DMA_WR_COMMANDS>)"
    R"(
        <MCE_COMMANDS>
            <!-- Mce Command 0 -->
            <COMMAND>
                <TYPE>ProgramMceStripe</TYPE>
                <!-- Agent type is WGT_STREAMER -->
                <AGENT_ID>0</AGENT_ID>
                <STRIPE_ID>0</STRIPE_ID>
                <PROGRAM_MCE_EXTRA_DATA>
                    <MUL_ENABLE_CE0>
                        <OG0>0x45</OG0>
                        <OG1>0x46</OG1>
                        <OG2>0x47</OG2>
                        <OG3>0x48</OG3>
                    </MUL_ENABLE_CE0>
                    <MUL_ENABLE_CE1>
                        <OG0>0x49</OG0>
                        <OG1>0x50</OG1>
                        <OG2>0x51</OG2>
                        <OG3>0x52</OG3>
                    </MUL_ENABLE_CE1>
                    <MUL_ENABLE_CE2>
                        <OG0>0x53</OG0>
                        <OG1>0x54</OG1>
                        <OG2>0x55</OG2>
                        <OG3>0x56</OG3>
                    </MUL_ENABLE_CE2>
                    <MUL_ENABLE_CE3>
                        <OG0>0x57</OG0>
                        <OG1>0x58</OG1>
                        <OG2>0x59</OG2>
                        <OG3>0x60</OG3>
                    </MUL_ENABLE_CE3>
                    <MUL_ENABLE_CE4>
                        <OG0>0x61</OG0>
                        <OG1>0x62</OG1>
                        <OG2>0x63</OG2>
                        <OG3>0x64</OG3>
                    </MUL_ENABLE_CE4>
                    <MUL_ENABLE_CE5>
                        <OG0>0x65</OG0>
                        <OG1>0x66</OG1>
                        <OG2>0x67</OG2>
                        <OG3>0x68</OG3>
                    </MUL_ENABLE_CE5>
                    <MUL_ENABLE_CE6>
                        <OG0>0x69</OG0>
                        <OG1>0x70</OG1>
                        <OG2>0x71</OG2>
                        <OG3>0x72</OG3>
                    </MUL_ENABLE_CE6>
                    <MUL_ENABLE_CE7>
                        <OG0>0x73</OG0>
                        <OG1>0x74</OG1>
                        <OG2>0x75</OG2>
                        <OG3>0x76</OG3>
                    </MUL_ENABLE_CE7>
                    <IFM_ROW_STRIDE>0x3423</IFM_ROW_STRIDE>
                    <IFM_CONFIG1>0xaa8daa</IFM_CONFIG1>
                    <IFM_PAD_NUM0>
                        <IG0>0x45</IG0>
                        <IG1>0x48</IG1>
                        <IG2>0x45</IG2>
                        <IG3>0x48</IG3>
                    </IFM_PAD_NUM0>
                    <IFM_PAD_NUM1>
                        <IG0>0x41</IG0>
                        <IG1>0x61</IG1>
                        <IG2>0x41</IG2>
                        <IG3>0x61</IG3>
                    </IFM_PAD_NUM1>
                    <IFM_PAD_NUM2>
                        <IG0>0x42</IG0>
                        <IG1>0x61</IG1>
                        <IG2>0x42</IG2>
                        <IG3>0x61</IG3>
                    </IFM_PAD_NUM2>
                    <IFM_PAD_NUM3>
                        <IG0>0x45</IG0>
                        <IG1>0x6a</IG1>
                        <IG2>0x42</IG2>
                        <IG3>0x61</IG3>
                    </IFM_PAD_NUM3>
                    <WIDE_KERNEL_OFFSET>0x998765</WIDE_KERNEL_OFFSET>
                    <IFM_TOP_SLOTS>0xee31</IFM_TOP_SLOTS>
                    <IFM_MID_SLOTS>0xe56654</IFM_MID_SLOTS>
                    <IFM_BOTTOM_SLOTS>0xf787</IFM_BOTTOM_SLOTS>
                    <IFM_SLOT_PAD_CONFIG>0x897</IFM_SLOT_PAD_CONFIG>
                    <OFM_STRIPE_SIZE>0xbb6</OFM_STRIPE_SIZE>
                    <OFM_CONFIG>0xa455435</OFM_CONFIG>
                    <WEIGHT_BASE_ADDR_OG0>0x34587</WEIGHT_BASE_ADDR_OG0>
                    <WEIGHT_BASE_ADDR_OG1>0xa</WEIGHT_BASE_ADDR_OG1>
                    <WEIGHT_BASE_ADDR_OG2>0x342</WEIGHT_BASE_ADDR_OG2>
                    <WEIGHT_BASE_ADDR_OG3>0xb</WEIGHT_BASE_ADDR_OG3>
                    <IFM_CONFIG2_CE0>
                        <IG0>0x145</IG0>
                        <IG1>0x246</IG1>
                        <IG2>0x145</IG2>
                        <IG3>0x246</IG3>
                    </IFM_CONFIG2_CE0>
                    <IFM_CONFIG2_CE1>
                        <IG0>0x149</IG0>
                        <IG1>0x250</IG1>
                        <IG2>0x149</IG2>
                        <IG3>0x250</IG3>
                    </IFM_CONFIG2_CE1>
                    <IFM_CONFIG2_CE2>
                        <IG0>0x153</IG0>
                        <IG1>0x254</IG1>
                        <IG2>0x153</IG2>
                        <IG3>0x254</IG3>
                    </IFM_CONFIG2_CE2>
                    <IFM_CONFIG2_CE3>
                        <IG0>0x157</IG0>
                        <IG1>0x258</IG1>
                        <IG2>0x157</IG2>
                        <IG3>0x258</IG3>
                    </IFM_CONFIG2_CE3>
                    <IFM_CONFIG2_CE4>
                        <IG0>0x161</IG0>
                        <IG1>0x262</IG1>
                        <IG2>0x161</IG2>
                        <IG3>0x262</IG3>
                    </IFM_CONFIG2_CE4>
                    <IFM_CONFIG2_CE5>
                        <IG0>0x165</IG0>
                        <IG1>0x266</IG1>
                        <IG2>0x165</IG2>
                        <IG3>0x266</IG3>
                    </IFM_CONFIG2_CE5>
                    <IFM_CONFIG2_CE6>
                        <IG0>0x169</IG0>
                        <IG1>0x270</IG1>
                        <IG2>0x169</IG2>
                        <IG3>0x270</IG3>
                    </IFM_CONFIG2_CE6>
                    <IFM_CONFIG2_CE7>
                        <IG0>0x173</IG0>
                        <IG1>0x274</IG1>
                        <IG2>0x173</IG2>
                        <IG3>0x274</IG3>
                    </IFM_CONFIG2_CE7>
                    <NUM_BLOCKS_PROGRAMMED_FOR_MCE>0x80</NUM_BLOCKS_PROGRAMMED_FOR_MCE>
                </PROGRAM_MCE_EXTRA_DATA>
            </COMMAND>
            <!-- Mce Command 1 -->
            <COMMAND>
                <TYPE>StartMceStripe</TYPE>
                <!-- Agent type is WGT_STREAMER -->
                <AGENT_ID>0</AGENT_ID>
                <STRIPE_ID>0</STRIPE_ID>
                <START_MCE_EXTRA_DATA>
                    <CE_ENABLES>74666</CE_ENABLES>
                </START_MCE_EXTRA_DATA>
            </COMMAND>
        </MCE_COMMANDS>)"
    R"(
        <PLE_COMMANDS>
            <!-- Ple Command 0 -->
            <COMMAND>
                <TYPE>WaitForAgent</TYPE>
                <!-- Agent type is WGT_STREAMER -->
                <AGENT_ID>0</AGENT_ID>
                <STRIPE_ID>0</STRIPE_ID>
            </COMMAND>
            <!-- Ple Command 1 -->
            <COMMAND>
                <TYPE>StartPleStripe</TYPE>
                <!-- Agent type is WGT_STREAMER -->
                <AGENT_ID>0</AGENT_ID>
                <STRIPE_ID>0</STRIPE_ID>
                <START_PLE_EXTRA_DATA>
                    <SCRATCH0>0x125aa</SCRATCH0>
                    <SCRATCH1>0x126aa</SCRATCH1>
                    <SCRATCH2>0x127aa</SCRATCH2>
                    <SCRATCH3>0x128aa</SCRATCH3>
                    <SCRATCH4>0x129aa</SCRATCH4>
                    <SCRATCH5>0x130aa</SCRATCH5>
                    <SCRATCH6>0x131aa</SCRATCH6>
                    <SCRATCH7>0x132aa</SCRATCH7>
                </START_PLE_EXTRA_DATA>
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
        cascading::AgentData{ cascading::WgtS{
            /* BufferId = */ uint16_t{ 3 },
        } },
    };

    /* Agent 1 = */
    cascading::Agent agent1 = {
        /* NumStripesTotal = */ uint16_t{ 96 },
        /* AgentData = */
        cascading::AgentData{ cascading::IfmS{
            /* BufferId = */ uint16_t{ 3 },
            /* DMA_COMP_CONFIG0 = */ uint32_t{ 0x3534265 },
            /* DMA_STRIDE1 = */ uint32_t{ 0x23424 },
            /* DMA_STRIDE2 = */ uint32_t{ 0x213426 },
        } },
    };

    /* Agent 2 = */
    cascading::Agent agent2 = {
        /* NumStripesTotal = */ uint16_t{ 64 },
        /* AgentData = */
        cascading::AgentData{ cascading::OfmS{
            /* BufferId = */ uint16_t{ 0 },
            /* DMA_COMP_CONFIG0 = */ uint32_t{ 0x89679 },
            /* DMA_STRIDE1 = */ uint32_t{ 0x12346 },
            /* DMA_STRIDE2 = */ uint32_t{ 0x209347f },
        } },
    };

    /* Agent 3 = */
    cascading::Agent agent3 = {
        /* NumStripesTotal = */ uint16_t{ 64 },
        /* AgentData = */
        cascading::AgentData{ /* MceScheduler = */
                              cascading::MceS{
                                  /* MceOpMode = */ cascading::MceOperation::DEPTHWISE_CONVOLUTION,
                                  /* PleKernelId = */ cascading::PleKernelId::DOWNSAMPLE_2X2_16X16_1,
                                  /* ACTIVATION_CONFIG = */ uint32_t{ 0x12348235 },
                                  /* WIDE_KERNEL_CONTROL = */ uint32_t{ 0x87978 },
                                  /* FILTER = */ uint32_t{ 0x1234675 },
                                  /* IFM_ZERO_POINT = */ uint32_t{ 0x234235 },
                                  /* IFM_DEFAULT_SLOT_SIZE = */ uint32_t{ 0x234 },
                                  /* IFM_SLOT_STRIDE = */ uint32_t{ 0x8679 },
                                  /* STRIPE_BLOCK_CONFIG = */ uint32_t{ 0x1845768 },
                                  /* DEPTHWISE_CONTROL = */ uint32_t{ 0x11234 },
                                  /* IFM_SLOT_BASE_ADDRESS = */ uint32_t{ 0x32442335 },
                                  /* PLE_MCEIF_CONFIG = */ uint32_t{ 0x10098957 },
                              } },
    };

    /* Agent 4 = */
    cascading::Agent agent4 = {
        /* NumStripesTotal = */ uint16_t{ 64 },
        /* AgentData = */
        cascading::AgentData{ /* PleLoader = */
                              cascading::PleL{
                                  /* PleKernelId = */ cascading::PleKernelId::SIGMOID_16X8_1_S,
                              } },
    };

    /* Agent 5 = */
    cascading::Agent agent5 = {
        /* NumStripesTotal = */ uint16_t{ 64 },
        /* AgentData = */
        cascading::AgentData{ /* PleScheduler = */
                              cascading::PleS{
                                  /* InputMode = */ cascading::PleInputMode::MCE_ONE_OG,
                                  /* PleKernelId = */ cascading::PleKernelId::DOWNSAMPLE_2X2_16X16_1,
                                  /* PleKernelSramAddress = */ uint32_t{ 4096 },
                              } },
    };

    cascading::Command cascadingDmaRdCommand1 = { cascading::CommandType::LoadIfmStripe, 0, 0, 0 };
    cascading::Command cascadingDmaWrCommand1 = { cascading::CommandType::StoreOfmStripe, 2, 3, 0 };
    cascading::Command cascadingMceCommand1   = { cascading::CommandType::ProgramMceStripe, 0, 0, 0 };
    cascading::Command cascadingMceCommand2   = { cascading::CommandType::StartMceStripe, 0, 0, 0 };
    cascading::Command cascadingPleCommand1   = { cascading::CommandType::WaitForAgent, 0, 0, 0 };
    cascading::Command cascadingPleCommand2   = { cascading::CommandType::StartPleStripe, 0, 0, 0 };

    cascading::DmaExtraData cascadingDmaExtraData1 = {
        /* m_DramOffset = */ uint32_t{ 0x123412 },
        /* SRAM_ADDR = */ uint32_t{ 0x6543 },
        /* DMA_SRAM_STRIDE = */ uint32_t{ 0x2345 },
        /* DMA_STRIDE0 = */ uint32_t{ 0x7995 },
        /* DMA_STRIDE3 = */ uint32_t{ 0x23245 },
        /* DMA_CHANNELS = */ uint32_t{ 0x12345 },
        /* DMA_EMCS = */ uint32_t{ 0x989 },
        /* DMA_TOTAL_BYTES = */ uint32_t{ 0xfea },
        /* DMA_CMD = */ uint32_t{ 0xa },
        /* m_IsLastChunk = */ uint8_t{ 1 },
    };
    cascading::DmaExtraData cascadingDmaExtraData2 = {
        /* m_DramOffset = */ uint32_t{ 0xabe },
        /* SRAM_ADDR = */ uint32_t{ 0x6ee },
        /* DMA_SRAM_STRIDE = */ uint32_t{ 0xebbb5 },
        /* DMA_STRIDE0 = */ uint32_t{ 0x79aa },
        /* DMA_STRIDE3 = */ uint32_t{ 0xdef },
        /* DMA_CHANNELS = */ uint32_t{ 0xffeed },
        /* DMA_EMCS = */ uint32_t{ 0xdd2 },
        /* DMA_TOTAL_BYTES = */ uint32_t{ 0xfa12a },
        /* DMA_CMD = */ uint32_t{ 0x11a },
        /* m_IsLastChunk = */ uint8_t{ 0 },
    };
    cascading::ProgramMceExtraData cascadingProgramMceExtraData1 = {
        /* CE_CONTROL = */ uint32_t{ 0x54768 },
        /* MUL_ENABLE = */
        std::array<std::array<uint32_t, 4>, 8>{
            std::array<uint32_t, 4>{ 0x45, 0x46, 0x47, 0x48 },
            std::array<uint32_t, 4>{ 0x49, 0x50, 0x51, 0x52 },
            std::array<uint32_t, 4>{ 0x53, 0x54, 0x55, 0x56 },
            std::array<uint32_t, 4>{ 0x57, 0x58, 0x59, 0x60 },
            std::array<uint32_t, 4>{ 0x61, 0x62, 0x63, 0x64 },
            std::array<uint32_t, 4>{ 0x65, 0x66, 0x67, 0x68 },
            std::array<uint32_t, 4>{ 0x69, 0x70, 0x71, 0x72 },
            std::array<uint32_t, 4>{ 0x73, 0x74, 0x75, 0x76 },
        },
        /* IFM_ROW_STRIDE = */ uint32_t{ 0x3423 },
        /* IFM_CONFIG1 = */ uint32_t{ 0xaa8daa },
        /* IFM_PAD = */
        std::array<std::array<uint32_t, 4>, 4>{
            std::array<uint32_t, 4>{ 0x45, 0x48, 0x45, 0x48 },
            std::array<uint32_t, 4>{ 0x41, 0x61, 0x41, 0x61 },
            std::array<uint32_t, 4>{ 0x42, 0x61, 0x42, 0x61 },
            std::array<uint32_t, 4>{ 0x45, 0x6a, 0x42, 0x61 },
        },
        /* WIDE_KERNEL_OFFSET = */ uint32_t{ 0x998765 },
        /* IFM_TOP_SLOTS = */ uint32_t{ 0xee31 },
        /* IFM_MID_SLOTS = */ uint32_t{ 0xe56654 },
        /* IFM_BOTTOM_SLOTS = */ uint32_t{ 0xf787 },
        /* IFM_SLOT_PAD_CONFIG = */ uint32_t{ 0x0897 },
        /* OFM_STRIPE_SIZE = */ uint32_t{ 0xbb6 },
        /* OFM_CONFIG = */ uint32_t{ 0xa455435 },
        /* WEIGHT_BASE_ADDR = */ std::array<uint32_t, 4>{ 0x34587, 0xa, 0x342, 0xb },
        /* IFM_CONFIG2 = */
        std::array<std::array<uint32_t, 4>, 8>{
            std::array<uint32_t, 4>{ 0x145, 0x246, 0x145, 0x246 },
            std::array<uint32_t, 4>{ 0x149, 0x250, 0x149, 0x250 },
            std::array<uint32_t, 4>{ 0x153, 0x254, 0x153, 0x254 },
            std::array<uint32_t, 4>{ 0x157, 0x258, 0x157, 0x258 },
            std::array<uint32_t, 4>{ 0x161, 0x262, 0x161, 0x262 },
            std::array<uint32_t, 4>{ 0x165, 0x266, 0x165, 0x266 },
            std::array<uint32_t, 4>{ 0x169, 0x270, 0x169, 0x270 },
            std::array<uint32_t, 4>{ 0x173, 0x274, 0x173, 0x274 },
        },
        /* m_NumBlocksProgrammedForMce = */ uint32_t{ 128 },
    };
    cascading::StartMceExtraData cascadingStartMceExtraData1 = { /* CE_ENABLES = */ 0x123aa };
    cascading::StartPleExtraData cascadingStartPleExtraData1 = { /* SCRATCH = */
                                                                 0x125aa, 0x126aa, 0x127aa, 0x128aa,
                                                                 0x129aa, 0x130aa, 0x131aa, 0x132aa
    };

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
    AddCascade(buffer, { agent0, agent1, agent2, agent3, agent4, agent5 }, { cascadingDmaRdCommand1 },
               { cascadingDmaWrCommand1 }, { cascadingMceCommand1, cascadingMceCommand2 },
               { cascadingPleCommand1, cascadingPleCommand2 }, { cascadingDmaExtraData1, cascadingDmaExtraData2 },
               { cascadingProgramMceExtraData1 }, { cascadingStartMceExtraData1 }, { cascadingStartPleExtraData1 });
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
