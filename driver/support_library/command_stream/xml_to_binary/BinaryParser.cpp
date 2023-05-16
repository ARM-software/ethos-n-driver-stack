//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "BinaryParser.hpp"
#include <ethosn_command_stream/CommandStream.hpp>
#include <ethosn_command_stream/cascading/CommandStream.hpp>

#include <cassert>
#include <iomanip>
#include <string>
#include <type_traits>
#include <vector>

using namespace ethosn::command_stream;

namespace
{

std::vector<uint8_t> ReadBinaryData(std::istream& input)
{
    return std::vector<uint8_t>(std::istreambuf_iterator<char>{ input }, {});
}

void Parse(std::stringstream& parent, const char* const value, const int& tabs, const bool& newline)
{
    for (int i = 0; i < tabs; ++i)
    {
        parent << "    ";
    }

    parent << value;

    if (newline)
    {
        parent << "\n";
    }
}

void Parse(std::stringstream& parent, const std::string& value, const int tabs, const bool newline)
{
    Parse(parent, value.c_str(), tabs, newline);
}

std::string IntegersToString(std::ostringstream& oss)
{
    return oss.str();
}

template <typename I, typename... Is>
std::string IntegersToString(std::ostringstream& oss, const I value, const Is... more)
{
    using PrintType = std::conditional_t<std::is_unsigned<I>::value, unsigned int, int>;

    oss << static_cast<PrintType>(value);

    if (sizeof...(more) > 0)
    {
        oss << " ";
    }

    return IntegersToString(oss, more...);
}

template <int Base = 10, typename... Is>
std::string IntegersToString(const Is... ints)
{
    std::ostringstream oss;
    oss << std::setbase(Base);
    return IntegersToString(oss, ints...);
}

template <typename IntType>
std::enable_if_t<std::is_integral<IntType>::value> Parse(std::stringstream& parent, const IntType value)
{
    Parse(parent, IntegersToString(value));
}

void ParseAsHex(std::stringstream& parent, const uint32_t value)
{
    Parse(parent, "0x" + IntegersToString<16>(value), 0, false);
}

void ParseAsNum(std::stringstream& parent, const int32_t value)
{
    Parse(parent, IntegersToString<10>(value), 0, false);
}

void Parse(std::stringstream& parent, const DataType value)
{
    switch (value)
    {
        case DataType::U8:
        {
            Parse(parent, "U8", 0, false);
            break;
        }
        case DataType::S8:
        {
            Parse(parent, "S8", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid DataType in binary input: " + std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const DataFormat value)
{
    switch (value)
    {
        case DataFormat::NHWCB:
        {
            Parse(parent, "NHWCB", 0, false);
            break;
        }
        case DataFormat::NHWC:
        {
            Parse(parent, "NHWC", 0, false);
            break;
        }
        case DataFormat::NCHW:
        {
            Parse(parent, "NCHW", 0, false);
            break;
        }
        case DataFormat::WEIGHT_STREAM:
        {
            Parse(parent, "WEIGHT_STREAM", 0, false);
            break;
        }
        case DataFormat::FCAF_DEEP:
        {
            Parse(parent, "FCAF_DEEP", 0, false);
            break;
        }
        case DataFormat::FCAF_WIDE:
        {
            Parse(parent, "FCAF_WIDE", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid DataFormat in binary input: " + std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const SramAllocationStrategy value)
{
    switch (value)
    {
        case SramAllocationStrategy::STRATEGY_0:
        {
            Parse(parent, "STRATEGY_0", 0, false);
            break;
        }
        case SramAllocationStrategy::STRATEGY_1:
        {
            Parse(parent, "STRATEGY_1", 0, false);
            break;
        }
        case SramAllocationStrategy::STRATEGY_3:
        {
            Parse(parent, "STRATEGY_3", 0, false);
            break;
        }
        case SramAllocationStrategy::STRATEGY_4:
        {
            Parse(parent, "STRATEGY_4", 0, false);
            break;
        }
        case SramAllocationStrategy::STRATEGY_6:
        {
            Parse(parent, "STRATEGY_6", 0, false);
            break;
        }
        case SramAllocationStrategy::STRATEGY_7:
        {
            Parse(parent, "STRATEGY_7", 0, false);
            break;
        }
        case SramAllocationStrategy::STRATEGY_X:
        {
            Parse(parent, "STRATEGY_X", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid SramAllocationStrategy in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const UpsampleType value)
{
    switch (value)
    {
        case UpsampleType::OFF:
        {
            Parse(parent, "OFF", 0, false);
            break;
        }
        case UpsampleType::BILINEAR:
        {
            Parse(parent, "BILINEAR", 0, false);
            break;
        }
        case UpsampleType::NEAREST_NEIGHBOUR:
        {
            Parse(parent, "NEAREST_NEIGHBOUR", 0, false);
            break;
        }
        case UpsampleType::TRANSPOSE:
        {
            Parse(parent, "TRANSPOSE", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid UpsampleType in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const UpsampleEdgeMode& value)
{
    switch (value)
    {
        case UpsampleEdgeMode::DROP:
        {
            Parse(parent, "DROP", 0, false);
            break;
        }
        case UpsampleEdgeMode::GENERATE:
        {
            Parse(parent, "GENERATE", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid upsampleEdgeMode in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const MceAlgorithm value)
{
    switch (value)
    {
        case MceAlgorithm::DIRECT:
        {
            Parse(parent, "DIRECT", 0, false);
            break;
        }
        case MceAlgorithm::WINOGRAD:
        {
            Parse(parent, "WINOGRAD", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid MceAlgorithm in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const DataLocation value)
{
    switch (value)
    {
        case DataLocation::DRAM:
        {
            Parse(parent, "DRAM", 0, false);
            break;
        }
        case DataLocation::SRAM:
        {
            Parse(parent, "SRAM", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid DataLocation in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const MceOperation value)
{
    switch (value)
    {
        case MceOperation::CONVOLUTION:
        {
            Parse(parent, "CONVOLUTION", 0, false);
            break;
        }
        case MceOperation::DEPTHWISE_CONVOLUTION:
        {
            Parse(parent, "DEPTHWISE_CONVOLUTION", 0, false);
            break;
        }
        case MceOperation::FULLY_CONNECTED:
        {
            Parse(parent, "FULLY_CONNECTED", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid MceOperation in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const SectionType value)
{
    switch (value)
    {
        case SectionType::SISO:
        {
            Parse(parent, "SISO", 0, false);
            break;
        }
        case SectionType::SISO_CASCADED:
        {
            Parse(parent, "SISO_CASCADED", 0, false);
            break;
        }
        case SectionType::SIMO:
        {
            Parse(parent, "SIMO", 0, false);
            break;
        }
        case SectionType::SIMO_CASCADED:
        {
            Parse(parent, "SIMO_CASCADED", 0, false);
            break;
        }
        case SectionType::SISO_BRANCHED_CASCADED:
        {
            Parse(parent, "SISO_BRANCHED_CASCADED", 0, false);
            break;
        }
        case SectionType::MISO:
        {
            Parse(parent, "MISO", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid SectionType in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const TensorShape& value)
{
    const std::string text =
        IntegersToString(std::get<0>(value), std::get<1>(value), std::get<2>(value), std::get<3>(value));
    Parse(parent, text, 0, false);
}

void Parse(std::stringstream& parent, const TensorInfo& value)
{
    Parse(parent, "<DATA_TYPE>", 3, false);
    Parse(parent, value.m_DataType());
    Parse(parent, "</DATA_TYPE>", 0, true);

    Parse(parent, "<DATA_FORMAT>", 3, false);
    Parse(parent, value.m_DataFormat());
    Parse(parent, "</DATA_FORMAT>", 0, true);

    Parse(parent, "<TENSOR_SHAPE>", 3, false);
    Parse(parent, value.m_TensorShape());
    Parse(parent, "</TENSOR_SHAPE>", 0, true);

    Parse(parent, "<SUPERTENSOR_SHAPE>", 3, false);
    Parse(parent, value.m_SupertensorShape());
    Parse(parent, "</SUPERTENSOR_SHAPE>", 0, true);

    Parse(parent, "<SUPERTENSOR_OFFSET>", 3, false);
    Parse(parent, value.m_SupertensorOffset());
    Parse(parent, "</SUPERTENSOR_OFFSET>", 0, true);

    Parse(parent, "<STRIPE_SHAPE>", 3, false);
    Parse(parent, value.m_StripeShape());
    Parse(parent, "</STRIPE_SHAPE>", 0, true);
    // TileSize is represented as TILE_SHAPE in the XML, for compatibility with the prototype compiler and performance model.
    TensorShape tileShape{ value.m_TileSize(), 1, 1, 1 };
    Parse(parent, "<TILE_SHAPE>", 3, false);
    Parse(parent, tileShape);
    Parse(parent, "</TILE_SHAPE>", 0, true);

    Parse(parent, "<DRAM_BUFFER_ID>", 3, false);
    ParseAsNum(parent, value.m_DramBufferId());
    Parse(parent, "</DRAM_BUFFER_ID>", 0, true);

    Parse(parent, "<SRAM_OFFSET>", 3, false);
    ParseAsHex(parent, value.m_SramOffset());
    Parse(parent, "</SRAM_OFFSET>", 0, true);

    Parse(parent, "<ZERO_POINT>", 3, false);
    ParseAsNum(parent, value.m_ZeroPoint());
    Parse(parent, "</ZERO_POINT>", 0, true);

    Parse(parent, "<DATA_LOCATION>", 3, false);
    Parse(parent, value.m_DataLocation());
    Parse(parent, "</DATA_LOCATION>", 0, true);
}

void Parse(std::stringstream& parent, const SramConfig& value)
{
    Parse(parent, "<ALLOCATION_STRATEGY>", 3, false);
    Parse(parent, value.m_AllocationStrategy());
    Parse(parent, "</ALLOCATION_STRATEGY>", 0, true);
}

void Parse(std::stringstream& parent, const BlockConfig& value)
{
    Parse(parent, "<BLOCK_WIDTH>", 3, false);
    ParseAsNum(parent, value.m_BlockWidth());
    Parse(parent, "</BLOCK_WIDTH>", 0, true);

    Parse(parent, "<BLOCK_HEIGHT>", 3, false);
    ParseAsNum(parent, value.m_BlockHeight());
    Parse(parent, "</BLOCK_HEIGHT>", 0, true);
}

void Parse(std::stringstream& parent, const MceData& value)
{
    Parse(parent, "<MCE_OP_INFO>", 2, true);

    Parse(parent, "<STRIDE_X>", 3, false);
    ParseAsNum(parent, value.m_Stride().m_X());
    Parse(parent, "</STRIDE_X>", 0, true);

    Parse(parent, "<STRIDE_Y>", 3, false);
    ParseAsNum(parent, value.m_Stride().m_Y());
    Parse(parent, "</STRIDE_Y>", 0, true);

    Parse(parent, "<PAD_TOP>", 3, false);
    ParseAsNum(parent, value.m_PadTop());
    Parse(parent, "</PAD_TOP>", 0, true);

    Parse(parent, "<PAD_LEFT>", 3, false);
    ParseAsNum(parent, value.m_PadLeft());
    Parse(parent, "</PAD_LEFT>", 0, true);

    Parse(parent, "<UNINTERLEAVED_INPUT_SHAPE>", 3, false);
    Parse(parent, value.m_UninterleavedInputShape());
    Parse(parent, "</UNINTERLEAVED_INPUT_SHAPE>", 0, true);

    Parse(parent, "<OUTPUT_SHAPE>", 3, false);
    Parse(parent, value.m_OutputShape());
    Parse(parent, "</OUTPUT_SHAPE>", 0, true);

    Parse(parent, "<OUTPUT_STRIPE_SHAPE>", 3, false);
    Parse(parent, value.m_OutputStripeShape());
    Parse(parent, "</OUTPUT_STRIPE_SHAPE>", 0, true);

    Parse(parent, "<OPERATION>", 3, false);
    Parse(parent, value.m_Operation());
    Parse(parent, "</OPERATION>", 0, true);

    Parse(parent, "<ALGO>", 3, false);
    Parse(parent, value.m_Algorithm());
    Parse(parent, "</ALGO>", 0, true);

    Parse(parent, "<ACTIVATION_MIN>", 3, false);
    ParseAsNum(parent, value.m_ActivationMin());
    Parse(parent, "</ACTIVATION_MIN>", 0, true);

    Parse(parent, "<ACTIVATION_MAX>", 3, false);
    ParseAsNum(parent, value.m_ActivationMax());
    Parse(parent, "</ACTIVATION_MAX>", 0, true);

    Parse(parent, "<UPSAMPLE_TYPE>", 3, false);
    Parse(parent, value.m_UpsampleType());
    Parse(parent, "</UPSAMPLE_TYPE>", 0, true);

    Parse(parent, "<UPSAMPLE_EDGE_MODE_ROW>", 3, false);
    Parse(parent, value.m_UpsampleEdgeModeRow());
    Parse(parent, "</UPSAMPLE_EDGE_MODE_ROW>", 0, true);

    Parse(parent, "<UPSAMPLE_EDGE_MODE_COL>", 3, false);
    Parse(parent, value.m_UpsampleEdgeModeCol());
    Parse(parent, "</UPSAMPLE_EDGE_MODE_COL>", 0, true);

    Parse(parent, "</MCE_OP_INFO>", 2, true);
}

void Parse(std::stringstream& parent, const PleOperation value)
{
    switch (value)
    {
        case PleOperation::ADDITION:
        {
            Parse(parent, "ADDITION", 0, false);
            break;
        }
        case PleOperation::ADDITION_RESCALE:
        {
            Parse(parent, "ADDITION_RESCALE", 0, false);
            break;
        }
        case PleOperation::AVGPOOL_3X3_1_1_UDMA:
        {
            Parse(parent, "AVGPOOL_3X3_1_1_UDMA", 0, false);
            break;
        }
        case PleOperation::INTERLEAVE_2X2_2_2:
        {
            Parse(parent, "INTERLEAVE_2X2_2_2", 0, false);
            break;
        }
        case PleOperation::MAXPOOL_2X2_2_2:
        {
            Parse(parent, "MAXPOOL_2X2_2_2", 0, false);
            break;
        }
        case PleOperation::MAXPOOL_3X3_2_2_EVEN:
        {
            Parse(parent, "MAXPOOL_3X3_2_2_EVEN", 0, false);
            break;
        }
        case PleOperation::MAXPOOL_3X3_2_2_ODD:
        {
            Parse(parent, "MAXPOOL_3X3_2_2_ODD", 0, false);
            break;
        }
        case PleOperation::MEAN_XY_7X7:
        {
            Parse(parent, "MEAN_XY_7X7", 0, false);
            break;
        }
        case PleOperation::MEAN_XY_8X8:
        {
            Parse(parent, "MEAN_XY_8X8", 0, false);
            break;
        }
        case PleOperation::PASSTHROUGH:
        {
            Parse(parent, "PASSTHROUGH", 0, false);
            break;
        }
        case PleOperation::SIGMOID:
        {
            Parse(parent, "SIGMOID", 0, false);
            break;
        }
        case PleOperation::TRANSPOSE_XY:
        {
            Parse(parent, "TRANSPOSE_XY", 0, false);
            break;
        }
        case PleOperation::LEAKY_RELU:
        {
            Parse(parent, "LEAKY_RELU", 0, false);
            break;
        }
        case PleOperation::DOWNSAMPLE_2X2:
        {
            Parse(parent, "DOWNSAMPLE_2X2", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid PLE operation in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const PleData& value)
{
    Parse(parent, "<PLE_OP_INFO>", 2, true);

    Parse(parent, "<CE_SRAM>", 3, false);
    ParseAsHex(parent, value.m_CeSram());
    Parse(parent, "</CE_SRAM>", 0, true);

    Parse(parent, "<PLE_SRAM>", 3, false);
    ParseAsHex(parent, value.m_PleSram());
    Parse(parent, "</PLE_SRAM>", 0, true);

    Parse(parent, "<OPERATION>", 3, false);
    Parse(parent, value.m_Operation());
    Parse(parent, "</OPERATION>", 0, true);

    Parse(parent, "<RESCALE_MULTIPLIER0>", 3, false);
    ParseAsNum(parent, value.m_RescaleMultiplier0());
    Parse(parent, "</RESCALE_MULTIPLIER0>", 0, true);

    Parse(parent, "<RESCALE_SHIFT0>", 3, false);
    ParseAsNum(parent, value.m_RescaleShift0());
    Parse(parent, "</RESCALE_SHIFT0>", 0, true);

    Parse(parent, "<RESCALE_MULTIPLIER1>", 3, false);
    ParseAsNum(parent, value.m_RescaleMultiplier1());
    Parse(parent, "</RESCALE_MULTIPLIER1>", 0, true);

    Parse(parent, "<RESCALE_SHIFT1>", 3, false);
    ParseAsNum(parent, value.m_RescaleShift1());
    Parse(parent, "</RESCALE_SHIFT1>", 0, true);

    Parse(parent, "</PLE_OP_INFO>", 2, true);
}

void Parse(std::stringstream& parent, const McePle& value)
{
    Parse(parent, "<OPERATION_MCE_PLE>", 1, true);

    Parse(parent, "<INPUT_INFO>", 2, true);
    Parse(parent, value.m_InputInfo());
    Parse(parent, "</INPUT_INFO>", 2, true);

    Parse(parent, "<WEIGHT_INFO>", 2, true);
    Parse(parent, value.m_WeightInfo());
    Parse(parent, "</WEIGHT_INFO>", 2, true);

    Parse(parent, "<WEIGHTS_METADATA_BUFFER_ID>", 2, false);
    ParseAsNum(parent, value.m_WeightMetadataBufferId());
    Parse(parent, "</WEIGHTS_METADATA_BUFFER_ID>", 0, true);

    Parse(parent, "<OUTPUT_INFO>", 2, true);
    Parse(parent, value.m_OutputInfo());
    Parse(parent, "</OUTPUT_INFO>", 2, true);

    Parse(parent, "<SRAM_CONFIG>", 2, true);
    Parse(parent, value.m_SramConfig());
    Parse(parent, "</SRAM_CONFIG>", 2, true);

    Parse(parent, "<BLOCK_CONFIG>", 2, true);
    Parse(parent, value.m_BlockConfig());
    Parse(parent, "</BLOCK_CONFIG>", 2, true);

    Parse(parent, value.m_MceData());
    Parse(parent, value.m_PleData());

    Parse(parent, "</OPERATION_MCE_PLE>", 1, true);
}

void Parse(std::stringstream& parent, const PleOnly& value)
{
    Parse(parent, "<OPERATION_PLE>", 1, true);

    Parse(parent, "<INPUT_INFO>", 2, true);
    Parse(parent, value.m_InputInfo());
    Parse(parent, "</INPUT_INFO>", 2, true);
    if (value.m_NumInputInfos() == 2)
    {
        Parse(parent, "<INPUT_INFO>", 2, true);
        Parse(parent, value.m_InputInfo2());
        Parse(parent, "</INPUT_INFO>", 2, true);
    }
    Parse(parent, "<OUTPUT_INFO>", 2, true);
    Parse(parent, value.m_OutputInfo());
    Parse(parent, "</OUTPUT_INFO>", 2, true);

    Parse(parent, "<SRAM_CONFIG>", 2, true);
    Parse(parent, value.m_SramConfig());
    Parse(parent, "</SRAM_CONFIG>", 2, true);

    Parse(parent, value.m_PleData());

    Parse(parent, "</OPERATION_PLE>", 1, true);
}

void Parse(std::stringstream& parent, const Convert& value)
{
    Parse(parent, "<OPERATION_CONVERT>", 1, true);

    Parse(parent, "<INPUT_INFO>", 2, true);
    Parse(parent, value.m_InputInfo());
    Parse(parent, "</INPUT_INFO>", 2, true);

    Parse(parent, "<OUTPUT_INFO>", 2, true);
    Parse(parent, value.m_OutputInfo());
    Parse(parent, "</OUTPUT_INFO>", 2, true);

    Parse(parent, "</OPERATION_CONVERT>", 1, true);
}

void Parse(std::stringstream& parent, const SpaceToDepth& value)
{
    Parse(parent, "<OPERATION_SPACE_TO_DEPTH>", 1, true);

    Parse(parent, "<INPUT_INFO>", 2, true);
    Parse(parent, value.m_InputInfo());
    Parse(parent, "</INPUT_INFO>", 2, true);

    Parse(parent, "<OUTPUT_INFO>", 2, true);
    Parse(parent, value.m_OutputInfo());
    Parse(parent, "</OUTPUT_INFO>", 2, true);

    Parse(parent, "<USED_EMCS>", 2, false);
    ParseAsNum(parent, value.m_UsedEmcs());
    Parse(parent, "</USED_EMCS>", 0, true);

    Parse(parent, "<INTERMEDIATE_1_SIZE>", 2, false);
    ParseAsNum(parent, value.m_Intermediate1Size());
    Parse(parent, "</INTERMEDIATE_1_SIZE>", 0, true);

    Parse(parent, "<INTERMEDIATE_2_SIZE>", 2, false);
    ParseAsNum(parent, value.m_Intermediate2Size());
    Parse(parent, "</INTERMEDIATE_2_SIZE>", 0, true);

    Parse(parent, "</OPERATION_SPACE_TO_DEPTH>", 1, true);
}

void Parse(std::stringstream& parent, const Filename& value)
{
    char output[128];
    for (int i = 0; i < 128; ++i)
    {
        output[i] = value[i];
    }
    Parse(parent, output, 0, false);
}

void Parse(std::stringstream& parent, const DumpDram& value)
{
    Parse(parent, "<DUMP_DRAM>", 1, true);

    Parse(parent, "<DRAM_BUFFER_ID>", 2, false);
    ParseAsNum(parent, value.m_DramBufferId());
    Parse(parent, "</DRAM_BUFFER_ID>", 0, true);

    Parse(parent, "<FILENAME>", 2, false);
    Parse(parent, value.m_Filename());
    Parse(parent, "</FILENAME>", 0, true);

    Parse(parent, "</DUMP_DRAM>", 1, true);
}

void Parse(std::stringstream& parent, const DumpSram& value)
{
    Parse(parent, "<DUMP_SRAM>", 1, true);

    Parse(parent, "<PREFIX>", 2, false);
    Parse(parent, value.m_Filename());
    Parse(parent, "</PREFIX>", 0, true);

    Parse(parent, "</DUMP_SRAM>", 1, true);
}

void Parse(std::stringstream& parent, const Section& value)
{
    Parse(parent, "<SECTION>", 1, true);

    Parse(parent, "<TYPE>", 2, false);
    Parse(parent, value.m_Type());
    Parse(parent, "</TYPE>", 0, true);

    Parse(parent, "</SECTION>", 1, true);
}

void Parse(std::stringstream& parent, const Fence&)
{
    parent << "    <FENCE/>\n";
}

void Parse(std::stringstream& parent, const Delay& value)
{
    Parse(parent, "<DELAY>", 1, true);

    Parse(parent, "<VALUE>", 2, false);
    ParseAsNum(parent, value.m_Value());
    Parse(parent, "</VALUE>", 0, true);

    Parse(parent, "</DELAY>", 1, true);
}

void Parse(std::stringstream& parent, const cascading::IfmS& ifms)
{
    Parse(parent, "<IFM_STREAMER>", 4, true);

    Parse(parent, "<BUFFER_ID>", 5, false);
    ParseAsNum(parent, ifms.bufferId);
    Parse(parent, "</BUFFER_ID>", 0, true);

    Parse(parent, "<DMA_COMP_CONFIG0>", 5, false);
    ParseAsHex(parent, ifms.DMA_COMP_CONFIG0);
    Parse(parent, "</DMA_COMP_CONFIG0>", 0, true);

    Parse(parent, "<DMA_STRIDE1>", 5, false);
    ParseAsHex(parent, ifms.DMA_STRIDE1);
    Parse(parent, "</DMA_STRIDE1>", 0, true);

    Parse(parent, "<DMA_STRIDE2>", 5, false);
    ParseAsHex(parent, ifms.DMA_STRIDE2);
    Parse(parent, "</DMA_STRIDE2>", 0, true);

    Parse(parent, "</IFM_STREAMER>", 4, true);
}

void Parse(std::stringstream& parent, const cascading::OfmS& ofms)
{
    Parse(parent, "<OFM_STREAMER>", 4, true);

    Parse(parent, "<BUFFER_ID>", 5, false);
    ParseAsNum(parent, ofms.bufferId);
    Parse(parent, "</BUFFER_ID>", 0, true);

    Parse(parent, "<DMA_COMP_CONFIG0>", 5, false);
    ParseAsHex(parent, ofms.DMA_COMP_CONFIG0);
    Parse(parent, "</DMA_COMP_CONFIG0>", 0, true);

    Parse(parent, "<DMA_STRIDE1>", 5, false);
    ParseAsHex(parent, ofms.DMA_STRIDE1);
    Parse(parent, "</DMA_STRIDE1>", 0, true);

    Parse(parent, "<DMA_STRIDE2>", 5, false);
    ParseAsHex(parent, ofms.DMA_STRIDE2);
    Parse(parent, "</DMA_STRIDE2>", 0, true);

    Parse(parent, "</OFM_STREAMER>", 4, true);
}

void Parse(std::stringstream& parent, const cascading::WgtS& wgts)
{
    Parse(parent, "<WGT_STREAMER>", 4, true);

    Parse(parent, "<BUFFER_ID>", 5, false);
    ParseAsNum(parent, wgts.bufferId);
    Parse(parent, "</BUFFER_ID>", 0, true);

    Parse(parent, "</WGT_STREAMER>", 4, true);
}

void Parse(std::stringstream& parent, const cascading::MceOperation value)
{
    switch (value)
    {
        case cascading::MceOperation::CONVOLUTION:
        {
            Parse(parent, "CONVOLUTION", 0, false);
            break;
        }
        case cascading::MceOperation::DEPTHWISE_CONVOLUTION:
        {
            Parse(parent, "DEPTHWISE_CONVOLUTION", 0, false);
            break;
        }
        case cascading::MceOperation::FULLY_CONNECTED:
        {
            Parse(parent, "FULLY_CONNECTED", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid MceOperation in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const cascading::MceS& mces)
{
    Parse(parent, "<MCE_SCHEDULER>", 4, true);

    Parse(parent, "<MCE_OP_MODE>", 5, false);
    Parse(parent, mces.mceOpMode);
    Parse(parent, "</MCE_OP_MODE>", 0, true);

    Parse(parent, "<PLE_KERNEL_ID>", 5, false);
    Parse(parent, cascading::PleKernelId2String(mces.pleKernelId), 0, false);
    Parse(parent, "</PLE_KERNEL_ID>", 0, true);

    Parse(parent, "<ACTIVATION_CONFIG>", 5, false);
    ParseAsHex(parent, mces.ACTIVATION_CONFIG);
    Parse(parent, "</ACTIVATION_CONFIG>", 0, true);

    Parse(parent, "<WIDE_KERNEL_CONTROL>", 5, false);
    ParseAsHex(parent, mces.WIDE_KERNEL_CONTROL);
    Parse(parent, "</WIDE_KERNEL_CONTROL>", 0, true);

    Parse(parent, "<FILTER>", 5, false);
    ParseAsHex(parent, mces.FILTER);
    Parse(parent, "</FILTER>", 0, true);

    Parse(parent, "<IFM_ZERO_POINT>", 5, false);
    ParseAsHex(parent, mces.IFM_ZERO_POINT);
    Parse(parent, "</IFM_ZERO_POINT>", 0, true);

    Parse(parent, "<IFM_DEFAULT_SLOT_SIZE>", 5, false);
    ParseAsHex(parent, mces.IFM_DEFAULT_SLOT_SIZE);
    Parse(parent, "</IFM_DEFAULT_SLOT_SIZE>", 0, true);

    Parse(parent, "<IFM_SLOT_STRIDE>", 5, false);
    ParseAsHex(parent, mces.IFM_SLOT_STRIDE);
    Parse(parent, "</IFM_SLOT_STRIDE>", 0, true);

    Parse(parent, "<STRIPE_BLOCK_CONFIG>", 5, false);
    ParseAsHex(parent, mces.STRIPE_BLOCK_CONFIG);
    Parse(parent, "</STRIPE_BLOCK_CONFIG>", 0, true);

    Parse(parent, "<DEPTHWISE_CONTROL>", 5, false);
    ParseAsHex(parent, mces.DEPTHWISE_CONTROL);
    Parse(parent, "</DEPTHWISE_CONTROL>", 0, true);

    Parse(parent, "<IFM_SLOT_BASE_ADDRESS>", 5, false);
    ParseAsHex(parent, mces.IFM_SLOT_BASE_ADDRESS);
    Parse(parent, "</IFM_SLOT_BASE_ADDRESS>", 0, true);

    Parse(parent, "<PLE_MCEIF_CONFIG>", 5, false);
    ParseAsHex(parent, mces.PLE_MCEIF_CONFIG);
    Parse(parent, "</PLE_MCEIF_CONFIG>", 0, true);

    Parse(parent, "</MCE_SCHEDULER>", 4, true);
}

void Parse(std::stringstream& parent, const cascading::PleL& plel)
{
    Parse(parent, "<PLE_LOADER>", 4, true);

    Parse(parent, "<PLE_KERNEL_ID>", 5, false);
    Parse(parent, cascading::PleKernelId2String(plel.pleKernelId), 0, false);
    Parse(parent, "</PLE_KERNEL_ID>", 0, true);

    Parse(parent, "</PLE_LOADER>", 4, true);
}

void Parse(std::stringstream& parent, const cascading::PleInputMode value)
{
    switch (value)
    {
        case cascading::PleInputMode::MCE_ALL_OGS:
        {
            Parse(parent, "MCE_ALL_OGS", 0, false);
            break;
        }
        case cascading::PleInputMode::MCE_ONE_OG:
        {
            Parse(parent, "MCE_ONE_OG", 0, false);
            break;
        }
        case cascading::PleInputMode::SRAM_ONE_INPUT:
        {
            Parse(parent, "SRAM_ONE_INPUT", 0, false);
            break;
        }
        case cascading::PleInputMode::SRAM_TWO_INPUTS:
        {
            Parse(parent, "SRAM_TWO_INPUTS", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid PleInputMode in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const cascading::PleS& ples)
{
    Parse(parent, "<PLE_SCHEDULER>", 4, true);

    Parse(parent, "<INPUT_MODE>", 5, false);
    Parse(parent, ples.inputMode);
    Parse(parent, "</INPUT_MODE>", 0, true);

    Parse(parent, "<PLE_KERNEL_ID>", 5, false);
    Parse(parent, cascading::PleKernelId2String(ples.pleKernelId), 0, false);
    Parse(parent, "</PLE_KERNEL_ID>", 0, true);

    Parse(parent, "<PLE_KERNEL_SRAM_ADDR>", 5, false);
    ParseAsNum(parent, ples.pleKernelSramAddr);
    Parse(parent, "</PLE_KERNEL_SRAM_ADDR>", 0, true);

    Parse(parent, "</PLE_SCHEDULER>", 4, true);
}

void Parse(std::stringstream& parent, const cascading::AgentData& data)
{
    switch (data.type)
    {
        case cascading::AgentType::IFM_STREAMER:
            Parse(parent, data.ifm);
            break;
        case cascading::AgentType::WGT_STREAMER:
            Parse(parent, data.wgt);
            break;
        case cascading::AgentType::MCE_SCHEDULER:
            Parse(parent, data.mce);
            break;
        case cascading::AgentType::PLE_LOADER:
            Parse(parent, data.pleL);
            break;
        case cascading::AgentType::PLE_SCHEDULER:
            Parse(parent, data.pleS);
            break;
        case cascading::AgentType::OFM_STREAMER:
            Parse(parent, data.ofm);
            break;
        default:
        {
            // Bad binary
            throw ParseException("Invalid cascading agent type: " + std::to_string(static_cast<uint32_t>(data.type)));
        }
    };
}

void Parse(std::stringstream& parent, const cascading::Agent& agent)
{
    Parse(parent, "<AGENT>", 3, true);

    Parse(parent, "<NUM_STRIPES_TOTAL>", 4, false);
    ParseAsNum(parent, agent.numStripesTotal);
    Parse(parent, "</NUM_STRIPES_TOTAL>", 0, true);

    Parse(parent, agent.data);

    Parse(parent, "</AGENT>", 3, true);
}

const char* AgentTypeToString(cascading::AgentType t)
{
    switch (t)
    {
        case cascading::AgentType::IFM_STREAMER:
            return "IFM_STREAMER";
        case cascading::AgentType::WGT_STREAMER:
            return "WGT_STREAMER";
        case cascading::AgentType::MCE_SCHEDULER:
            return "MCE_SCHEDULER";
        case cascading::AgentType::PLE_LOADER:
            return "PLE_LOADER";
        case cascading::AgentType::PLE_SCHEDULER:
            return "PLE_SCHEDULER";
        case cascading::AgentType::OFM_STREAMER:
            return "OFM_STREAMER";
        default:
            throw ParseException("Invalid cascading agent type: " + std::to_string(static_cast<uint32_t>(t)));
    }
}

void Parse(std::stringstream& parent, const cascading::WaitForAgentCommand& waitCommand, const cascading::Agent* agents)
{
    using namespace ethosn::command_stream::cascading;

    Parse(parent, "<WAIT_FOR_AGENT_COMMAND>", 3, true);

    // Add helpful comment to indicate the agent type being waited for
    Parse(parent,
          ("<!-- Waited-for agent type is " + std::string(AgentTypeToString(agents[waitCommand.agentId].data.type)) +
           " -->")
              .c_str(),
          4, true);
    Parse(parent, "<AGENT_ID>", 4, false);
    ParseAsNum(parent, waitCommand.agentId);
    Parse(parent, "</AGENT_ID>", 0, true);

    Parse(parent, "<STRIPE_ID>", 4, false);
    ParseAsNum(parent, waitCommand.stripeId);
    Parse(parent, "</STRIPE_ID>", 0, true);

    Parse(parent, "</WAIT_FOR_AGENT_COMMAND>", 3, true);
}

void Parse(std::stringstream& parent, const cascading::DmaCommand& dmaCommand, const cascading::Agent* agents)
{
    using namespace ethosn::command_stream::cascading;

    Parse(parent, "<DMA_COMMAND>", 3, true);

    // Add helpful comment to indicate the agent type (DmaCommands are used for several different kinds of agent)
    Parse(
        parent,
        ("<!-- Agent type is " + std::string(AgentTypeToString(agents[dmaCommand.agentId].data.type)) + " -->").c_str(),
        4, true);
    Parse(parent, "<AGENT_ID>", 4, false);
    ParseAsNum(parent, dmaCommand.agentId);
    Parse(parent, "</AGENT_ID>", 0, true);

    Parse(parent, "<STRIPE_ID>", 4, false);
    ParseAsNum(parent, dmaCommand.stripeId);
    Parse(parent, "</STRIPE_ID>", 0, true);

    Parse(parent, "<DRAM_OFFSET>", 4, false);
    ParseAsHex(parent, dmaCommand.m_DramOffset);
    Parse(parent, "</DRAM_OFFSET>", 0, true);

    Parse(parent, "<SRAM_ADDR>", 4, false);
    ParseAsHex(parent, dmaCommand.SRAM_ADDR);
    Parse(parent, "</SRAM_ADDR>", 0, true);

    Parse(parent, "<DMA_SRAM_STRIDE>", 4, false);
    ParseAsHex(parent, dmaCommand.DMA_SRAM_STRIDE);
    Parse(parent, "</DMA_SRAM_STRIDE>", 0, true);

    Parse(parent, "<DMA_STRIDE0>", 4, false);
    ParseAsHex(parent, dmaCommand.DMA_STRIDE0);
    Parse(parent, "</DMA_STRIDE0>", 0, true);

    Parse(parent, "<DMA_STRIDE3>", 4, false);
    ParseAsHex(parent, dmaCommand.DMA_STRIDE3);
    Parse(parent, "</DMA_STRIDE3>", 0, true);

    Parse(parent, "<DMA_CHANNELS>", 4, false);
    ParseAsHex(parent, dmaCommand.DMA_CHANNELS);
    Parse(parent, "</DMA_CHANNELS>", 0, true);

    Parse(parent, "<DMA_EMCS>", 4, false);
    ParseAsHex(parent, dmaCommand.DMA_EMCS);
    Parse(parent, "</DMA_EMCS>", 0, true);

    Parse(parent, "<DMA_TOTAL_BYTES>", 4, false);
    ParseAsHex(parent, dmaCommand.DMA_TOTAL_BYTES);
    Parse(parent, "</DMA_TOTAL_BYTES>", 0, true);

    Parse(parent, "<DMA_CMD>", 4, false);
    ParseAsHex(parent, dmaCommand.DMA_CMD);
    Parse(parent, "</DMA_CMD>", 0, true);

    Parse(parent, "<IS_LAST_CHUNK>", 4, false);
    ParseAsNum(parent, dmaCommand.m_IsLastChunk);
    Parse(parent, "</IS_LAST_CHUNK>", 0, true);

    Parse(parent, "</DMA_COMMAND>", 3, true);
}

void Parse(std::stringstream& parent, const cascading::ProgramMceStripeCommand& programMceCommand)
{
    using namespace ethosn::command_stream::cascading;

    Parse(parent, "<PROGRAM_MCE_STRIPE_COMMAND>", 3, true);

    Parse(parent, "<AGENT_ID>", 4, false);
    ParseAsNum(parent, programMceCommand.agentId);
    Parse(parent, "</AGENT_ID>", 0, true);

    Parse(parent, "<STRIPE_ID>", 4, false);
    ParseAsNum(parent, programMceCommand.stripeId);
    Parse(parent, "</STRIPE_ID>", 0, true);

    for (size_t ce = 0; ce < programMceCommand.MUL_ENABLE.size(); ++ce)
    {
        std::string beginElementName = std::string("<MUL_ENABLE_CE") + std::to_string(ce) + ">";
        Parse(parent, beginElementName.c_str(), 4, true);

        for (size_t og = 0; og < programMceCommand.MUL_ENABLE[ce].size(); ++og)
        {
            std::string beginElementName = std::string("<OG") + std::to_string(og) + ">";
            Parse(parent, beginElementName.c_str(), 5, false);

            ParseAsHex(parent, programMceCommand.MUL_ENABLE[ce][og]);

            std::string endElementName = std::string("</OG") + std::to_string(og) + ">";
            Parse(parent, endElementName.c_str(), 0, true);
        }

        std::string endElementName = std::string("</MUL_ENABLE_CE") + std::to_string(ce) + ">";
        Parse(parent, endElementName.c_str(), 4, true);
    }

    Parse(parent, "<IFM_ROW_STRIDE>", 4, false);
    ParseAsHex(parent, programMceCommand.IFM_ROW_STRIDE);
    Parse(parent, "</IFM_ROW_STRIDE>", 0, true);

    Parse(parent, "<IFM_CONFIG1>", 4, false);
    ParseAsHex(parent, programMceCommand.IFM_CONFIG1);
    Parse(parent, "</IFM_CONFIG1>", 0, true);

    for (size_t num = 0; num < programMceCommand.IFM_PAD.size(); ++num)
    {
        std::string beginElementName = std::string("<IFM_PAD_NUM") + std::to_string(num) + ">";
        Parse(parent, beginElementName.c_str(), 4, true);

        for (size_t ig = 0; ig < programMceCommand.IFM_PAD[num].size(); ++ig)
        {
            std::string beginElementName = std::string("<IG") + std::to_string(ig) + ">";
            Parse(parent, beginElementName.c_str(), 5, false);

            ParseAsHex(parent, programMceCommand.IFM_PAD[num][ig]);

            std::string endElementName = std::string("</IG") + std::to_string(ig) + ">";
            Parse(parent, endElementName.c_str(), 0, true);
        }

        std::string endElementName = std::string("</IFM_PAD_NUM") + std::to_string(num) + ">";
        Parse(parent, endElementName.c_str(), 4, true);
    }

    Parse(parent, "<WIDE_KERNEL_OFFSET>", 4, false);
    ParseAsHex(parent, programMceCommand.WIDE_KERNEL_OFFSET);
    Parse(parent, "</WIDE_KERNEL_OFFSET>", 0, true);

    Parse(parent, "<IFM_TOP_SLOTS>", 4, false);
    ParseAsHex(parent, programMceCommand.IFM_TOP_SLOTS);
    Parse(parent, "</IFM_TOP_SLOTS>", 0, true);

    Parse(parent, "<IFM_MID_SLOTS>", 4, false);
    ParseAsHex(parent, programMceCommand.IFM_MID_SLOTS);
    Parse(parent, "</IFM_MID_SLOTS>", 0, true);

    Parse(parent, "<IFM_BOTTOM_SLOTS>", 4, false);
    ParseAsHex(parent, programMceCommand.IFM_BOTTOM_SLOTS);
    Parse(parent, "</IFM_BOTTOM_SLOTS>", 0, true);

    Parse(parent, "<IFM_SLOT_PAD_CONFIG>", 4, false);
    ParseAsHex(parent, programMceCommand.IFM_SLOT_PAD_CONFIG);
    Parse(parent, "</IFM_SLOT_PAD_CONFIG>", 0, true);

    Parse(parent, "<OFM_STRIPE_SIZE>", 4, false);
    ParseAsHex(parent, programMceCommand.OFM_STRIPE_SIZE);
    Parse(parent, "</OFM_STRIPE_SIZE>", 0, true);

    Parse(parent, "<OFM_CONFIG>", 4, false);
    ParseAsHex(parent, programMceCommand.OFM_CONFIG);
    Parse(parent, "</OFM_CONFIG>", 0, true);

    for (size_t og = 0; og < programMceCommand.WEIGHT_BASE_ADDR.size(); ++og)
    {
        std::string beginElementName = std::string("<WEIGHT_BASE_ADDR_OG") + std::to_string(og) + ">";
        Parse(parent, beginElementName.c_str(), 4, false);
        ParseAsHex(parent, programMceCommand.WEIGHT_BASE_ADDR[og]);
        std::string endElementName = std::string("</WEIGHT_BASE_ADDR_OG") + std::to_string(og) + ">";
        Parse(parent, endElementName.c_str(), 0, true);
    }

    for (size_t ce = 0; ce < programMceCommand.IFM_CONFIG2.size(); ++ce)
    {
        std::string beginElementName = std::string("<IFM_CONFIG2_CE") + std::to_string(ce) + ">";
        Parse(parent, beginElementName.c_str(), 4, true);

        for (size_t ig = 0; ig < programMceCommand.IFM_CONFIG2[ce].size(); ++ig)
        {
            std::string beginElementName = std::string("<IG") + std::to_string(ig) + ">";
            Parse(parent, beginElementName.c_str(), 5, false);

            ParseAsHex(parent, programMceCommand.IFM_CONFIG2[ce][ig]);

            std::string endElementName = std::string("</IG") + std::to_string(ig) + ">";
            Parse(parent, endElementName.c_str(), 0, true);
        }

        std::string endElementName = std::string("</IFM_CONFIG2_CE") + std::to_string(ce) + ">";
        Parse(parent, endElementName.c_str(), 4, true);
    }

    Parse(parent, "<NUM_BLOCKS_PROGRAMMED_FOR_MCE>", 4, false);
    ParseAsHex(parent, programMceCommand.m_NumBlocksProgrammedForMce);
    Parse(parent, "</NUM_BLOCKS_PROGRAMMED_FOR_MCE>", 0, true);

    Parse(parent, "</PROGRAM_MCE_STRIPE_COMMAND>", 3, true);
}

void Parse(std::stringstream& parent, const cascading::StartMceStripeCommand& startMceStripeCommand)
{
    using namespace ethosn::command_stream::cascading;

    Parse(parent, "<START_MCE_STRIPE_COMMAND>", 3, true);

    Parse(parent, "<AGENT_ID>", 4, false);
    ParseAsNum(parent, startMceStripeCommand.agentId);
    Parse(parent, "</AGENT_ID>", 0, true);

    Parse(parent, "<STRIPE_ID>", 4, false);
    ParseAsNum(parent, startMceStripeCommand.stripeId);
    Parse(parent, "</STRIPE_ID>", 0, true);

    Parse(parent, "<CE_ENABLES>", 4, false);
    ParseAsNum(parent, startMceStripeCommand.CE_ENABLES);
    Parse(parent, "</CE_ENABLES>", 0, true);

    Parse(parent, "</START_MCE_STRIPE_COMMAND>", 3, true);
}

void Parse(std::stringstream& parent, const cascading::StartPleStripeCommand& startPleStripeCommand)
{
    using namespace ethosn::command_stream::cascading;

    Parse(parent, "<START_PLE_STRIPE_COMMAND>", 3, true);

    Parse(parent, "<AGENT_ID>", 4, false);
    ParseAsNum(parent, startPleStripeCommand.agentId);
    Parse(parent, "</AGENT_ID>", 0, true);

    Parse(parent, "<STRIPE_ID>", 4, false);
    ParseAsNum(parent, startPleStripeCommand.stripeId);
    Parse(parent, "</STRIPE_ID>", 0, true);

    for (size_t i = 0; i < startPleStripeCommand.SCRATCH.size(); ++i)
    {
        std::string beginElementName = std::string("<SCRATCH") + std::to_string(i) + ">";
        Parse(parent, beginElementName.c_str(), 4, false);
        ParseAsHex(parent, startPleStripeCommand.SCRATCH[i]);
        std::string endElementName = std::string("</SCRATCH") + std::to_string(i) + ">";
        Parse(parent, endElementName.c_str(), 0, true);
    }

    Parse(parent, "</START_PLE_STRIPE_COMMAND>", 3, true);
}

void Parse(std::stringstream& parent, const cascading::Command& cmd, const cascading::Agent* agents)
{
    using namespace ethosn::command_stream::cascading;

    switch (cmd.type)
    {
        case CommandType::WaitForAgent:
            Parse(parent, static_cast<const WaitForAgentCommand&>(cmd), agents);
            break;
        case CommandType::LoadIfmStripe:
            Parse(parent, static_cast<const DmaCommand&>(cmd), agents);
            break;
        case CommandType::LoadWgtStripe:
            Parse(parent, static_cast<const DmaCommand&>(cmd), agents);
            break;
        case CommandType::ProgramMceStripe:
            Parse(parent, static_cast<const ProgramMceStripeCommand&>(cmd));
            break;
        case CommandType::StartMceStripe:
            Parse(parent, static_cast<const StartMceStripeCommand&>(cmd));
            break;
        case CommandType::LoadPleCode:
            Parse(parent, static_cast<const DmaCommand&>(cmd), agents);
            break;
        case CommandType::StartPleStripe:
            Parse(parent, static_cast<const StartPleStripeCommand&>(cmd));
            break;
        case CommandType::StoreOfmStripe:
            Parse(parent, static_cast<const DmaCommand&>(cmd), agents);
            break;
        default:
            throw ParseException("Invalid cascading command type: " + std::to_string(static_cast<uint32_t>(cmd.type)));
    }
}

void Parse(std::stringstream& parent, const Cascade& value)
{
    using namespace cascading;
    using Command = cascading::Command;

    Parse(parent, "<CASCADE>", 1, true);

    // Calculate pointers to the agent array and command lists
    const Agent* agentsArray          = value.GetAgentsArray();
    const Command* dmaRdCommandsBegin = value.GetDmaRdCommandsBegin();
    const Command* dmaWrCommandsBegin = value.GetDmaWrCommandsBegin();
    const Command* mceCommandsBegin   = value.GetMceCommandsBegin();
    const Command* pleCommandsBegin   = value.GetPleCommandsBegin();

    // Moves command pointer to next Command (each Command has different length)
    auto getNextCommand = [](const Command* c) {
        return reinterpret_cast<const Command*>(reinterpret_cast<const char*>(c) + c->GetSize());
    };

    Parse(parent, "<AGENTS>", 2, true);
    for (uint32_t agentId = 0; agentId < value.NumAgents; ++agentId)
    {
        // Add helpful comment to indicate the agent ID (very useful for long command streams)
        Parse(parent, "<!-- Agent " + std::to_string(agentId) + " -->", 3, true);
        Parse(parent, agentsArray[agentId]);
    }
    Parse(parent, "</AGENTS>", 2, true);

    Parse(parent, "<DMA_RD_COMMANDS>", 2, true);
    for (uint32_t commandIdx = 0; commandIdx < value.NumDmaRdCommands; ++commandIdx)
    {
        // Add helpful comment to indicate the command idx (very useful for long command streams)
        Parse(parent, "<!-- DmaRd Command " + std::to_string(commandIdx) + " -->", 3, true);
        Parse(parent, *dmaRdCommandsBegin, agentsArray);
        dmaRdCommandsBegin = getNextCommand(dmaRdCommandsBegin);
    }
    Parse(parent, "</DMA_RD_COMMANDS>", 2, true);

    Parse(parent, "<DMA_WR_COMMANDS>", 2, true);
    for (uint32_t commandIdx = 0; commandIdx < value.NumDmaWrCommands; ++commandIdx)
    {
        // Add helpful comment to indicate the command idx (very useful for long command streams)
        Parse(parent, "<!-- DmaWr Command " + std::to_string(commandIdx) + " -->", 3, true);
        Parse(parent, *dmaWrCommandsBegin, agentsArray);
        dmaWrCommandsBegin = getNextCommand(dmaWrCommandsBegin);
    }
    Parse(parent, "</DMA_WR_COMMANDS>", 2, true);

    Parse(parent, "<MCE_COMMANDS>", 2, true);
    for (uint32_t commandIdx = 0; commandIdx < value.NumMceCommands; ++commandIdx)
    {
        // Add helpful comment to indicate the command idx (very useful for long command streams)
        Parse(parent, "<!-- Mce Command " + std::to_string(commandIdx) + " -->", 3, true);
        Parse(parent, *mceCommandsBegin, agentsArray);
        mceCommandsBegin = getNextCommand(mceCommandsBegin);
    }
    Parse(parent, "</MCE_COMMANDS>", 2, true);

    Parse(parent, "<PLE_COMMANDS>", 2, true);
    for (uint32_t commandIdx = 0; commandIdx < value.NumPleCommands; ++commandIdx)
    {
        // Add helpful comment to indicate the command idx (very useful for long command streams)
        Parse(parent, "<!-- Ple Command " + std::to_string(commandIdx) + " -->", 3, true);
        Parse(parent, *pleCommandsBegin, agentsArray);
        pleCommandsBegin = getNextCommand(pleCommandsBegin);
    }
    Parse(parent, "</PLE_COMMANDS>", 2, true);

    Parse(parent, "</CASCADE>", 1, true);
}
}    // namespace

void ParseBinary(CommandStream& cstream, std::stringstream& output)
{
    output << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
    output << "<STREAM VERSION_MAJOR="
           << "\"" << std::to_string(cstream.GetVersionMajor()).c_str() << "\"";
    output << " VERSION_MINOR="
           << "\"" << std::to_string(cstream.GetVersionMinor()).c_str() << "\"";
    output << " VERSION_PATCH="
           << "\"" << std::to_string(cstream.GetVersionPatch()).c_str() << "\">\n";

    uint32_t commandCounter = 0;
    for (const CommandHeader& header : cstream)
    {
        Opcode command = header.m_Opcode();
        output << ("    <!-- Command " + std::to_string(commandCounter) + " -->\n").c_str();
        switch (command)
        {
            case Opcode::OPERATION_MCE_PLE:
            {
                Parse(output, header.GetCommand<Opcode::OPERATION_MCE_PLE>()->m_Data());
                break;
            }
            case Opcode::OPERATION_PLE_ONLY:
            {
                Parse(output, header.GetCommand<Opcode::OPERATION_PLE_ONLY>()->m_Data());
                break;
            }
            case Opcode::OPERATION_CONVERT:
            {
                Parse(output, header.GetCommand<Opcode::OPERATION_CONVERT>()->m_Data());
                break;
            }
            case Opcode::OPERATION_SPACE_TO_DEPTH:
            {
                Parse(output, header.GetCommand<Opcode::OPERATION_SPACE_TO_DEPTH>()->m_Data());
                break;
            }
            case Opcode::DUMP_DRAM:
            {
                Parse(output, header.GetCommand<Opcode::DUMP_DRAM>()->m_Data());
                break;
            }
            case Opcode::DUMP_SRAM:
            {
                Parse(output, header.GetCommand<Opcode::DUMP_SRAM>()->m_Data());
                break;
            }
            case Opcode::FENCE:
            {
                Parse(output, Fence{});
                break;
            }
            case Opcode::SECTION:
            {
                Parse(output, header.GetCommand<Opcode::SECTION>()->m_Data());
                break;
            }
            case Opcode::DELAY:
            {
                Parse(output, header.GetCommand<Opcode::DELAY>()->m_Data());
                break;
            }
            case Opcode::CASCADE:
            {
                Parse(output, header.GetCommand<Opcode::CASCADE>()->m_Data());
                break;
            }
            default:
            {
                // Bad binary
                throw ParseException("Invalid Opcode in binary input: " +
                                     std::to_string(static_cast<uint32_t>(command)));
            }
        }
        ++commandCounter;
    }
    output << "</STREAM>\n";
}

BinaryParser::BinaryParser(std::istream& input)
{
    std::vector<uint8_t> data = ReadBinaryData(input);

    CommandStream cstream(data.data(), data.data() + data.size());
    ParseBinary(cstream, out);
}

BinaryParser::BinaryParser(const std::vector<uint32_t>& data)
{
    CommandStream cstream(data.data(), data.data() + data.size());
    ParseBinary(cstream, out);
}

void BinaryParser::WriteXml(std::ostream& output)
{
    std::string temp = out.str();
    output.write(temp.c_str(), temp.size());
}
