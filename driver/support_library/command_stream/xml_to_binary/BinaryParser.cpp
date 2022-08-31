//
// Copyright © 2018-2022 Arm Limited.
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

void Parse(std::stringstream& parent, const Softmax& value)
{
    Parse(parent, "<OPERATION_SOFTMAX>", 1, true);

    Parse(parent, "<INPUT_INFO>", 2, true);
    Parse(parent, value.m_InputInfo());
    Parse(parent, "</INPUT_INFO>", 2, true);

    Parse(parent, "<OUTPUT_INFO>", 2, true);
    Parse(parent, value.m_OutputInfo());
    Parse(parent, "</OUTPUT_INFO>", 2, true);

    Parse(parent, "<SCALED_DIFF>", 2, false);
    ParseAsNum(parent, value.m_ScaledDiff());
    Parse(parent, "</SCALED_DIFF>", 0, true);

    Parse(parent, "<EXP_ACCUMULATION>", 2, false);
    ParseAsNum(parent, value.m_ExpAccumulation());
    Parse(parent, "</EXP_ACCUMULATION>", 0, true);

    Parse(parent, "<INPUT_BETA_MULTIPLIER>", 2, false);
    ParseAsNum(parent, value.m_InputBetaMultiplier());
    Parse(parent, "</INPUT_BETA_MULTIPLIER>", 0, true);

    Parse(parent, "<INPUT_BETA_LEFT_SHIFT>", 2, false);
    ParseAsNum(parent, value.m_InputBetaLeftShift());
    Parse(parent, "</INPUT_BETA_LEFT_SHIFT>", 0, true);

    Parse(parent, "<DIFF_MIN>", 2, false);
    ParseAsNum(parent, value.m_DiffMin());
    Parse(parent, "</DIFF_MIN>", 0, true);

    Parse(parent, "</OPERATION_SOFTMAX>", 1, true);
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

template <typename T>
void Parse(std::stringstream& parent, const cascading::TensorSize<T>& size)
{
    Parse(parent, "<HEIGHT>", 5, false);
    ParseAsNum(parent, size.height);
    Parse(parent, "</HEIGHT>", 0, true);

    Parse(parent, "<WIDTH>", 5, false);
    ParseAsNum(parent, size.width);
    Parse(parent, "</WIDTH>", 0, true);

    Parse(parent, "<CHANNELS>", 5, false);
    ParseAsNum(parent, size.channels);
    Parse(parent, "</CHANNELS>", 0, true);
}

template <typename T>
void Parse(std::stringstream& parent, const cascading::SupertensorSize<T>& size)
{
    Parse(parent, "<WIDTH>", 5, false);
    ParseAsNum(parent, size.width);
    Parse(parent, "</WIDTH>", 0, true);

    Parse(parent, "<CHANNELS>", 5, false);
    ParseAsNum(parent, size.channels);
    Parse(parent, "</CHANNELS>", 0, true);
}

void Parse(std::stringstream& parent, const cascading::Tile& tile)
{
    Parse(parent, "<BASE_ADDR>", 5, false);
    ParseAsNum(parent, tile.baseAddr);
    Parse(parent, "</BASE_ADDR>", 0, true);

    Parse(parent, "<NUM_SLOTS>", 5, false);
    ParseAsNum(parent, tile.numSlots);
    Parse(parent, "</NUM_SLOTS>", 0, true);

    Parse(parent, "<SLOT_SIZE>", 5, false);
    ParseAsNum(parent, tile.slotSize);
    Parse(parent, "</SLOT_SIZE>", 0, true);
}

void Parse(std::stringstream& parent, const cascading::FmsDataType& dataType)
{
    switch (dataType)
    {
        case cascading::FmsDataType::NHWC:
        {
            Parse(parent, "NHWC", 0, false);
            break;
        }
        case cascading::FmsDataType::FCAF_WIDE:
        {
            Parse(parent, "FCAF_WIDE", 0, false);
            break;
        }
        case cascading::FmsDataType::FCAF_DEEP:
        {
            Parse(parent, "FCAF_DEEP", 0, false);
            break;
        }
        case cascading::FmsDataType::NHWCB:
        {
            Parse(parent, "NHWCB", 0, false);
            break;
        }
        default:
        {
            throw ParseException("Invalid Data Type in binary input: " +
                                 std::to_string(static_cast<uint32_t>(dataType)));
        }
    }
}

void Parse(std::stringstream& parent, const cascading::FcafInfo& fcafInfo)
{
    Parse(parent, "<ZERO_POINT>", 5, false);
    ParseAsNum(parent, fcafInfo.zeroPoint);
    Parse(parent, "</ZERO_POINT>", 0, true);

    Parse(parent, "<SIGNED_ACTIVATION>", 5, false);
    ParseAsNum(parent, fcafInfo.signedActivation);
    Parse(parent, "</SIGNED_ACTIVATION>", 0, true);
}

void Parse(std::stringstream& parent, const cascading::FmSData& fmData)
{
    Parse(parent, "<DRAM_OFFSET>", 4, false);
    ParseAsNum(parent, fmData.dramOffset);
    Parse(parent, "</DRAM_OFFSET>", 0, true);

    Parse(parent, "<BUFFER_ID>", 4, false);
    ParseAsNum(parent, fmData.bufferId);
    Parse(parent, "</BUFFER_ID>", 0, true);

    Parse(parent, "<DATA_TYPE>", 4, false);
    Parse(parent, fmData.dataType);
    Parse(parent, "</DATA_TYPE>", 0, true);

    Parse(parent, "<FCAF_INFO>", 4, true);
    Parse(parent, fmData.fcafInfo);
    Parse(parent, "</FCAF_INFO>", 4, true);

    Parse(parent, "<TILE>", 4, true);
    Parse(parent, fmData.tile);
    Parse(parent, "</TILE>", 4, true);

    Parse(parent, "<DFLT_STRIPE_SIZE>", 4, true);
    Parse(parent, fmData.dfltStripeSize);
    Parse(parent, "</DFLT_STRIPE_SIZE>", 4, true);

    Parse(parent, "<EDGE_STRIPE_SIZE>", 4, true);
    Parse(parent, fmData.edgeStripeSize);
    Parse(parent, "</EDGE_STRIPE_SIZE>", 4, true);

    Parse(parent, "<SUPERTENSOR_SIZE_IN_CELLS>", 4, true);
    Parse(parent, fmData.supertensorSizeInCells);
    Parse(parent, "</SUPERTENSOR_SIZE_IN_CELLS>", 4, true);

    Parse(parent, "<NUM_STRIPES>", 4, true);
    Parse(parent, fmData.numStripes);
    Parse(parent, "</NUM_STRIPES>", 4, true);

    Parse(parent, "<STRIPE_ID_STRIDES>", 4, true);
    Parse(parent, fmData.stripeIdStrides);
    Parse(parent, "</STRIPE_ID_STRIDES>", 4, true);
}

void Parse(std::stringstream& parent, const cascading::PackedBoundaryThickness& value)
{
    Parse(parent, "<LEFT>", 5, false);
    ParseAsNum(parent, value.left);
    Parse(parent, "</LEFT>", 0, true);

    Parse(parent, "<TOP>", 5, false);
    ParseAsNum(parent, value.top);
    Parse(parent, "</TOP>", 0, true);

    Parse(parent, "<RIGHT>", 5, false);
    ParseAsNum(parent, value.right);
    Parse(parent, "</RIGHT>", 0, true);

    Parse(parent, "<BOTTOM>", 5, false);
    ParseAsNum(parent, value.bottom);
    Parse(parent, "</BOTTOM>", 0, true);
}

void Parse(std::stringstream& parent, const cascading::IfmS& ifms)
{
    Parse(parent, "<IFM_STREAMER>", 3, true);
    Parse(parent, ifms.fmData);

    Parse(parent, "<PACKED_BOUNDARY_THICKNESS>", 4, true);
    Parse(parent, ifms.packedBoundaryThickness);
    Parse(parent, "</PACKED_BOUNDARY_THICKNESS>", 4, true);

    Parse(parent, "</IFM_STREAMER>", 3, true);
}

void Parse(std::stringstream& parent, const cascading::OfmS& ofms)
{
    Parse(parent, "<OFM_STREAMER>", 3, true);
    Parse(parent, ofms.fmData);
    Parse(parent, "</OFM_STREAMER>", 3, true);
}

void Parse(std::stringstream& parent, const cascading::WgtSWorkSize<uint16_t>& size)
{
    Parse(parent, "<OFM_CHANNELS>", 5, false);
    ParseAsNum(parent, size.ofmChannels);
    Parse(parent, "</OFM_CHANNELS>", 0, true);

    Parse(parent, "<IFM_CHANNELS>", 5, false);
    ParseAsNum(parent, size.ifmChannels);
    Parse(parent, "</IFM_CHANNELS>", 0, true);
}

void Parse(std::stringstream& parent, const cascading::WgtS& wgts)
{
    Parse(parent, "<WGT_STREAMER>", 3, true);

    Parse(parent, "<BUFFER_ID>", 4, false);
    ParseAsNum(parent, wgts.bufferId);
    Parse(parent, "</BUFFER_ID>", 0, true);

    Parse(parent, "<METADATA_BUFFER_ID>", 4, false);
    ParseAsNum(parent, wgts.metadataBufferId);
    Parse(parent, "</METADATA_BUFFER_ID>", 0, true);

    Parse(parent, "<TILE>", 4, true);
    Parse(parent, wgts.tile);
    Parse(parent, "</TILE>", 4, true);

    Parse(parent, "<NUM_STRIPES>", 4, true);
    Parse(parent, wgts.numStripes);
    Parse(parent, "</NUM_STRIPES>", 4, true);

    Parse(parent, "<STRIPE_ID_STRIDES>", 4, true);
    Parse(parent, wgts.stripeIdStrides);
    Parse(parent, "</STRIPE_ID_STRIDES>", 4, true);

    Parse(parent, "</WGT_STREAMER>", 3, true);
}

void Parse(std::stringstream& parent, const cascading::BlockSize& size)
{
    Parse(parent, "<HEIGHT>", 5, false);
    ParseAsNum(parent, size.height);
    Parse(parent, "</HEIGHT>", 0, true);

    Parse(parent, "<WIDTH>", 5, false);
    ParseAsNum(parent, size.width);
    Parse(parent, "</WIDTH>", 0, true);
}

void Parse(std::stringstream& parent, const cascading::MceSWorkSize<uint16_t>& size)
{
    Parse(parent, "<OFM_HEIGHT>", 5, false);
    ParseAsNum(parent, size.ofmHeight);
    Parse(parent, "</OFM_HEIGHT>", 0, true);

    Parse(parent, "<OFM_WIDTH>", 5, false);
    ParseAsNum(parent, size.ofmWidth);
    Parse(parent, "</OFM_WIDTH>", 0, true);

    Parse(parent, "<OFM_CHANNELS>", 5, false);
    ParseAsNum(parent, size.ofmChannels);
    Parse(parent, "</OFM_CHANNELS>", 0, true);

    Parse(parent, "<IFM_CHANNELS>", 5, false);
    ParseAsNum(parent, size.ifmChannels);
    Parse(parent, "</IFM_CHANNELS>", 0, true);
}

void Parse(std::stringstream& parent, const cascading::StrideXy<uint8_t>& size)
{
    Parse(parent, "<X>", 5, false);
    ParseAsNum(parent, size.x);
    Parse(parent, "</X>", 0, true);

    Parse(parent, "<Y>", 5, false);
    ParseAsNum(parent, size.y);
    Parse(parent, "</Y>", 0, true);
}

void Parse(std::stringstream& parent, const cascading::ReluActivation& relu)
{
    Parse(parent, "<MIN>", 5, false);
    ParseAsNum(parent, relu.min);
    Parse(parent, "</MIN>", 0, true);

    Parse(parent, "<MAX>", 5, false);
    ParseAsNum(parent, relu.max);
    Parse(parent, "</MAX>", 0, true);
}

void Parse(std::stringstream& parent, const cascading::UpsampleType& value)
{
    switch (value)
    {
        case cascading::UpsampleType::TRANSPOSE:
        {
            Parse(parent, "TRANSPOSE", 0, false);
            break;
        }
        case cascading::UpsampleType::NEAREST_NEIGHBOUR:
        {
            Parse(parent, "NEAREST NEIGHBOUR", 0, false);
            break;
        }
        case cascading::UpsampleType::BILINEAR:
        {
            Parse(parent, "BILINEAR", 0, false);
            break;
        }
        case cascading::UpsampleType::OFF:
        {
            Parse(parent, "OFF", 0, false);
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid upsampleType in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const cascading::UpsampleEdgeMode& value)
{
    switch (value)
    {
        case cascading::UpsampleEdgeMode::DROP:
        {
            Parse(parent, "DROP", 0, false);
            break;
        }
        case cascading::UpsampleEdgeMode::GENERATE:
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

void Parse(std::stringstream& parent, const cascading::UpsampleEdgeModeType& value)
{
    Parse(parent, "<ROW>", 5, false);
    Parse(parent, value.row);
    Parse(parent, "</ROW>", 0, true);

    Parse(parent, "<COL>", 5, false);
    Parse(parent, value.col);
    Parse(parent, "</COL>", 0, true);
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

void Parse(std::stringstream& parent, const cascading::MceAlgorithm value)
{
    switch (value)
    {
        case cascading::MceAlgorithm::DIRECT:
        {
            Parse(parent, "DIRECT", 0, false);
            break;
        }
        case cascading::MceAlgorithm::WINOGRAD:
        {
            Parse(parent, "WINOGRAD", 0, false);
            break;
        }
        default:
        {
            throw ParseException("Invalid MceAlgorithm in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(std::stringstream& parent, const cascading::FilterShape& filterShape)
{
    Parse(parent, "<WIDTH>", 6, false);
    ParseAsNum(parent, filterShape.width);
    Parse(parent, "</WIDTH>", 0, true);

    Parse(parent, "<HEIGHT>", 6, false);
    ParseAsNum(parent, filterShape.height);
    Parse(parent, "</HEIGHT>", 0, true);
}

void Parse(std::stringstream& parent, const std::array<cascading::FilterShape, 4>(&filterShape))
{
    int idx = 0;
    for (const auto& value : filterShape)
    {
        Parse(parent, ("<VALUE_" + std::to_string(idx) + ">").c_str(), 5, true);
        Parse(parent, value);
        Parse(parent, ("</VALUE_" + std::to_string(idx) + ">").c_str(), 5, true);
        idx++;
    }
}

void Parse(std::stringstream& parent, const cascading::Padding& padding)
{
    Parse(parent, "<LEFT>", 6, false);
    ParseAsNum(parent, padding.left);
    Parse(parent, "</LEFT>", 0, true);

    Parse(parent, "<TOP>", 6, false);
    ParseAsNum(parent, padding.top);
    Parse(parent, "</TOP>", 0, true);
}

void Parse(std::stringstream& parent, const std::array<cascading::Padding, 4>(&padding))
{
    int idx = 0;
    for (const auto& value : padding)
    {
        Parse(parent, ("<VALUE_" + std::to_string(idx) + ">").c_str(), 5, true);
        Parse(parent, value);
        Parse(parent, ("</VALUE_" + std::to_string(idx) + ">").c_str(), 5, true);
        idx++;
    }
}

void Parse(std::stringstream& parent, const cascading::IfmDelta& ifmDelta)
{
    Parse(parent, "<WIDTH>", 6, false);
    ParseAsNum(parent, ifmDelta.width);
    Parse(parent, "</WIDTH>", 0, true);

    Parse(parent, "<HEIGHT>", 6, false);
    ParseAsNum(parent, ifmDelta.height);
    Parse(parent, "</HEIGHT>", 0, true);
}

void Parse(std::stringstream& parent, const std::array<cascading::IfmDelta, 4>(&ifmDelta))
{
    int idx = 0;
    for (const auto& value : ifmDelta)
    {
        Parse(parent, ("<VALUE_" + std::to_string(idx) + ">").c_str(), 5, true);
        Parse(parent, value);
        Parse(parent, ("</VALUE_" + std::to_string(idx) + ">").c_str(), 5, true);
        idx++;
    }
}

void Parse(std::stringstream& parent, const cascading::IfmStripeShape& ifmStripeShape)
{
    Parse(parent, "<WIDTH>", 5, false);
    ParseAsNum(parent, ifmStripeShape.width);
    Parse(parent, "</WIDTH>", 0, true);

    Parse(parent, "<HEIGHT>", 5, false);
    ParseAsNum(parent, ifmStripeShape.height);
    Parse(parent, "</HEIGHT>", 0, true);
}

void Parse(std::stringstream& parent, const cascading::MceS& mces)
{
    Parse(parent, "<MCE_SCHEDULER>", 3, true);

    Parse(parent, "<IFM_TILE>", 4, true);
    Parse(parent, mces.ifmTile);
    Parse(parent, "</IFM_TILE>", 4, true);

    Parse(parent, "<WGT_TILE>", 4, true);
    Parse(parent, mces.wgtTile);
    Parse(parent, "</WGT_TILE>", 4, true);

    Parse(parent, "<BLOCK_SIZE>", 4, true);
    Parse(parent, mces.blockSize);
    Parse(parent, "</BLOCK_SIZE>", 4, true);

    Parse(parent, "<DFLT_STRIPE_SIZE>", 4, true);
    Parse(parent, mces.dfltStripeSize);
    Parse(parent, "</DFLT_STRIPE_SIZE>", 4, true);

    Parse(parent, "<EDGE_STRIPE_SIZE>", 4, true);
    Parse(parent, mces.edgeStripeSize);
    Parse(parent, "</EDGE_STRIPE_SIZE>", 4, true);

    Parse(parent, "<NUM_STRIPES>", 4, true);
    Parse(parent, mces.numStripes);
    Parse(parent, "</NUM_STRIPES>", 4, true);

    Parse(parent, "<STRIPE_ID_STRIDES>", 4, true);
    Parse(parent, mces.stripeIdStrides);
    Parse(parent, "</STRIPE_ID_STRIDES>", 4, true);

    Parse(parent, "<CONV_STRIDE_XY>", 4, true);
    Parse(parent, mces.convStrideXy);
    Parse(parent, "</CONV_STRIDE_XY>", 4, true);

    Parse(parent, "<IFM_ZERO_POINT>", 4, false);
    ParseAsNum(parent, mces.ifmZeroPoint);
    Parse(parent, "</IFM_ZERO_POINT>", 0, true);

    Parse(parent, "<IS_IFM_SIGNED>", 4, false);
    ParseAsNum(parent, mces.isIfmSigned);
    Parse(parent, "</IS_IFM_SIGNED>", 0, true);

    Parse(parent, "<IS_OFM_SIGNED>", 4, false);
    ParseAsNum(parent, mces.isOfmSigned);
    Parse(parent, "</IS_OFM_SIGNED>", 0, true);

    Parse(parent, "<UPSAMPLE_TYPE>", 4, false);
    Parse(parent, mces.upsampleType);
    Parse(parent, "</UPSAMPLE_TYPE>", 0, true);

    Parse(parent, "<UPSAMPLE_EDGE_MODE>", 4, true);
    Parse(parent, mces.upsampleEdgeMode);
    Parse(parent, "</UPSAMPLE_EDGE_MODE>", 4, true);

    Parse(parent, "<MCE_OP_MODE>", 4, false);
    Parse(parent, mces.mceOpMode);
    Parse(parent, "</MCE_OP_MODE>", 0, true);

    Parse(parent, "<ALGORITHM>", 4, false);
    Parse(parent, mces.algorithm);
    Parse(parent, "</ALGORITHM>", 0, true);

    Parse(parent, "<IS_WIDE_FILTER>", 4, false);
    ParseAsNum(parent, mces.isWideFilter);
    Parse(parent, "</IS_WIDE_FILTER>", 0, true);

    Parse(parent, "<IS_EXTRA_IFM_STRIPE_AT_RIGHT_EDGE>", 4, false);
    ParseAsNum(parent, mces.isExtraIfmStripeAtRightEdge);
    Parse(parent, "</IS_EXTRA_IFM_STRIPE_AT_RIGHT_EDGE>", 0, true);

    Parse(parent, "<IS_EXTRA_IFM_STRIPE_AT_BOTTOM_EDGE>", 4, false);
    ParseAsNum(parent, mces.isExtraIfmStripeAtBottomEdge);
    Parse(parent, "</IS_EXTRA_IFM_STRIPE_AT_BOTTOM_EDGE>", 0, true);

    Parse(parent, "<IS_PACKED_BOUNDARY_X>", 4, false);
    ParseAsNum(parent, mces.isPackedBoundaryX);
    Parse(parent, "</IS_PACKED_BOUNDARY_X>", 0, true);

    Parse(parent, "<IS_PACKED_BOUNDARY_Y>", 4, false);
    ParseAsNum(parent, mces.isPackedBoundaryY);
    Parse(parent, "</IS_PACKED_BOUNDARY_Y>", 0, true);

    Parse(parent, "<FILTER_SHAPE>", 4, true);
    Parse(parent, mces.filterShape);
    Parse(parent, "</FILTER_SHAPE>", 4, true);

    Parse(parent, "<PADDING>", 4, true);
    Parse(parent, mces.padding);
    Parse(parent, "</PADDING>", 4, true);

    Parse(parent, "<IFM_DELTA_DEFAULT>", 4, true);
    Parse(parent, mces.ifmDeltaDefault);
    Parse(parent, "</IFM_DELTA_DEFAULT>", 4, true);

    Parse(parent, "<IFM_DELTA_EDGE>", 4, true);
    Parse(parent, mces.ifmDeltaEdge);
    Parse(parent, "</IFM_DELTA_EDGE>", 4, true);

    Parse(parent, "<IFM_STRIPE_SHAPE_DEFAULT>", 4, true);
    Parse(parent, mces.ifmStripeShapeDefault);
    Parse(parent, "</IFM_STRIPE_SHAPE_DEFAULT>", 4, true);

    Parse(parent, "<IFM_STRIPE_SHAPE_EDGE>", 4, true);
    Parse(parent, mces.ifmStripeShapeEdge);
    Parse(parent, "</IFM_STRIPE_SHAPE_EDGE>", 4, true);

    Parse(parent, "<RELU_ACTIV>", 4, true);
    Parse(parent, mces.reluActiv);
    Parse(parent, "</RELU_ACTIV>", 4, true);

    Parse(parent, "<PLE_KERNEL_ID>", 4, false);
    Parse(parent, cascading::PleKernelId2String(mces.pleKernelId), 0, false);
    Parse(parent, "</PLE_KERNEL_ID>", 0, true);

    Parse(parent, "</MCE_SCHEDULER>", 3, true);
}

void Parse(std::stringstream& parent, const cascading::PleL& plel)
{
    Parse(parent, "<PLE_LOADER>", 3, true);

    Parse(parent, "<PLE_KERNEL_ID>", 4, false);
    Parse(parent, cascading::PleKernelId2String(plel.pleKernelId), 0, false);
    Parse(parent, "</PLE_KERNEL_ID>", 0, true);

    Parse(parent, "<SRAM_ADDR>", 4, false);
    ParseAsNum(parent, plel.sramAddr);
    Parse(parent, "</SRAM_ADDR>", 0, true);

    Parse(parent, "</PLE_LOADER>", 3, true);
}

void Parse(std::stringstream& parent, const cascading::PleIfmInfo& info)
{
    Parse(parent, "<ZERO_POINT>", 5, false);
    ParseAsNum(parent, info.zeroPoint);
    Parse(parent, "</ZERO_POINT>", 0, true);

    Parse(parent, "<MULTIPLIER>", 5, false);
    ParseAsNum(parent, info.multiplier);
    Parse(parent, "</MULTIPLIER>", 0, true);

    Parse(parent, "<SHIFT>", 5, false);
    ParseAsNum(parent, info.shift);
    Parse(parent, "</SHIFT>", 0, true);
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
        case cascading::PleInputMode::SRAM:
        {
            Parse(parent, "SRAM", 0, false);
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
    Parse(parent, "<PLE_SCHEDULER>", 3, true);

    Parse(parent, "<OFM_TILE>", 4, true);
    Parse(parent, ples.ofmTile);
    Parse(parent, "</OFM_TILE>", 4, true);

    Parse(parent, "<OFM_ZERO_POINT>", 4, false);
    ParseAsNum(parent, ples.ofmZeroPoint);
    Parse(parent, "</OFM_ZERO_POINT>", 0, true);

    Parse(parent, "<DFLT_STRIPE_SIZE>", 4, true);
    Parse(parent, ples.dfltStripeSize);
    Parse(parent, "</DFLT_STRIPE_SIZE>", 4, true);

    Parse(parent, "<EDGE_STRIPE_SIZE>", 4, true);
    Parse(parent, ples.edgeStripeSize);
    Parse(parent, "</EDGE_STRIPE_SIZE>", 4, true);

    Parse(parent, "<NUM_STRIPES>", 4, true);
    Parse(parent, ples.numStripes);
    Parse(parent, "</NUM_STRIPES>", 4, true);

    Parse(parent, "<STRIPE_ID_STRIDES>", 4, true);
    Parse(parent, ples.stripeIdStrides);
    Parse(parent, "</STRIPE_ID_STRIDES>", 4, true);

    Parse(parent, "<INPUT_MODE>", 4, false);
    Parse(parent, ples.inputMode);
    Parse(parent, "</INPUT_MODE>", 0, true);

    Parse(parent, "<PLE_KERNEL_ID>", 4, false);
    Parse(parent, cascading::PleKernelId2String(ples.pleKernelId), 0, false);
    Parse(parent, "</PLE_KERNEL_ID>", 0, true);

    Parse(parent, "<PLE_KERNEL_SRAM_ADDR>", 4, false);
    ParseAsNum(parent, ples.pleKernelSramAddr);
    Parse(parent, "</PLE_KERNEL_SRAM_ADDR>", 0, true);

    Parse(parent, "<IFM_TILE_0>", 4, true);
    Parse(parent, ples.ifmTile0);
    Parse(parent, "</IFM_TILE_0>", 4, true);

    Parse(parent, "<IFM_INFO_0>", 4, true);
    Parse(parent, ples.ifmInfo0);
    Parse(parent, "</IFM_INFO_0>", 4, true);

    Parse(parent, "<IFM_TILE_1>", 4, true);
    Parse(parent, ples.ifmTile1);
    Parse(parent, "</IFM_TILE_1>", 4, true);

    Parse(parent, "<IFM_INFO_1>", 4, true);
    Parse(parent, ples.ifmInfo1);
    Parse(parent, "</IFM_INFO_1>", 4, true);

    Parse(parent, "</PLE_SCHEDULER>", 3, true);
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

void Parse(std::stringstream& parent, const cascading::Ratio& ratio)
{
    Parse(parent, "<OTHER>", 5, false);
    ParseAsNum(parent, ratio.other);
    Parse(parent, "</OTHER>", 0, true);

    Parse(parent, "<SELF>", 5, false);
    ParseAsNum(parent, ratio.self);
    Parse(parent, "</SELF>", 0, true);
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
    };
}

void Parse(std::stringstream& parent,
           const char* depName,
           const cascading::Dependency& dep,
           const cascading::CommandStream cascadeCmdStream,
           uint32_t agentId)
{
    if (dep.relativeAgentId == 0)
    {
        return;
    }

    Parse(parent, "<" + static_cast<std::string>(depName) + ">", 3, true);

    Parse(parent, "<RELATIVE_AGENT_ID>", 4, false);
    ParseAsNum(parent, dep.relativeAgentId);
    Parse(parent, "</RELATIVE_AGENT_ID>", 0, true);

    // Add helpful comment to indicate the agent this one depends on
    const uint32_t otherAgentId =
        strcmp(depName, "READ_DEPENDENCY") == 0 ? agentId - dep.relativeAgentId : agentId + dep.relativeAgentId;
    Parse(parent,
          dep.relativeAgentId == 0 ? "<!-- DISABLED -->"
                                   : ("<!-- Other agent ID is " + std::to_string(otherAgentId) + " (" +
                                      AgentTypeToString(cascadeCmdStream[otherAgentId].data.type) + ") -->")
                                         .c_str(),
          4, true);

    Parse(parent, "<OUTER_RATIO>", 4, true);
    Parse(parent, dep.outerRatio);
    Parse(parent, "</OUTER_RATIO>", 4, true);

    Parse(parent, "<INNER_RATIO>", 4, true);
    Parse(parent, dep.innerRatio);
    Parse(parent, "</INNER_RATIO>", 4, true);

    Parse(parent, "<BOUNDARY>", 4, false);
    ParseAsNum(parent, dep.boundary);
    Parse(parent, "</BOUNDARY>", 0, true);

    Parse(parent, "</" + static_cast<std::string>(depName) + ">", 3, true);
}

void Parse(std::stringstream& parent,
           const cascading::AgentDependencyInfo& agentInfo,
           const cascading::CommandStream cascadeCmdStream,
           uint32_t agentId)
{
    Parse(parent, "<NUM_STRIPES_TOTAL>", 3, false);
    ParseAsNum(parent, agentInfo.numStripesTotal);
    Parse(parent, "</NUM_STRIPES_TOTAL>", 0, true);

    for (auto& dep : agentInfo.scheduleDependencies)
    {
        Parse(parent, "SCHEDULE_DEPENDENCY", dep, cascadeCmdStream, agentId);
    }
    for (auto& dep : agentInfo.readDependencies)
    {
        Parse(parent, "READ_DEPENDENCY", dep, cascadeCmdStream, agentId);
    }
    for (auto& dep : agentInfo.writeDependencies)
    {
        Parse(parent, "WRITE_DEPENDENCY", dep, cascadeCmdStream, agentId);
    }
}

void Parse(std::stringstream& parent, const Cascade& value)
{
    Parse(parent, "<CASCADE>", 1, true);

    Parse(parent, "<NUM_AGENTS>", 2, false);
    ParseAsNum(parent, value.m_NumAgents());
    Parse(parent, "</NUM_AGENTS>", 0, true);

    const void* const cascadeBegin = &value + 1U;
    const cascading::CommandStream cascade{ static_cast<const cascading::Agent*>(cascadeBegin), value.m_NumAgents() };

    uint32_t agentId = 0;
    for (auto& agent : cascade)
    {
        // Add helpful comment to indicate the agent ID (very useful for long command streams)
        Parse(parent, "<!-- Agent " + std::to_string(agentId) + " -->", 2, true);
        Parse(parent, "<AGENT>", 2, true);
        Parse(parent, agent.data);
        Parse(parent, agent.info, cascade, agentId);
        Parse(parent, "</AGENT>", 2, true);

        ++agentId;
    }
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
            case Opcode::OPERATION_SOFTMAX:
            {
                Parse(output, header.GetCommand<Opcode::OPERATION_SOFTMAX>()->m_Data());
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
    const std::string& temp = out.str();
    const char* cstr        = temp.c_str();

    output.write(cstr, strlen(cstr));
}
