//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "XmlParser.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace ethosn::command_stream;

namespace
{
void ParseIntegers(std::istringstream& iss)
{
    if (iss.fail())
    {
        throw ParseException("Wrong integer format in text: " + iss.str());
    }

    if (!iss.eof())
    {
        throw ParseException("Garbage characters after integer values: " + iss.str());
    }
}

template <typename I, typename... Is>
void ParseIntegers(std::istringstream& iss, I& value, Is&... more)
{
    using RawInt = std::conditional_t<std::is_unsigned<I>::value, unsigned int, int>;

    RawInt raw;
    iss >> raw;

    value = static_cast<I>(raw);

    if (value != raw)
    {
        RawInt min = static_cast<RawInt>(std::numeric_limits<I>::lowest());
        RawInt max = static_cast<RawInt>(std::numeric_limits<I>::max());
        std::ostringstream oss;
        oss << "Integer range [" << min << ", " << max << "] exceeded: " << iss.str();
        throw ParseException(oss.str());
    }

    ParseIntegers(iss, more...);
}

template <int Base = 10, typename... Is>
void ParseIntegers(const std::string& text, Is&... ints)
{
    std::istringstream iss(text);
    iss >> std::setbase(Base);
    ParseIntegers(iss, ints...);
}
}    // namespace

XmlParser::XmlParser(std::istream& input)
{
    std::string inputString(std::istreambuf_iterator<char>(input), {});
    XmlHandle xml(mxmlSAXLoadString(nullptr, inputString.c_str(), MXML_OPAQUE_CALLBACK, &SaxCallback, this));

    if (xml == nullptr)
    {
        throw ParseException("Invalid XML");
    }
}

void XmlParser::WriteBinary(std::ostream& output)
{
    const std::vector<uint32_t>& csData = m_CSBuffer.GetData();
    output.write(reinterpret_cast<const char*>(csData.data()), sizeof(csData[0]) * csData.size());

    if (!output.good())
    {
        throw IOException("IO error on binary write");
    }
}

template <Opcode O>
void XmlParser::Parse()
{
    CommandData<O> data;
    Pop(data);
    m_CSBuffer.EmplaceBack(data);
}

void XmlParser::Push(std::string key, std::string value)
{
    if (m_XmlData.find(key) == m_XmlData.end())
    {
        m_XmlData.emplace(key, std::queue<std::string>());
    }
    m_XmlData.at(key).push(value);
}

void XmlParser::Pop(DumpDram& value)
{
    Pop("DUMP_DRAM/DRAM_BUFFER_ID", value.m_DramBufferId());
    Pop("DUMP_DRAM/FILENAME", value.m_Filename());
}

void XmlParser::Pop(DumpSram& value)
{
    Pop("DUMP_SRAM/PREFIX", value.m_Filename());
}

std::string XmlParser::Pop(const std::string& key)
{
    std::string str;

    try
    {
        str = m_XmlData.at(key).front();
        m_XmlData.at(key).pop();
    }
    catch (const std::out_of_range&)
    {
        throw std::out_of_range(key + " not found");
    }

    return str;
}

void XmlParser::Pop(const std::string& key, Filename& value)
{
    std::string str = Pop(key);
    if (str.length() > value.size())
    {
        throw ParseException("Filename is too long");
    }
    std::copy(str.begin(), str.end(), value.data());
}

void XmlParser::Pop(McePle& value)
{
    Pop("INPUT_INFO/", value.m_InputInfo());
    Pop("WEIGHT_INFO/", value.m_WeightInfo());
    Pop("OPERATION_MCE_PLE/WEIGHTS_METADATA_BUFFER_ID", value.m_WeightMetadataBufferId());
    Pop("OUTPUT_INFO/", value.m_OutputInfo());
    Pop("SRAM_CONFIG/", value.m_SramConfig());
    Pop("BLOCK_CONFIG/", value.m_BlockConfig());
    Pop(value.m_MceData());
    Pop(value.m_PleData());
}

void XmlParser::Pop(PleOnly& value)
{
    value.m_NumInputInfos() = static_cast<uint32_t>(m_XmlData.at("INPUT_INFO/DATA_TYPE").size());
    Pop("INPUT_INFO/", value.m_InputInfo());
    if (value.m_NumInputInfos() == 2)
    {
        Pop("INPUT_INFO/", value.m_InputInfo2());
    }
    Pop("OUTPUT_INFO/", value.m_OutputInfo());
    Pop("SRAM_CONFIG/", value.m_SramConfig());
    Pop(value.m_PleData());
}

void XmlParser::Pop(Softmax& value)
{
    Pop("INPUT_INFO/", value.m_InputInfo());
    Pop("OUTPUT_INFO/", value.m_OutputInfo());
    Pop("OPERATION_SOFTMAX/SCALED_DIFF", value.m_ScaledDiff());
    Pop("OPERATION_SOFTMAX/EXP_ACCUMULATION", value.m_ExpAccumulation());
    Pop("OPERATION_SOFTMAX/INPUT_BETA_MULTIPLIER", value.m_InputBetaMultiplier());
    Pop("OPERATION_SOFTMAX/INPUT_BETA_LEFT_SHIFT", value.m_InputBetaLeftShift());
    Pop("OPERATION_SOFTMAX/DIFF_MIN", value.m_DiffMin());
}

void XmlParser::Pop(Convert& value)
{
    Pop("INPUT_INFO/", value.m_InputInfo());
    Pop("OUTPUT_INFO/", value.m_OutputInfo());
}

void XmlParser::Pop(SpaceToDepth& value)
{
    Pop("INPUT_INFO/", value.m_InputInfo());
    Pop("OUTPUT_INFO/", value.m_OutputInfo());
    Pop("OPERATION_SPACE_TO_DEPTH/USED_EMCS", value.m_UsedEmcs());
    Pop("OPERATION_SPACE_TO_DEPTH/INTERMEDIATE_1_SIZE", value.m_Intermediate1Size());
    Pop("OPERATION_SPACE_TO_DEPTH/INTERMEDIATE_2_SIZE", value.m_Intermediate2Size());
}

void XmlParser::Pop(const std::string& keyPrefix, TensorInfo& value)
{
    Pop(keyPrefix + "DATA_TYPE", value.m_DataType());
    Pop(keyPrefix + "DATA_FORMAT", value.m_DataFormat());
    Pop(keyPrefix + "TENSOR_SHAPE", value.m_TensorShape());
    Pop(keyPrefix + "SUPERTENSOR_SHAPE", value.m_SupertensorShape());
    Pop(keyPrefix + "SUPERTENSOR_OFFSET", value.m_SupertensorOffset());
    Pop(keyPrefix + "STRIPE_SHAPE", value.m_StripeShape());
    // TileSize is represented as TILE_SHAPE in the XML, for compatibility with the prototype compiler and performance model.
    TensorShape tileShape;
    Pop(keyPrefix + "TILE_SHAPE", tileShape);
    value.m_TileSize() = tileShape[0] * tileShape[1] * tileShape[2] * tileShape[3];
    Pop(keyPrefix + "DRAM_BUFFER_ID", value.m_DramBufferId());
    Pop(keyPrefix + "SRAM_OFFSET", value.m_SramOffset());
    Pop(keyPrefix + "ZERO_POINT", value.m_ZeroPoint());
    Pop(keyPrefix + "DATA_LOCATION", value.m_DataLocation());
}

void XmlParser::Pop(const std::string& keyPrefix, SramConfig& value)
{
    Pop(keyPrefix + "ALLOCATION_STRATEGY", value.m_AllocationStrategy());
}

void XmlParser::Pop(const std::string& keyPrefix, BlockConfig& value)
{
    Pop(keyPrefix + "BLOCK_WIDTH", value.m_BlockWidth());
    Pop(keyPrefix + "BLOCK_HEIGHT", value.m_BlockHeight());
}

void XmlParser::Pop(MceData& value)
{
    Pop("MCE_OP_INFO/STRIDE_X", value.m_Stride().m_X());
    Pop("MCE_OP_INFO/STRIDE_Y", value.m_Stride().m_Y());
    Pop("MCE_OP_INFO/PAD_TOP", value.m_PadTop());
    Pop("MCE_OP_INFO/PAD_LEFT", value.m_PadLeft());
    Pop("MCE_OP_INFO/UNINTERLEAVED_INPUT_SHAPE", value.m_UninterleavedInputShape());
    Pop("MCE_OP_INFO/OUTPUT_SHAPE", value.m_OutputShape());
    Pop("MCE_OP_INFO/OUTPUT_STRIPE_SHAPE", value.m_OutputStripeShape());
    Pop("MCE_OP_INFO/OPERATION", value.m_Operation());
    Pop("MCE_OP_INFO/ALGO", value.m_Algorithm());
    Pop("MCE_OP_INFO/ACTIVATION_MIN", value.m_ActivationMin());
    Pop("MCE_OP_INFO/ACTIVATION_MAX", value.m_ActivationMax());
    Pop("MCE_OP_INFO/UPSAMPLE_MODE", value.m_UpsampleMode());
}

void XmlParser::Pop(PleData& value)
{
    Pop("PLE_OP_INFO/CE_SRAM", value.m_CeSram());
    Pop("PLE_OP_INFO/PLE_SRAM", value.m_PleSram());
    Pop("PLE_OP_INFO/OPERATION", value.m_Operation());
    Pop("PLE_OP_INFO/RESCALE_MULTIPLIER0", value.m_RescaleMultiplier0());
    Pop("PLE_OP_INFO/RESCALE_SHIFT0", value.m_RescaleShift0());
    Pop("PLE_OP_INFO/RESCALE_MULTIPLIER1", value.m_RescaleMultiplier1());
    Pop("PLE_OP_INFO/RESCALE_SHIFT1", value.m_RescaleShift1());
}

void XmlParser::Pop(Section& value)
{
    Pop("SECTION/TYPE", value.m_Type());
}

void XmlParser::Pop(Delay& value)
{
    Pop("DELAY/VALUE", value.m_Value());
}

void XmlParser::Pop(const std::string& key, TensorShape& value)
{
    ParseIntegers(Pop(key), std::get<0>(value), std::get<1>(value), std::get<2>(value), std::get<3>(value));
}

void XmlParser::Pop(const std::string& key, bool& value)
{
    std::string str = Pop(key);

    if ((str == "true") || (str == "1"))
    {
        value = true;
    }
    else if ((str == "false") || (str == "0"))
    {
        value = false;
    }
    else
    {
        throw ParseException(key + " is not boolean type: " + str);
    }
}

void XmlParser::Pop(const std::string& key, uint8_t& value)
{
    ParseIntegers(Pop(key), value);
}

void XmlParser::Pop(const std::string& key, uint16_t& value)
{
    ParseIntegers(Pop(key), value);
}

void XmlParser::Pop(const std::string& key, int16_t& value)
{
    ParseIntegers(Pop(key), value);
}

void XmlParser::Pop(const std::string& key, int32_t& value)
{
    ParseIntegers(Pop(key), value);
}

void XmlParser::Pop(const std::string& key, uint32_t& value)
{
    ParseIntegers<0>(Pop(key), value);
}

void XmlParser::Pop(const std::string& key, DataType& value)
{
    std::string str = Pop(key);

    if (str == "U8")
    {
        value = DataType::U8;
    }
    else if (str == "S8")
    {
        value = DataType::S8;
    }
    else
    {
        throw ParseException(key + " is not a DataType: " + str);
    }
}

void XmlParser::Pop(const std::string& key, DataFormat& value)
{
    std::string str = Pop(key);

    if (str == "NHWCB_COMPRESSED")
    {
        value = DataFormat::NHWCB_COMPRESSED;
    }
    else if (str == "NHWCB")
    {
        value = DataFormat::NHWCB;
    }
    else if (str == "NHWC")
    {
        value = DataFormat::NHWC;
    }
    else if (str == "NCHW")
    {
        value = DataFormat::NCHW;
    }
    else if (str == "WEIGHT_STREAM")
    {
        value = DataFormat::WEIGHT_STREAM;
    }
    else if (str == "FCAF_DEEP")
    {
        value = DataFormat::FCAF_DEEP;
    }
    else if (str == "FCAF_WIDE")
    {
        value = DataFormat::FCAF_WIDE;
    }
    else
    {
        throw ParseException(key + " is not a DataFormat: " + str);
    }
}

void XmlParser::Pop(const std::string& key, SectionType& value)
{
    std::string str = Pop(key);

    if (str == "SISO")
    {
        value = SectionType::SISO;
    }
    else if (str == "SISO_CASCADED")
    {
        value = SectionType::SISO_CASCADED;
    }
    else if (str == "SIMO")
    {
        value = SectionType::SIMO;
    }
    else if (str == "SIMO_CASCADED")
    {
        value = SectionType::SIMO_CASCADED;
    }
    else if (str == "SISO_BRANCHED_CASCADED")
    {
        value = SectionType::SISO_BRANCHED_CASCADED;
    }
    else if (str == "MISO")
    {
        value = SectionType::MISO;
    }
    else
    {
        throw ParseException(key + " is not a SectionType: " + str);
    }
}

void XmlParser::Pop(const std::string& key, PleOperation& value)
{
    const std::string str = Pop(key);

#define POP_PLE_OPERATION_CASE(op)                                                                                     \
    if (str == #op)                                                                                                    \
    {                                                                                                                  \
        value = PleOperation::op;                                                                                      \
        return;                                                                                                        \
    }

    POP_PLE_OPERATION_CASE(ADDITION)
    POP_PLE_OPERATION_CASE(ADDITION_RESCALE)
    POP_PLE_OPERATION_CASE(AVGPOOL_3X3_1_1_UDMA)
    POP_PLE_OPERATION_CASE(INTERLEAVE_2X2_2_2)
    POP_PLE_OPERATION_CASE(MAXPOOL_2X2_2_2)
    POP_PLE_OPERATION_CASE(MAXPOOL_3X3_2_2_EVEN)
    POP_PLE_OPERATION_CASE(MAXPOOL_3X3_2_2_ODD)
    POP_PLE_OPERATION_CASE(MEAN_XY_7X7)
    POP_PLE_OPERATION_CASE(MEAN_XY_8X8)
    POP_PLE_OPERATION_CASE(PASSTHROUGH)
    POP_PLE_OPERATION_CASE(TRANSPOSE_XY)
    POP_PLE_OPERATION_CASE(LEAKY_RELU)
    POP_PLE_OPERATION_CASE(DOWNSAMPLE_2X2)

    throw ParseException(key + " is not a SectionType: " + str);
}

void XmlParser::Pop(const std::string& key, SramAllocationStrategy& value)
{
    std::string str = Pop(key);

    if (str == "STRATEGY_0")
    {
        value = SramAllocationStrategy::STRATEGY_0;
    }
    else if (str == "STRATEGY_1")
    {
        value = SramAllocationStrategy::STRATEGY_1;
    }
    else if (str == "STRATEGY_3")
    {
        value = SramAllocationStrategy::STRATEGY_3;
    }
    else if (str == "STRATEGY_4")
    {
        value = SramAllocationStrategy::STRATEGY_4;
    }
    else if (str == "STRATEGY_6")
    {
        value = SramAllocationStrategy::STRATEGY_6;
    }
    else if (str == "STRATEGY_7")
    {
        value = SramAllocationStrategy::STRATEGY_7;
    }
    else if (str == "STRATEGY_X")
    {
        value = SramAllocationStrategy::STRATEGY_X;
    }
    else
    {
        throw ParseException(key + " is not a SramAllocationStrategy: " + str);
    }
}

void XmlParser::Pop(const std::string& key, UpsampleType& value)
{
    std::string str = Pop(key);

    if (str == "OFF")
    {
        value = UpsampleType::OFF;
    }
    else if (str == "BILINEAR")
    {
        value = UpsampleType::BILINEAR;
    }
    else if (str == "NEAREST_NEIGHBOUR")
    {
        value = UpsampleType::NEAREST_NEIGHBOUR;
    }
    else if (str == "TRANSPOSE")
    {
        value = UpsampleType::TRANSPOSE;
    }
    else
    {
        throw ParseException(key + " is not a UpsampleType: " + str);
    }
}

void XmlParser::Pop(const std::string& key, MceOperation& value)
{
    std::string str = Pop(key);

    if (str == "CONVOLUTION")
    {
        value = MceOperation::CONVOLUTION;
    }
    else if (str == "DEPTHWISE_CONVOLUTION")
    {
        value = MceOperation::DEPTHWISE_CONVOLUTION;
    }
    else if (str == "FULLY_CONNECTED")
    {
        value = MceOperation::FULLY_CONNECTED;
    }
    else
    {
        throw ParseException(key + " is not an MceOperation: " + str);
    }
}

void XmlParser::Pop(const std::string& key, MceAlgorithm& value)
{
    std::string str = Pop(key);

    if (str == "DIRECT")
    {
        value = MceAlgorithm::DIRECT;
    }
    else if (str == "WINOGRAD")
    {
        value = MceAlgorithm::WINOGRAD;
    }
    else
    {
        throw ParseException(key + " is not an MceAlgorithm: " + str);
    }
}

void XmlParser::Pop(const std::string& key, DataLocation& value)
{
    std::string str = Pop(key);

    if (str == "DRAM")
    {
        value = DataLocation::DRAM;
    }
    else if (str == "SRAM")
    {
        value = DataLocation::SRAM;
    }
    else
    {
        throw ParseException(key + " is not a DataLocation: " + str);
    }
}

void XmlParser::SaxCallback(mxml_node_t* node, mxml_sax_event_t event, void* parserAsVoid)
{
    const auto parser = static_cast<XmlParser*>(parserAsVoid);

    if (mxmlGetParent(node) == nullptr)
    {
        // Retain root node so mxmlSAXLoadFile doesn't return NULL on success
        mxmlRetain(node);
    }

    switch (event)
    {
        case MXML_SAX_ELEMENT_OPEN:
        {
            if (strcmp(mxmlGetElement(node), g_XmlRootName) == 0)
            {
                const char* majorStr = mxmlElementGetAttr(node, "VERSION_MAJOR");
                const char* minorStr = mxmlElementGetAttr(node, "VERSION_MINOR");
                const char* patchStr = mxmlElementGetAttr(node, "VERSION_PATCH");

                if (majorStr == nullptr || std::to_string(ETHOSN_COMMAND_STREAM_VERSION_MAJOR) != majorStr ||
                    minorStr == nullptr || std::to_string(ETHOSN_COMMAND_STREAM_VERSION_MINOR) != minorStr ||
                    patchStr == nullptr || std::to_string(ETHOSN_COMMAND_STREAM_VERSION_PATCH) != patchStr)
                {
                    throw ParseException("Unsupported version");
                }
            }
            // Nothing to do
            break;
        }
        case MXML_SAX_DATA:
        {
            const std::string text = mxmlGetOpaque(node);

            if (!std::all_of(text.begin(), text.end(), isspace))
            {
                mxml_node_t* const element = mxmlGetParent(node);

                if ((element == nullptr) || (mxmlGetParent(element) == nullptr))
                {
                    throw ParseException("Bad XML structure");
                }

                std::string prefix = mxmlGetElement(mxmlGetParent(element));

                const std::string name = prefix + "/" + mxmlGetElement(element);

                parser->Push(name, text);
            }

            break;
        }
        case MXML_SAX_ELEMENT_CLOSE:
        {
            static const char* const rootName = "STREAM";

            mxml_node_t* const parent    = mxmlGetParent(node);
            const std::string parentName = (parent != nullptr) ? mxmlGetElement(parent) : "";

            if (parentName == rootName)
            {
                const std::string name = mxmlGetElement(node);

                if (name == "SECTION")
                {
                    parser->Parse<Opcode::SECTION>();
                }
                else if (name == "OPERATION_MCE_PLE")
                {
                    parser->Parse<Opcode::OPERATION_MCE_PLE>();
                }
                else if (name == "OPERATION_PLE")
                {
                    parser->Parse<Opcode::OPERATION_PLE_ONLY>();
                }
                else if (name == "OPERATION_SOFTMAX")
                {
                    parser->Parse<Opcode::OPERATION_SOFTMAX>();
                }
                else if (name == "OPERATION_CONVERT")
                {
                    parser->Parse<Opcode::OPERATION_CONVERT>();
                }
                else if (name == "OPERATION_SPACE_TO_DEPTH")
                {
                    parser->Parse<Opcode::OPERATION_SPACE_TO_DEPTH>();
                }
                else if (name == "FENCE")
                {
                    parser->m_CSBuffer.EmplaceBack(Fence{});
                }
                else if (name == "DUMP_DRAM")
                {
                    parser->Parse<Opcode::DUMP_DRAM>();
                }
                else if (name == "DUMP_SRAM")
                {
                    parser->Parse<Opcode::DUMP_SRAM>();
                }
                else if (name == "DELAY")
                {
                    parser->Parse<Opcode::DELAY>();
                }
                else
                {
                    // Ignore unknown element
                    parser->Push("UNKNOWN", name);
                }
            }

            break;
        }
        default:
        {
            // Do nothing
            break;
        }
    }
}
