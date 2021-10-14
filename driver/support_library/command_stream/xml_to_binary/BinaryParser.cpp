//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "BinaryParser.hpp"

#include <ethosn_command_stream/CommandStream.hpp>

#include <cassert>
#include <iomanip>
#include <sstream>
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

const char* XmlSaveCallback(mxml_node_t* const node, const int where)
{
    static size_t currentIndent = 0;

    bool childIsNull         = mxmlGetFirstChild(node) == nullptr;
    bool childChildIsNotNull = mxmlGetFirstChild(mxmlGetFirstChild(node)) != nullptr;

    if (where == MXML_WS_BEFORE_OPEN)
    {
        ++currentIndent;
    }

    if (where == MXML_WS_AFTER_CLOSE || ((where == MXML_WS_AFTER_OPEN) && childIsNull))
    {
        --currentIndent;
    }

    if ((mxmlGetParent(node) == nullptr) || (std::string(mxmlGetElement(node)) == g_XmlRootName))
    {
        currentIndent = 0;
    }

    // Measure indent by taking distance from the _end_ of a whitespace string
    // (stops weird valgrind error from previous constexpr return)
    static const char indents[] = "                    ";
    const char* indent          = indents + sizeof(indents) - 1 - (currentIndent * 2);
    assert((indent >= &indents[0]) && "Insufficient indent space in string");

    switch (where)
    {
        case MXML_WS_BEFORE_OPEN:
            return indent;
        case MXML_WS_AFTER_OPEN:
            return (childIsNull || childChildIsNotNull) ? "\n" : nullptr;
        case MXML_WS_BEFORE_CLOSE:
            return childChildIsNotNull ? indent : nullptr;
        case MXML_WS_AFTER_CLOSE:
            return "\n";
        default:
            return nullptr;
    }
}

void Parse(mxml_node_t& parent, const char* const value)
{
    mxmlNewText(&parent, 0, value);
}

void Parse(mxml_node_t& parent, const std::string& value)
{
    Parse(parent, value.c_str());
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
std::enable_if_t<std::is_integral<IntType>::value> Parse(mxml_node_t& parent, const IntType value)
{
    Parse(parent, IntegersToString(value));
}

void ParseAsHex(mxml_node_t& parent, const uint32_t value)
{
    Parse(parent, "0x" + IntegersToString<16>(value));
}

void Parse(mxml_node_t& parent, const DataType value)
{
    switch (value)
    {
        case DataType::U8:
        {
            Parse(parent, "U8");
            break;
        }
        case DataType::S8:
        {
            Parse(parent, "S8");
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid DataType in binary input: " + std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(mxml_node_t& parent, const DataFormat value)
{
    switch (value)
    {
        case DataFormat::NHWCB:
        {
            Parse(parent, "NHWCB");
            break;
        }
        case DataFormat::NHWC:
        {
            Parse(parent, "NHWC");
            break;
        }
        case DataFormat::NCHW:
        {
            Parse(parent, "NCHW");
            break;
        }
        case DataFormat::WEIGHT_STREAM:
        {
            Parse(parent, "WEIGHT_STREAM");
            break;
        }
        case DataFormat::FCAF_DEEP:
        {
            Parse(parent, "FCAF_DEEP");
            break;
        }
        case DataFormat::FCAF_WIDE:
        {
            Parse(parent, "FCAF_WIDE");
            break;
        }
        default:
        {
            // Bad binary
            throw ParseException("Invalid DataFormat in binary input: " + std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(mxml_node_t& parent, const SramAllocationStrategy value)
{
    switch (value)
    {
        case SramAllocationStrategy::STRATEGY_0:
        {
            Parse(parent, "STRATEGY_0");
            break;
        }
        case SramAllocationStrategy::STRATEGY_1:
        {
            Parse(parent, "STRATEGY_1");
            break;
        }
        case SramAllocationStrategy::STRATEGY_3:
        {
            Parse(parent, "STRATEGY_3");
            break;
        }
        case SramAllocationStrategy::STRATEGY_4:
        {
            Parse(parent, "STRATEGY_4");
            break;
        }
        case SramAllocationStrategy::STRATEGY_6:
        {
            Parse(parent, "STRATEGY_6");
            break;
        }
        case SramAllocationStrategy::STRATEGY_7:
        {
            Parse(parent, "STRATEGY_7");
            break;
        }
        case SramAllocationStrategy::STRATEGY_X:
        {
            Parse(parent, "STRATEGY_X");
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

void Parse(mxml_node_t& parent, const UpsampleType value)
{
    switch (value)
    {
        case UpsampleType::OFF:
        {
            Parse(parent, "OFF");
            break;
        }
        case UpsampleType::BILINEAR:
        {
            Parse(parent, "BILINEAR");
            break;
        }
        case UpsampleType::NEAREST_NEIGHBOUR:
        {
            Parse(parent, "NEAREST_NEIGHBOUR");
            break;
        }
        case UpsampleType::TRANSPOSE:
        {
            Parse(parent, "TRANSPOSE");
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

void Parse(mxml_node_t& parent, const MceAlgorithm value)
{
    switch (value)
    {
        case MceAlgorithm::DIRECT:
        {
            Parse(parent, "DIRECT");
            break;
        }
        case MceAlgorithm::WINOGRAD:
        {
            Parse(parent, "WINOGRAD");
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

void Parse(mxml_node_t& parent, const DataLocation value)
{
    switch (value)
    {
        case DataLocation::DRAM:
        {
            Parse(parent, "DRAM");
            break;
        }
        case DataLocation::SRAM:
        {
            Parse(parent, "SRAM");
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

void Parse(mxml_node_t& parent, const MceOperation value)
{
    switch (value)
    {
        case MceOperation::CONVOLUTION:
        {
            Parse(parent, "CONVOLUTION");
            break;
        }
        case MceOperation::DEPTHWISE_CONVOLUTION:
        {
            Parse(parent, "DEPTHWISE_CONVOLUTION");
            break;
        }
        case MceOperation::FULLY_CONNECTED:
        {
            Parse(parent, "FULLY_CONNECTED");
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

void Parse(mxml_node_t& parent, const SectionType value)
{
    switch (value)
    {
        case SectionType::SISO:
        {
            Parse(parent, "SISO");
            break;
        }
        case SectionType::SISO_CASCADED:
        {
            Parse(parent, "SISO_CASCADED");
            break;
        }
        case SectionType::SIMO:
        {
            Parse(parent, "SIMO");
            break;
        }
        case SectionType::SIMO_CASCADED:
        {
            Parse(parent, "SIMO_CASCADED");
            break;
        }
        case SectionType::SISO_BRANCHED_CASCADED:
        {
            Parse(parent, "SISO_BRANCHED_CASCADED");
            break;
        }
        case SectionType::MISO:
        {
            Parse(parent, "MISO");
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

void Parse(mxml_node_t& parent, const TensorShape& value)
{
    const std::string text =
        IntegersToString(std::get<0>(value), std::get<1>(value), std::get<2>(value), std::get<3>(value));
    Parse(parent, text);
}

void Parse(mxml_node_t& parent, const TensorInfo& value)
{
    Parse(*mxmlNewElement(&parent, "DATA_TYPE"), value.m_DataType());
    Parse(*mxmlNewElement(&parent, "DATA_FORMAT"), value.m_DataFormat());
    Parse(*mxmlNewElement(&parent, "TENSOR_SHAPE"), value.m_TensorShape());
    Parse(*mxmlNewElement(&parent, "SUPERTENSOR_SHAPE"), value.m_SupertensorShape());
    Parse(*mxmlNewElement(&parent, "SUPERTENSOR_OFFSET"), value.m_SupertensorOffset());
    Parse(*mxmlNewElement(&parent, "STRIPE_SHAPE"), value.m_StripeShape());
    // TileSize is represented as TILE_SHAPE in the XML, for compatibility with the prototype compiler and performance model.
    TensorShape tileShape{ value.m_TileSize(), 1, 1, 1 };
    Parse(*mxmlNewElement(&parent, "TILE_SHAPE"), tileShape);
    Parse(*mxmlNewElement(&parent, "DRAM_BUFFER_ID"), value.m_DramBufferId());
    ParseAsHex(*mxmlNewElement(&parent, "SRAM_OFFSET"), value.m_SramOffset());
    Parse(*mxmlNewElement(&parent, "ZERO_POINT"), value.m_ZeroPoint());
    Parse(*mxmlNewElement(&parent, "DATA_LOCATION"), value.m_DataLocation());
}

void Parse(mxml_node_t& parent, const SramConfig& value)
{
    Parse(*mxmlNewElement(&parent, "ALLOCATION_STRATEGY"), value.m_AllocationStrategy());
}

void Parse(mxml_node_t& parent, const BlockConfig& value)
{
    Parse(*mxmlNewElement(&parent, "BLOCK_WIDTH"), value.m_BlockWidth());
    Parse(*mxmlNewElement(&parent, "BLOCK_HEIGHT"), value.m_BlockHeight());
}

void Parse(mxml_node_t& parent, const MceData& value)
{
    mxml_node_t& mce = *mxmlNewElement(&parent, "MCE_OP_INFO");

    Parse(*mxmlNewElement(&mce, "STRIDE_X"), value.m_Stride().m_X());
    Parse(*mxmlNewElement(&mce, "STRIDE_Y"), value.m_Stride().m_Y());
    Parse(*mxmlNewElement(&mce, "PAD_TOP"), value.m_PadTop());
    Parse(*mxmlNewElement(&mce, "PAD_LEFT"), value.m_PadLeft());
    Parse(*mxmlNewElement(&mce, "UNINTERLEAVED_INPUT_SHAPE"), value.m_UninterleavedInputShape());
    Parse(*mxmlNewElement(&mce, "OUTPUT_SHAPE"), value.m_OutputShape());
    Parse(*mxmlNewElement(&mce, "OUTPUT_STRIPE_SHAPE"), value.m_OutputStripeShape());
    Parse(*mxmlNewElement(&mce, "OPERATION"), value.m_Operation());
    Parse(*mxmlNewElement(&mce, "ALGO"), value.m_Algorithm());
    Parse(*mxmlNewElement(&mce, "ACTIVATION_MIN"), value.m_ActivationMin());
    Parse(*mxmlNewElement(&mce, "ACTIVATION_MAX"), value.m_ActivationMax());
    Parse(*mxmlNewElement(&mce, "UPSAMPLE_TYPE"), value.m_UpsampleType());
}

void Parse(mxml_node_t& parent, const PleOperation value)
{
    switch (value)
    {
#define PARSE_PLE_OPERATION_CASE(op)                                                                                   \
    case PleOperation::op:                                                                                             \
    {                                                                                                                  \
        Parse(parent, #op);                                                                                            \
        break;                                                                                                         \
    }

        PARSE_PLE_OPERATION_CASE(ADDITION)
        PARSE_PLE_OPERATION_CASE(ADDITION_RESCALE)
        PARSE_PLE_OPERATION_CASE(AVGPOOL_3X3_1_1_UDMA)
        PARSE_PLE_OPERATION_CASE(INTERLEAVE_2X2_2_2)
        PARSE_PLE_OPERATION_CASE(MAXPOOL_2X2_2_2)
        PARSE_PLE_OPERATION_CASE(MAXPOOL_3X3_2_2_EVEN)
        PARSE_PLE_OPERATION_CASE(MAXPOOL_3X3_2_2_ODD)
        PARSE_PLE_OPERATION_CASE(MEAN_XY_7X7)
        PARSE_PLE_OPERATION_CASE(MEAN_XY_8X8)
        PARSE_PLE_OPERATION_CASE(PASSTHROUGH)
        PARSE_PLE_OPERATION_CASE(SIGMOID)
        PARSE_PLE_OPERATION_CASE(TRANSPOSE_XY)
        PARSE_PLE_OPERATION_CASE(LEAKY_RELU)
        PARSE_PLE_OPERATION_CASE(DOWNSAMPLE_2X2)

        default:
        {
            // Bad binary
            throw ParseException("Invalid PLE operation in binary input: " +
                                 std::to_string(static_cast<uint32_t>(value)));
        }
    }
}

void Parse(mxml_node_t& parent, const PleData& value)
{
    mxml_node_t& compute = *mxmlNewElement(&parent, "PLE_OP_INFO");

    ParseAsHex(*mxmlNewElement(&compute, "CE_SRAM"), value.m_CeSram());
    ParseAsHex(*mxmlNewElement(&compute, "PLE_SRAM"), value.m_PleSram());
    Parse(*mxmlNewElement(&compute, "OPERATION"), value.m_Operation());
    Parse(*mxmlNewElement(&compute, "RESCALE_MULTIPLIER0"), value.m_RescaleMultiplier0());
    Parse(*mxmlNewElement(&compute, "RESCALE_SHIFT0"), value.m_RescaleShift0());
    Parse(*mxmlNewElement(&compute, "RESCALE_MULTIPLIER1"), value.m_RescaleMultiplier1());
    Parse(*mxmlNewElement(&compute, "RESCALE_SHIFT1"), value.m_RescaleShift1());
}

void Parse(mxml_node_t& parent, const McePle& value)
{
    mxml_node_t& operation = *mxmlNewElement(&parent, "OPERATION_MCE_PLE");
    Parse(*mxmlNewElement(&operation, "INPUT_INFO"), value.m_InputInfo());
    Parse(*mxmlNewElement(&operation, "WEIGHT_INFO"), value.m_WeightInfo());
    Parse(*mxmlNewElement(&operation, "WEIGHTS_METADATA_BUFFER_ID"), value.m_WeightMetadataBufferId());
    Parse(*mxmlNewElement(&operation, "OUTPUT_INFO"), value.m_OutputInfo());
    Parse(*mxmlNewElement(&operation, "SRAM_CONFIG"), value.m_SramConfig());
    Parse(*mxmlNewElement(&operation, "BLOCK_CONFIG"), value.m_BlockConfig());
    Parse(operation, value.m_MceData());
    Parse(operation, value.m_PleData());
}

void Parse(mxml_node_t& parent, const PleOnly& value)
{
    mxml_node_t& operation = *mxmlNewElement(&parent, "OPERATION_PLE");

    Parse(*mxmlNewElement(&operation, "INPUT_INFO"), value.m_InputInfo());
    if (value.m_NumInputInfos() == 2)
    {
        Parse(*mxmlNewElement(&operation, "INPUT_INFO"), value.m_InputInfo2());
    }
    Parse(*mxmlNewElement(&operation, "OUTPUT_INFO"), value.m_OutputInfo());
    Parse(*mxmlNewElement(&operation, "SRAM_CONFIG"), value.m_SramConfig());

    Parse(operation, value.m_PleData());
}

void Parse(mxml_node_t& parent, const Softmax& value)
{
    mxml_node_t& operation = *mxmlNewElement(&parent, "OPERATION_SOFTMAX");

    Parse(*mxmlNewElement(&operation, "INPUT_INFO"), value.m_InputInfo());
    Parse(*mxmlNewElement(&operation, "OUTPUT_INFO"), value.m_OutputInfo());
    Parse(*mxmlNewElement(&operation, "SCALED_DIFF"), value.m_ScaledDiff());
    Parse(*mxmlNewElement(&operation, "EXP_ACCUMULATION"), value.m_ExpAccumulation());
    Parse(*mxmlNewElement(&operation, "INPUT_BETA_MULTIPLIER"), value.m_InputBetaMultiplier());
    Parse(*mxmlNewElement(&operation, "INPUT_BETA_LEFT_SHIFT"), value.m_InputBetaLeftShift());
    Parse(*mxmlNewElement(&operation, "DIFF_MIN"), value.m_DiffMin());
}

void Parse(mxml_node_t& parent, const Convert& value)
{
    mxml_node_t& operation = *mxmlNewElement(&parent, "OPERATION_CONVERT");

    Parse(*mxmlNewElement(&operation, "INPUT_INFO"), value.m_InputInfo());
    Parse(*mxmlNewElement(&operation, "OUTPUT_INFO"), value.m_OutputInfo());
}

void Parse(mxml_node_t& parent, const SpaceToDepth& value)
{
    mxml_node_t& operation = *mxmlNewElement(&parent, "OPERATION_SPACE_TO_DEPTH");

    Parse(*mxmlNewElement(&operation, "INPUT_INFO"), value.m_InputInfo());
    Parse(*mxmlNewElement(&operation, "OUTPUT_INFO"), value.m_OutputInfo());
    Parse(*mxmlNewElement(&operation, "USED_EMCS"), value.m_UsedEmcs());
    Parse(*mxmlNewElement(&operation, "INTERMEDIATE_1_SIZE"), value.m_Intermediate1Size());
    Parse(*mxmlNewElement(&operation, "INTERMEDIATE_2_SIZE"), value.m_Intermediate2Size());
}

void Parse(mxml_node_t& parent, const DumpDram& value)
{
    mxml_node_t& operation = *mxmlNewElement(&parent, "DUMP_DRAM");

    Parse(*mxmlNewElement(&operation, "DRAM_BUFFER_ID"), value.m_DramBufferId());
    Parse(*mxmlNewElement(&operation, "FILENAME"), value.m_Filename().data());
}

void Parse(mxml_node_t& parent, const DumpSram& value)
{
    mxml_node_t& operation = *mxmlNewElement(&parent, "DUMP_SRAM");

    Parse(*mxmlNewElement(&operation, "PREFIX"), value.m_Filename().data());
}

void Parse(mxml_node_t& parent, const Section& value)
{
    mxml_node_t* operation = mxmlNewElement(&parent, "SECTION");

    Parse(*mxmlNewElement(operation, "TYPE"), value.m_Type());
}

void Parse(mxml_node_t& parent, const Fence&)
{
    mxmlNewElement(&parent, "FENCE");
}

void Parse(mxml_node_t& parent, const Delay& value)
{
    mxml_node_t* operation = mxmlNewElement(&parent, "DELAY");

    Parse(*mxmlNewElement(operation, "VALUE"), value.m_Value());
}

void Parse(mxml_node_t& parent, const Cascade& value)
{
    mxml_node_t* operation = mxmlNewElement(&parent, "CASCADE");

    Parse(*mxmlNewElement(operation, "SIZE"), value.m_Size());
}
}    // namespace

BinaryParser::BinaryParser(std::istream& input)
    : m_XmlDoc(mxmlNewXML("1.0"))
{
    mxml_node_t* xmlRoot = mxmlNewElement(m_XmlDoc.get(), g_XmlRootName);

    std::vector<uint8_t> data = ReadBinaryData(input);

    CommandStream cstream(data.data(), data.data() + data.size());
    mxmlElementSetAttr(xmlRoot, "VERSION_MAJOR", std::to_string(cstream.GetVersionMajor()).c_str());
    mxmlElementSetAttr(xmlRoot, "VERSION_MINOR", std::to_string(cstream.GetVersionMinor()).c_str());
    mxmlElementSetAttr(xmlRoot, "VERSION_PATCH", std::to_string(cstream.GetVersionPatch()).c_str());

    uint32_t commandCounter = 0;
    for (const CommandHeader& header : cstream)
    {
        Opcode command = header.m_Opcode();
        mxmlNewElement(xmlRoot, ("!-- Command " + std::to_string(commandCounter) + "--").c_str());
        switch (command)
        {
            case Opcode::OPERATION_MCE_PLE:
            {
                Parse(*xmlRoot, header.GetCommand<Opcode::OPERATION_MCE_PLE>()->m_Data());
                break;
            }
            case Opcode::OPERATION_PLE_ONLY:
            {
                Parse(*xmlRoot, header.GetCommand<Opcode::OPERATION_PLE_ONLY>()->m_Data());
                break;
            }
            case Opcode::OPERATION_SOFTMAX:
            {
                Parse(*xmlRoot, header.GetCommand<Opcode::OPERATION_SOFTMAX>()->m_Data());
                break;
            }
            case Opcode::OPERATION_CONVERT:
            {
                Parse(*xmlRoot, header.GetCommand<Opcode::OPERATION_CONVERT>()->m_Data());
                break;
            }
            case Opcode::OPERATION_SPACE_TO_DEPTH:
            {
                Parse(*xmlRoot, header.GetCommand<Opcode::OPERATION_SPACE_TO_DEPTH>()->m_Data());
                break;
            }
            case Opcode::DUMP_DRAM:
            {
                Parse(*xmlRoot, header.GetCommand<Opcode::DUMP_DRAM>()->m_Data());
                break;
            }
            case Opcode::DUMP_SRAM:
            {
                Parse(*xmlRoot, header.GetCommand<Opcode::DUMP_SRAM>()->m_Data());
                break;
            }
            case Opcode::FENCE:
            {
                Parse(*xmlRoot, Fence{});
                break;
            }
            case Opcode::SECTION:
            {
                Parse(*xmlRoot, header.GetCommand<Opcode::SECTION>()->m_Data());
                break;
            }
            case Opcode::DELAY:
            {
                Parse(*xmlRoot, header.GetCommand<Opcode::DELAY>()->m_Data());
                break;
            }
            case Opcode::CASCADE:
            {
                Parse(*xmlRoot, header.GetCommand<Opcode::CASCADE>()->m_Data());
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
}

void BinaryParser::WriteXml(std::ostream& output, int wrapMargin)
{
    mxmlSetWrapMargin(wrapMargin);

    char dummyBuffer[1];
    const int bufferSize = mxmlSaveString(m_XmlDoc.get(), dummyBuffer, sizeof(dummyBuffer), &XmlSaveCallback);
    std::unique_ptr<char[]> buffer(std::make_unique<char[]>(bufferSize));

    const int bytesWritten = mxmlSaveString(m_XmlDoc.get(), buffer.get(), bufferSize, &XmlSaveCallback);

    if (bytesWritten != bufferSize)
    {
        throw IOException("IO error on XML write");
    }

    buffer[bufferSize - 1] = '\n';    // Replace NUL-terminator with newline so that the XML file ends in a newline
    output.write(buffer.get(), bufferSize);
    if (!output.good())
    {
        throw IOException("IO error on XML write");
    }
}
