//
// Copyright © 2018-2022 Arm Limited.
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

void XmlParser::Pop(cascading::AgentDependencyInfo& info)
{
    Pop("AGENT/NUM_STRIPES_TOTAL", info.numStripesTotal);

    for (size_t i = 0; !m_XmlData["SCHEDULE_DEPENDENCY/RELATIVE_AGENT_ID"].empty(); ++i)
    {
        Pop("SCHEDULE_DEPENDENCY", info.scheduleDependencies.at(i));
    }

    for (size_t i = 0; !m_XmlData["READ_DEPENDENCY/RELATIVE_AGENT_ID"].empty(); ++i)
    {
        Pop("READ_DEPENDENCY", info.readDependencies.at(i));
    }

    for (size_t i = 0; !m_XmlData["WRITE_DEPENDENCY/RELATIVE_AGENT_ID"].empty(); ++i)
    {
        Pop("WRITE_DEPENDENCY", info.writeDependencies.at(i));
    }
}

template <typename T>
void XmlParser::ParseKnownAgent()
{
    T data;
    Pop(data);
    cascading::AgentDependencyInfo info{};
    Pop(info);
    // We can't add to the command stream immediately, otherwise agents would
    // be added before the CASCADE command.
    m_PendingAgents.push_back(cascading::Agent{ data, info });
}

void XmlParser::ParseAgent()
{
    std::string agentType = Pop("AGENT_TYPE");

    if (agentType == "IFM_STREAMER")
    {
        ParseKnownAgent<cascading::IfmS>();
    }
    else if (agentType == "WGT_STREAMER")
    {
        ParseKnownAgent<cascading::WgtS>();
    }
    else if (agentType == "MCE_SCHEDULER")
    {
        ParseKnownAgent<cascading::MceS>();
    }
    else if (agentType == "PLE_LOADER")
    {
        ParseKnownAgent<cascading::PleL>();
    }
    else if (agentType == "PLE_SCHEDULER")
    {
        ParseKnownAgent<cascading::PleS>();
    }
    else if (agentType == "OFM_STREAMER")
    {
        ParseKnownAgent<cascading::OfmS>();
    }
    else
    {
        throw ParseException(agentType + " is not in cascading::AgentData union");
    }
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
    if (m_XmlData[key].empty())
    {
        throw std::out_of_range(key + " not found");
    }

    const std::string str = m_XmlData[key].front();
    m_XmlData[key].pop();

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
    Pop("MCE_OP_INFO/UPSAMPLE_TYPE", value.m_UpsampleType());
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

void XmlParser::Pop(Cascade& value)
{
    Pop("CASCADE/NUM_AGENTS", value.m_NumAgents());
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::FmSData& value)
{
    Pop(keyPrefix + "/DRAM_OFFSET", value.dramOffset);
    Pop(keyPrefix + "/BUFFER_ID", value.bufferId);
    Pop(keyPrefix + "/DATA_TYPE", value.dataType);
    Pop("FCAF_INFO", value.fcafInfo);
    Pop("TILE", value.tile);
    Pop("DFLT_STRIPE_SIZE", value.dfltStripeSize);
    Pop("EDGE_STRIPE_SIZE", value.edgeStripeSize);
    Pop("SUPERTENSOR_SIZE_IN_CELLS", value.supertensorSizeInCells);
    Pop("NUM_STRIPES", value.numStripes);
    Pop("STRIPE_ID_STRIDES", value.stripeIdStrides);
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::PackedBoundaryThickness& value)
{
    Pop(keyPrefix + "/LEFT", value.left);
    Pop(keyPrefix + "/TOP", value.top);
    Pop(keyPrefix + "/RIGHT", value.right);
    Pop(keyPrefix + "/BOTTOM", value.bottom);
}

void XmlParser::Pop(cascading::IfmS& value)
{
    Pop("IFM_STREAMER", value.fmData);
    Pop("PACKED_BOUNDARY_THICKNESS", value.packedBoundaryThickness);
}

void XmlParser::Pop(cascading::OfmS& value)
{
    Pop("OFM_STREAMER", value.fmData);
}

void XmlParser::Pop(cascading::WgtS& value)
{
    Pop("WGT_STREAMER/BUFFER_ID", value.bufferId);
    Pop("WGT_STREAMER/METADATA_BUFFER_ID", value.metadataBufferId);
    Pop("TILE", value.tile);
    Pop("NUM_STRIPES", value.numStripes);
    Pop("STRIPE_ID_STRIDES", value.stripeIdStrides);
}

void XmlParser::Pop(cascading::MceS& value)
{
    Pop("IFM_TILE", value.ifmTile);
    Pop("WGT_TILE", value.wgtTile);
    Pop("BLOCK_SIZE", value.blockSize);
    Pop("DFLT_STRIPE_SIZE", value.dfltStripeSize);
    Pop("EDGE_STRIPE_SIZE", value.edgeStripeSize);
    Pop("NUM_STRIPES", value.numStripes);
    Pop("STRIPE_ID_STRIDES", value.stripeIdStrides);
    Pop("CONV_STRIDE_XY", value.convStrideXy);
    Pop("MCE_SCHEDULER/IFM_ZERO_POINT", value.ifmZeroPoint);
    Pop("MCE_SCHEDULER/IS_IFM_SIGNED", value.isIfmSigned);
    Pop("MCE_SCHEDULER/IS_OFM_SIGNED", value.isOfmSigned);
    Pop("MCE_SCHEDULER/UPSAMPLE_TYPE", value.upsampleType);
    Pop("UPSAMPLE_EDGE_MODE", value.upsampleEdgeMode);
    Pop("MCE_SCHEDULER/MCE_OP_MODE", value.mceOpMode);
    Pop("MCE_SCHEDULER/ALGORITHM", value.algorithm);
    Pop("MCE_SCHEDULER/IS_WIDE_FILTER", value.isWideFilter);
    Pop("MCE_SCHEDULER/IS_EXTRA_IFM_STRIPE_AT_RIGHT_EDGE", value.isExtraIfmStripeAtRightEdge);
    Pop("MCE_SCHEDULER/IS_EXTRA_IFM_STRIPE_AT_BOTTOM_EDGE", value.isExtraIfmStripeAtBottomEdge);
    Pop("MCE_SCHEDULER/IS_PACKED_BOUNDARY_X", value.isPackedBoundaryX);
    Pop("MCE_SCHEDULER/IS_PACKED_BOUNDARY_Y", value.isPackedBoundaryY);
    Pop("FILTER_SHAPE", value.filterShape);
    Pop("PADDING", value.padding);
    Pop("IFM_DELTA_DEFAULT", value.ifmDeltaDefault);
    Pop("IFM_DELTA_EDGE", value.ifmDeltaEdge);
    Pop("IFM_STRIPE_SHAPE_DEFAULT", value.ifmStripeShapeDefault);
    Pop("IFM_STRIPE_SHAPE_EDGE", value.ifmStripeShapeEdge);
    Pop("RELU_ACTIV", value.reluActiv);
    Pop("MCE_SCHEDULER/PLE_KERNEL_ID", value.pleKernelId);
}

void XmlParser::Pop(cascading::PleL& value)
{
    Pop("PLE_LOADER/PLE_KERNEL_ID", value.pleKernelId);
    Pop("PLE_LOADER/SRAM_ADDR", value.sramAddr);
}

void XmlParser::Pop(cascading::PleS& value)
{
    Pop("OFM_TILE", value.ofmTile);
    Pop("PLE_SCHEDULER/OFM_ZERO_POINT", value.ofmZeroPoint);
    Pop("DFLT_STRIPE_SIZE", value.dfltStripeSize);
    Pop("EDGE_STRIPE_SIZE", value.edgeStripeSize);
    Pop("NUM_STRIPES", value.numStripes);
    Pop("STRIPE_ID_STRIDES", value.stripeIdStrides);
    Pop("PLE_SCHEDULER/INPUT_MODE", value.inputMode);
    Pop("PLE_SCHEDULER/PLE_KERNEL_ID", value.pleKernelId);
    Pop("PLE_SCHEDULER/PLE_KERNEL_SRAM_ADDR", value.pleKernelSramAddr);
    Pop("IFM_TILE_0", value.ifmTile0);
    Pop("IFM_INFO_0", value.ifmInfo0);
    Pop("IFM_TILE_1", value.ifmTile1);
    Pop("IFM_INFO_1", value.ifmInfo1);
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::Dependency& value)
{
    Pop(keyPrefix + "/RELATIVE_AGENT_ID", value.relativeAgentId);
    Pop("OUTER_RATIO/OTHER", value.outerRatio.other);
    Pop("OUTER_RATIO/SELF", value.outerRatio.self);
    Pop("INNER_RATIO/OTHER", value.innerRatio.other);
    Pop("INNER_RATIO/SELF", value.innerRatio.self);
    Pop(keyPrefix + "/BOUNDARY", value.boundary);
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::BlockSize& value)
{
    Pop(keyPrefix + "/HEIGHT", value.height);
    Pop(keyPrefix + "/WIDTH", value.width);
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::MceSWorkSize<uint16_t>& value)
{
    Pop(keyPrefix + "/OFM_HEIGHT", value.ofmHeight);
    Pop(keyPrefix + "/OFM_WIDTH", value.ofmWidth);
    Pop(keyPrefix + "/OFM_CHANNELS", value.ofmChannels);
    Pop(keyPrefix + "/IFM_CHANNELS", value.ifmChannels);
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::StrideXy<uint8_t>& value)
{
    Pop(keyPrefix + "/X", value.x);
    Pop(keyPrefix + "/Y", value.y);
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::ReluActivation& value)
{
    Pop(keyPrefix + "/MIN", value.min);
    Pop(keyPrefix + "/MAX", value.max);
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::WgtSWorkSize<uint16_t>& value)
{
    Pop(keyPrefix + "/OFM_CHANNELS", value.ofmChannels);
    Pop(keyPrefix + "/IFM_CHANNELS", value.ifmChannels);
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::PleIfmInfo& value)
{
    Pop(keyPrefix + "/ZERO_POINT", value.zeroPoint);
    Pop(keyPrefix + "/MULTIPLIER", value.multiplier);
    Pop(keyPrefix + "/SHIFT", value.shift);
}

void XmlParser::Pop(const std::string& key, cascading::UpsampleType& value)
{
    std::string str = Pop(key);

    if (str == "BILINEAR")
    {
        value = cascading::UpsampleType::BILINEAR;
    }
    else if (str == "NEAREST_NEIGHBOUR")
    {
        value = cascading::UpsampleType::NEAREST_NEIGHBOUR;
    }
    else if (str == "TRANSPOSE")
    {
        value = cascading::UpsampleType::TRANSPOSE;
    }
    else if (str == "OFF")
    {
        value = cascading::UpsampleType::TRANSPOSE;
    }
    else
    {
        throw ParseException(key + " is not a upsampleType: " + str);
    }
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::UpsampleEdgeModeType& value)
{
    Pop(keyPrefix + "/ROW", value.row);
    Pop(keyPrefix + "/COL", value.col);
}

void XmlParser::Pop(const std::string& key, cascading::UpsampleEdgeMode& value)
{
    std::string str = Pop(key);

    if (str == "GENERATE")
    {
        value = cascading::UpsampleEdgeMode::GENERATE;
    }
    else if (str == "DROP")
    {
        value = cascading::UpsampleEdgeMode::DROP;
    }
    else
    {
        throw ParseException(key + " is not an upsampleEdgeModeType: " + str);
    }
}

void XmlParser::Pop(const std::string& key, cascading::MceOperation& value)
{
    std::string str = Pop(key);

    if (str == "CONVOLUTION")
    {
        value = cascading::MceOperation::CONVOLUTION;
    }
    else if (str == "DEPTHWISE_CONVOLUTION")
    {
        value = cascading::MceOperation::DEPTHWISE_CONVOLUTION;
    }
    else if (str == "FULLY_CONNECTED")
    {
        value = cascading::MceOperation::FULLY_CONNECTED;
    }
    else
    {
        throw ParseException(key + " is not a MceOperation: " + str);
    }
}

void XmlParser::Pop(const std::string& key, cascading::MceAlgorithm& value)
{
    std::string str = Pop(key);

    if (str == "DIRECT")
    {
        value = cascading::MceAlgorithm::DIRECT;
    }
    else if (str == "WINOGRAD")
    {
        value = cascading::MceAlgorithm::WINOGRAD;
    }
    else
    {
        throw ParseException(key + " is not an MceAlgorithm: " + str);
    }
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::FilterShape& value)
{
    Pop(keyPrefix + "/WIDTH", value.width);
    Pop(keyPrefix + "/HEIGHT", value.height);
}

void XmlParser::Pop(const std::string&, std::array<cascading::FilterShape, 4>(&value))
{
    int idx = 0;
    for (auto& filterShape : value)
    {
        Pop("VALUE_" + std::to_string(idx), filterShape);
        idx++;
    }
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::Padding& value)
{
    Pop(keyPrefix + "/LEFT", value.left);
    Pop(keyPrefix + "/TOP", value.top);
}

void XmlParser::Pop(const std::string&, std::array<cascading::Padding, 4>(&value))
{
    int idx = 0;
    for (auto& padding : value)
    {
        Pop("VALUE_" + std::to_string(idx), padding);
        idx++;
    }
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::IfmDelta& value)
{
    Pop(keyPrefix + "/WIDTH", value.width);
    Pop(keyPrefix + "/HEIGHT", value.height);
}

void XmlParser::Pop(const std::string&, std::array<cascading::IfmDelta, 4>(&value))
{
    int idx = 0;
    for (auto& ifmDelta : value)
    {
        Pop("VALUE_" + std::to_string(idx), ifmDelta);
        idx++;
    }
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::IfmStripeShape& value)
{
    Pop(keyPrefix + "/WIDTH", value.width);
    Pop(keyPrefix + "/HEIGHT", value.height);
}

void XmlParser::Pop(const std::string& key, cascading::PleInputMode& value)
{
    std::string str = Pop(key);

    if (str == "MCE_ALL_OGS")
    {
        value = cascading::PleInputMode::MCE_ALL_OGS;
    }
    else if (str == "MCE_ONE_OG")
    {
        value = cascading::PleInputMode::MCE_ONE_OG;
    }
    else if (str == "SRAM")
    {
        value = cascading::PleInputMode::SRAM;
    }
    else
    {
        throw ParseException(key + " is not a PleInputMode: " + str);
    }
}

void XmlParser::Pop(const std::string& key, cascading::PleKernelId& value)
{
    std::string str = Pop(key);
    value           = cascading::String2PleKernelId(str.c_str());
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

void XmlParser::Pop(const std::string& key, int8_t& value)
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

    if (str == "NHWCB")
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

template <typename T>
void XmlParser::Pop(const std::string& keyPrefix, cascading::TensorSize<T>& value)
{
    Pop(keyPrefix + "/HEIGHT", value.height);
    Pop(keyPrefix + "/WIDTH", value.width);
    Pop(keyPrefix + "/CHANNELS", value.channels);
}

template <typename T>
void XmlParser::Pop(const std::string& keyPrefix, cascading::SupertensorSize<T>& value)
{
    Pop(keyPrefix + "/WIDTH", value.width);
    Pop(keyPrefix + "/CHANNELS", value.channels);
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::Tile& value)
{
    Pop(keyPrefix + "/BASE_ADDR", value.baseAddr);
    Pop(keyPrefix + "/NUM_SLOTS", value.numSlots);
    Pop(keyPrefix + "/SLOT_SIZE", value.slotSize);
}

void XmlParser::Pop(const std::string& keyPrefix, cascading::FcafInfo& value)
{
    Pop(keyPrefix + "/ZERO_POINT", value.zeroPoint);
    Pop(keyPrefix + "/SIGNED_ACTIVATION", value.signedActivation);
}

void XmlParser::Pop(const std::string& key, cascading::FmsDataType& value)
{
    const std::string str = Pop(key);

#define POP_DATA_TYPE_CASE(op)                                                                                         \
    if (str == #op)                                                                                                    \
    {                                                                                                                  \
        value = cascading::FmsDataType::op;                                                                            \
        return;                                                                                                        \
    }

    POP_DATA_TYPE_CASE(NHWC)
    POP_DATA_TYPE_CASE(FCAF_WIDE)
    POP_DATA_TYPE_CASE(FCAF_DEEP)
    POP_DATA_TYPE_CASE(NHWCB)

    throw ParseException(key + " is not a valid data type: " + str);
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
            // First child of Agent
            else if (((parser->GetXmlData().count("AGENT_TYPE") == 0) ||
                      parser->GetXmlData().at("AGENT_TYPE").empty()) &&
                     (strcmp(mxmlGetElement(mxmlGetParent(node)), "AGENT") == 0))
            {
                parser->Push("AGENT_TYPE", mxmlGetElement(node));
            }
            else
            {
                // Nothing to do
            }
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

            // The mxml parser has just finished an element - check if this was an element
            // which we need to parse (a command or an agent in a cascade)
            if (parentName == rootName || parentName == "CASCADE")
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
                else if (name == "CASCADE")
                {
                    parser->Parse<Opcode::CASCADE>();

                    // Add all agents parsed as part of this CASCADE element
                    // (we can't add them as we parse them, otherwise they'd
                    //  end up before the CASCADE command).
                    for (const cascading::Agent& a : parser->m_PendingAgents)
                    {
                        parser->m_CSBuffer.EmplaceBack(a);
                    }
                    parser->m_PendingAgents.clear();
                }
                else if (name == "AGENT")
                {
                    parser->ParseAgent();
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
