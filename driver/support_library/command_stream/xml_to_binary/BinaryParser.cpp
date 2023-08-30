//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "BinaryParser.hpp"
#include <ethosn_command_stream/CommandStream.hpp>

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

void Parse(std::stringstream& parent, const IfmS& ifms)
{
    Parse(parent, "<IFM_STREAMER>", 2, true);

    Parse(parent, "<BUFFER_ID>", 3, false);
    ParseAsNum(parent, ifms.bufferId);
    Parse(parent, "</BUFFER_ID>", 0, true);

    Parse(parent, "<DMA_COMP_CONFIG0>", 3, false);
    ParseAsHex(parent, ifms.DMA_COMP_CONFIG0);
    Parse(parent, "</DMA_COMP_CONFIG0>", 0, true);

    Parse(parent, "<DMA_STRIDE1>", 3, false);
    ParseAsHex(parent, ifms.DMA_STRIDE1);
    Parse(parent, "</DMA_STRIDE1>", 0, true);

    Parse(parent, "<DMA_STRIDE2>", 3, false);
    ParseAsHex(parent, ifms.DMA_STRIDE2);
    Parse(parent, "</DMA_STRIDE2>", 0, true);

    Parse(parent, "</IFM_STREAMER>", 2, true);
}

void Parse(std::stringstream& parent, const OfmS& ofms)
{
    Parse(parent, "<OFM_STREAMER>", 2, true);

    Parse(parent, "<BUFFER_ID>", 3, false);
    ParseAsNum(parent, ofms.bufferId);
    Parse(parent, "</BUFFER_ID>", 0, true);

    Parse(parent, "<DMA_COMP_CONFIG0>", 3, false);
    ParseAsHex(parent, ofms.DMA_COMP_CONFIG0);
    Parse(parent, "</DMA_COMP_CONFIG0>", 0, true);

    Parse(parent, "<DMA_STRIDE1>", 3, false);
    ParseAsHex(parent, ofms.DMA_STRIDE1);
    Parse(parent, "</DMA_STRIDE1>", 0, true);

    Parse(parent, "<DMA_STRIDE2>", 3, false);
    ParseAsHex(parent, ofms.DMA_STRIDE2);
    Parse(parent, "</DMA_STRIDE2>", 0, true);

    Parse(parent, "</OFM_STREAMER>", 2, true);
}

void Parse(std::stringstream& parent, const WgtS& wgts)
{
    Parse(parent, "<WGT_STREAMER>", 2, true);

    Parse(parent, "<BUFFER_ID>", 3, false);
    ParseAsNum(parent, wgts.bufferId);
    Parse(parent, "</BUFFER_ID>", 0, true);

    Parse(parent, "</WGT_STREAMER>", 2, true);
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

void Parse(std::stringstream& parent, const MceS& mces)
{
    Parse(parent, "<MCE_SCHEDULER>", 2, true);

    Parse(parent, "<MCE_OP_MODE>", 3, false);
    Parse(parent, mces.mceOpMode);
    Parse(parent, "</MCE_OP_MODE>", 0, true);

    Parse(parent, "<PLE_KERNEL_ID>", 3, false);
    Parse(parent, PleKernelId2String(mces.pleKernelId), 0, false);
    Parse(parent, "</PLE_KERNEL_ID>", 0, true);

    Parse(parent, "<ACTIVATION_CONFIG>", 3, false);
    ParseAsHex(parent, mces.ACTIVATION_CONFIG);
    Parse(parent, "</ACTIVATION_CONFIG>", 0, true);

    Parse(parent, "<WIDE_KERNEL_CONTROL>", 3, false);
    ParseAsHex(parent, mces.WIDE_KERNEL_CONTROL);
    Parse(parent, "</WIDE_KERNEL_CONTROL>", 0, true);

    Parse(parent, "<FILTER>", 3, false);
    ParseAsHex(parent, mces.FILTER);
    Parse(parent, "</FILTER>", 0, true);

    Parse(parent, "<IFM_ZERO_POINT>", 3, false);
    ParseAsHex(parent, mces.IFM_ZERO_POINT);
    Parse(parent, "</IFM_ZERO_POINT>", 0, true);

    Parse(parent, "<IFM_DEFAULT_SLOT_SIZE>", 3, false);
    ParseAsHex(parent, mces.IFM_DEFAULT_SLOT_SIZE);
    Parse(parent, "</IFM_DEFAULT_SLOT_SIZE>", 0, true);

    Parse(parent, "<IFM_SLOT_STRIDE>", 3, false);
    ParseAsHex(parent, mces.IFM_SLOT_STRIDE);
    Parse(parent, "</IFM_SLOT_STRIDE>", 0, true);

    Parse(parent, "<STRIPE_BLOCK_CONFIG>", 3, false);
    ParseAsHex(parent, mces.STRIPE_BLOCK_CONFIG);
    Parse(parent, "</STRIPE_BLOCK_CONFIG>", 0, true);

    Parse(parent, "<DEPTHWISE_CONTROL>", 3, false);
    ParseAsHex(parent, mces.DEPTHWISE_CONTROL);
    Parse(parent, "</DEPTHWISE_CONTROL>", 0, true);

    Parse(parent, "<IFM_SLOT_BASE_ADDRESS>", 3, false);
    ParseAsHex(parent, mces.IFM_SLOT_BASE_ADDRESS);
    Parse(parent, "</IFM_SLOT_BASE_ADDRESS>", 0, true);

    Parse(parent, "<PLE_MCEIF_CONFIG>", 3, false);
    ParseAsHex(parent, mces.PLE_MCEIF_CONFIG);
    Parse(parent, "</PLE_MCEIF_CONFIG>", 0, true);

    Parse(parent, "</MCE_SCHEDULER>", 2, true);
}

void Parse(std::stringstream& parent, const PleL& plel)
{
    Parse(parent, "<PLE_LOADER>", 2, true);

    Parse(parent, "<PLE_KERNEL_ID>", 3, false);
    Parse(parent, PleKernelId2String(plel.pleKernelId), 0, false);
    Parse(parent, "</PLE_KERNEL_ID>", 0, true);

    Parse(parent, "</PLE_LOADER>", 2, true);
}

void Parse(std::stringstream& parent, const PleInputMode value)
{
    switch (value)
    {
        case PleInputMode::MCE_ALL_OGS:
        {
            Parse(parent, "MCE_ALL_OGS", 0, false);
            break;
        }
        case PleInputMode::MCE_ONE_OG:
        {
            Parse(parent, "MCE_ONE_OG", 0, false);
            break;
        }
        case PleInputMode::SRAM_ONE_INPUT:
        {
            Parse(parent, "SRAM_ONE_INPUT", 0, false);
            break;
        }
        case PleInputMode::SRAM_TWO_INPUTS:
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

void Parse(std::stringstream& parent, const PleS& ples)
{
    Parse(parent, "<PLE_SCHEDULER>", 2, true);

    Parse(parent, "<INPUT_MODE>", 3, false);
    Parse(parent, ples.inputMode);
    Parse(parent, "</INPUT_MODE>", 0, true);

    Parse(parent, "<PLE_KERNEL_ID>", 3, false);
    Parse(parent, PleKernelId2String(ples.pleKernelId), 0, false);
    Parse(parent, "</PLE_KERNEL_ID>", 0, true);

    Parse(parent, "<PLE_KERNEL_SRAM_ADDR>", 3, false);
    ParseAsNum(parent, ples.pleKernelSramAddr);
    Parse(parent, "</PLE_KERNEL_SRAM_ADDR>", 0, true);

    Parse(parent, "</PLE_SCHEDULER>", 2, true);
}

void Parse(std::stringstream& parent, const Agent& data)
{
    switch (data.type)
    {
        case AgentType::IFM_STREAMER:
            Parse(parent, data.ifm);
            break;
        case AgentType::WGT_STREAMER:
            Parse(parent, data.wgt);
            break;
        case AgentType::MCE_SCHEDULER:
            Parse(parent, data.mce);
            break;
        case AgentType::PLE_LOADER:
            Parse(parent, data.pleL);
            break;
        case AgentType::PLE_SCHEDULER:
            Parse(parent, data.pleS);
            break;
        case AgentType::OFM_STREAMER:
            Parse(parent, data.ofm);
            break;
        default:
        {
            // Bad binary
            throw ParseException("Invalid agent type: " + std::to_string(static_cast<uint32_t>(data.type)));
        }
    };
}

const char* CounterNameToString(CounterName c)
{
    switch (c)
    {
        case CounterName::DmaRd:
            return "DmaRd";
        case CounterName::DmaWr:
            return "DmaWr";
        case CounterName::Mceif:
            return "Mceif";
        case CounterName::MceStripe:
            return "MceStripe";
        case CounterName::PleCodeLoadedIntoPleSram:
            return "PleCodeLoadedIntoPleSram";
        case CounterName::PleStripe:
            return "PleStripe";
        default:
            throw ParseException("Invalid counter name: " + std::to_string(static_cast<uint32_t>(c)));
    }
}

void Parse(std::stringstream& parent, const WaitForCounterCommand& waitCommand)
{
    Parse(parent, "<WAIT_FOR_COUNTER_COMMAND>", 2, true);

    Parse(parent, "<COUNTER_NAME>", 3, false);
    Parse(parent, CounterNameToString(waitCommand.counterName), 0, false);
    Parse(parent, "</COUNTER_NAME>", 0, true);

    Parse(parent, "<COUNTER_VALUE>", 3, false);
    ParseAsNum(parent, waitCommand.counterValue);
    Parse(parent, "</COUNTER_VALUE>", 0, true);

    Parse(parent, "</WAIT_FOR_COUNTER_COMMAND>", 2, true);
}

const char* CommandTypeToString(CommandType t)
{
    switch (t)
    {
        case CommandType::WaitForCounter:
            return "WaitForCounter";
        case CommandType::LoadIfmStripe:
            return "LoadIfmStripe";
        case CommandType::LoadWgtStripe:
            return "LoadWgtStripe";
        case CommandType::ProgramMceStripe:
            return "ProgramMceStripe";
        case CommandType::ConfigMceif:
            return "ConfigMceif";
        case CommandType::StartMceStripe:
            return "StartMceStripe";
        case CommandType::LoadPleCodeIntoSram:
            return "LoadPleCodeIntoSram";
        case CommandType::LoadPleCodeIntoPleSram:
            return "LoadPleCodeIntoPleSram";
        case CommandType::StartPleStripe:
            return "StartPleStripe";
        case CommandType::StoreOfmStripe:
            return "StoreOfmStripe";
        default:
            throw std::runtime_error("Invalid command type: " + std::to_string(static_cast<uint32_t>(t)));
    }
}

void Parse(std::stringstream& parent, const DmaCommand& dmaCommand)
{
    // Add helpful comment to indicate the command type (DmaCommands are used as the storage for several different kinds of command)
    Parse(parent, ("<!-- Command type is " + std::string(CommandTypeToString(dmaCommand.type)) + " -->").c_str(), 2,
          true);
    Parse(parent, "<DMA_COMMAND>", 2, true);

    Parse(parent, "<AGENT_ID>", 3, false);
    ParseAsNum(parent, dmaCommand.agentId);
    Parse(parent, "</AGENT_ID>", 0, true);

    Parse(parent, "<DRAM_OFFSET>", 3, false);
    ParseAsHex(parent, dmaCommand.m_DramOffset);
    Parse(parent, "</DRAM_OFFSET>", 0, true);

    Parse(parent, "<SRAM_ADDR>", 3, false);
    ParseAsHex(parent, dmaCommand.SRAM_ADDR);
    Parse(parent, "</SRAM_ADDR>", 0, true);

    Parse(parent, "<DMA_SRAM_STRIDE>", 3, false);
    ParseAsHex(parent, dmaCommand.DMA_SRAM_STRIDE);
    Parse(parent, "</DMA_SRAM_STRIDE>", 0, true);

    Parse(parent, "<DMA_STRIDE0>", 3, false);
    ParseAsHex(parent, dmaCommand.DMA_STRIDE0);
    Parse(parent, "</DMA_STRIDE0>", 0, true);

    Parse(parent, "<DMA_STRIDE3>", 3, false);
    ParseAsHex(parent, dmaCommand.DMA_STRIDE3);
    Parse(parent, "</DMA_STRIDE3>", 0, true);

    Parse(parent, "<DMA_CHANNELS>", 3, false);
    ParseAsHex(parent, dmaCommand.DMA_CHANNELS);
    Parse(parent, "</DMA_CHANNELS>", 0, true);

    Parse(parent, "<DMA_EMCS>", 3, false);
    ParseAsHex(parent, dmaCommand.DMA_EMCS);
    Parse(parent, "</DMA_EMCS>", 0, true);

    Parse(parent, "<DMA_TOTAL_BYTES>", 3, false);
    ParseAsHex(parent, dmaCommand.DMA_TOTAL_BYTES);
    Parse(parent, "</DMA_TOTAL_BYTES>", 0, true);

    Parse(parent, "<DMA_CMD>", 3, false);
    ParseAsHex(parent, dmaCommand.DMA_CMD);
    Parse(parent, "</DMA_CMD>", 0, true);

    Parse(parent, "</DMA_COMMAND>", 2, true);
}

void Parse(std::stringstream& parent, const ProgramMceStripeCommand& programMceCommand)
{
    Parse(parent, "<PROGRAM_MCE_STRIPE_COMMAND>", 2, true);

    Parse(parent, "<AGENT_ID>", 3, false);
    ParseAsNum(parent, programMceCommand.agentId);
    Parse(parent, "</AGENT_ID>", 0, true);

    for (size_t ce = 0; ce < programMceCommand.MUL_ENABLE.size(); ++ce)
    {
        std::string beginElementName = std::string("<MUL_ENABLE_CE") + std::to_string(ce) + ">";
        Parse(parent, beginElementName.c_str(), 3, true);

        for (size_t og = 0; og < programMceCommand.MUL_ENABLE[ce].size(); ++og)
        {
            std::string beginElementName = std::string("<OG") + std::to_string(og) + ">";
            Parse(parent, beginElementName.c_str(), 4, false);

            ParseAsHex(parent, programMceCommand.MUL_ENABLE[ce][og]);

            std::string endElementName = std::string("</OG") + std::to_string(og) + ">";
            Parse(parent, endElementName.c_str(), 0, true);
        }

        std::string endElementName = std::string("</MUL_ENABLE_CE") + std::to_string(ce) + ">";
        Parse(parent, endElementName.c_str(), 3, true);
    }

    Parse(parent, "<IFM_ROW_STRIDE>", 3, false);
    ParseAsHex(parent, programMceCommand.IFM_ROW_STRIDE);
    Parse(parent, "</IFM_ROW_STRIDE>", 0, true);

    Parse(parent, "<IFM_CONFIG1>", 3, false);
    ParseAsHex(parent, programMceCommand.IFM_CONFIG1);
    Parse(parent, "</IFM_CONFIG1>", 0, true);

    for (size_t num = 0; num < programMceCommand.IFM_PAD.size(); ++num)
    {
        std::string beginElementName = std::string("<IFM_PAD_NUM") + std::to_string(num) + ">";
        Parse(parent, beginElementName.c_str(), 3, true);

        for (size_t ig = 0; ig < programMceCommand.IFM_PAD[num].size(); ++ig)
        {
            std::string beginElementName = std::string("<IG") + std::to_string(ig) + ">";
            Parse(parent, beginElementName.c_str(), 4, false);

            ParseAsHex(parent, programMceCommand.IFM_PAD[num][ig]);

            std::string endElementName = std::string("</IG") + std::to_string(ig) + ">";
            Parse(parent, endElementName.c_str(), 0, true);
        }

        std::string endElementName = std::string("</IFM_PAD_NUM") + std::to_string(num) + ">";
        Parse(parent, endElementName.c_str(), 3, true);
    }

    Parse(parent, "<WIDE_KERNEL_OFFSET>", 3, false);
    ParseAsHex(parent, programMceCommand.WIDE_KERNEL_OFFSET);
    Parse(parent, "</WIDE_KERNEL_OFFSET>", 0, true);

    Parse(parent, "<IFM_TOP_SLOTS>", 3, false);
    ParseAsHex(parent, programMceCommand.IFM_TOP_SLOTS);
    Parse(parent, "</IFM_TOP_SLOTS>", 0, true);

    Parse(parent, "<IFM_MID_SLOTS>", 3, false);
    ParseAsHex(parent, programMceCommand.IFM_MID_SLOTS);
    Parse(parent, "</IFM_MID_SLOTS>", 0, true);

    Parse(parent, "<IFM_BOTTOM_SLOTS>", 3, false);
    ParseAsHex(parent, programMceCommand.IFM_BOTTOM_SLOTS);
    Parse(parent, "</IFM_BOTTOM_SLOTS>", 0, true);

    Parse(parent, "<IFM_SLOT_PAD_CONFIG>", 3, false);
    ParseAsHex(parent, programMceCommand.IFM_SLOT_PAD_CONFIG);
    Parse(parent, "</IFM_SLOT_PAD_CONFIG>", 0, true);

    Parse(parent, "<OFM_STRIPE_SIZE>", 3, false);
    ParseAsHex(parent, programMceCommand.OFM_STRIPE_SIZE);
    Parse(parent, "</OFM_STRIPE_SIZE>", 0, true);

    Parse(parent, "<OFM_CONFIG>", 3, false);
    ParseAsHex(parent, programMceCommand.OFM_CONFIG);
    Parse(parent, "</OFM_CONFIG>", 0, true);

    for (size_t og = 0; og < programMceCommand.WEIGHT_BASE_ADDR.size(); ++og)
    {
        std::string beginElementName = std::string("<WEIGHT_BASE_ADDR_OG") + std::to_string(og) + ">";
        Parse(parent, beginElementName.c_str(), 3, false);
        ParseAsHex(parent, programMceCommand.WEIGHT_BASE_ADDR[og]);
        std::string endElementName = std::string("</WEIGHT_BASE_ADDR_OG") + std::to_string(og) + ">";
        Parse(parent, endElementName.c_str(), 0, true);
    }

    for (size_t ce = 0; ce < programMceCommand.IFM_CONFIG2.size(); ++ce)
    {
        std::string beginElementName = std::string("<IFM_CONFIG2_CE") + std::to_string(ce) + ">";
        Parse(parent, beginElementName.c_str(), 3, true);

        for (size_t ig = 0; ig < programMceCommand.IFM_CONFIG2[ce].size(); ++ig)
        {
            std::string beginElementName = std::string("<IG") + std::to_string(ig) + ">";
            Parse(parent, beginElementName.c_str(), 4, false);

            ParseAsHex(parent, programMceCommand.IFM_CONFIG2[ce][ig]);

            std::string endElementName = std::string("</IG") + std::to_string(ig) + ">";
            Parse(parent, endElementName.c_str(), 0, true);
        }

        std::string endElementName = std::string("</IFM_CONFIG2_CE") + std::to_string(ce) + ">";
        Parse(parent, endElementName.c_str(), 3, true);
    }

    Parse(parent, "<NUM_BLOCKS_PROGRAMMED_FOR_MCE>", 3, false);
    ParseAsHex(parent, programMceCommand.m_NumBlocksProgrammedForMce);
    Parse(parent, "</NUM_BLOCKS_PROGRAMMED_FOR_MCE>", 0, true);

    Parse(parent, "</PROGRAM_MCE_STRIPE_COMMAND>", 2, true);
}

void Parse(std::stringstream& parent, const ConfigMceifCommand& configMceifCommand)
{
    Parse(parent, "<CONFIG_MCEIF_COMMAND>", 2, true);

    Parse(parent, "<AGENT_ID>", 3, false);
    ParseAsNum(parent, configMceifCommand.agentId);
    Parse(parent, "</AGENT_ID>", 0, true);

    Parse(parent, "</CONFIG_MCEIF_COMMAND>", 2, true);
}

void Parse(std::stringstream& parent, const StartMceStripeCommand& startMceStripeCommand)
{
    Parse(parent, "<START_MCE_STRIPE_COMMAND>", 2, true);

    Parse(parent, "<AGENT_ID>", 3, false);
    ParseAsNum(parent, startMceStripeCommand.agentId);
    Parse(parent, "</AGENT_ID>", 0, true);

    Parse(parent, "<CE_ENABLES>", 3, false);
    ParseAsNum(parent, startMceStripeCommand.CE_ENABLES);
    Parse(parent, "</CE_ENABLES>", 0, true);

    Parse(parent, "</START_MCE_STRIPE_COMMAND>", 2, true);
}

void Parse(std::stringstream& parent, const LoadPleCodeIntoPleSramCommand& c)
{
    Parse(parent, "<LOAD_PLE_CODE_INTO_PLE_SRAM_COMMAND>", 2, true);

    Parse(parent, "<AGENT_ID>", 3, false);
    ParseAsNum(parent, c.agentId);
    Parse(parent, "</AGENT_ID>", 0, true);

    Parse(parent, "</LOAD_PLE_CODE_INTO_PLE_SRAM_COMMAND>", 2, true);
}

void Parse(std::stringstream& parent, const StartPleStripeCommand& startPleStripeCommand)
{
    Parse(parent, "<START_PLE_STRIPE_COMMAND>", 2, true);

    Parse(parent, "<AGENT_ID>", 3, false);
    ParseAsNum(parent, startPleStripeCommand.agentId);
    Parse(parent, "</AGENT_ID>", 0, true);

    for (size_t i = 0; i < startPleStripeCommand.SCRATCH.size(); ++i)
    {
        std::string beginElementName = std::string("<SCRATCH") + std::to_string(i) + ">";
        Parse(parent, beginElementName.c_str(), 3, false);
        ParseAsHex(parent, startPleStripeCommand.SCRATCH[i]);
        std::string endElementName = std::string("</SCRATCH") + std::to_string(i) + ">";
        Parse(parent, endElementName.c_str(), 0, true);
    }

    Parse(parent, "</START_PLE_STRIPE_COMMAND>", 2, true);
}

void Parse(std::stringstream& parent, const Command& cmd)
{
    switch (cmd.type)
    {
        case CommandType::WaitForCounter:
            Parse(parent, static_cast<const WaitForCounterCommand&>(cmd));
            break;
        case CommandType::LoadIfmStripe:
            Parse(parent, static_cast<const DmaCommand&>(cmd));
            break;
        case CommandType::LoadWgtStripe:
            Parse(parent, static_cast<const DmaCommand&>(cmd));
            break;
        case CommandType::ProgramMceStripe:
            Parse(parent, static_cast<const ProgramMceStripeCommand&>(cmd));
            break;
        case CommandType::ConfigMceif:
            Parse(parent, static_cast<const ConfigMceifCommand&>(cmd));
            break;
        case CommandType::StartMceStripe:
            Parse(parent, static_cast<const StartMceStripeCommand&>(cmd));
            break;
        case CommandType::LoadPleCodeIntoSram:
            Parse(parent, static_cast<const DmaCommand&>(cmd));
            break;
        case CommandType::LoadPleCodeIntoPleSram:
            Parse(parent, static_cast<const LoadPleCodeIntoPleSramCommand&>(cmd));
            break;
        case CommandType::StartPleStripe:
            Parse(parent, static_cast<const StartPleStripeCommand&>(cmd));
            break;
        case CommandType::StoreOfmStripe:
            Parse(parent, static_cast<const DmaCommand&>(cmd));
            break;
        default:
            throw ParseException("Invalid command type: " + std::to_string(static_cast<uint32_t>(cmd.type)));
    }
}

void Parse(std::stringstream& parent, const CommandStream& value)
{
    using Command = Command;

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

    Parse(parent, "<AGENTS>", 1, true);
    for (uint32_t agentId = 0; agentId < value.NumAgents; ++agentId)
    {
        // Add helpful comment to indicate the agent ID (very useful for long command streams)
        Parse(parent, "<!-- Agent " + std::to_string(agentId) + " -->", 2, true);
        Parse(parent, agentsArray[agentId]);
    }
    Parse(parent, "</AGENTS>", 1, true);

    Parse(parent, "<DMA_RD_COMMANDS>", 1, true);
    for (uint32_t commandIdx = 0; commandIdx < value.NumDmaRdCommands; ++commandIdx)
    {
        // Add helpful comment to indicate the command idx (very useful for long command streams)
        Parse(parent, "<!-- DmaRd Command " + std::to_string(commandIdx) + " -->", 2, true);
        Parse(parent, *dmaRdCommandsBegin);
        dmaRdCommandsBegin = getNextCommand(dmaRdCommandsBegin);
    }
    Parse(parent, "</DMA_RD_COMMANDS>", 1, true);

    Parse(parent, "<DMA_WR_COMMANDS>", 1, true);
    for (uint32_t commandIdx = 0; commandIdx < value.NumDmaWrCommands; ++commandIdx)
    {
        // Add helpful comment to indicate the command idx (very useful for long command streams)
        Parse(parent, "<!-- DmaWr Command " + std::to_string(commandIdx) + " -->", 2, true);
        Parse(parent, *dmaWrCommandsBegin);
        dmaWrCommandsBegin = getNextCommand(dmaWrCommandsBegin);
    }
    Parse(parent, "</DMA_WR_COMMANDS>", 1, true);

    Parse(parent, "<MCE_COMMANDS>", 1, true);
    for (uint32_t commandIdx = 0; commandIdx < value.NumMceCommands; ++commandIdx)
    {
        // Add helpful comment to indicate the command idx (very useful for long command streams)
        Parse(parent, "<!-- Mce Command " + std::to_string(commandIdx) + " -->", 2, true);
        Parse(parent, *mceCommandsBegin);
        mceCommandsBegin = getNextCommand(mceCommandsBegin);
    }
    Parse(parent, "</MCE_COMMANDS>", 1, true);

    Parse(parent, "<PLE_COMMANDS>", 1, true);
    for (uint32_t commandIdx = 0; commandIdx < value.NumPleCommands; ++commandIdx)
    {
        // Add helpful comment to indicate the command idx (very useful for long command streams)
        Parse(parent, "<!-- Ple Command " + std::to_string(commandIdx) + " -->", 2, true);
        Parse(parent, *pleCommandsBegin);
        pleCommandsBegin = getNextCommand(pleCommandsBegin);
    }
    Parse(parent, "</PLE_COMMANDS>", 1, true);
}
}    // namespace

void ParseBinary(CommandStreamParser& parser, std::stringstream& output)
{
    if (!parser.IsValid())
    {
        throw std::runtime_error("Invalid command stream");
    }

    output << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
    output << "<COMMAND_STREAM VERSION_MAJOR="
           << "\"" << std::to_string(parser.GetVersionMajor()).c_str() << "\"";
    output << " VERSION_MINOR="
           << "\"" << std::to_string(parser.GetVersionMinor()).c_str() << "\"";
    output << " VERSION_PATCH="
           << "\"" << std::to_string(parser.GetVersionPatch()).c_str() << "\">\n";

    Parse(output, *parser.GetData());

    output << "</COMMAND_STREAM>\n";
}

BinaryParser::BinaryParser(std::istream& input)
{
    std::vector<uint8_t> data = ReadBinaryData(input);

    CommandStreamParser parser(data.data(), data.data() + data.size());
    ParseBinary(parser, out);
}

BinaryParser::BinaryParser(const std::vector<uint32_t>& data)
{
    CommandStreamParser parser(data.data(), data.data() + data.size());
    ParseBinary(parser, out);
}

void BinaryParser::WriteXml(std::ostream& output)
{
    std::string temp = out.str();
    output.write(temp.c_str(), temp.size());
}
