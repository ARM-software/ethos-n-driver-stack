//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../BinaryParser.hpp"
#include "../CMMParser.hpp"

#include <ethosn_utils/Strings.hpp>

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuilder.hpp>

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
<COMMAND_STREAM VERSION_MAJOR="%VERSION_MAJOR%" VERSION_MINOR="%VERSION_MINOR%" VERSION_PATCH="%VERSION_PATCH%">
    <AGENTS>
        <!-- Agent 0 -->
        <WGT_STREAMER>
            <BUFFER_ID>3</BUFFER_ID>
        </WGT_STREAMER>
        <!-- Agent 1 -->
        <IFM_STREAMER>
            <BUFFER_ID>3</BUFFER_ID>
            <DMA_COMP_CONFIG0>0x3534265</DMA_COMP_CONFIG0>
            <DMA_STRIDE1>0x23424</DMA_STRIDE1>
            <DMA_STRIDE2>0x213426</DMA_STRIDE2>
        </IFM_STREAMER>
        <!-- Agent 2 -->
        <OFM_STREAMER>
            <BUFFER_ID>0</BUFFER_ID>
            <DMA_COMP_CONFIG0>0x89679</DMA_COMP_CONFIG0>
            <DMA_STRIDE1>0x12346</DMA_STRIDE1>
            <DMA_STRIDE2>0x209347f</DMA_STRIDE2>
        </OFM_STREAMER>
        <!-- Agent 3 -->
        <MCE_SCHEDULER>
            <MCE_OP_MODE>DEPTHWISE_CONVOLUTION</MCE_OP_MODE>
            <PLE_KERNEL_ID>V2442_DOWNSAMPLE_2X2_bw16_bh16_bm1</PLE_KERNEL_ID>
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
        <!-- Agent 4 -->
        <PLE_LOADER>
            <PLE_KERNEL_ID>V2442_SIGMOID_bw16_bh16_bm1_s8</PLE_KERNEL_ID>
        </PLE_LOADER>
        <!-- Agent 5 -->
        <PLE_SCHEDULER>
            <INPUT_MODE>MCE_ONE_OG</INPUT_MODE>
            <PLE_KERNEL_ID>V2442_DOWNSAMPLE_2X2_bw16_bh16_bm1</PLE_KERNEL_ID>
            <PLE_KERNEL_SRAM_ADDR>4096</PLE_KERNEL_SRAM_ADDR>
        </PLE_SCHEDULER>
    </AGENTS>)"
    R"(
    <DMA_RD_COMMANDS>
        <!-- DmaRd Command 0 -->
        <!-- Command type is LoadIfmStripe -->
        <DMA_COMMAND>
            <AGENT_ID>0</AGENT_ID>
            <DRAM_OFFSET>0x123412</DRAM_OFFSET>
            <SRAM_ADDR>0x6543</SRAM_ADDR>
            <DMA_SRAM_STRIDE>0x2345</DMA_SRAM_STRIDE>
            <DMA_STRIDE0>0x7995</DMA_STRIDE0>
            <DMA_STRIDE3>0x23245</DMA_STRIDE3>
            <DMA_CHANNELS>0x12345</DMA_CHANNELS>
            <DMA_EMCS>0x989</DMA_EMCS>
            <DMA_TOTAL_BYTES>0xfea</DMA_TOTAL_BYTES>
            <DMA_CMD>0xa</DMA_CMD>
        </DMA_COMMAND>
    </DMA_RD_COMMANDS>
    <DMA_WR_COMMANDS>
        <!-- DmaWr Command 0 -->
        <!-- Command type is StoreOfmStripe -->
        <DMA_COMMAND>
            <AGENT_ID>2</AGENT_ID>
            <DRAM_OFFSET>0xabe</DRAM_OFFSET>
            <SRAM_ADDR>0x6ee</SRAM_ADDR>
            <DMA_SRAM_STRIDE>0xebbb5</DMA_SRAM_STRIDE>
            <DMA_STRIDE0>0x79aa</DMA_STRIDE0>
            <DMA_STRIDE3>0xdef</DMA_STRIDE3>
            <DMA_CHANNELS>0xffeed</DMA_CHANNELS>
            <DMA_EMCS>0xdd2</DMA_EMCS>
            <DMA_TOTAL_BYTES>0xfa12a</DMA_TOTAL_BYTES>
            <DMA_CMD>0x11a</DMA_CMD>
        </DMA_COMMAND>
    </DMA_WR_COMMANDS>)"
    R"(
    <MCE_COMMANDS>
        <!-- Mce Command 0 -->
        <PROGRAM_MCE_STRIPE_COMMAND>
            <AGENT_ID>0</AGENT_ID>
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
        </PROGRAM_MCE_STRIPE_COMMAND>
        <!-- Mce Command 1 -->
        <CONFIG_MCEIF_COMMAND>
            <AGENT_ID>0</AGENT_ID>
        </CONFIG_MCEIF_COMMAND>
        <!-- Mce Command 2 -->
        <START_MCE_STRIPE_COMMAND>
            <AGENT_ID>0</AGENT_ID>
            <CE_ENABLES>74666</CE_ENABLES>
        </START_MCE_STRIPE_COMMAND>
    </MCE_COMMANDS>
    <PLE_COMMANDS>
        <!-- Ple Command 0 -->
        <WAIT_FOR_COUNTER_COMMAND>
            <COUNTER_NAME>DmaRd</COUNTER_NAME>
            <COUNTER_VALUE>0</COUNTER_VALUE>
        </WAIT_FOR_COUNTER_COMMAND>
        <!-- Ple Command 1 -->
        <LOAD_PLE_CODE_INTO_PLE_SRAM_COMMAND>
            <AGENT_ID>0</AGENT_ID>
        </LOAD_PLE_CODE_INTO_PLE_SRAM_COMMAND>
        <!-- Ple Command 2 -->
        <START_PLE_STRIPE_COMMAND>
            <AGENT_ID>0</AGENT_ID>
            <SCRATCH0>0x125aa</SCRATCH0>
            <SCRATCH1>0x126aa</SCRATCH1>
            <SCRATCH2>0x127aa</SCRATCH2>
            <SCRATCH3>0x128aa</SCRATCH3>
            <SCRATCH4>0x129aa</SCRATCH4>
            <SCRATCH5>0x130aa</SCRATCH5>
            <SCRATCH6>0x131aa</SCRATCH6>
            <SCRATCH7>0x132aa</SCRATCH7>
        </START_PLE_STRIPE_COMMAND>
    </PLE_COMMANDS>
</COMMAND_STREAM>
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
    /* Agent 0 = */
    Agent agent0 = {
        /* AgentData = */
        { WgtS{
            /* BufferId = */ uint16_t{ 3 },
        } },
    };

    /* Agent 1 = */
    Agent agent1 = {
        /* AgentData = */
        { IfmS{
            /* BufferId = */ uint16_t{ 3 },
            /* DMA_COMP_CONFIG0 = */ uint32_t{ 0x3534265 },
            /* DMA_STRIDE1 = */ uint32_t{ 0x23424 },
            /* DMA_STRIDE2 = */ uint32_t{ 0x213426 },
        } },
    };

    /* Agent 2 = */
    Agent agent2 = {
        /* AgentData = */
        { OfmS{
            /* BufferId = */ uint16_t{ 0 },
            /* DMA_COMP_CONFIG0 = */ uint32_t{ 0x89679 },
            /* DMA_STRIDE1 = */ uint32_t{ 0x12346 },
            /* DMA_STRIDE2 = */ uint32_t{ 0x209347f },
        } },
    };

    /* Agent 3 = */
    Agent agent3 = {
        /* AgentData = */
        { /* MceScheduler = */
          MceS{
              /* MceOpMode = */ MceOperation::DEPTHWISE_CONVOLUTION,
              /* PleKernelId = */ PleKernelId::V2442_DOWNSAMPLE_2X2_bw16_bh16_bm1,
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
    Agent agent4 = {
        /* AgentData = */
        { /* PleLoader = */
          PleL{
              /* PleKernelId = */ PleKernelId::V2442_SIGMOID_bw16_bh16_bm1_s8,
          } },
    };

    /* Agent 5 = */
    Agent agent5 = {
        /* AgentData = */
        { /* PleScheduler = */
          PleS{
              /* InputMode = */ PleInputMode::MCE_ONE_OG,
              /* PleKernelId = */ PleKernelId::V2442_DOWNSAMPLE_2X2_bw16_bh16_bm1,
              /* PleKernelSramAddress = */ uint32_t{ 4096 },
          } },
    };

    std::vector<ethosn::command_stream::CommandVariant> dmaRdCommands;
    std::vector<ethosn::command_stream::CommandVariant> dmaWrCommands;
    std::vector<ethosn::command_stream::CommandVariant> mceCommands;
    std::vector<ethosn::command_stream::CommandVariant> pleCommands;

    DmaCommand dmaRdCommand1      = {};
    dmaRdCommand1.type            = CommandType::LoadIfmStripe;
    dmaRdCommand1.agentId         = 0;
    dmaRdCommand1.m_DramOffset    = uint32_t{ 0x123412 };
    dmaRdCommand1.SRAM_ADDR       = uint32_t{ 0x6543 };
    dmaRdCommand1.DMA_SRAM_STRIDE = uint32_t{ 0x2345 };
    dmaRdCommand1.DMA_STRIDE0     = uint32_t{ 0x7995 };
    dmaRdCommand1.DMA_STRIDE3     = uint32_t{ 0x23245 };
    dmaRdCommand1.DMA_CHANNELS    = uint32_t{ 0x12345 };
    dmaRdCommand1.DMA_EMCS        = uint32_t{ 0x989 };
    dmaRdCommand1.DMA_TOTAL_BYTES = uint32_t{ 0xfea };
    dmaRdCommand1.DMA_CMD         = uint32_t{ 0xa };
    dmaRdCommands.push_back(ethosn::command_stream::CommandVariant(dmaRdCommand1));

    DmaCommand dmaWrCommand1      = {};
    dmaWrCommand1.type            = CommandType::StoreOfmStripe;
    dmaWrCommand1.agentId         = 2;
    dmaWrCommand1.m_DramOffset    = uint32_t{ 0xabe };
    dmaWrCommand1.SRAM_ADDR       = uint32_t{ 0x6ee };
    dmaWrCommand1.DMA_SRAM_STRIDE = uint32_t{ 0xebbb5 };
    dmaWrCommand1.DMA_STRIDE0     = uint32_t{ 0x79aa };
    dmaWrCommand1.DMA_STRIDE3     = uint32_t{ 0xdef };
    dmaWrCommand1.DMA_CHANNELS    = uint32_t{ 0xffeed };
    dmaWrCommand1.DMA_EMCS        = uint32_t{ 0xdd2 };
    dmaWrCommand1.DMA_TOTAL_BYTES = uint32_t{ 0xfa12a };
    dmaWrCommand1.DMA_CMD         = uint32_t{ 0x11a };
    dmaWrCommands.push_back(ethosn::command_stream::CommandVariant(dmaWrCommand1));

    ProgramMceStripeCommand mceCommand1 = {};
    mceCommand1.type                    = CommandType::ProgramMceStripe;
    mceCommand1.agentId                 = 0;
    mceCommand1.CE_CONTROL              = uint32_t{ 0x54768 };
    mceCommand1.MUL_ENABLE =
        std::array<std::array<uint32_t, 4>, 8>{
            std::array<uint32_t, 4>{ 0x45, 0x46, 0x47, 0x48 }, std::array<uint32_t, 4>{ 0x49, 0x50, 0x51, 0x52 },
            std::array<uint32_t, 4>{ 0x53, 0x54, 0x55, 0x56 }, std::array<uint32_t, 4>{ 0x57, 0x58, 0x59, 0x60 },
            std::array<uint32_t, 4>{ 0x61, 0x62, 0x63, 0x64 }, std::array<uint32_t, 4>{ 0x65, 0x66, 0x67, 0x68 },
            std::array<uint32_t, 4>{ 0x69, 0x70, 0x71, 0x72 }, std::array<uint32_t, 4>{ 0x73, 0x74, 0x75, 0x76 },
        },
    mceCommand1.IFM_ROW_STRIDE = uint32_t{ 0x3423 };
    mceCommand1.IFM_CONFIG1    = uint32_t{ 0xaa8daa };
    mceCommand1.IFM_PAD =
        std::array<std::array<uint32_t, 4>, 4>{
            std::array<uint32_t, 4>{ 0x45, 0x48, 0x45, 0x48 },
            std::array<uint32_t, 4>{ 0x41, 0x61, 0x41, 0x61 },
            std::array<uint32_t, 4>{ 0x42, 0x61, 0x42, 0x61 },
            std::array<uint32_t, 4>{ 0x45, 0x6a, 0x42, 0x61 },
        },
    mceCommand1.WIDE_KERNEL_OFFSET  = uint32_t{ 0x998765 };
    mceCommand1.IFM_TOP_SLOTS       = uint32_t{ 0xee31 };
    mceCommand1.IFM_MID_SLOTS       = uint32_t{ 0xe56654 };
    mceCommand1.IFM_BOTTOM_SLOTS    = uint32_t{ 0xf787 };
    mceCommand1.IFM_SLOT_PAD_CONFIG = uint32_t{ 0x0897 };
    mceCommand1.OFM_STRIPE_SIZE     = uint32_t{ 0xbb6 };
    mceCommand1.OFM_CONFIG          = uint32_t{ 0xa455435 };
    mceCommand1.WEIGHT_BASE_ADDR    = std::array<uint32_t, 4>{ 0x34587, 0xa, 0x342, 0xb };
    mceCommand1.IFM_CONFIG2 =
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
    mceCommand1.m_NumBlocksProgrammedForMce = uint32_t{ 128 };
    mceCommands.push_back(ethosn::command_stream::CommandVariant(mceCommand1));

    ConfigMceifCommand mceCommand3 = {};
    mceCommand3.type               = CommandType::ConfigMceif;
    mceCommand3.agentId            = 0;
    mceCommands.push_back(ethosn::command_stream::CommandVariant(mceCommand3));

    StartMceStripeCommand mceCommand2 = {};
    mceCommand2.type                  = CommandType::StartMceStripe;
    mceCommand2.agentId               = 0;
    mceCommand2.CE_ENABLES            = 0x123aa;
    mceCommands.push_back(ethosn::command_stream::CommandVariant(mceCommand2));

    WaitForCounterCommand pleCommand1 = {};
    pleCommand1.type                  = CommandType::WaitForCounter;
    pleCommand1.counterName           = CounterName::DmaRd;
    pleCommand1.counterValue          = 0;
    pleCommands.push_back(ethosn::command_stream::CommandVariant(pleCommand1));

    LoadPleCodeIntoPleSramCommand pleCommand3 = {};
    pleCommand3.type                          = CommandType::LoadPleCodeIntoPleSram;
    pleCommand3.agentId                       = 0;
    pleCommands.push_back(ethosn::command_stream::CommandVariant(pleCommand3));

    StartPleStripeCommand pleCommand2 = {};
    pleCommand2.type                  = CommandType::StartPleStripe;
    pleCommand2.agentId               = 0;
    pleCommand2.SCRATCH               = { 0x125aa, 0x126aa, 0x127aa, 0x128aa, 0x129aa, 0x130aa, 0x131aa, 0x132aa };
    pleCommands.push_back(ethosn::command_stream::CommandVariant(pleCommand2));

    std::string xmlStr = ReplaceVersionNumbers(g_XmlStr);
    std::stringstream inputXml(xmlStr);

    const std::vector<uint32_t> commandStreamBinary = BuildCommandStream(
        { agent0, agent1, agent2, agent3, agent4, agent5 }, dmaRdCommands, dmaWrCommands, mceCommands, pleCommands);

    BinaryParser binaryParser(commandStreamBinary);
    std::stringstream outputXml;
    binaryParser.WriteXml(outputXml);

    std::string inputString  = inputXml.str();
    std::string outputString = outputXml.str();

    if (inputString != outputString)
    {
        {
            std::ofstream expected("expected.txt");
            expected << inputString;
            std::ofstream actual("actual.txt");
            actual << outputString;
        }
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
    const std::vector<uint32_t> commandStreamBinary = BuildCommandStream({}, {}, {}, {}, {});

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
