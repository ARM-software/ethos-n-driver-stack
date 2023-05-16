//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "TestUtils.hpp"

#include "../src/cascading/DmaRegisters.hpp"
#include "../src/cascading/RegistersLayout.hpp"

#include <ethosn_command_stream/cascading/CommandStream.hpp>

#include <catch.hpp>

using namespace ethosn::support_library;
using namespace ethosn::support_library::registers;
using namespace ethosn::command_stream::cascading;

TEST_CASE("Cascading/DmaRdCmdWeights/Emcs")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    // Checks that the DMA_DMA_EMCS mask is set correctly for weights transfers

    const std::vector<WeightsMetadata> metaData = {
        { 0x1000, 0x100 }, { 0x2000, 0x100 }, { 0x3000, 0x100 },
        { 0x4000, 0x100 }, { 0x5000, 0x100 }, { 0x6000, 0x100 },
    };

    WgtSDesc wgts                    = {};
    wgts.bufferId                    = 0;
    wgts.metadata                    = &metaData;
    wgts.tile.baseAddr               = 0x2000U;
    wgts.tile.numSlots               = 2;
    wgts.tile.slotSize               = 0X1000;
    wgts.numStripes.ofmChannels      = 1;
    wgts.numStripes.ifmChannels      = 1;
    wgts.stripeIdStrides.ofmChannels = 1;
    wgts.stripeIdStrides.ifmChannels = 1;

    // All stripes are copied to all EMCs (0xFFFF). Even for the last stripe which might
    // contain less OFMs than the number of EMCs, because it is padded by the support library
    DmaCommand data1 = GenerateDmaCommandForLoadWgtStripe(wgts, 0, 0, caps, 0);
    CHECK(data1.DMA_EMCS == 0xFFFF);

    DmaCommand data2 = GenerateDmaCommandForLoadWgtStripe(wgts, 0, wgts.numStripes.ofmChannels - 1U, caps, 0);
    CHECK(data2.DMA_EMCS == 0xFFFF);
}

uint32_t ExpectedCmdRegRd(uint32_t rdIdAdd)
{
    dma_rd_cmd_r rdCmd;
    rdCmd.set_format(dma_format_read_t::NHWCB);
    rdCmd.set_rd_id(0 + rdIdAdd);
    return rdCmd.word;
}

uint32_t ExpectedCmdRegWr(uint32_t wrIdAdd, bool weightStreaming = true)
{
    dma_wr_cmd_r wrCmd;
    wrCmd.set_format(weightStreaming ? dma_format_write_t::NHWCB_WEIGHT_STREAMING : dma_format_write_t::NHWCB);
    wrCmd.set_wr_id(4 + wrIdAdd);
    return wrCmd.word;
}

TEST_CASE("Cascading/Dma_Rd_Wr_CmdNhwcb")
{
    SECTION("IfmS ~ 24x50x16/0x16x0/24x34x16/8x16x32")
    {
        const HardwareCapabilities caps =
            GENERATE(GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO));

        const uint32_t numEmcs  = caps.GetNumberOfSrams();    // Either 8 or 16
        const uint32_t stripeId = GENERATE(0U, 1U, 3U);       // Covers stripes 0,0,0, 0,1,0, and 1,0,0

        const FmsDataType format = FmsDataType::NHWCB;

        // Tensor Data in HWC order
        // Supertensor:    24, 50, 16
        // Tensor offset:  0,  16, 0
        // Tensor size:    24, 34, 16
        // Default stripe: 8,  16, 32
        // Edge stripe:    8,  2,  16

        IfmSDesc ifmsData = {};
        // Each brick group has a size 8x8x16 and a tensor offset
        // of (0, 16, 0) is equivalent of an offset of 2 brick groups.
        // Therefore, dramOffset = 8x8x16x2
        ifmsData.fmData.dramOffset                = 2048U;
        ifmsData.fmData.bufferId                  = 0;
        ifmsData.fmData.dataType                  = format;
        ifmsData.fmData.fcafInfo.zeroPoint        = 0;
        ifmsData.fmData.fcafInfo.signedActivation = false;
        ifmsData.fmData.tile.baseAddr             = 0x2000U;
        ifmsData.fmData.tile.numSlots             = 2;
        ifmsData.fmData.tile.slotSize =
            static_cast<uint16_t>(numEmcs > 8 ? 256 : 512);    // 512 at 8 EMCs, 256 at 16 EMCs
        ifmsData.fmData.defaultStripeSize               = { 8, 16, 32 };
        ifmsData.fmData.edgeStripeSize                  = { 8, 2, 16 };
        ifmsData.fmData.supertensorSizeInCells.width    = 7;
        ifmsData.fmData.supertensorSizeInCells.channels = 1;
        ifmsData.fmData.numStripes                      = { 3, 3, 1 };
        ifmsData.fmData.stripeIdStrides                 = { 3, 1, 1 };
        ifmsData.packedBoundaryThickness                = { 0, 0, 0, 0 };
        ifmsData.isExtraPackedBoundaryDataOnRightEdge   = 0;
        ifmsData.isExtraPackedBoundaryDataOnBottomEdge  = 0;

        // Test flow:
        // 1. Initialize test
        // 2. Input manually calculated register values in "expected" map
        // 3. Log data
        // 4. Call Handle(DmaRdCmdIfm)
        // 5. Verify last write register was to DMA_RD_CMD or DMA_WR_CMD
        // 6. Verify registers written as expected
        // 7. Call Handle() again with same data, and verify rd/wr_id is changed

        // Log data
        INFO("test command ");
        CAPTURE(stripeId);
        CAPTURE(caps.GetNumberOfEngines(), caps.GetOgsPerEngine(), caps.GetNumberofSramsPerEngine(),
                caps.GetNumberOfPleLanes());

        // Call Handle()
        DmaCommand data = GenerateDmaCommandForLoadIfmStripe(ifmsData, 0, stripeId, 0, caps, 0);

        {
            // Offset from fmData, not calculated by firmware
            uint32_t dramOffset = ifmsData.fmData.dramOffset;

            switch (stripeId)
            {
                case 0:
                    dramOffset += 0U;
                    break;
                case 1:
                    dramOffset += 2048U;
                    break;
                case 3:
                    dramOffset += 7168U;
                    break;
                default:
                    FAIL("StripeId not a tested value.");
                    break;
            }

            CHECK(data.m_DramOffset == dramOffset);
        }
        {
            sram_addr_r sramAddr;
            // Stripe Ids: 0, 1, 3. 2 slots. First slot for 0, second for the rest.
            // Tile size: 256 for 16 EMCs, 512 for 8 EMCs
            sramAddr.set_address(stripeId == 0 ? 0x2000U : numEmcs > 8 ? 0x2100U : 0x2200U);
            CHECK(data.SRAM_ADDR == sramAddr.word);
        }
        {
            // All EMCs active due to stripe channels, either 8 or 16 EMCs
            dma_emcs_r emcs;
            emcs.set_emcs(numEmcs > 8 ? 0xFFFF : 0x00FF);
            CHECK(data.DMA_EMCS == emcs.word);
        }
        {
            // Stripe channels
            dma_channels_r channels;
            channels.set_channels(16);
            CHECK(data.DMA_CHANNELS == channels.word);
        }
        {
            // Tested stripes are full-size
            dma_total_bytes_r tot;
            tot.set_total_bytes(2048U);
            CHECK(data.DMA_TOTAL_BYTES == tot.word);
        }
        CHECK(data.DMA_CMD == ExpectedCmdRegRd(0));
    }
    SECTION("IfmS ~ 32x32x32/0x0x0/32x32x32/16x16x16 Chunkified")
    {
        const HardwareCapabilities caps =
            GENERATE(GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO));

        const uint32_t numEmcs  = caps.GetNumberOfSrams();    // Either 8 or 16
        const uint32_t stripeId = GENERATE(0U, 1U, 3U);       // Covers stripes 0,0,0, 0,0,1, and 0,1,1

        // Tensor Data in HWC order
        // Supertensor:    32, 32, 32
        // Tensor offset:  0,  0, 0
        // Tensor size:    32, 32, 32
        // Default stripe: 16,  16, 16
        // Edge stripe:    16,  16,  16

        IfmSDesc ifmsData                         = {};
        ifmsData.fmData.dramOffset                = 0U;
        ifmsData.fmData.bufferId                  = 0;
        ifmsData.fmData.dataType                  = FmsDataType::NHWCB;
        ifmsData.fmData.fcafInfo.zeroPoint        = 0;
        ifmsData.fmData.fcafInfo.signedActivation = false;
        ifmsData.fmData.tile.baseAddr             = 0x2000U;
        ifmsData.fmData.tile.numSlots             = 2;
        ifmsData.fmData.tile.slotSize =
            static_cast<uint16_t>(numEmcs > 8 ? 256 : 512),    // 512 at 8 EMCs, 256 at 16 EMCs
            ifmsData.fmData.defaultStripeSize           = { 16, 16, 16 };
        ifmsData.fmData.edgeStripeSize                  = { 16, 16, 16 };
        ifmsData.fmData.supertensorSizeInCells.width    = 4;
        ifmsData.fmData.supertensorSizeInCells.channels = 2;
        ifmsData.fmData.numStripes                      = { 2, 2, 2 };
        ifmsData.fmData.stripeIdStrides                 = { 4, 2, 1 };
        ifmsData.packedBoundaryThickness                = { 0, 0, 0, 0 };
        ifmsData.isExtraPackedBoundaryDataOnRightEdge   = 0;
        ifmsData.isExtraPackedBoundaryDataOnBottomEdge  = 0;

        // Test flow:
        // 1. Initialize test
        // 2. Input manually calculated register values in "expected" map
        // 3. Log data
        // 4. Call Handle(DmaRdCmdIfm)
        // 5. Verify last write register was to DMA_RD_CMD or DMA_WR_CMD
        // 6. Verify registers written as expected
        // 7. Call Handle() again with same data, and verify rd/wr_id is changed

        // Log data
        INFO("test command ");
        CAPTURE(stripeId);
        CAPTURE(caps.GetNumberOfEngines(), caps.GetOgsPerEngine(), caps.GetNumberofSramsPerEngine(),
                caps.GetNumberOfPleLanes());

        // Offset from fmData, not calculated by firmware
        uint32_t dramOffset = ifmsData.fmData.dramOffset;

        switch (stripeId)
        {
            case 0:
                dramOffset += 0U;
                break;
            case 1:
                dramOffset += 1024U;
                break;
            case 3:
                dramOffset += 5120U;
                break;
            default:
                FAIL("StripeId not a tested value.");
                break;
        }

        uint32_t sramAddr = stripeId == 0 ? 0x2000U : numEmcs > 8 ? 0x2100U : 0x2200U;

        // Call Handle() four times
        // res should only be complete on the last call
        // sramAddr and dramOffset should be offset more each call
        for (uint8_t chunkId = 0; chunkId < 4; chunkId++)
        {
            DmaCommand data = GenerateDmaCommandForLoadIfmStripe(ifmsData, 0, stripeId, chunkId, caps, 0);

            // Verify registers written

            // Dram and Sram addresses are offset based on the chunk being loaded
            sram_addr_r sramReg;

            {
                CHECK(data.m_DramOffset == static_cast<uint32_t>(dramOffset));
            }
            {
                // Stripe Ids: 0, 1, 3. 2 slots. First slot for 0, second for the rest.
                // Tile size: 256 for 16 EMCs, 512 for 8 EMCs
                sramReg.set_address(sramAddr);
                CHECK(data.SRAM_ADDR == sramReg.word);
            }
            {
                // All EMCs active due to stripe channels, either 8 or 16 EMCs
                dma_emcs_r emcs;
                emcs.set_emcs(numEmcs > 8 ? 0xFFFF : 0x00FF);
                CHECK(data.DMA_EMCS == emcs.word);
            }
            {
                // Stripe channels
                dma_channels_r channels;
                channels.set_channels(16);
                CHECK(data.DMA_CHANNELS == channels.word);
            }
            {
                // Tested stripes are chunkified
                // Total bytes across all chunks is 4096
                // One chunk is 1024
                dma_total_bytes_r tot;
                tot.set_total_bytes(1024U);
                CHECK(data.DMA_TOTAL_BYTES == tot.word);
            }
            CHECK(data.DMA_CMD == ExpectedCmdRegRd(0));

            dramOffset += chunkId == 1 ? 6144U : 2048U;
            sramAddr += 1024 / numEmcs;
        }
    }
    SECTION("IfmS ~ 32x32x16/0x0x0/32x32x16/16x16x16 Chunkified")
    {
        const HardwareCapabilities caps =
            GENERATE(GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO));

        const uint32_t numEmcs  = caps.GetNumberOfSrams();    // Either 8 or 16
        const uint32_t stripeId = GENERATE(0U, 1U, 3U);       // Covers stripes 0,0,0, 0,0,1, and 0,1,1

        // Tensor Data in HWC order
        // Supertensor:    32, 32, 16
        // Tensor offset:  0,  0,  0
        // Tensor size:    32, 32, 16
        // Default stripe: 16, 16, 16
        // Edge stripe:    16, 16, 16

        IfmSDesc ifmsData                         = {};
        ifmsData.fmData.dramOffset                = 0U;
        ifmsData.fmData.bufferId                  = 0;
        ifmsData.fmData.dataType                  = FmsDataType::NHWCB;
        ifmsData.fmData.fcafInfo.zeroPoint        = 0;
        ifmsData.fmData.fcafInfo.signedActivation = false;
        ifmsData.fmData.tile.baseAddr             = 0x2000U;
        ifmsData.fmData.tile.numSlots             = 2;
        ifmsData.fmData.tile.slotSize =
            static_cast<uint16_t>(numEmcs > 8 ? 256 : 512),    // 512 at 8 EMCs, 256 at 16 EMCs
            ifmsData.fmData.defaultStripeSize           = { 16, 16, 16 };
        ifmsData.fmData.edgeStripeSize                  = { 16, 16, 16 };
        ifmsData.fmData.supertensorSizeInCells.width    = 4;
        ifmsData.fmData.supertensorSizeInCells.channels = 1;
        ifmsData.fmData.numStripes                      = { 2, 2, 1 };
        ifmsData.fmData.stripeIdStrides                 = { 2, 1, 1 };
        ifmsData.packedBoundaryThickness                = { 0, 0, 0, 0 };
        ifmsData.isExtraPackedBoundaryDataOnRightEdge   = 0;
        ifmsData.isExtraPackedBoundaryDataOnBottomEdge  = 0;

        // Log data
        INFO("test command ");
        CAPTURE(stripeId);
        CAPTURE(caps.GetNumberOfEngines(), caps.GetOgsPerEngine(), caps.GetNumberofSramsPerEngine(),
                caps.GetNumberOfPleLanes());

        // Test flow:
        // 1. Initialize test
        // 2. Input manually calculated register values in "expected" map
        // 3. Log data
        // 4. Call Handle(DmaRdCmdIfm)
        // 5. Verify last write register was to DMA_RD_CMD or DMA_WR_CMD
        // 6. Verify registers written as expected
        // 7. Call Handle() again with same data, and verify rd/wr_id is changed

        // Offset from ifmsData, not calculated by firmware
        uint32_t dramOffset = ifmsData.fmData.dramOffset;

        switch (stripeId)
        {
            case 0:
                dramOffset += 0U;
                break;
            case 1:
                dramOffset += 2048U;
                break;
            case 3:
                dramOffset += 10240U;
                break;
            default:
                FAIL("StripeId not a tested value.");
                break;
        }

        uint32_t sramAddr = stripeId == 0 ? 0x2000U : numEmcs > 8 ? 0x2100U : 0x2200U;

        // Test setup: initialize HAL, HwAbstraction

        // Call Handle() two times
        // res should only be complete on the last call
        // sramAddr and dramOffset should be offset more each call
        for (uint8_t chunkId = 0; chunkId < 2; chunkId++)
        {
            DmaCommand data = GenerateDmaCommandForLoadIfmStripe(ifmsData, 0, stripeId, chunkId, caps, 0);

            // Dram and Sram addresses are offset based on the chunk being loaded
            sram_addr_r sramReg;

            {
                CHECK(data.m_DramOffset == dramOffset);
            }
            {
                // Stripe Ids: 0, 1, 3. 2 slots. First slot for 0, second for the rest.
                // Tile size: 256 for 16 EMCs, 512 for 8 EMCs
                sramReg.set_address(sramAddr);
                CHECK(data.SRAM_ADDR == sramReg.word);
            }
            {
                // All EMCs active due to stripe channels, either 8 or 16 EMCs
                dma_emcs_r emcs;
                emcs.set_emcs(numEmcs > 8 ? 0xFFFF : 0x00FF);
                CHECK(data.DMA_EMCS == emcs.word);
            }
            {
                // Stripe channels
                dma_channels_r channels;
                channels.set_channels(16);
                CHECK(data.DMA_CHANNELS == channels.word);
            }
            {
                // Tested stripes are chunkified
                // Total bytes across all chunks is 4096
                // One chunk is 2048
                dma_total_bytes_r tot;
                tot.set_total_bytes(2048U);
                CHECK(data.DMA_TOTAL_BYTES == tot.word);
            }
            CHECK(data.DMA_CMD == ExpectedCmdRegRd(0));

            dramOffset += 4096U;
            sramAddr += 2048U / numEmcs;
        }
    }
    SECTION("IfmS ~ 32x16x32/0x0x0/32x16x32/16x16x16 Chunkified")
    {
        const HardwareCapabilities caps =
            GENERATE(GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO));

        const uint32_t numEmcs  = caps.GetNumberOfSrams();    // Either 8 or 16
        const uint32_t stripeId = GENERATE(0U, 1U, 3U);       // Covers stripes 0,0,0, 0,0,1, and 0,1,1

        // Tensor Data in HWC order
        // Supertensor:    32, 16, 32
        // Tensor offset:  0,  0, 0
        // Tensor size:    32, 16, 32
        // Default stripe: 16,  16, 16
        // Edge stripe:    16,  16,  16

        IfmSDesc ifmsData                         = {};
        ifmsData.fmData.dramOffset                = 0U;
        ifmsData.fmData.bufferId                  = 0;
        ifmsData.fmData.dataType                  = FmsDataType::NHWCB;
        ifmsData.fmData.fcafInfo.zeroPoint        = 0;
        ifmsData.fmData.fcafInfo.signedActivation = false;
        ifmsData.fmData.tile.baseAddr             = 0x2000U;
        ifmsData.fmData.tile.numSlots             = 2;
        ifmsData.fmData.tile.slotSize =
            static_cast<uint16_t>(numEmcs > 8 ? 256 : 512),    // 512 at 8 EMCs, 256 at 16 EMCs
            ifmsData.fmData.defaultStripeSize           = { 16, 16, 16 };
        ifmsData.fmData.edgeStripeSize                  = { 16, 16, 16 };
        ifmsData.fmData.supertensorSizeInCells.width    = 2;
        ifmsData.fmData.supertensorSizeInCells.channels = 2;
        ifmsData.fmData.numStripes                      = { 2, 1, 2 };
        ifmsData.fmData.stripeIdStrides                 = { 2, 1, 1 };
        ifmsData.packedBoundaryThickness                = { 0, 0, 0, 0 };
        ifmsData.isExtraPackedBoundaryDataOnRightEdge   = 0;
        ifmsData.isExtraPackedBoundaryDataOnBottomEdge  = 0;

        // Log data
        INFO("test command ");
        CAPTURE(stripeId);
        CAPTURE(caps.GetNumberOfEngines(), caps.GetOgsPerEngine(), caps.GetNumberofSramsPerEngine(),
                caps.GetNumberOfPleLanes());

        // Test flow:
        // 1. Initialize test
        // 2. Input manually calculated register values in "expected" map
        // 3. Log data
        // 4. Call Handle(DmaRdCmdIfm)
        // 5. Verify last write register was to DMA_RD_CMD or DMA_WR_CMD
        // 6. Verify registers written as expected
        // 7. Call Handle() again with same data, and verify rd/wr_id is changed

        // Test setup: initialize HAL, HwAbstraction

        // Dram and Sram addresses are offset based on the chunk being loaded
        // Offset from ifmsData, not calculated by firmware
        uint32_t dramOffset = ifmsData.fmData.dramOffset;

        switch (stripeId)
        {
            case 0:
                dramOffset += 0U;
                break;
            case 1:
                dramOffset += 1024U;
                break;
            case 3:
                dramOffset += 9216U;
                break;
            default:
                FAIL("StripeId not a tested value.");
                break;
        }

        sram_addr_r sramReg;
        uint32_t sramAddr = stripeId == 0 ? 0x2000U : numEmcs > 8 ? 0x2100U : 0x2200U;

        // Call Handle() four times
        // res should only be complete on the last call
        // sramAddr and dramOffset should be offset more each call
        for (uint8_t chunkId = 0; chunkId < 4; chunkId++)
        {
            DmaCommand data = GenerateDmaCommandForLoadIfmStripe(ifmsData, 0, stripeId, chunkId, caps, 0);

            {
                CHECK(data.m_DramOffset == dramOffset);
            }
            {
                // Stripe Ids: 0, 1, 3. 2 slots. First slot for 0, second for the rest.
                // Tile size: 256 for 16 EMCs, 512 for 8 EMCs
                sramReg.set_address(sramAddr);
                CHECK(data.SRAM_ADDR == sramReg.word);
            }
            {
                // All EMCs active due to stripe channels, either 8 or 16 EMCs
                dma_emcs_r emcs;
                emcs.set_emcs(numEmcs > 8 ? 0xFFFF : 0x00FF);
                CHECK(data.DMA_EMCS == emcs.word);
            }
            {
                // Stripe channels
                dma_channels_r channels;
                channels.set_channels(16);
                CHECK(data.DMA_CHANNELS == channels.word);
            }
            {
                // Tested stripes are chunkified
                // Total bytes across all chunks is 4096
                // One chunk is 1024
                dma_total_bytes_r tot;
                tot.set_total_bytes(1024U);
                CHECK(data.DMA_TOTAL_BYTES == tot.word);
            }
            CHECK(data.DMA_CMD == ExpectedCmdRegRd(0));

            CHECK(data.DMA_CMD == ExpectedCmdRegRd(0));

            dramOffset += 2048U;
            sramAddr += 1024 / numEmcs;
        }
    }
    SECTION("OfmS ~ 24x50x16/0x16x0/24x34x16/8x16x32")
    {
        const HardwareCapabilities caps =
            GENERATE(GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO));

        const uint32_t numEmcs  = caps.GetNumberOfSrams();    // Either 8 or 16
        const uint32_t stripeId = GENERATE(0U, 1U, 3U);       // Covers stripes 0,0,0, 0,1,0, and 1,0,0

        const FmsDataType format = FmsDataType::NHWCB;

        // Tensor Data in HWC order
        // Supertensor:    24, 50, 16
        // Tensor offset:  0,  16, 0
        // Tensor size:    24, 34, 16
        // Default stripe: 8,  16, 32
        // Edge stripe:    8,  2,  16

        OfmSDesc ofmsData = {};
        // Each brick group has a size 8x8x16 and a tensor offset
        // of (0, 16, 0) is equivalent of an offset of 2 brick groups.
        // Therefore, dramOffset = 8x8x16x2
        ofmsData.fmData.dramOffset                = 2048U;
        ofmsData.fmData.bufferId                  = 0;
        ofmsData.fmData.dataType                  = format;
        ofmsData.fmData.fcafInfo.zeroPoint        = 0;
        ofmsData.fmData.fcafInfo.signedActivation = false;
        ofmsData.fmData.tile.baseAddr             = 0x2000U;
        ofmsData.fmData.tile.numSlots             = 2;
        ofmsData.fmData.tile.slotSize =
            static_cast<uint16_t>(numEmcs > 8 ? 256 : 512),    // 512 at 8 EMCs, 256 at 16 EMCs
            ofmsData.fmData.defaultStripeSize           = { 8, 16, 32 };
        ofmsData.fmData.edgeStripeSize                  = { 8, 2, 16 };
        ofmsData.fmData.supertensorSizeInCells.width    = 7;
        ofmsData.fmData.supertensorSizeInCells.channels = 1;
        ofmsData.fmData.numStripes                      = { 3, 3, 1 };
        ofmsData.fmData.stripeIdStrides                 = { 3, 1, 1 };

        // Log data
        INFO("test command ");
        CAPTURE(stripeId);
        CAPTURE(caps.GetNumberOfEngines(), caps.GetOgsPerEngine(), caps.GetNumberofSramsPerEngine(),
                caps.GetNumberOfPleLanes());

        // Test flow:
        // 1. Initialize test
        // 2. Input manually calculated register values in "expected" map
        // 3. Log data
        // 4. Call Handle(DmaWrCmdOfm)
        // 5. Verify last write register was to DMA_RD_CMD or DMA_WR_CMD
        // 6. Verify registers written as expected
        // 7. Call Handle() again with same data, and verify rd/wr_id is changed

        // Test setup: initialize HAL, HwAbstraction
        DmaCommand data = GenerateDmaCommandForStoreOfmStripe(ofmsData, 0, stripeId, 0, caps, 4);

        {
            // Offset from fmData, not calculated by firmware
            uint32_t dramOffset = ofmsData.fmData.dramOffset;

            // cppcheck-suppress danglingTemporaryLifetime
            switch (stripeId)
            {
                case 0:
                    dramOffset += 0U;
                    break;
                case 1:
                    dramOffset += 2048U;
                    break;
                case 3:
                    dramOffset += 7168U;
                    break;
                default:
                    FAIL("StripeId not a tested value.");
                    break;
            }

            CHECK(data.m_DramOffset == dramOffset);
        }
        {
            sram_addr_r sramAddr;
            // Stripe Ids: 0, 1, 3. 2 slots. First slot for 0, second for the rest.
            // Tile size: 256 for 16 EMCs, 512 for 8 EMCs
            sramAddr.set_address(stripeId == 0 ? 0x2000U : numEmcs > 8 ? 0x2100U : 0x2200U);
            CHECK(data.SRAM_ADDR == sramAddr.word);
        }
        {
            // All EMCs active due to stripe channels, either 8 or 16 EMCs
            dma_emcs_r emcs;
            emcs.set_emcs(numEmcs > 8 ? 0xFFFF : 0x00FF);
            CHECK(data.DMA_EMCS == emcs.word);
        }
        {
            // Stripe channels
            dma_channels_r channels;
            channels.set_channels(16);
            CHECK(data.DMA_CHANNELS == channels.word);
        }
        {
            // Tested stripes are full-size
            dma_total_bytes_r tot;
            tot.set_total_bytes(2048U);
            CHECK(data.DMA_TOTAL_BYTES == tot.word);
        }
        CHECK(data.DMA_CMD == ExpectedCmdRegWr(0, false));
    }
    SECTION("OfmS ~ 32x32x32/0x0x0/32x32x32/16x16x16 Strided & Chunkified")
    {
        const HardwareCapabilities caps =
            GENERATE(GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO));

        const uint32_t numEmcs  = caps.GetNumberOfSrams();    // Either 8 or 16
        const uint32_t stripeId = GENERATE(0U, 1U, 3U);       // Covers stripes 0,0,0, 0,0,1, and 0,1,1

        // Tensor Data in HWC order
        // Supertensor:    32, 32, 32
        // Tensor offset:  0,  0, 0
        // Tensor size:    32, 32, 32
        // Default stripe: 16,  16, 16
        // Edge stripe:    16,  16,  16

        OfmSDesc ofmsData                         = {};
        ofmsData.fmData.dramOffset                = 0U;
        ofmsData.fmData.bufferId                  = 0;
        ofmsData.fmData.dataType                  = FmsDataType::NHWCB;
        ofmsData.fmData.fcafInfo.zeroPoint        = 0;
        ofmsData.fmData.fcafInfo.signedActivation = false;
        ofmsData.fmData.tile.baseAddr             = 0x2000U;
        ofmsData.fmData.tile.numSlots             = 2;
        ofmsData.fmData.tile.slotSize =
            static_cast<uint16_t>(numEmcs > 8 ? 256 : 512),    // 512 at 8 EMCs, 256 at 16 EMCs
            ofmsData.fmData.defaultStripeSize           = { 16, 16, 16 };
        ofmsData.fmData.edgeStripeSize                  = { 16, 16, 16 };
        ofmsData.fmData.supertensorSizeInCells.width    = 4;
        ofmsData.fmData.supertensorSizeInCells.channels = 2;
        ofmsData.fmData.numStripes                      = { 2, 2, 2 };
        ofmsData.fmData.stripeIdStrides                 = { 4, 2, 1 };

        // Log data
        INFO("test command ");
        CAPTURE(stripeId);
        CAPTURE(caps.GetNumberOfEngines(), caps.GetOgsPerEngine(), caps.GetNumberofSramsPerEngine(),
                caps.GetNumberOfPleLanes());

        // Test flow:
        // 1. Initialize test
        // 2. Input manually calculated register values in "expected" map
        // 3. Log data
        // 4. Call Handle(DmaWrCmdOfm)
        // 5. Verify last write register was to DMA_RD_CMD or DMA_WR_CMD
        // 6. Verify registers written as expected
        // 7. Call Handle() again with same data, and verify rd/wr_id is changed

        // Test setup: initialize HAL, HwAbstraction

        // Dram and Sram addresses are offset based on the chunk being loaded
        // Offset from fmData, not calculated by firmware
        uint32_t dramOffset = ofmsData.fmData.dramOffset;

        switch (stripeId)
        {
            case 0:
                dramOffset += 0U;
                break;
            case 1:
                dramOffset += 1024U;
                break;
            case 3:
                dramOffset += 5120U;
                break;
            default:
                FAIL("StripeId not a tested value.");
                break;
        }

        sram_addr_r sramReg;
        // cppcheck-suppress danglingTemporaryLifetime
        uint32_t sramAddr = stripeId == 0 ? 0x2000U : numEmcs > 8 ? 0x2100U : 0x2200U;

        // Call Handle() twice
        // res should only be complete on the last call
        // sramAddr and dramOffset should be offset after each call
        for (uint8_t chunkId = 0; chunkId < 2; chunkId++)
        {
            DmaCommand data = GenerateDmaCommandForStoreOfmStripe(ofmsData, 0, stripeId, chunkId, caps, 4);

            {
                CHECK(data.m_DramOffset == dramOffset);
            }
            {
                // Stripe Ids: 0, 1, 3. 2 slots. First slot for 0, second for the rest.
                // Tile size: 256 for 16 EMCs, 512 for 8 EMCs
                sramReg.set_address(sramAddr);
                CHECK(data.SRAM_ADDR == sramReg.word);
            }
            {
                // All EMCs active due to stripe channels, either 8 or 16 EMCs
                dma_emcs_r emcs;
                emcs.set_emcs(numEmcs > 8 ? 0xFFFF : 0x00FF);
                CHECK(data.DMA_EMCS == emcs.word);
            }
            {
                // Stripe channels
                dma_channels_r channels;
                channels.set_channels(16);
                CHECK(data.DMA_CHANNELS == channels.word);
            }
            {
                // NHWCB can dram stride on output
                dma_stride0_r stride0;
                stride0.set_inner_stride(1024U);
                CHECK(data.DMA_STRIDE0 == stride0.word);
            }
            {
                // Tested stripes are chunkified
                // Total bytes across all chunks is 4096
                // One chunk is 2048
                dma_total_bytes_r tot;
                tot.set_total_bytes(2048U);
                CHECK(data.DMA_TOTAL_BYTES == tot.word);
            }
            CHECK(data.DMA_CMD == ExpectedCmdRegWr(0));

            dramOffset += 8192U;
            sramAddr += 2048U / numEmcs;
        }
    }
    SECTION("OfmS ~ 32x16x32/0x0x0/32x16x32/16x16x16 Fully Strided")
    {
        const HardwareCapabilities caps =
            GENERATE(GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO),
                     GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO));

        const uint32_t numEmcs  = caps.GetNumberOfSrams();    // Either 8 or 16
        const uint32_t stripeId = GENERATE(0U, 1U, 3U);       // Covers stripes 0,0,0, 0,0,1, and 0,1,1

        // Tensor Data in HWC order
        // Supertensor:    32, 16, 32
        // Tensor offset:  0,  0, 0
        // Tensor size:    32, 16, 32
        // Default stripe: 16,  16, 16
        // Edge stripe:    16,  16,  16

        OfmSDesc ofmsData                         = {};
        ofmsData.fmData.dramOffset                = 0U;
        ofmsData.fmData.bufferId                  = 0;
        ofmsData.fmData.dataType                  = FmsDataType::NHWCB;
        ofmsData.fmData.fcafInfo.zeroPoint        = 0;
        ofmsData.fmData.fcafInfo.signedActivation = false;
        ofmsData.fmData.tile.baseAddr             = 0x2000U;
        ofmsData.fmData.tile.numSlots             = 2;
        ofmsData.fmData.tile.slotSize =
            static_cast<uint16_t>(numEmcs > 8 ? 256 : 512),    // 512 at 8 EMCs, 256 at 16 EMCs
            ofmsData.fmData.defaultStripeSize           = { 16, 16, 16 };
        ofmsData.fmData.edgeStripeSize                  = { 16, 16, 16 };
        ofmsData.fmData.supertensorSizeInCells.width    = 2;
        ofmsData.fmData.supertensorSizeInCells.channels = 2;
        ofmsData.fmData.numStripes                      = { 2, 1, 2 };
        ofmsData.fmData.stripeIdStrides                 = { 2, 1, 1 };

        // Log data
        INFO("test command ");
        CAPTURE(stripeId);
        CAPTURE(caps.GetNumberOfEngines(), caps.GetOgsPerEngine(), caps.GetNumberofSramsPerEngine(),
                caps.GetNumberOfPleLanes());

        // Test flow:
        // 1. Initialize test
        // 2. Input manually calculated register values in "expected" map
        // 3. Log data
        // 4. Call Handle(DmaWrCmdOfm)
        // 5. Verify last write register was to DMA_RD_CMD or DMA_WR_CMD
        // 6. Verify registers written as expected

        // Test setup: initialize HAL, HwAbstraction

        // Input manually calculated register values in "expected" map

        // Call Handle() once
        // res should be complete on the first call
        // sramAddr and dramOffset should be offset after each call
        DmaCommand data = GenerateDmaCommandForStoreOfmStripe(ofmsData, 0, stripeId, 0, caps, 4);

        // Dram and Sram addresses are offset based on the chunk being loaded
        sram_addr_r sramReg;
        // cppcheck-suppress danglingTemporaryLifetime
        uint32_t sramAddr = stripeId == 0 ? 0x2000U : numEmcs > 8 ? 0x2100U : 0x2200U;

        {
            // Offset from fmData, not calculated by firmware
            uint32_t dramOffset = ofmsData.fmData.dramOffset;

            switch (stripeId)
            {
                case 0:
                    dramOffset += 0U;
                    break;
                case 1:
                    dramOffset += 1024U;
                    break;
                case 3:
                    dramOffset += 9216U;
                    break;
                default:
                    FAIL("StripeId not a tested value.");
                    break;
            }

            CHECK(data.m_DramOffset == dramOffset);
        }
        {
            // Stripe Ids: 0, 1, 3. 2 slots. First slot for 0, second for the rest.
            // Tile size: 256 for 16 EMCs, 512 for 8 EMCs
            sramReg.set_address(sramAddr);
            CHECK(data.SRAM_ADDR == sramReg.word);
        }
        {
            // All EMCs active due to stripe channels, either 8 or 16 EMCs
            dma_emcs_r emcs;
            emcs.set_emcs(numEmcs > 8 ? 0xFFFF : 0x00FF);
            CHECK(data.DMA_EMCS == emcs.word);
        }
        {
            // Stripe channels
            dma_channels_r channels;
            channels.set_channels(16);
            CHECK(data.DMA_CHANNELS == channels.word);
        }
        {
            // NHWCB can dram stride on output
            dma_stride0_r stride0;
            stride0.set_inner_stride(1024U);
            CHECK(data.DMA_STRIDE0 == stride0.word);
        }
        {
            // Tested stripes are fully strided
            dma_total_bytes_r tot;
            tot.set_total_bytes(4096U);
            CHECK(data.DMA_TOTAL_BYTES == tot.word);
        }
        CHECK(data.DMA_CMD == ExpectedCmdRegWr(0));
    }
}
