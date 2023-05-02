//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../../unprivileged/HwAbstraction.hpp"
#include "TestUtils.hpp"
#include <model/LoggingHal.hpp>

#include <common/FirmwareApi.hpp>

#include <catch.hpp>

using namespace ethosn::control_unit;
using namespace ethosn::control_unit::tests;
using namespace ethosn::command_stream;

#ifdef CONTROL_UNIT_ASSERTS
TEST_CASE("DmaRd_Wr_CmdNhwc/InvalidBufferType")
{
    // Buffer table where the input buffer has output type and the output buffer has input type which isn't allowed
    const ethosn_buffer_desc bufferTableData[2] = { { 0x1000, 0x1000, ETHOSN_BUFFER_OUTPUT },
                                                    { 0x3000, 0x1000, ETHOSN_BUFFER_INPUT } };
    const BufferTable bufferTable(std::begin(bufferTableData), std::end(bufferTableData));

    LoggingHal hal(LoggingHal::Options{});
    Pmu<LoggingHal> pmu(hal);
    profiling::ProfilingData<LoggingHal> profilingData(pmu);
    HwAbstraction<LoggingHal> hwAbs(bufferTable, 0, hal, profilingData);

    // The only interesting part here for the test is FmsData bufferId. The rest is just to setup a valid command.

    SECTION("Invalid DMA read with output buffer type")
    {
        IfmS ifmS             = {};
        ifmS.bufferId         = 0;
        DmaCommand dmaCommand = {};
        RequireFatalCall([&]() { hwAbs.HandleDmaRdCmdIfm(ifmS, dmaCommand); });
    }

    SECTION("Invalid DMA write with input buffer type")
    {
        // Change to use the second buffer in the buffer table
        OfmS ofmS             = {};
        ofmS.bufferId         = 1;
        DmaCommand dmaCommand = {};
        RequireFatalCall([&]() { hwAbs.HandleDmaWrCmdOfm(ofmS, dmaCommand); });
    }
}
#endif    // CONTROL_UNIT_ASSERTS
