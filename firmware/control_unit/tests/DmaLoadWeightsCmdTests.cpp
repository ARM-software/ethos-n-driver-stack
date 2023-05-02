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

const ethosn_buffer_desc bufferTableData[] = {
    { 0x1000, 0x1000, ETHOSN_BUFFER_CONSTANT },
};

#ifdef CONTROL_UNIT_ASSERTS
TEST_CASE("DmaRdCmdWeights/InvalidBufferType")
{
    const ethosn_buffer_desc invalidBufferTableData[] = {
        { 0x1000, 0x1000, ETHOSN_BUFFER_INPUT },
    };
    const BufferTable bufferTable(std::begin(invalidBufferTableData), std::end(invalidBufferTableData));

    LoggingHal hal(LoggingHal::Options{});
    Pmu<LoggingHal> pmu(hal);
    profiling::ProfilingData<LoggingHal> profilingData(pmu);
    HwAbstraction<LoggingHal> hwAbs(bufferTable, 0, hal, profilingData);

    WgtS wgts = {
        .bufferId = 0,
    };
    DmaCommand dmaCommand = {};

    RequireFatalCall([&]() { hwAbs.HandleDmaRdCmdWeights(wgts, dmaCommand); });
}
#endif    // CONTROL_UNIT_ASSERTS
