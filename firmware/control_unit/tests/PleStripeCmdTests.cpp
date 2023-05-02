//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../../unprivileged/HwAbstraction.hpp"
#include <model/LoggingHal.hpp>

#include <catch.hpp>

using namespace ethosn::control_unit;
using namespace ethosn::command_stream;

TEST_CASE("PleStripeCmd enables PLE MCU events when a stripe starts")
{
    LoggingHal hal(LoggingHal::Options{});
    Pmu<LoggingHal> pmu(hal);
    profiling::ProfilingData<LoggingHal> profilingData(pmu);
    const BufferTable bufferTable(nullptr, nullptr);
    HwAbstraction<LoggingHal> hwAbs(bufferTable, 0, hal, profilingData);

    PleS pleS{};
    pleS.inputMode = PleInputMode::SRAM_TWO_INPUTS;
    StartPleStripeCommand startPleCommand{};

    REQUIRE(hwAbs.TrySetCeEnables(CeEnables::AllEnabledForPleOnly));

    // Call the function under test
    hwAbs.HandlePleStripeCmd(pleS, startPleCommand);

    // Confirm that PLE MCU events have been enabled
    ple_control_1_r expectedPleCtrl1;
    expectedPleCtrl1.set_mcu_setevent(1);
    expectedPleCtrl1.set_mceif_event(1);
    expectedPleCtrl1.set_udma_event(1);
    expectedPleCtrl1.set_txev_ncu(1);
    CHECK(hal.GetFinalValue(TOP_REG(CE_RP, CE_PLE_CONTROL_1)) == expectedPleCtrl1.word);
}
