//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <common/Inference.hpp>

#include "Controller.hpp"

namespace ethosn::control_unit
{

template <typename HwAbstractionT>
bool RunCommandStream(const command_stream::CommandStream& cmdStream, HwAbstractionT&& hwAbstraction)
{
    Controller controller{ std::forward<HwAbstractionT>(hwAbstraction), cmdStream };

    // Main processing loop. This keeps looping until we are finished running the entire command stream.
    while (true)
    {
        if (controller.GetHwAbstraction().HasErrors())
        {
            return false;
        }

        while (controller.Spin())
        {
        }

        // At this point the HW is busy doing stuff, or there is nothing left to do
        if (controller.IsDone())
        {
            break;
        }

        // Controller has processed everything it can,
        // so we must be waiting for the HW to process stuff. Go to sleep and wait for it to wake us up,
        // at which point we will immediately check if we can run some more stuff on the HW.
        controller.WaitForEvents();
    }

    if (!controller.GetHwAbstraction().IsFinished())
    {
        controller.GetHwAbstraction().GetLogger().Error(
            "Could not complete inference (HwAbstraction has pending commands)");
        return false;
    }

    return true;
}

}    // namespace ethosn::control_unit
