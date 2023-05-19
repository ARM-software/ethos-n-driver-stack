//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "CombinerDFS.hpp"
#include "Compiler.hpp"
#include "Part.hpp"
#include "cascading/CascadingCommandStreamGenerator.hpp"

namespace ethosn
{
namespace support_library
{

class HardwareCapabilities;
struct EstimationOptions;
struct DebuggingContext;

struct RunCascadingResult
{
    OpGraph opGraph;
    /// This is necessary to keep data alive which is referenced inside `compiledOpGraph` and `opGraph`.
    Combination combination;
    /// Some fields of this will be empty/null if estimation was requested.
    cascading_compiler::CompiledOpGraph compiledOpGraph;

    const NetworkPerformanceData& GetNetworkPerformanceData() const
    {
        return compiledOpGraph.m_EstimatedOpGraph.m_PerfData;
    }
};

/// Estimation and Compilation share a lot of the same code path, so this function is used to run both.
/// The presence (or lack) of `estOpt` determines if estimation or compilation is performed.
RunCascadingResult RunCascading(const Network& network,
                                utils::Optional<const EstimationOptions&> estOpt,
                                const CompilationOptions& compOpt,
                                const HardwareCapabilities& caps,
                                DebuggingContext& debuggingContext);

}    // namespace support_library
}    // namespace ethosn
