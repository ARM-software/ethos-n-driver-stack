//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../../include/ethosn_support_library/Support.hpp"

namespace ethosn
{
namespace support_library
{

class Graph;
struct EstimationOptions;
class HardwareCapabilities;

NetworkPerformanceData
    NonCascadingEstimate(Graph& graph, const EstimationOptions& estOpt, const HardwareCapabilities& hwCap);

}    // namespace support_library

}    // namespace ethosn
