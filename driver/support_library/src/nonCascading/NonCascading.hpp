//
// Copyright © 2018-2022 Arm Limited.
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

NetworkPerformanceData NonCascadingEstimate(Graph& graph, const EstimationOptions& estOpt);

}    // namespace support_library

}    // namespace ethosn
