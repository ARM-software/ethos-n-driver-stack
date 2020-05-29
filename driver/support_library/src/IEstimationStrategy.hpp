//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ethosn_support_library/Support.hpp"

namespace ethosn
{
namespace support_library
{

class Graph;
struct EstimationOptions;
class HardwareCapabilities;
struct DebuggingContext;

class IEstimationStrategy
{
public:
    IEstimationStrategy(const EstimationOptions& estOpt,
                        const HardwareCapabilities& hwCap,
                        const DebuggingContext& debuggingContext)
        : m_EstimationOptions(estOpt)
        , m_Capabilities(hwCap)
        , m_DebuggingContext(debuggingContext){};
    virtual NetworkPerformanceData Estimate(Graph& graph) = 0;
    virtual ~IEstimationStrategy()
    {}
    IEstimationStrategy(const IEstimationStrategy&) = delete;
    IEstimationStrategy& operator=(const IEstimationStrategy&) = delete;

    EstimationOptions GetEstimationOptions(void)
    {
        return m_EstimationOptions;
    }

protected:
    const EstimationOptions& m_EstimationOptions;
    const HardwareCapabilities& m_Capabilities;
    const DebuggingContext& m_DebuggingContext;
};

}    // namespace support_library
}    // namespace ethosn
