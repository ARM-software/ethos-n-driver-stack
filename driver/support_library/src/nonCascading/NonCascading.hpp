//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "IEstimationStrategy.hpp"

namespace ethosn
{
namespace support_library
{

class NonCascading : public IEstimationStrategy
{
public:
    NonCascading(const EstimationOptions& estOpt,
                 const HardwareCapabilities& hwCap,
                 const DebuggingContext& debuggingContext);
    NetworkPerformanceData Estimate(Graph& graph) override;

private:
    void EstimateCascading();
    NetworkPerformanceData m_PerformanceStream;
};

}    // namespace support_library

}    // namespace ethosn
