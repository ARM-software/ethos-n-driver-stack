//
// Copyright © 2018-2021 Arm Limited.
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
    NonCascading(const EstimationOptions& estOpt, const CompilationOptions& compOpt, const HardwareCapabilities& hwCap);
    NetworkPerformanceData Estimate(Graph& graph) override;

private:
    NetworkPerformanceData m_PerformanceStream;
};

}    // namespace support_library

}    // namespace ethosn
