//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "CombinerDFS.hpp"
#include "Part.hpp"

namespace ethosn
{
namespace support_library
{

class Graph;
class HardwareCapabilities;
struct EstimationOptions;
struct DebuggingContext;

class Cascading
{
public:
    Cascading(const EstimationOptions& estOpt,
              const CompilationOptions& compOpt,
              const HardwareCapabilities& caps,
              const DebuggingContext&);

    const GraphOfParts& GetGraphOfParts() const;

    NetworkPerformanceData EstimateNetwork(const Network& network);

    const Combination& GetBestCombination();

private:
    void EstimatePerformance();

    const EstimationOptions& m_EstimationOptions;
    const CompilationOptions& m_CompilationOptions;
    const HardwareCapabilities& m_Capabilities;
    const DebuggingContext& m_DebuggingContext;

    NetworkPerformanceData m_PerformanceStream;
    Combiner m_Combiner;
    GraphOfParts m_GraphOfParts;
};

GraphOfParts CreateGraphOfParts(const Network& network,
                                const HardwareCapabilities& capabilities,
                                const EstimationOptions& estOpt,
                                const CompilationOptions& compOpt);

}    // namespace support_library
}    // namespace ethosn
