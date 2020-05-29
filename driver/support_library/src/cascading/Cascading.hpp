//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../Utils.hpp"
#include "Combiner.hpp"
#include "IEstimationStrategy.hpp"
#include "Part.hpp"

namespace ethosn
{
namespace support_library
{

class Graph;

class Cascading : public IEstimationStrategy
{
public:
    Cascading(const EstimationOptions& estOpt,
              const HardwareCapabilities& caps,
              const DebuggingContext& debuggingContext);
    virtual ~Cascading();

    NetworkPerformanceData Estimate(Graph& graph) override;
    void EstimatePerformance();
    NetworkPerformanceData EstimateCombination(const Combination&);

    Combinations Combine(const GraphOfParts&);

    Combination GetOptimalCombination(const Combinations&)
    {
        Combination result;

        // TODO

        return result;
    }

    const GraphOfParts& getGraphOfParts() const;

private:
    NetworkPerformanceData m_PerformanceStream;
    Metadata m_Metadata;
    Combinations m_ValidCombinations;
    GraphOfParts m_GraphOfParts;
};

GraphOfParts CreateGraphOfParts(const Graph& graph);
NetworkPerformanceData EstimateCombination(const Combination& combination,
                                           const GraphOfParts& parts,
                                           const HardwareCapabilities& capabilities);

}    // namespace support_library
}    // namespace ethosn
