//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Combiner.hpp"
#include "IEstimationStrategy.hpp"
#include "Part.hpp"

namespace ethosn
{
namespace support_library
{

class Graph;
class HardwareCapabilities;
struct EstimationOptions;
struct DebuggingContext;

class Cascading : public IEstimationStrategy
{
public:
    Cascading(const EstimationOptions& estOpt, const CompilationOptions& compOpt, const HardwareCapabilities& caps);
    virtual ~Cascading();

    const GraphOfParts& GetGraphOfParts() const;

    Combinations Combine(const GraphOfParts&);
    NetworkPerformanceData Estimate(Graph& graph) override;

    const Combination* GetBestCombination();

private:
    void EstimatePerformance();

    NetworkPerformanceData m_PerformanceStream;
    const Combination* m_BestCombination;
    Metadata m_Metadata;
    Combinations m_ValidCombinations;
    GraphOfParts m_GraphOfParts;
};

GraphOfParts CreateGraphOfParts(const Graph& graph,
                                const EstimationOptions& estOpt,
                                const CompilationOptions& compOpt,
                                const HardwareCapabilities& capabilities);

}    // namespace support_library
}    // namespace ethosn
