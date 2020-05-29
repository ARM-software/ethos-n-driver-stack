//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../Graph.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

template <typename D, typename B>
bool IsObjectOfType(const B* obj)
{
    return (dynamic_cast<const D*>(obj) != nullptr);
}

using Plans          = std::vector<std::unique_ptr<Plan>>;
using StripeSizeType = TensorShape::value_type;

class Part : public DebuggableObject
{
public:
    using Nodes = std::vector<Node*>;

    Part()
        : DebuggableObject("Part")
    {}

    void CreatePlans(const HardwareCapabilities& caps);
    const Plan& GetPlan(const PlanId id) const;
    size_t GetNumPlans() const;
    std::vector<const Edge*> GetInputs() const;
    std::vector<const Edge*> GetOutputs() const;

    // SubGraph of Nodes for this Part
    Nodes m_SubGraph;

    // All valid plans for this Part
    Plans m_Plans;

private:
    uint32_t CalculateSizeInBytes(const TensorShape& shape) const;
};

using Parts = std::vector<std::unique_ptr<Part>>;

using InPart  = std::pair<bool, PartId>;
using OutPart = std::pair<bool, PartId>;

class GraphOfParts
{
public:
    GraphOfParts() = default;
    size_t GetNumParts() const;
    const Part& GetPart(const PartId id) const;
    const Parts& GetParts() const;

    InPart GetInputPart(const Edge& e) const;
    OutPart GetOutputPart(const Edge& e) const;

    Parts m_Parts;
};

}    // namespace support_library
}    // namespace ethosn
