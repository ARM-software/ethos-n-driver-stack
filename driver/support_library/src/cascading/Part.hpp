//
// Copyright © 2018-2021 Arm Limited.
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
D* GetObjectAs(B* obj)
{
    return dynamic_cast<D*>(obj);
}

template <typename D, typename B>
const D* GetObjectAs(const B* obj)
{
    return dynamic_cast<const D*>(obj);
}

template <typename D, typename B>
bool IsObjectOfType(const B* obj)
{
    return (GetObjectAs<D>(obj) != nullptr);
}

using Plans          = std::vector<std::unique_ptr<Plan>>;
using StripeSizeType = TensorShape::value_type;

class WeightEncoderCache;

class Part : public DebuggableObject
{
public:
    using Nodes = std::vector<Node*>;

    using NumStripesType = uint32_t;
    struct NumStripes
    {
        NumStripesType minInputStripes;
        NumStripesType maxInputStripes;
        NumStripesType minOutputStripes;
        NumStripesType maxOutputStripes;
        NumStripesType minWeightStripes;
        NumStripesType maxWeightStripes;

        bool operator<(const NumStripes& rhs) const;
    };

    struct StripeInfos
    {
        TensorShape m_InputStripeShape;
        TensorShape m_OutputStripeShape;
        NumStripes m_NumStripes;
        command_stream::BlockConfig m_BlockConfig = { 8U, 8U };
        Lifetime m_Lifetime                       = Lifetime::Cascade;

        bool operator<(const StripeInfos& rhs) const;
    };

    Part(const EstimationOptions& estOpt, const CompilationOptions& compOpt, const HardwareCapabilities& capabilities)
        : DebuggableObject("Part")
        , m_EstimationOptions(estOpt)
        , m_CompilationOptions(compOpt)
        , m_Capabilities(capabilities)
    {}

    void CreatePlans();
    const Plan& GetPlan(const PlanId id) const;
    size_t GetNumPlans() const;
    std::vector<const Edge*> GetInputs() const;
    std::vector<const Edge*> GetOutputs() const;

    // SubGraph of Nodes for this Part
    Nodes m_SubGraph;

    // All valid plans for this Part
    Plans m_Plans;
    size_t m_NumInvalidPlans;

private:
    void AddNewPlan(Plan::InputMapping&& inputMappings, Plan::OutputMapping&& outputMappings, OwnedOpGraph&& opGraph);
    void CreatePlanForInputNode(Node* node, Lifetime lifetime, TraversalOrder order);
    void CreatePlanForOutputNode(Node* node, Lifetime lifetime, TraversalOrder order);
    void CreatePlanForNode(Node* node,
                           Lifetime lifetime,
                           TraversalOrder order,
                           TensorShape inputShape,
                           TensorShape outputShape,
                           NumStripesType numInputStripes,
                           NumStripesType numOutputStripes,
                           NumStripesType numWeightStripes,
                           Location inputBufferLocaton,
                           Location outputBufferLocation,
                           command_stream::BlockConfig blockConfig,
                           WeightEncoderCache& weightEncoderCache);
    void GenerateWithTraversalOrders(Node* node, WeightEncoderCache& weightEncoderCache);
    void GenerateWithStripeSizes(Node* node,
                                 const std::vector<command_stream::BlockConfig>& blockConfigs,
                                 TraversalOrder order,
                                 WeightEncoderCache& weightEncoderCache);
    void GenerateWithNumStripes(Node* node,
                                TraversalOrder order,
                                const std::set<StripeInfos>& stripeInfo,
                                WeightEncoderCache& weightEncoderCache);
    void GenerateWithNumStripesForLocation(Node* node,
                                           TraversalOrder order,
                                           const std::set<StripeInfos>& stripeInfos,
                                           Location inputBufferLocaton,
                                           Location outputBufferLocation,
                                           WeightEncoderCache& weightEncoderCache);

    Buffer* AddIdentityMceOpForSubGraph(OwnedOpGraph& opGraph,
                                        const TensorShape& inputShape,
                                        const QuantizationInfo& inpQuantInfo,
                                        Lifetime lifetime,
                                        TraversalOrder order,
                                        TensorShape inputStripe,
                                        TensorShape outputStripe,
                                        NumStripesType numInputStripes,
                                        NumStripesType numWeightStripes,
                                        command_stream::BlockConfig blockConfig,
                                        WeightEncoderCache& weightEncoderCache);

    void CreatePlanWithIdentityMceOp(FuseOnlyPleOperationNode* node,
                                     Lifetime lifetime,
                                     TraversalOrder order,
                                     TensorShape inputStripe,
                                     TensorShape outputStripe,
                                     NumStripesType numOutputStripes,
                                     command_stream::BlockConfig blockConfig,
                                     WeightEncoderCache& weightEncoderCache);

    void AddOpToOpGraphWithInputOutputBuffers(OwnedOpGraph& opGraph,
                                              Node* node,
                                              Lifetime lifetime,
                                              TraversalOrder order,
                                              TensorShape inputStripe,
                                              TensorShape outputStripe,
                                              NumStripesType numInputStripes,
                                              NumStripesType numOutputStripes,
                                              Location inputBufferLocaton,
                                              Location outputBufferLocation,
                                              command_stream::BlockConfig blockConfig,
                                              Plan::InputMapping& inputMappings,
                                              Plan::OutputMapping& outputMappings);

    void CreatePlanWithIdentityPleOp(Node* node,
                                     Lifetime lifetime,
                                     TraversalOrder order,
                                     TensorShape inputStripe,
                                     TensorShape outputStripe,
                                     NumStripesType numInputStripes,
                                     NumStripesType numOutputStripes,
                                     NumStripesType numWeightStripes,
                                     Location inputBufferLocaton,
                                     Location outputBufferLocation,
                                     command_stream::BlockConfig blockConfig,
                                     WeightEncoderCache& weightEncoderCache);

    const EstimationOptions& m_EstimationOptions;
    const CompilationOptions& m_CompilationOptions;
    const HardwareCapabilities& m_Capabilities;
};

using Parts = std::vector<std::unique_ptr<Part>>;

using InPart  = std::pair<bool, PartId>;
using OutPart = std::pair<bool, PartId>;

class GraphOfParts
{
public:
    GraphOfParts() = default;

    size_t GetNumInvalidPlans() const
    {
        size_t result = 0;
        for (const auto& part : m_Parts)
        {
            result += part->m_NumInvalidPlans;
        }
        return result;
    }

    size_t GetNumParts() const;
    const Part& GetPart(const PartId id) const;
    const Parts& GetParts() const;

    InPart GetInputPart(const Edge& e) const;
    OutPart GetOutputPart(const Edge& e) const;

    Parts m_Parts;
};

uint32_t CalculateTileSize(Node* node,
                           const HardwareCapabilities& caps,
                           const TensorShape& inputTensorShape,
                           const TensorShape& inputStripeShape,
                           const TensorShape& outputStripeShape,
                           uint32_t numStripes);

}    // namespace support_library
}    // namespace ethosn
