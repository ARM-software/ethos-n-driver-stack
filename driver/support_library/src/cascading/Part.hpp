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

using PartId         = size_t;
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
        NumStripesType m_Min;
        NumStripesType m_Max;
        bool operator<(const NumStripes& rhs) const;
    };

    struct MceStripesInfo
    {
        TensorShape m_Input;
        TensorShape m_Output;
        TensorShape m_Weight;
        command_stream::BlockConfig m_BlockConfig = { 8U, 8U };

        bool operator<(const MceStripesInfo& rhs) const;
    };

    struct PleStripesInfo
    {
        TensorShape m_Input;
        TensorShape m_Output;
        command_stream::BlockConfig m_BlockConfig = { 8U, 8U };
        bool operator<(const PleStripesInfo& rhs) const;
    };

    struct MemoryStripeInfo
    {
        NumStripes m_Range;
        TensorShape m_Shape;
        bool operator<(const MemoryStripeInfo& rhs) const;
    };

    struct MemoryStripesInfo
    {
        MemoryStripeInfo m_Input;
        MemoryStripeInfo m_Output;
        MemoryStripeInfo m_Weight;
        MemoryStripeInfo m_PleInput;
        bool operator<(const MemoryStripesInfo& rhs) const;
    };

    struct NumMemoryStripes
    {
        NumStripesType m_Input;
        NumStripesType m_Output;
        NumStripesType m_Weight;
        NumStripesType m_PleInput;
        bool operator<(const NumMemoryStripes& rhs) const;
    };

    // The following structs are intermediate representations of plans
    // describing the size of compute stripes and the size and number of memory stripes

    // A representation of plans with both mce and ple operations
    // this is to enable plans which need identity mce or identity ple operations
    struct MceAndPleInfo
    {
        MceStripesInfo m_MceCompute;
        PleStripesInfo m_PleCompute;
        MemoryStripesInfo m_Memory;
        Lifetime m_Lifetime = Lifetime::Cascade;

        bool operator<(const MceAndPleInfo& rhs) const;
    };

    // A representation of plans without an identity PLE operation
    // this is to enable fusing with subsequent ple operations
    struct MceOnlyInfo
    {
        MceStripesInfo m_MceCompute;
        MemoryStripesInfo m_Memory;
        Lifetime m_Lifetime = Lifetime::Cascade;

        bool operator<(const MceOnlyInfo& rhs) const;
    };

    // A representation of plans without an identity MCE operation
    // this is to enable fusing with preceding mce operations
    struct PleOnlyInfo
    {
        PleStripesInfo m_PleCompute;
        MemoryStripesInfo m_Memory;
        Lifetime m_Lifetime = Lifetime::Cascade;

        bool operator<(const PleOnlyInfo& rhs) const;
    };

    // A representation of plans that only use DMA and thus only
    // have information about memory
    struct DmaOnlyInfo
    {
        MemoryStripeInfo m_Input;
        MemoryStripeInfo m_Output;
        Lifetime m_Lifetime = Lifetime::Cascade;

        bool operator<(const DmaOnlyInfo& rhs) const;
    };

    struct StripeInfos
    {
        std::set<MceAndPleInfo> m_MceAndPleInfos;
        std::set<MceOnlyInfo> m_MceOnlyInfos;
        std::set<PleOnlyInfo> m_PleOnlyInfos;
        std::set<DmaOnlyInfo> m_DmaOnlyInfos;
    };

    Part(PartId id,
         const EstimationOptions& estOpt,
         const CompilationOptions& compOpt,
         const HardwareCapabilities& capabilities)
        : DebuggableObject("Part")
        , m_PartId(id)
        , m_NumInvalidPlans(0)
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
    PartId m_PartId;
    size_t m_NumInvalidPlans;

private:
    void AddNewPlan(Plan::InputMapping&& inputMappings, Plan::OutputMapping&& outputMappings, OwnedOpGraph&& opGraph);
    void CreateOpGraphAndPlan(Node* node,
                              DmaOnlyInfo& dmaInfo,
                              NumMemoryStripes& numMemoryStripes,
                              TraversalOrder order,
                              Location input,
                              Location output);
    void CreatePlanForInputNode(Node* node, Lifetime lifetime, TraversalOrder order);
    void CreatePlanForOutputNode(Node* node, Lifetime lifetime, TraversalOrder order);
    void GenerateWithTraversalOrders(Node* node, WeightEncoderCache& weightEncoderCache);
    void GenerateWithStripeSizes(Node* node,
                                 const std::vector<command_stream::BlockConfig>& blockConfigs,
                                 TraversalOrder order,
                                 WeightEncoderCache& weightEncoderCache);
    void GenerateWithNumStripes(Node* node,
                                TraversalOrder order,
                                StripeInfos& stripeInfos,
                                WeightEncoderCache& weightEncoderCache);
    void GenerateMcePlans(Node* node,
                          TraversalOrder order,
                          StripeInfos& stripeInfos,
                          WeightEncoderCache& weightEncoderCache);
    void GenerateFuseOnlyPlePlans(Node* node,
                                  TraversalOrder order,
                                  StripeInfos& stripeInfos,
                                  WeightEncoderCache& weightEncoderCache);
    void GenerateFormatConversionPlans(Node* node,
                                       TraversalOrder order,
                                       StripeInfos& stripeInfos,
                                       Location inputBufferLocaton,
                                       Location outputBufferLocation);
    void CreateMceAndIdentityPlePlans(Node* node,
                                      const MceAndPleInfo& info,
                                      TraversalOrder order,
                                      WeightEncoderCache& weightEncoderCache);
    void CreateMceOnlyPlans(Node* node,
                            const MceOnlyInfo& info,
                            TraversalOrder order,
                            WeightEncoderCache& weightEncoderCache);
    void CreateIdentityMceAndFusedPlePlans(Node* node,
                                           const MceAndPleInfo& info,
                                           TraversalOrder order,
                                           WeightEncoderCache& weightEncoderCache);
    void CreateFuseOnlyPlans(Node* node, const PleOnlyInfo& info, TraversalOrder order);

    void CreateComputePlans(Node* node,
                            StripeInfos& stripeInfos,
                            TraversalOrder order,
                            WeightEncoderCache& weightEncoderCache);

    void CreateFormatConversionPlans(Node* node,
                                     DmaOnlyInfo& dmaInfo,
                                     NumMemoryStripes& numMemoryStripes,
                                     TraversalOrder order,
                                     Location inputBufferLocaton,
                                     Location outputBufferLocation);

    void CreateVirtualSramPlans(Node* node,
                                DmaOnlyInfo& dmaInfo,
                                NumMemoryStripes& numMemoryStripes,
                                TraversalOrder order);

    std::pair<Buffer*, Buffer*> AddIdentityMceOpForSubGraph(OwnedOpGraph& opGraph,
                                                            Lifetime lifetime,
                                                            const Part::MceStripesInfo& mceComputeInfo,
                                                            const Part::NumMemoryStripes& numMemoryStripes,
                                                            const Part::MemoryStripesInfo& memoryStripes,
                                                            const TensorShape& inpShape,
                                                            const QuantizationInfo& inpQuantInfo,
                                                            TraversalOrder order,
                                                            WeightEncoderCache& weightEncoderCache);

    void AddOpToOpGraphWithInputOutputBuffers(OwnedOpGraph& opGraph,
                                              Node* node,
                                              TraversalOrder order,
                                              DmaOnlyInfo& stripeInfos,
                                              NumMemoryStripes& numMemoryStripes,
                                              Location inputBufferLocation,
                                              Location outputBufferLocation,
                                              Plan::InputMapping& inputMappings,
                                              Plan::OutputMapping& outputMappings);

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

    PartId GeneratePartId()
    {
        PartId currId = m_NextPartId;
        ++m_NextPartId;
        return currId;
    }

    Parts m_Parts;
    PartId m_NextPartId = 0;
};

uint32_t CalculateTileSize(Node* node,
                           const HardwareCapabilities& caps,
                           const TensorShape& inputTensorShape,
                           const TensorShape& inputStripeShape,
                           const TensorShape& outputStripeShape,
                           uint32_t numStripes);

}    // namespace support_library
}    // namespace ethosn
