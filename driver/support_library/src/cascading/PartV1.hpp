//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

class WeightEncoderCache;

class PartV1 : public BasePart
{
public:
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

    struct StripeInfos
    {
        std::set<MceAndPleInfo> m_MceAndPleInfos;
        std::set<MceOnlyInfo> m_MceOnlyInfos;
        std::set<PleOnlyInfo> m_PleOnlyInfos;
        std::set<DmaOnlyInfo> m_DmaOnlyInfos;
    };

    PartV1(PartId id,
           const CompilerDataFormat& compilerDataFormat,
           const QuantizationInfo& quantizationInfo,
           const std::set<uint32_t>& correspondingOperationIds,
           const EstimationOptions& estOpt,
           const CompilationOptions& compOpt,
           const HardwareCapabilities& capabilities)
        : BasePart(id, compilerDataFormat, quantizationInfo, correspondingOperationIds, estOpt, compOpt, capabilities)
    {}

    virtual Plans GetPlans(CascadeType cascadeType,
                           ethosn::command_stream::BlockConfig blockConfig,
                           Buffer* sramBuffer,
                           uint32_t numWeightStripes) const;

    virtual utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const;

    std::vector<const Edge*> GetInputs() const;
    std::vector<const Edge*> GetOutputs() const;

    // SubGraph of Nodes for this Part
    Nodes m_SubGraph;

private:
    void GenerateWithTraversalOrders(CascadeType cascadeType,
                                     Buffer* sramBuffer,
                                     uint32_t numWeightStripes,
                                     Node* node,
                                     WeightEncoderCache& weightEncoderCache,
                                     Plans& plans) const;
    void GenerateWithStripeSizes(Node* node,
                                 const std::vector<command_stream::BlockConfig>& blockConfigs,
                                 TraversalOrder order,
                                 WeightEncoderCache& weightEncoderCache,
                                 Plans& plans) const;
    void GenerateWithNumStripes(Node* node,
                                TraversalOrder order,
                                StripeInfos& stripeInfos,
                                WeightEncoderCache& weightEncoderCache,
                                Plans& plans) const;
    void GenerateMcePlans(Node* node,
                          TraversalOrder order,
                          StripeInfos& stripeInfos,
                          WeightEncoderCache& weightEncoderCache,
                          Plans& plans) const;
    void GenerateFuseOnlyPlePlans(Node* node,
                                  TraversalOrder order,
                                  StripeInfos& stripeInfos,
                                  WeightEncoderCache& weightEncoderCache,
                                  Plans& plans) const;
    void GenerateFormatConversionPlans(Node* node,
                                       TraversalOrder order,
                                       StripeInfos& stripeInfos,
                                       Location inputBufferLocaton,
                                       Location outputBufferLocation,
                                       Plans& plans) const;
    void CreateMceAndIdentityPlePlans(Node* node,
                                      const MceAndPleInfo& info,
                                      TraversalOrder order,
                                      WeightEncoderCache& weightEncoderCache,
                                      Plans& plans) const;
    void CreateMceOnlyPlans(Node* node,
                            const MceOnlyInfo& info,
                            TraversalOrder order,
                            WeightEncoderCache& weightEncoderCache,
                            Plans& plans) const;
    void CreateIdentityMceAndFusedPlePlans(Node* node,
                                           const MceAndPleInfo& info,
                                           TraversalOrder order,
                                           WeightEncoderCache& weightEncoderCache,
                                           Plans& plans) const;
    void CreateFuseOnlyPlans(Node* node, const PleOnlyInfo& info, TraversalOrder order, Plans& plans) const;

    void CreateFormatConversionPlans(Node* node,
                                     DmaOnlyInfo& dmaInfo,
                                     NumMemoryStripes& numMemoryStripes,
                                     TraversalOrder order,
                                     Location inputBufferLocaton,
                                     Location outputBufferLocation,
                                     Plans& plans) const;

    void CreateVirtualSramPlans(
        Node* node, DmaOnlyInfo& dmaInfo, NumMemoryStripes& numMemoryStripes, TraversalOrder order, Plans& plans) const;

    std::pair<Buffer*, Buffer*> AddIdentityMceOpForSubGraph(OwnedOpGraph& opGraph,
                                                            Lifetime lifetime,
                                                            const PartV1::MceStripesInfo& mceComputeInfo,
                                                            const NumMemoryStripes& numMemoryStripes,
                                                            const MemoryStripesInfo& memoryStripes,
                                                            const TensorShape& inpShape,
                                                            const QuantizationInfo& inpQuantInfo,
                                                            TraversalOrder order,
                                                            WeightEncoderCache& weightEncoderCache) const;
};

}    // namespace support_library
}    // namespace ethosn
