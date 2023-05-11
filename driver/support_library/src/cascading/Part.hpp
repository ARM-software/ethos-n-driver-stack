//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../Graph.hpp"
#include "../Utils.hpp"
#include "DebuggableObject.hpp"

#include <functional>
#include <map>
#include <typeinfo>
#include <utility>
#include <vector>

namespace ethosn
{
namespace support_library
{

class Buffer;

struct BoundaryRequirements
{
    bool m_NeedsBeforeX = false;
    bool m_NeedsAfterX  = false;
    bool m_NeedsBeforeY = false;
    bool m_NeedsAfterY  = false;
};

enum class CascadeType
{
    Beginning,
    Middle,
    End,
    Lonely
};

enum class CascadingBufferFormat
{
    NHWC,
    NCHW,
    NHWCB,
    WEIGHT,
    FCAF_DEEP,
    FCAF_WIDE
};

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
    return typeid(*obj) == typeid(D);
}

using PartId         = uint32_t;
using StripeSizeType = TensorShape::value_type;
using Plans          = std::vector<Plan>;
using InPart         = std::pair<bool, PartId>;
using OutPart        = std::pair<bool, PartId>;
using Nodes          = std::vector<Node*>;

// Object which represents the input to a part
// This consists of the PartId of the part connected
// and the index of that input into the part
struct PartInputSlot
{
    PartId m_PartId;
    uint32_t m_InputIndex;
    bool operator==(const PartInputSlot& r) const
    {
        return m_PartId == r.m_PartId && m_InputIndex == r.m_InputIndex;
    }
    bool operator<(const PartInputSlot& r) const
    {
        if (m_PartId < r.m_PartId)
            return true;
        if (r.m_PartId < m_PartId)
            return false;
        if (m_InputIndex < r.m_InputIndex)
            return true;
        if (r.m_InputIndex < m_InputIndex)
            return false;
        return false;
    }
};

// Object which represents the output to a part
// This consists of the PartId of the part connected
// and the index of that output out of the part
struct PartOutputSlot
{
    PartId m_PartId;
    uint32_t m_OutputIndex;
    bool operator==(const PartOutputSlot& r) const
    {
        return m_PartId == r.m_PartId && m_OutputIndex == r.m_OutputIndex;
    }
    bool operator<(const PartOutputSlot& r) const
    {
        if (m_PartId < r.m_PartId)
            return true;
        if (r.m_PartId < m_PartId)
            return false;
        if (m_OutputIndex < r.m_OutputIndex)
            return true;
        if (r.m_OutputIndex < m_OutputIndex)
            return false;
        return false;
    }
};

using PartInputMapping  = std::map<Buffer*, PartInputSlot>;
using PartOutputMapping = std::map<Buffer*, PartOutputSlot>;
class OwnedOpGraph;

class BasePart : public DebuggableObject
{
public:
    BasePart(PartId id,
             const char* debugPartType,
             const EstimationOptions& estOpt,
             const CompilationOptions& compOpt,
             const HardwareCapabilities& capabilities)
        // Explicitly set the debug tag based on the Part ID, to make the identifiers consistent.
        : DebuggableObject(DebuggableObject::ExplicitDebugTag(),
                           (std::string(debugPartType) + " " + std::to_string(id)).c_str())
        , m_PartId{ id }
        , m_EstimationOptions{ estOpt }
        , m_CompilationOptions{ compOpt }
        , m_Capabilities{ capabilities }
    {}
    template <typename Ids>
    BasePart(PartId id,
             const char* debugPartType,
             Ids&& correspondingOperationIds,
             const EstimationOptions& estOpt,
             const CompilationOptions& compOpt,
             const HardwareCapabilities& capabilities)
        // Explicitly set the debug tag based on the Part ID, to make the identifiers consistent.
        : DebuggableObject(DebuggableObject::ExplicitDebugTag(),
                           (std::string(debugPartType) + " " + std::to_string(id)).c_str())
        , m_PartId{ id }
        , m_CorrespondingOperationIds{ std::forward<Ids>(correspondingOperationIds) }
        , m_EstimationOptions{ estOpt }
        , m_CompilationOptions{ compOpt }
        , m_Capabilities{ capabilities }
    {}
    PartId GetPartId() const;
    virtual Plans GetPlans(CascadeType cascadeType,
                           ethosn::command_stream::BlockConfig blockConfig,
                           Buffer* sramBuffer,
                           uint32_t numWeightStripes) const = 0;
    virtual utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const;

    virtual bool HasActivationBounds() const;
    virtual void ApplyActivationBounds(int16_t lowerBound, int16_t upperBound);
    virtual bool CanDoubleBufferWeights() const;
    virtual bool IsOutputGuaranteedNhwc() const;

    /// Sets the requirements that we store for each output of this part,
    /// so that we generate plans with the correct requirements for consuming parts.
    void SetOutputRequirements(std::vector<BoundaryRequirements> boundaryReqs, std::vector<bool> canTakePleInputSram);
    /// For each input of this Part, do we require boundary data for that input.
    virtual std::vector<BoundaryRequirements> GetInputBoundaryRequirements() const = 0;
    /// For each input of this Part, can it take a PleInputSram buffer.
    virtual std::vector<bool> CanInputsTakePleInputSram() const = 0;

    DotAttributes GetDotAttributes(DetailLevel) const override;

    virtual ~BasePart()
    {}

    std::set<uint32_t> GetOperationIds() const
    {
        return m_CorrespondingOperationIds;
    }
    void AddOperationId(uint32_t operationId);

    /// Gets the weights matrix for this part if it is a 'channel selector', otherwise an empty optional.
    ///
    /// A channel selector part is one which fulfils the following conditions:
    ///  * Single single, single output
    ///  * The output width and height are the same as the input width and height
    ///  * The input and output quantization info are the same
    ///  * The weights quantization zero point must be 0
    ///  * Each channel of the output is either
    ///     - A copy of one of the input channels (i.e. each output channel 'selects' an input channel)
    ///     - or, entirely zero (in real space, so may be non-zero in quant space depending on zero point)
    ///  * Note that an input channel may not necessarily be selected by any output channel(s), and so would be lost.
    ///  * Note that multiple output channels could select the same input channel.
    ///
    /// With these conditions met, the weights matrix returned is guaranteed to be mostly zero, with each
    /// output channel having at most one non-zero value, corresponding to the input channel which it selects.
    ///
    /// These properties enable an optimization where we can merge channel selector parts with the preceding
    /// or following MceParts, by merging the weights of the two layers.
    virtual utils::Optional<utils::ConstTensorData> GetChannelSelectorWeights() const
    {
        // By default, assume that this part is not a channel selector. Derived part types must override this as necessary.
        return {};
    }

    /// If it is possible and efficient to do so, modifies this Part so that it includes the effect
    /// of a preceding channel selector part (see GetChannelSelectorWeights above), allowing that
    /// channel selector part to be removed from the graph.
    virtual bool MergeWithChannelSelectorBefore(const utils::ConstTensorData& channelSelectorWeights)
    {
        // By default, assume that this part cannot be merged with a channel selector. Derived part types must override this as necessary.
        ETHOSN_UNUSED(channelSelectorWeights);
        return false;
    }

    /// If it is possible and efficient to do so, modifies this Part so that it includes the effect
    /// of a following channel selector part (see GetChannelSelectorWeights above), allowing that
    /// channel selector part to be removed from the graph.
    virtual bool MergeWithChannelSelectorAfter(const utils::ConstTensorData& channelSelectorWeights)
    {
        // By default, assume that this part cannot be merged with a channel selector. Derived part types must override this as necessary.
        ETHOSN_UNUSED(channelSelectorWeights);
        return false;
    }

    void ChangePartId(PartId newId);

protected:
    PartId m_PartId;
    std::set<uint32_t> m_CorrespondingOperationIds;
    const EstimationOptions& m_EstimationOptions;
    const CompilationOptions& m_CompilationOptions;
    const HardwareCapabilities& m_Capabilities;
    /// For each output slot, should we generate plans with boundary data or not.
    std::vector<BoundaryRequirements> m_OutputBoundaryRequirements;
    /// For each output slot, should we generate plans with the output buffer in PLE input SRAM or not.
    std::vector<bool> m_OutputCanTakePleInputSram;

    void AddNewPlan(PartInputMapping&& inputMappings,
                    PartOutputMapping&& outputMappings,
                    OwnedOpGraph&& opGraph,
                    Plans& plans) const;
};

using Parts = std::map<PartId, std::unique_ptr<BasePart>>;

class WeightEncoderCache;

}    // namespace support_library

}    // namespace ethosn
