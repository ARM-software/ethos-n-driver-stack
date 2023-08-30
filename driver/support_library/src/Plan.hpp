//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "DebuggableObject.hpp"
#include "OpGraph.hpp"
#include "Part.hpp"
#include "WeightEncoder.hpp"

#include "../include/ethosn_support_library/Optional.hpp"
#include <ethosn_command_stream/PleKernelIds.hpp>

#include <map>
#include <unordered_map>

namespace ethosn
{
namespace support_library
{

struct SizeInBytes
{
    uint32_t m_Tot       = 0;
    uint32_t m_TotAtomic = 0;
};

struct PleKernelInfo
{
    uint32_t m_Size;
    PleOp* m_PleOp;
};

class Plan : public DebuggableObject
{
public:
    Plan();
    Plan(PartInputMapping&& inputMappings, PartOutputMapping&& outputMappings);
    Plan(Plan&&) = default;
    virtual ~Plan()
    {}

    /// Gets the Buffer corresponding to the given part's input slot, which should be an input to the Part that this Plan is for.
    /// Returns nullptr if the slot is unrecognised.
    Buffer* GetInputBuffer(const PartInputSlot& partInputSlot) const;
    /// Gets the Buffer corresponding to the given part's output slot, which should be an output from the Part that this Plan is for.
    /// Returns nullptr if the slot is unrecognised.
    Buffer* GetOutputBuffer(const PartOutputSlot& partOutputSlot) const;

    PleKernelInfo GetPleKernelInfo(const HardwareCapabilities& cap) const;

    /// The graph of Ops and Buffers which define how this plan would be executed.
    OwnedOpGraph m_OpGraph;

    /// Specifies which of the Buffers in the above OpGraph are inputs to this plan, and which Part inputs
    /// these correspond to
    PartInputMapping m_InputMappings;
    /// Specifies which of the Buffers in the above OpGraph are outputs from this plan, and which Part outputs
    /// these correspond to.
    PartOutputMapping m_OutputMappings;

    /// Have the SRAM buffers for this plan already been allocated in SRAM?
    /// Note that this only makes sense for Lonely plans, and prevents the Combiner from doing its own allocation.
    bool m_IsPreallocated = false;

    /// For plans which have the concept of a block config. This is used by the combiner to ensure
    /// a consistent block config throughout a section.
    utils::Optional<BlockConfig> m_BlockConfig;
};

bool IsOutputBufferInDram(const Plan& plan, const PartOutputSlot& outputSlot);
bool IsInputBufferInSram(const Plan& plan, const PartInputSlot& inputSlot);
bool IsOutputBufferInSram(const Plan& plan, const PartOutputSlot& outputSlot);

SizeInBytes GetTotSizeInBytes(const Plan& plan);
SizeInBytes GetInputsSizeInBytes(const Plan& plan);

}    // namespace support_library
}    // namespace ethosn
