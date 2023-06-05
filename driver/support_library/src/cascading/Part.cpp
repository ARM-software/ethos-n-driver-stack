//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Part.hpp"

#include "../Utils.hpp"
#include "GraphOfParts.hpp"
#include "Plan.hpp"
#include "WeightEncoder.hpp"
#include "WeightEncoderCache.hpp"

#include <unordered_map>

using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{

using namespace utils;

PartId BasePart::GetPartId() const
{
    return m_PartId;
}

utils::Optional<ethosn::command_stream::MceOperation> BasePart::GetMceOperation() const
{
    utils::Optional<ethosn::command_stream::MceOperation> mceOperationWithNoValue;
    return mceOperationWithNoValue;
}

bool IsPlanValid(const HardwareCapabilities& caps, const Plan& plan)
{
    const uint32_t pleSize     = plan.GetPleKernelInfo(caps).m_Size;
    const uint32_t sizeInBytes = GetTotSizeInBytes(plan).m_Tot + pleSize * caps.GetNumberOfSrams();

    if (sizeInBytes > caps.GetTotalSramSize())
    {
        // There is no space
        return false;
    }

    return true;
}

void BasePart::ChangePartId(PartId newId)
{
    m_PartId      = newId;
    size_t offset = m_DebugTag.find_last_of(' ');
    m_DebugTag    = m_DebugTag.substr(0, offset) + " " + std::to_string(newId);
}

void BasePart::AddNewPlan(PartInputMapping&& inputMappings,
                          PartOutputMapping&& outputMappings,
                          OwnedOpGraph&& opGraph,
                          utils::Optional<command_stream::BlockConfig> blockConfig,
                          Plans& plans) const
{
    Plan plan(std::move(inputMappings), std::move(outputMappings));
    plan.m_OpGraph     = std::move(opGraph);
    plan.m_BlockConfig = blockConfig;

    if (IsPlanValid(m_Capabilities, plan))
    {
        plans.push_back(std::move(plan));
    }
}

ethosn::support_library::DotAttributes BasePart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = DebuggableObject::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "\n";
        result.m_Label += "CorrespondingOperationIds = " + ArrayToString(m_CorrespondingOperationIds) + "\n";
    }
    return result;
}

bool BasePart::HasActivationBounds() const
{
    return false;
}

bool BasePart::CanDoubleBufferWeights() const
{
    return false;
}

bool BasePart::IsOutputGuaranteedNhwc() const
{
    return false;
}

void BasePart::ApplyActivationBounds(int16_t, int16_t)
{}

void BasePart::AddOperationId(uint32_t operationId)
{
    m_CorrespondingOperationIds.insert(operationId);
}

void BasePart::SetOutputRequirements(std::vector<BoundaryRequirements> boundaryReqs,
                                     std::vector<bool> canTakePleInputSram)
{
    m_OutputBoundaryRequirements = std::move(boundaryReqs);
    m_OutputCanTakePleInputSram  = std::move(canTakePleInputSram);
}

}    // namespace support_library
}    // namespace ethosn
