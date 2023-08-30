//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Plan.hpp"

using namespace std;
using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{

Plan::Plan()
    : Plan({}, {})
{}

Plan::Plan(PartInputMapping&& inputMappings, PartOutputMapping&& outputMappings)
    : DebuggableObject("Plan")
    , m_InputMappings(std::move(inputMappings))
    , m_OutputMappings(std::move(outputMappings))
{}

Buffer* Plan::GetInputBuffer(const PartInputSlot& partInputSlot) const
{
    for (const auto& pair : m_InputMappings)
    {
        if (pair.second == partInputSlot)
        {
            return pair.first;
        }
    }
    return nullptr;
}

Buffer* Plan::GetOutputBuffer(const PartOutputSlot& partOutputSlot) const
{
    for (const auto& pair : m_OutputMappings)
    {
        if (pair.second == partOutputSlot)
        {
            return pair.first;
        }
    }
    return nullptr;
}

PleKernelInfo Plan::GetPleKernelInfo(const HardwareCapabilities& cap) const
{
    PleKernelInfo pleKernelInfo;
    pleKernelInfo.m_Size  = 0;
    pleKernelInfo.m_PleOp = nullptr;

    for (auto& op : m_OpGraph.GetOps())
    {
        if (IsObjectOfType<PleOp>(op))
        {
            PleOp* pleOp          = static_cast<PleOp*>(op);
            pleKernelInfo.m_Size  = cap.GetMaxPleSize();
            pleKernelInfo.m_PleOp = pleOp;
            break;
        }
    }

    return pleKernelInfo;
}

bool IsOutputBufferInDram(const Plan& plan, const PartOutputSlot& outputSlot)
{
    const Buffer* buf = plan.GetOutputBuffer(outputSlot);
    return (buf == nullptr) ? true : ((buf->m_Location) == Location::Dram);
}

bool IsInputBufferInSram(const Plan& plan, const PartInputSlot& inputSlot)
{
    const Buffer* buf = plan.GetInputBuffer(inputSlot);
    return (buf == nullptr) ? false : ((buf->m_Location) == Location::Sram);
}

bool IsOutputBufferInSram(const Plan& plan, const PartOutputSlot& outputSlot)
{
    const Buffer* buf = plan.GetOutputBuffer(outputSlot);
    return (buf == nullptr) ? false : ((buf->m_Location) == Location::Sram);
}

SizeInBytes GetTotSizeInBytes(const Plan& plan)
{
    SizeInBytes result;
    const OpGraph::BufferList& bufs        = plan.m_OpGraph.GetBuffers();
    OpGraph::BufferList::const_iterator it = bufs.begin();
    while (it != bufs.end())
    {
        const Buffer* buf   = *it;
        const uint32_t size = buf->m_SizeInBytes;
        if (buf->m_Location == Location::Sram)
        {
            result.m_Tot += size;
        }
        ++it;
    }
    assert(result.m_TotAtomic <= result.m_Tot);
    return result;
}

SizeInBytes GetInputsSizeInBytes(const Plan& plan)
{
    SizeInBytes result;
    const PartInputMapping in           = plan.m_InputMappings;
    PartInputMapping::const_iterator it = in.begin();
    while (it != in.end())
    {
        const Buffer* buf   = it->first;
        const uint32_t size = buf->m_SizeInBytes;
        if (buf->m_Location == Location::Sram)
        {
            result.m_Tot += size;
        }
        ++it;
    }
    assert(result.m_TotAtomic <= result.m_Tot);
    return result;
}

}    // namespace support_library
}    // namespace ethosn
