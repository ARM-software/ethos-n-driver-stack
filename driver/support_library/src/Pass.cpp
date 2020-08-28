//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Pass.hpp"

#include "Compiler.hpp"
#include "Utils.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

namespace ethosn
{
namespace support_library
{

using namespace utils;

command_stream::DataLocation GetCommandDataLocation(BufferLocation bufferLocation)
{
    assert(bufferLocation == BufferLocation::Dram || bufferLocation == BufferLocation::Sram);

    if (bufferLocation == BufferLocation::Sram)
    {
        return command_stream::DataLocation::SRAM;
    }
    else
    {
        return command_stream::DataLocation::DRAM;
    }
}

namespace
{
std::string GetParentIds(const Node& node);

std::string GetIdOfPass(const Node& node)
{
    if (node.GetPass() != nullptr)
    {
        return std::to_string(node.GetPass()->GetId());
    }

    return GetParentIds(node);
}

std::string GetParentIds(const Node& node)
{
    std::stringstream ss;

    ss << '[';
    for (auto it = node.GetInputs().begin(); it != node.GetInputs().end(); ++it)
    {
        const bool isLast = it == std::prev(node.GetInputs().end());
        ss << ' ' << GetIdOfPass(*(*it)->GetSource()) << (isLast ? ' ' : ',');
    }
    ss << ']';

    return ss.str();
}
}    // namespace

void Pass::Estimate(std::vector<PassPerformanceData>& perfStream, const EstimationOptions& estimationOptions)
{
    PassPerformanceData perfData;

    perfData.m_OperationIds = GetCorrespondingOperationIds();
    perfData.m_ParentIds    = GetParentIds(*m_Nodes.front());
    perfData.m_Stats        = GetStats(estimationOptions);

    perfStream.emplace_back(std::move(perfData));

    m_IsEstimated = true;
}

void Pass::PreGenerate(command_stream::CommandStreamBuffer& cmdStream)
{
    m_CommandStreamFirstCommandIdx = cmdStream.GetCount();
}

void Pass::PostGenerate(command_stream::CommandStreamBuffer& cmdStream, bool dumpRam)
{
    m_IsGenerated = true;

    if (dumpRam)
    {
        if (m_Nodes.back()->GetLocation() == ethosn::support_library::BufferLocation::Dram)
        {
            // In order for the end-to-end tests to only validate dram dumps when the output is actually in DRAM,
            // have a different dumpName for passes that have static outputs (output in SRAM).
            const char* const ignoreStr = (m_Nodes.back()->GetLocation() == BufferLocation::Sram) ? "IGNORE_" : "";
            const std::string dumpName  = ignoreStr + std::to_string(m_Nodes.back()->GetShape()[0]) + "_" +
                                         std::to_string(m_Nodes.back()->GetShape()[1]) + "_" +
                                         std::to_string(m_Nodes.back()->GetShape()[2]) + "_" +
                                         std::to_string(m_Nodes.back()->GetShape()[3]) + "_CommandStream_Operation_" +
                                         std::to_string(m_Id) + "_OutputModel_NHWCB.hex";

            ethosn::command_stream::DumpDram cmdStrDumpDram;
            cmdStrDumpDram.m_DramBufferId() = m_Nodes.back()->GetBufferId();

            std::copy(dumpName.begin(), dumpName.end(), cmdStrDumpDram.m_Filename().begin());
            cmdStream.EmplaceBack(cmdStrDumpDram);
        }

        ethosn::command_stream::DumpSram cmdStrDumpSram;
        const std::string dumpName = "output_ce_" + std::to_string(m_Id);
        std::copy(dumpName.begin(), dumpName.end(), cmdStrDumpSram.m_Filename().begin());
        cmdStream.EmplaceBack(cmdStrDumpSram);
    }

    m_CommandStreamLastCommandIdx = cmdStream.GetCount() - 1;
}

std::set<uint32_t> Pass::GetCorrespondingOperationIds() const
{
    std::set<uint32_t> result;
    for (const Node* n : m_Nodes)
    {
        std::set<uint32_t> nodeOperationIds = n->GetCorrespondingOperationIds();
        result.insert(nodeOperationIds.begin(), nodeOperationIds.end());
    }
    return result;
}

ethosn::support_library::DotAttributes Pass::GetDotAttributes()
{
    std::stringstream stream;
    stream << std::hex << m_Nodes.back()->GetOutputSramOffset();
    std::string outputSramOffset =
        m_Nodes.back()->GetLocation() == BufferLocation::Sram ? "\nOutputSramOffset " + stream.str() : "";
    return DotAttributes(std::to_string(m_Id),
                         "Pass " + std::to_string(m_Id) + "\nCommands " +
                             std::to_string(m_CommandStreamFirstCommandIdx) + "-" +
                             std::to_string(m_CommandStreamLastCommandIdx) + "\nOutputSramOffset " + outputSramOffset,
                         "black");
}

ConcatNode* FindConcatNode(Node* node)
{
    for (const auto& n : node->GetOutputs())
    {
        if (dynamic_cast<ConcatNode*>(n->GetDestination()))
        {
            return dynamic_cast<ConcatNode*>(n->GetDestination());
        }
    }
    return nullptr;
}

std::pair<TensorShape, TensorShape> CalculateConcatSupertensorInfo(Node* inputToConcat, ConcatNode* concatNode)
{
    assert(inputToConcat);
    assert(concatNode);
    uint32_t axis = concatNode->GetAxis();

    TensorShape offset = { 0, 0, 0, 0 };
    for (uint32_t inputIdx = 0; inputIdx < concatNode->GetInputs().size(); ++inputIdx)
    {
        if (concatNode->GetInput(inputIdx)->GetSource() == inputToConcat)
        {
            break;
        }
        offset[axis] += concatNode->GetInputShape(inputIdx)[axis];
    }
    std::pair<TensorShape, TensorShape> res;
    res.first  = offset;
    res.second = concatNode->GetShape();
    return res;
}

}    // namespace support_library
}    // namespace ethosn
