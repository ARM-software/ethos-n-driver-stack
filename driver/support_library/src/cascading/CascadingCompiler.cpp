//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../Compiler.hpp"
#include "CascadingCompiler.hpp"

#include <memory>

namespace ethosn
{
namespace support_library
{

CascadingCompiler::CascadingCompiler(const OpGraph& mergedOpGraph,
                                     const std::set<uint32_t>& operationIds,
                                     const HardwareCapabilities& capabilities,
                                     const CompilationOptions& compilationOptions)
    : m_MergedOpGraph{ mergedOpGraph }
    , m_OperationIds{ operationIds }
    , m_Capabilities{ capabilities }
    , m_CompilationOptions{ compilationOptions }
{}

CascadingCompiler::~CascadingCompiler()
{}

std::unique_ptr<CompiledNetwork> CascadingCompiler::Compile()
{
    OpGraph::OpList opsInExecutionOrder = m_MergedOpGraph.GetOps();

    assert(opsInExecutionOrder.size() != 0);

    try
    {
        for (auto currentOp : opsInExecutionOrder)
        {

            if (IsObjectOfType<DmaOp>(currentOp))
            {
                ProcessDmaOp(currentOp);
            }
            else if (IsObjectOfType<MceOp>(currentOp))
            {
                ProcessMceOp(currentOp);
            }
            else if (IsObjectOfType<PleOp>(currentOp))
            {
                ProcessPleOp(currentOp);
            }
            else if (IsObjectOfType<ConcatOp>(currentOp))
            {
                ProcessConcatOp(currentOp);
            }
            else
            {
                throw NotSupportedException("Op is not currently supported by the Cascading Compiler");
            }
        }
    }
    catch (const NotSupportedException& e)
    {
        g_Logger.Error("Error: %s", e.what());
        return std::unique_ptr<CompiledNetworkImpl>(nullptr);
    }

    m_BufferManager.AddCommandStream(m_CommandStream);

    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = std::make_unique<CompiledNetworkImpl>(
        m_BufferManager.GetConstantDmaData(), m_BufferManager.GetConstantControlUnitData(),
        m_BufferManager.GetBuffers(), m_OperationIds);

    return compiledNetwork;
}

void CascadingCompiler::ProcessDmaOp(Op* const ptrDmaOp)
{
    ETHOSN_UNUSED(ptrDmaOp);
}

void CascadingCompiler::ProcessMceOp(Op* const ptrMceOp)
{
    ETHOSN_UNUSED(ptrMceOp);
}

void CascadingCompiler::ProcessPleOp(Op* const ptrPleOp)
{
    ETHOSN_UNUSED(ptrPleOp);
}

void CascadingCompiler::ProcessConcatOp(Op* const ptrConcatOp)
{
    ETHOSN_UNUSED(ptrConcatOp);
}

void CascadingCompiler::ProcessSplitOp(Op* const ptrSplitOp)
{
    ETHOSN_UNUSED(ptrSplitOp);
}

void CascadingCompiler::ProcessSpaceToDepthOp(Op* const ptrSpaceToDepthOp)
{
    ETHOSN_UNUSED(ptrSpaceToDepthOp);
}

void CascadingCompiler::ProcessTransposeOp(Op* const ptrTransposeOp)
{
    ETHOSN_UNUSED(ptrTransposeOp);
}

}    // namespace support_library
}    // namespace ethosn
