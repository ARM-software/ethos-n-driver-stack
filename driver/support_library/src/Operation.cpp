//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Operation.hpp"

#include "Network.hpp"

namespace ethosn
{
namespace support_library
{

Operation::Operation(const detail::PosInNetwork pos,
                     uint32_t opId,
                     const std::vector<Operand*>& inputs,
                     const std::vector<TensorInfo>& outputTensorInfos)
    : m_Pos(pos)
    , m_OperationId(opId)
    , m_Inputs(inputs)
{
    m_Outputs.reserve(outputTensorInfos.size());
    uint32_t indexInOp = 0;

    for (const TensorInfo& outputInfo : outputTensorInfos)
    {
        m_Outputs.emplace_back(*this, indexInOp, outputInfo);
        ++indexInOp;
    }

    size_t i = 0;
    for (Operand* operand : inputs)
    {
        assert(operand != nullptr);
        operand->AddConsumer(*this, i);
        ++i;
    }
}

}    // namespace support_library

}    // namespace ethosn
