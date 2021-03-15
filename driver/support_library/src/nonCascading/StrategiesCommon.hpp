//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Pass.hpp"
#include "SramAllocator.hpp"

namespace ethosn
{
namespace support_library
{

namespace
{

struct AllocationResult
{
    bool m_Success;
    uint32_t m_InputOffset;
    uint32_t m_WeightOffset;
    uint32_t m_OutputOffset;
    uint32_t m_PleOffset;
};

AllocationResult FitsInSram(SramAllocator& sramAllocator,
                            const HardwareCapabilities& capabilities,
                            const uint32_t input,
                            const uint32_t weight,
                            const uint32_t output,
                            std::pair<const bool, const uint32_t> inputStaticAndOffset)
{
    AllocationResult res;
    res.m_Success          = true;
    auto pleAllocateResult = sramAllocator.Allocate(capabilities.GetMaxPleSize(), AllocationPreference::Start, "ple");
    res.m_Success &= pleAllocateResult.first;
    res.m_PleOffset = pleAllocateResult.second;

    if (inputStaticAndOffset.first)
    {
        res.m_InputOffset = inputStaticAndOffset.second;
    }
    else
    {
        assert(input > 0);
        auto inputAllocateResult =
            sramAllocator.Allocate(input / capabilities.GetNumberOfSrams(), AllocationPreference::Start, "input");
        res.m_Success &= inputAllocateResult.first;
        res.m_InputOffset = inputAllocateResult.second;
    }

    // Try to allocate output and input tiles in opposite ends of SRAM, so we can overlap loading/saving
    AllocationPreference outputAllocationPreference;
    AllocationPreference weightAllocationPreference;
    if (res.m_InputOffset <= (capabilities.GetTotalSramSize() / capabilities.GetNumberOfSrams()) / 2)
    {
        outputAllocationPreference = AllocationPreference::End;
        weightAllocationPreference = AllocationPreference::Start;
    }
    else
    {
        outputAllocationPreference = AllocationPreference::Start;
        weightAllocationPreference = AllocationPreference::End;
    }

    // There are passes without weights but still need to decide on strategies i.e. PlePasses
    // We don't allocate anything if there are no weights.
    assert(weight > 0);
    auto weightAllocateResult =
        sramAllocator.Allocate(weight / capabilities.GetNumberOfSrams(), weightAllocationPreference, "weights");
    res.m_Success &= weightAllocateResult.first;
    res.m_WeightOffset = weightAllocateResult.second;

    assert(output > 0);
    auto outputAllocateResult =
        sramAllocator.Allocate(output / capabilities.GetNumberOfSrams(), outputAllocationPreference, "outputs");
    res.m_Success &= outputAllocateResult.first;
    res.m_OutputOffset = outputAllocateResult.second;

    return res;
}

void FillStrategyConfigOffsets(const AllocationResult& allocationResults, StrategyConfig& outStrategyConfig)
{
    outStrategyConfig.pleAllocation.offset     = allocationResults.m_PleOffset;
    outStrategyConfig.inputAllocation.offset   = allocationResults.m_InputOffset;
    outStrategyConfig.weightsAllocation.offset = allocationResults.m_WeightOffset;
    outStrategyConfig.outputAllocation.offset  = allocationResults.m_OutputOffset;
}

// Helper function to account for the fact that if the output stripe in a dimension is the entire tensor
// we need to use the full input tensor in that dimension
uint32_t AccountForFullDimension(const uint32_t outputTensorDim,
                                 const uint32_t inputTensorDim,
                                 const uint32_t outputStripeDim,
                                 const utils::Fraction multiplier)
{
    if (outputStripeDim >= outputTensorDim)
    {
        return inputTensorDim;
    }
    else
    {
        return outputStripeDim / multiplier;
    }
}

}    // namespace

}    // namespace support_library
}    // namespace ethosn
