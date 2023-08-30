//
// Copyright © 2020-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "Plan.hpp"
#include "WeightEncoder.hpp"

#include <vector>

namespace ethosn
{
namespace support_library
{

class Buffer;
class HardwareCapabilities;

struct ConversionData
{
    ConversionData() = default;
    TensorShape tensorShape;
    TensorShape stripeShape;
    bool isNhwc;
};

PassStats GetConversionStats(const ConversionData& input, const ConversionData& output, bool isDramToDram);

PleStats GetPleStats(const HardwareCapabilities& caps,
                     const std::vector<TensorShape>& inputShapes,
                     const TensorShape& outputShape,
                     const PleOperation& pleOperation,
                     uint32_t blockMultiplier,
                     uint32_t blockWidth,
                     uint32_t blockHeight);

InputStats GetInputStats(const SramBuffer& ifmBuffer,
                         const TensorShape& weightsShape,
                         utils::Optional<BufferFormat> dramBufferFormat);

OutputStats GetOutputStats(const SramBuffer& ofmSramBuffer, utils::Optional<BufferFormat> dramBufferFormat);

InputStats AccountForActivationCompression(InputStats stats, float spaceSavingRatio);

/// Increases the number of stripes in the given stats if the transfer between the two buffers provided
/// would result in the DMA having to be split into multiple chunks. This is useful as the performance estimate
/// will then take this into account, and prefer to choose strategies that don't require chunking.
StripesStats AccountForDmaChunking(StripesStats stats,
                                   const SramBuffer& sramBuffer,
                                   const DramBuffer& dramBuffer,
                                   bool dramStridingAllowed);

struct PassDesc
{
    // For MCE passes, input 0 is the IFM and input 1 is the weights.
    // For standalone PLE passes, input 1 could be a second IFM (e.g. for Addition).

    Buffer* m_Input0     = nullptr;    ///< Either an SRAM or DRAM buffer
    Buffer* m_Input0Dram = nullptr;    ///< nullptr if the input is in SRAM.
    DmaOp* m_Input0Dma   = nullptr;    ///< nullptr if the input is in SRAM.
    Buffer* m_Input0Sram = nullptr;    ///< nullptr if the input is in DRAM.

    Buffer* m_Input1     = nullptr;    ///< Either an SRAM or DRAM buffer
    Buffer* m_Input1Dram = nullptr;    ///< nullptr if the input is in SRAM.
    DmaOp* m_Input1Dma   = nullptr;    ///< nullptr if the input is in SRAM.
    Buffer* m_Input1Sram = nullptr;    ///< nullptr if the input is in DRAM.

    MceOp* m_Mce           = nullptr;
    Buffer* m_PleInputSram = nullptr;
    PleOp* m_Ple           = nullptr;

    Buffer* m_OutputSram = nullptr;    ///< nullptr if the output is in DRAM
    DmaOp* m_OutputDma   = nullptr;    ///< nullptr if the output is in SRAM.
    Buffer* m_OutputDram = nullptr;    ///< nullptr if the output is in SRAM.
    Buffer* m_Output     = nullptr;    ///< Either an SRAM or DRAM buffer
};

double CalculateMetric(const PassStats& legacyPerfData, const PassDesc& passDesc, std::string* outDebugInfo = nullptr);

}    //namespace support_library
}    //namespace ethosn
