//
// Copyright © 2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_ple/Common.hpp"
#include "../include/ethosn_ple/DfcSramTraversal.hpp"
#include "../include/ethosn_ple/SignedSupport.hpp"

// Performs elementwise multiplication between two input tensors
// as a standalone PLE kernel.
// This kernel handles the inputs and outputs all having different quantization parameters
// The quantization multiplication can be model as
// out = (s_i0 * s_i1 * 1/s_out) * (i0 - z_i0) * (i1 - z_i1) + z_out
// Where:
//  s_i and z_i is the quantization scale and zero point of an input
//  s_out z_out is the quantization scale and zero point of the output
//  i0 and i1 are the quantized input
//  out is the quantized output

namespace
{

template <unsigned Dst, unsigned Src1, unsigned Src2, unsigned int post_cc = 0>
void Mul8()
{
    if (k_IsSigned)
    {
        ve_smul_8<Dst, Src1, Src2, post_cc>();
    }
    else
    {
        ve_umul_8<Dst, Src1, Src2, post_cc>();
    }
}

ncu_ple_interface::PleMsg::StripeDone ProcessStripe(EnumBitset<Event>& activeEvents)
{
    // Read stripe parameters from scratch registers
    Xyz outputSizeInElements = {
        static_cast<uint32_t>(*reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH0)) & 0x0000ffff),
        static_cast<uint32_t>(*reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH0)) & 0xffff0000) >>
            16,
        static_cast<uint32_t>(*reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH1)) & 0x0000ffff),
    };
    Xyz inputSizeInElements = outputSizeInElements;
    // Number of channels to be processed by this PLE, with Z including all SRAMS and lanes.
    uint32_t numChannels = DivRoundUp(std::max(outputSizeInElements.z, g_CeId) - g_CeId, NUM_CES);
    Xy inputSizeInGroups = DivRoundUp(Xy(inputSizeInElements), Xy::Dup(ELEMENTS_PER_GROUP_1D));

    uint16_t overallMultiplier =
        static_cast<uint16_t>(*reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH3)) & 0x0000ffff);
    uint16_t overallShift = static_cast<uint16_t>(
        (*reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH3)) & 0xffff0000) >> 16);
    int16_t input0Zeropoint =
        static_cast<int16_t>(*reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH4)) & 0x0000ffff);
    int16_t input1Zeropoint = static_cast<int16_t>(
        (*reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH4)) & 0xffff0000) >> 16);
    int16_t outputZeropoint =
        static_cast<int16_t>(*reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH2)) & 0x0000ffff);
    // Update the 4 x ASR instructions with the shift value which we only know at runtime
    volatile Cdp2Inst* asrSat;
    // cppcheck-suppress uninitvar
    __ASM("ADR %[asrSat], INSTRUCTION_FOR_MODIFICATION_0" : [asrSat] "=r"(asrSat));
    asrSat->SetRm(overallShift);
    __ASM("ADR %[asrSat], INSTRUCTION_FOR_MODIFICATION_1" : [asrSat] "=r"(asrSat));
    asrSat->SetRm(overallShift);
    __ASM("ADR %[asrSat], INSTRUCTION_FOR_MODIFICATION_2" : [asrSat] "=r"(asrSat));
    asrSat->SetRm(overallShift);
    __ASM("ADR %[asrSat], INSTRUCTION_FOR_MODIFICATION_3" : [asrSat] "=r"(asrSat));
    asrSat->SetRm(overallShift);

    uint32_t inDfcAddrBase0 = *reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH5));
    uint32_t inDfcAddrBase1 = *reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH6));
    uint32_t outDfcAddrBase = *reinterpret_cast<volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH7));

    udma::UdmaLoader udmaLoader(activeEvents);
    udma::UdmaStorer udmaStorer(activeEvents);

    // Set UDMA parameters which we never need to change. We load/store a single group (2x2 patches) at a time.
    udma::Params udmaParams;
    udmaParams.colGrpCountMinusOne = 0U;
    udmaParams.rowGrpCountMinusOne = 0U;
    udmaParams.colGrpStride        = 0U;    // Irrelevant as we're only copying one group at a time
    udmaParams.rowGrpStride        = 0U;    // Irrelevant as we're only copying one group at a time

    SetStoreParams<PATCHES_PER_GROUP>(udmaParams);
    SetLoadParams<PATCHES_PER_GROUP>(udmaParams);

    // The distance between spatially adjacent groups in bytes
    Xyz outputGroupStrideBytes =
        Xyz(dfcsram::GetNhwcbGroupStride(outputSizeInElements) * ELEMENTS_PER_PATCH, ELEMENTS_PER_GROUP);
    Xyz inputGroupStrideBytes =
        Xyz(dfcsram::GetNhwcbGroupStride(inputSizeInElements) * ELEMENTS_PER_PATCH, ELEMENTS_PER_GROUP);

    // Process each SRAM in turnasdasd
    // Each PLE lane automatically processes a separate SRAM. We only need to program the first lane and
    // the other follows, so we skip some SRAMs
    for (unsigned dfc = 0; dfc < NUM_SRAMS; dfc += NUM_PLE_LANES)
    {
        // Default to both lanes being used
        SetPleLanesInUse(NUM_PLE_LANES);

        // Process each depth for this SRAM in turn
        unsigned depthForThisSram = DivRoundUp(std::max(numChannels, dfc) - dfc, NUM_SRAMS);
        unsigned depthForNextSram = DivRoundUp(std::max(numChannels, (dfc + 1)) - (dfc + 1), NUM_SRAMS);
        for (unsigned z = 0; z < depthForThisSram; ++z)
        {
            // If there is a second lane, but it isn't needed because this is the last pair of channels but there is an odd number, disable it
            if (z >= depthForNextSram)
            {
                SetPleLanesInUse(1U);
            }

            // Loop over each row
            for (unsigned y = 0; y < inputSizeInGroups.y; ++y)
            {
                uint32_t inDfcAddr0 = inDfcAddrBase0 + z * inputGroupStrideBytes.z + y * inputGroupStrideBytes.y;
                uint32_t inDfcAddr1 = inDfcAddrBase1 + z * inputGroupStrideBytes.z + y * inputGroupStrideBytes.y;
                uint32_t outDfcAddr = outDfcAddrBase + z * outputGroupStrideBytes.z + y * outputGroupStrideBytes.y;

                // Loop over each group in the row
                for (unsigned x = 0; x < inputSizeInGroups.x; ++x)
                {
                    // Load one group of input 0 from regular SRAM into PLE input SRAM (at address 0)
                    udma::Address udmaInAddr;
                    udmaInAddr.dfcAddrWords = inDfcAddr0 / 4;
                    udmaInAddr.pleAddr      = 0;
                    udmaLoader.Load(dfc, udmaInAddr);
                    udmaLoader.WaitForUdma();

                    // Load input_0 into VE registers 1, 3, 5, 7
                    // We leave a space between them so we can sign extend the input
                    lsu::LoadHalfInramRf<0>(dfc, { 0U * 4U, 1U * WORDS_PER_REGISTER });
                    lsu::LoadHalfInramRf<0>(dfc, { 1U * 4U, 3U * WORDS_PER_REGISTER });
                    lsu::LoadHalfInramRf<0>(dfc, { 2U * 4U, 5U * WORDS_PER_REGISTER });
                    lsu::LoadHalfInramRf<0>(dfc, { 3U * 4U, 7U * WORDS_PER_REGISTER });

                    // Sign extend the input by shifting down 8 bits
                    SR16<0, 0, 8>();
                    SR16<2, 2, 8>();
                    SR16<4, 4, 8>();
                    SR16<6, 6, 8>();

                    // Load one group of input 1 from regular SRAM into PLE input SRAM (at address 0)
                    udmaInAddr.dfcAddrWords = inDfcAddr1 / 4;
                    udmaInAddr.pleAddr      = 0;
                    udmaLoader.Load(dfc, udmaInAddr);
                    udmaLoader.WaitForUdma();

                    // Load input_1 into VE registers 9, 11, 13, 15
                    // We leave a space between them so we can sign extend the input
                    lsu::LoadHalfInramRf<0>(dfc, { 0U * 4U, 9U * WORDS_PER_REGISTER });
                    lsu::LoadHalfInramRf<0>(dfc, { 1U * 4U, 11U * WORDS_PER_REGISTER });
                    lsu::LoadHalfInramRf<0>(dfc, { 2U * 4U, 13U * WORDS_PER_REGISTER });
                    lsu::LoadHalfInramRf<0>(dfc, { 3U * 4U, 15U * WORDS_PER_REGISTER });

                    // Sign extend the input by shifting down 8 bits
                    SR16<8, 8, 8>();
                    SR16<10, 10, 8>();
                    SR16<12, 12, 8>();
                    SR16<14, 14, 8>();

                    // out = (s_i0 * s_i1 * 1/s_out) * (i0 - z_i0) * (i1 - z_i1) + z_out
                    // Subtract the zero points (i0 - z_i0) and (i1 - z_i1)
                    // Register 16 and 18 holds the zero points
                    ve_regrep_16<16>(static_cast<uint32_t>(input0Zeropoint));
                    ve_regrep_16<18>(static_cast<uint32_t>(input1Zeropoint));

                    ve_sub_16<0, 0, 16>();
                    ve_sub_16<2, 2, 16>();
                    ve_sub_16<4, 4, 16>();
                    ve_sub_16<6, 6, 16>();

                    ve_sub_16<8, 8, 18>();
                    ve_sub_16<10, 10, 18>();
                    ve_sub_16<12, 12, 18>();
                    ve_sub_16<14, 14, 18>();

                    // Multiply (i0 - z_i0) * (i1 - z_i1)
                    // The input is 9 bits (8 bits originally + 1 for the added zero point)
                    // We only extract the bottom 16 bits of the multiplication
                    // It can technically have 18 bits of precision (9+9)
                    // There may be precision issues for inputs at the maximum range + a maximum zero point.
                    // e.g. zero point of 255 (if unsigned) with inputs of 255 * 255, means a real value of
                    // 510*510 = 260100. This requires 18 bits of precision (262143).
                    nop<2>();
                    ve_smul_16<8, 0, 8>();
                    nop<2>();
                    ve_smul_16<10, 2, 10>();
                    nop<2>();
                    ve_smul_16<12, 4, 12>();
                    nop<2>();
                    ve_smul_16<14, 6, 14>();
                    nop<2>();

                    // Scale to the output quantization space
                    // First half of (s_i0 * s_i1) / s_out is the multipler
                    // Register 18 holds the multiplier#
                    // 0-4, 4-7, 8-11, 12-15 hold the 32 bit results
                    ve_regrep_16<18>(static_cast<uint32_t>(overallMultiplier));
                    nop<2>();
                    ve_umull_16<0, 8, 18>();
                    nop<3>();
                    ve_umull_16<4, 10, 18>();
                    nop<3>();
                    ve_umull_16<8, 12, 18>();
                    nop<3>();
                    ve_umull_16<12, 14, 18>();
                    nop<2>();

                    // Shift right and saturate to 16-bit is the second half of the scale.
                    // The shift amount here is set to zero, but is replaced at runtime by self-modifying code above.
                    // The result is a 16 bit number held in:
                    // 0-1, 2-3, 4-5, 6-7.
                    __ASM volatile("INSTRUCTION_FOR_MODIFICATION_0:");
                    ve_lsrsat_32_16<0, 0, 0>();
                    nop<1>();
                    __ASM volatile("INSTRUCTION_FOR_MODIFICATION_1:");
                    ve_lsrsat_32_16<2, 4, 0>();
                    nop<1>();
                    __ASM volatile("INSTRUCTION_FOR_MODIFICATION_2:");
                    ve_lsrsat_32_16<4, 8, 0>();
                    nop<1>();
                    __ASM volatile("INSTRUCTION_FOR_MODIFICATION_3:");
                    ve_lsrsat_32_16<6, 12, 0>();
                    nop<3>();

                    // Add the output zero point
                    // (z_out)
                    // Register 20 holds the output zero point
                    // The result is a 16 bit number held in:
                    // 0-1, 2-3, 4-5, 6-7.
                    ve_regrep_16<20>(static_cast<uint32_t>(outputZeropoint));
                    nop<2>();
                    ve_add_16<0, 0, 20>();
                    nop<2>();
                    ve_add_16<2, 2, 20>();
                    nop<2>();
                    ve_add_16<4, 4, 20>();
                    nop<2>();
                    ve_add_16<6, 6, 20>();
                    nop<1>();

                    // We only need to store the 8 bit values
                    // Move register 0, 2, 4, 6 into PLE output SRAM.
                    lsu::StoreHalfRfOutram<0>({ 0U, 0U * WORDS_PER_REGISTER });
                    lsu::StoreHalfRfOutram<0>({ 4U, 2U * WORDS_PER_REGISTER });
                    lsu::StoreHalfRfOutram<0>({ 8U, 4U * WORDS_PER_REGISTER });
                    lsu::StoreHalfRfOutram<0>({ 12U, 6U * WORDS_PER_REGISTER });

                    // Store one group from PLE output SRAM to regular SRAM
                    udma::Address udmaOutAddr;
                    udmaOutAddr.dfcAddrWords = outDfcAddr / 4;
                    udmaOutAddr.pleAddr      = 0;
                    udmaStorer.Store(dfc, udmaOutAddr);
                    udmaStorer.WaitForUdma();
                    outDfcAddr += outputGroupStrideBytes.x;

                    // Move to next group in regular SRAM
                    inDfcAddr0 += inputGroupStrideBytes.x;
                    inDfcAddr1 += inputGroupStrideBytes.x;
                }
            }
        }
    }

    return ncu_ple_interface::PleMsg::StripeDone{};
}

}    // namespace

extern "C" __attribute__((noreturn)) void main()
{
    EnumBitset<Event> activeEvents;
    Main([&]() { WaitForEvent<Event::SETIRQ_EVENT>(activeEvents); }, [&]() { return ProcessStripe(activeEvents); });
}
