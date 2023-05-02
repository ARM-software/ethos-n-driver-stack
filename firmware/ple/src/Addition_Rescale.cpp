//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "../include/ethosn_ple/BlockConstants.hpp"
#include "../include/ethosn_ple/CExec.hpp"
#include "../include/ethosn_ple/Common.hpp"

namespace
{
constexpr unsigned kGroupsPerBlock   = 4U;
constexpr unsigned kGroupSizeInWords = WORDS_PER_REGISTER * PATCHES_PER_GROUP;
constexpr unsigned kBlockSizeInWords = kGroupsPerBlock * kGroupSizeInWords;

constexpr unsigned kRegScratch0 = 8;
constexpr unsigned kRegScratch1 = 10;
constexpr unsigned kRegMultTmp  = 12;
constexpr unsigned kRegZp0      = 16;
constexpr unsigned kRegZp1      = 18;
constexpr unsigned kRegMult0    = 20;
constexpr unsigned kRegMult1    = 22;

class PleState2
{
public:
    PleState2()
        : m_ActiveEvents{}
    {}

    /// Waits until a specific HW event has happened since this method was last called.
    template <Event E>
    void WaitForEvent()
    {
#pragma unroll 1
        while (!(m_ActiveEvents |= EnumBitset<Event>{ read_reg(PLE_REG(CE_RP, CE_PLE_STATUS)) })[E])
        {
            __WFE();
        }
        m_ActiveEvents[E] = false;
    }

private:
    EnumBitset<Event> m_ActiveEvents;
};

__attribute__((noinline)) void SetEncodedShift(volatile Cdp2Inst& asrSat, const unsigned shift)
{
    asrSat.SetRm(shift);
}

template <unsigned I>
void SetEncodedShift(const unsigned shift)
{
    // This modifies the CDP2 instructions stored at the address of the
    // label shift_inst_0 to execute the correct amount of right shift.
    // This is done by modifying the Rm field of the CDP2 instruction
    volatile Cdp2Inst* asrSat;
    // cppcheck-suppress uninitvar
    __ASM("ADR %[asrSat], ASRSat_32_16_%c[I]" : [asrSat] "=r"(asrSat) : [I] "n"(I));
    SetEncodedShift(*asrSat, shift);
}

template <unsigned I0, unsigned... Is>
void SetEncodedShift(std::index_sequence<Is...>, const unsigned shift)
{
    (..., SetEncodedShift<I0 + Is>(shift));
}

template <unsigned Dst, unsigned Src, unsigned ZeroPoint, unsigned Multiplier>
auto ConvertTo16bAndRescale()
{
    return std::tuple_cat(cexec::ConvertTo16b<Dst, Src>(),
                          cexec::Rescale<Dst, Dst, ZeroPoint, Multiplier, 0, kRegMultTmp, Src>());
}

template <unsigned I>
auto ProcessHalfGroup16b(const unsigned outZeroPoint)
{
    using namespace cexec;

    constexpr unsigned i0p0 = 2 * I;
    constexpr unsigned i0p1 = i0p0 + 1;

    constexpr unsigned i1p0 = i0p0 + PATCHES_PER_GROUP;
    constexpr unsigned i1p1 = i1p0 + 1;

    constexpr unsigned i0p0_16b = i0p0;
    constexpr unsigned i0p1_16b = kRegScratch1;

    constexpr unsigned i1p0_16b = kRegScratch0;
    constexpr unsigned i1p1_16b = i1p0;

    const auto rescale = std::tuple_cat(ConvertTo16bAndRescale<i0p1_16b, i0p1, kRegZp0, kRegMult0>(),
                                        ConvertTo16bAndRescale<i0p0_16b, i0p0, kRegZp0, kRegMult0>(),
                                        ConvertTo16bAndRescale<i1p0_16b, i1p0, kRegZp1, kRegMult1>(),
                                        ConvertTo16bAndRescale<i1p1_16b, i1p1, kRegZp1, kRegMult1>());

    const std::tuple add = {
        RegrepAdd16<i0p0_16b, i0p0_16b>(outZeroPoint),
        Add16<i0p0, i0p0_16b, i1p0_16b>{},

        RegrepAdd16<i0p1_16b, i0p1_16b>(outZeroPoint),
        Add16<i1p0, i0p1_16b, i1p1_16b>{},
    };

    return std::tuple_cat(rescale, add);
}

template <unsigned... Is>
auto ProcessHalfGroup16b(std::index_sequence<Is...>, const unsigned outZeroPoint)
{
    return std::tuple_cat(ProcessHalfGroup16b<Is>(outZeroPoint)...);
}

template <unsigned I>
auto Sat()
{
    using namespace cexec;

    constexpr unsigned Dst = I;
    constexpr unsigned Src = I + ((I % 2) * (PATCHES_PER_GROUP - 1));

    if constexpr (k_IsSigned)
    {
        return std::tuple{
            ASRSat_16_8<Dst, Src, 0>{},
        };
    }
    else
    {
        return std::tuple{
            SMax16<Src, Src, kRegScratch0>{},
            LSRSat_16_8<Dst, Src, 0>{},
        };
    }
}

template <unsigned... Is>
auto Sat(std::index_sequence<Is...>)
{
    return std::tuple_cat(Sat<Is>()...);
}

__attribute__((noinline)) void ProcessGroup(const unsigned dfc, const lsu::Address lsuAddr, const unsigned outZeroPoint)
{
    using namespace cexec;

    constexpr lsu::Stride offsetBetweenInputs = { .ramStride = kBlockSizeInWords - kGroupSizeInWords };

    const std::tuple load = {
        LoadInramRf<0>(dfc, lsuAddr),
        LoadInramRf<2>(dfc, lsuAddr),
        LoadInramRf<4>(dfc, lsuAddr + offsetBetweenInputs),
        LoadInramRf<6>(dfc, lsuAddr + offsetBetweenInputs),
    };
    const auto processGroup16b = ProcessHalfGroup16b(std::make_index_sequence<2>{}, outZeroPoint);
    const auto sat             = Sat(std::make_index_sequence<PATCHES_PER_GROUP>{});
    const std::tuple store     = {
        StoreRfOutram<0>(lsuAddr),
        StoreRfOutram<2>(lsuAddr),
    };

    if constexpr (k_IsSigned)
    {
        Exec(std::tuple_cat(load, processGroup16b, sat, store));
    }
    else
    {
        const std::tuple initRegScratch0ToZero = {
            Regrep16<kRegScratch0>(0U),
        };
        Exec(std::tuple_cat(load, processGroup16b, initRegScratch0ToZero, sat, store));
    }
}

__attribute__((always_inline)) void
    ProcessGroups(const unsigned dfc, const unsigned pleAddr, const unsigned numGroups, const unsigned outZeroPoint)
{
    constexpr lsu::Stride groupStride = { .ramStride = kGroupSizeInWords };

    lsu::Address lsuAddr = { .ramAddr = pleAddr };

    for (unsigned group = 0; group < numGroups; ++group)
    {
        ProcessGroup(dfc, lsuAddr, outZeroPoint);
        lsuAddr += groupStride;
    }
}

__attribute__((always_inline)) void ScheduleUdmaLoad(PleState2& pleState,
                                                     const unsigned strideBetweenInputsInWords,
                                                     const unsigned dfc,
                                                     const unsigned dfcAddr,
                                                     const unsigned pleAddr)
{
    {
        const udma::Address addr = {
            .dfcAddrWords = dfcAddr,
            .pleAddr      = pleAddr,
        };
        udma::Transfer<udma::Direction::DFC_INRAM>(dfc, addr);
    }
    {
        const udma::Address addr = {
            .dfcAddrWords = dfcAddr + strideBetweenInputsInWords,
            .pleAddr      = pleAddr + kBlockSizeInWords,
        };
        pleState.WaitForEvent<Event::UDMA_LOAD_DONE>();
        udma::Transfer<udma::Direction::DFC_INRAM>(dfc, addr);
    }
}

__attribute__((noinline)) ncu_ple_interface::PleMsg::StripeDone ProcessStripe(PleState2 pleState)
{
    unsigned numFullZ;
    unsigned numEdgeZ;
    unsigned numFullBlocks;
    unsigned numEdgeGroups;
    unsigned inDfcAddrZ;
    unsigned outDfcAddrZ;
    unsigned groupStrideInPatches;
    unsigned blockStrideInWords;
    unsigned strideBetweenInputsInWords;
    unsigned outZeroPoint;

    {
        const OperatorInfo opInfo = GetOperatorInfo<>();

        const auto& input0 = opInfo.inputs[0];
        const auto& input1 = opInfo.inputs[1];

        ve_regrep_16<kRegZp0>(static_cast<unsigned>(input0.zeroPoint));
        ve_regrep_16<kRegZp1>(static_cast<unsigned>(input1.zeroPoint));
        ve_regrep_16<kRegMult0>(static_cast<unsigned>(input0.multiplier));
        ve_regrep_16<kRegMult1>(static_cast<unsigned>(input1.multiplier));

        SetEncodedShift<0>(std::make_index_sequence<PATCHES_PER_GROUP>{}, input0.shift - 1);

        SetEncodedShift<PATCHES_PER_GROUP>(std::make_index_sequence<PATCHES_PER_GROUP>{}, input1.shift - 1);

        {
            const unsigned numZ = DivRoundUp(std::max(opInfo.sizeInElements.z, g_CeId) - g_CeId, NUM_CES);

            numFullZ = numZ / NUM_SRAMS;
            numEdgeZ = numZ % NUM_SRAMS;
        }

        {
            const unsigned numGroups = TotalSize(DivRoundUp(Xy(opInfo.sizeInElements), Xy::Dup(ELEMENTS_PER_GROUP_1D)));

            numFullBlocks = numGroups / kGroupsPerBlock;
            numEdgeGroups = numGroups % kGroupsPerBlock;
        }

        inDfcAddrZ                 = input0.dfcAddr;
        outDfcAddrZ                = opInfo.output.dfcAddr;
        groupStrideInPatches       = PATCHES_PER_GROUP * DivRoundUp(opInfo.sizeInElements.z, TOTAL_NUM_SRAMS);
        blockStrideInWords         = (WORDS_PER_REGISTER * kGroupsPerBlock) * groupStrideInPatches;
        strideBetweenInputsInWords = input1.dfcAddr - input0.dfcAddr;
        outZeroPoint               = static_cast<unsigned>(opInfo.output.zeroPoint);

        {
            const unsigned groupsPerBlock = (numFullBlocks != 0) ? kGroupsPerBlock : numEdgeGroups;

            const udma::Params udmaParams = {
                .colGrpStride        = groupStrideInPatches - PATCHES_PER_GROUP,
                .colGrpCountMinusOne = groupsPerBlock - 1U,
            };

            udma::SetLoadParams<PATCHES_PER_GROUP>(udmaParams);
        }
    }

    unsigned pleAddr = 0;

    const auto ProcessDfc = [&](const unsigned dfc) __attribute__((always_inline))
    {

        unsigned inDfcAddr = inDfcAddrZ;

        ScheduleUdmaLoad(pleState, strideBetweenInputsInWords, dfc, inDfcAddr, pleAddr);

        unsigned outDfcAddr = outDfcAddrZ;

        {
            const udma::Params udmaParams = {
                .colGrpStride        = groupStrideInPatches - PATCHES_PER_GROUP,
                .colGrpCountMinusOne = kGroupsPerBlock - 1U,
            };

            udma::SetStoreParams<PATCHES_PER_GROUP>(udmaParams);
        }

        for (unsigned block = numFullBlocks; block > 0; --block)
        {
            const unsigned nextPleAddr = pleAddr ^ (2U * kBlockSizeInWords);

            pleState.WaitForEvent<Event::UDMA_LOAD_DONE>();

            if ((block > 1U) || (numEdgeGroups > 0))
            {
                inDfcAddr += blockStrideInWords;
                ScheduleUdmaLoad(pleState, strideBetweenInputsInWords, dfc, inDfcAddr, nextPleAddr);
            }

            ProcessGroups(dfc, pleAddr, kGroupsPerBlock, outZeroPoint);

            if (block != numFullBlocks)
            {
                pleState.WaitForEvent<Event::UDMA_STORE_DONE>();
            }
            udma::Transfer<udma::Direction::OUTRAM_DFC>(dfc, {
                                                                 .dfcAddrWords = outDfcAddr,
                                                                 .pleAddr      = pleAddr,
                                                             });

            outDfcAddr += blockStrideInWords;
            pleAddr = nextPleAddr;
        }

        if (numEdgeGroups > 0)
        {
            pleState.WaitForEvent<Event::UDMA_LOAD_DONE>();

            ProcessGroups(dfc, pleAddr, numEdgeGroups, outZeroPoint);

            if (numFullBlocks != 0)
            {
                pleState.WaitForEvent<Event::UDMA_STORE_DONE>();
            }
            {
                const udma::Params udmaParams = {
                    .colGrpStride        = groupStrideInPatches - PATCHES_PER_GROUP,
                    .colGrpCountMinusOne = numEdgeGroups - 1U,
                };

                udma::SetStoreParams<PATCHES_PER_GROUP>(udmaParams);
            }
            udma::Transfer<udma::Direction::OUTRAM_DFC>(dfc, {
                                                                 .dfcAddrWords = outDfcAddr,
                                                                 .pleAddr      = pleAddr,
                                                             });
        }

        pleState.WaitForEvent<Event::UDMA_STORE_DONE>();
    };

    for (unsigned z = numFullZ; z > 0; --z)
    {
        for (unsigned dfc = 0; dfc < NUM_SRAMS; dfc += NUM_PLE_LANES)
        {
            ProcessDfc(dfc);
        }

        inDfcAddrZ += kGroupSizeInWords;
        outDfcAddrZ += kGroupSizeInWords;
    }

    for (unsigned dfc = 0; dfc < NUM_SRAMS; dfc += NUM_PLE_LANES)
    {
        if (dfc >= numEdgeZ)
        {
            break;
        }

        if ((numEdgeZ - dfc) == 1U)
        {
            SetPleLanesInUse(1U);
        }

        ProcessDfc(dfc);
    }

    return ncu_ple_interface::PleMsg::StripeDone{};
}
}    // namespace

extern "C" __attribute__((noreturn)) void main()
{
    PleState2 pleState;
    Main([&]() { pleState.WaitForEvent<Event::SETIRQ_EVENT>(); }, [&]() { return ProcessStripe(pleState); });
}
