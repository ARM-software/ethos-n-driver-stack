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

constexpr unsigned kRegZero      = 16;
constexpr unsigned kRegZeroPoint = 18;

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

template <unsigned I>
auto ConvertTo16b()
{
    constexpr unsigned Dst = ((I % 2) == 0) ? I + (2 * PATCHES_PER_GROUP) : I - 1;
    return cexec::ConvertTo16b<Dst, I>();
}

template <unsigned... Is>
auto ConvertTo16b(std::index_sequence<Is...>)
{
    return std::tuple_cat(ConvertTo16b<Is>()...);
}

template <unsigned I>
auto Add()
{
    using namespace cexec;

    constexpr unsigned Input0 = ((I % 2) == 0) ? I + (2 * PATCHES_PER_GROUP) : I - 1;
    constexpr unsigned Input1 = Input0 + PATCHES_PER_GROUP;

    return std::tuple{
        Sub16<Input0, Input0, kRegZeroPoint>(),
        Add16<Input0, Input0, Input1>{},
    };
}

template <unsigned... Is>
auto Add(std::index_sequence<Is...>)
{
    return std::tuple_cat(Add<Is>()...);
}

template <unsigned I>
auto Sat()
{
    using namespace cexec;

    constexpr unsigned Src = ((I % 2) == 0) ? I + (2 * PATCHES_PER_GROUP) : I - 1;

    if constexpr (k_IsSigned)
    {
        return std::tuple{
            ASRSat_16_8<I, Src, 0>{},
        };
    }
    else
    {
        return std::tuple{
            SMax16<Src, Src, kRegZero>{},
            LSRSat_16_8<I, Src, 0>{},
        };
    }
}

template <unsigned... Is>
auto Sat(std::index_sequence<Is...>)
{
    return std::tuple_cat(Sat<PATCHES_PER_GROUP - 1 - Is>()...);
}

__attribute__((noinline)) void ProcessGroup(const unsigned dfc, const lsu::Address lsuAddr)
{
    using namespace cexec;

    constexpr lsu::Stride offsetBetweenInputs = { .ramStride = kBlockSizeInWords - kGroupSizeInWords };

    const std::tuple load = {
        LoadInramRf<0>(dfc, lsuAddr),
        LoadInramRf<2>(dfc, lsuAddr),
        LoadInramRf<4>(dfc, lsuAddr + offsetBetweenInputs),
        LoadInramRf<6>(dfc, lsuAddr + offsetBetweenInputs),
    };
    const auto convertTo16b = ConvertTo16b(std::make_index_sequence<2 * PATCHES_PER_GROUP>{});
    const auto add          = Add(std::make_index_sequence<PATCHES_PER_GROUP>{});
    const auto sat          = Sat(std::make_index_sequence<PATCHES_PER_GROUP>{});
    const std::tuple store  = {
        StoreRfOutram<0>(lsuAddr),
        StoreRfOutram<2>(lsuAddr),
    };

    Exec(std::tuple_cat(load, convertTo16b, add, sat, store));
}

__attribute__((always_inline)) void ProcessGroups(const unsigned dfc, const unsigned pleAddr, const unsigned numGroups)
{
    constexpr lsu::Stride groupStride = { .ramStride = kGroupSizeInWords };

    lsu::Address lsuAddr = { .ramAddr = pleAddr };

    for (unsigned group = 0; group < numGroups; ++group)
    {
        ProcessGroup(dfc, lsuAddr);
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

    {
        const OperatorInfo opInfo = GetOperatorInfo<>();

        ve_regrep_16<kRegZeroPoint>(static_cast<uint32_t>(opInfo.output.zeroPoint));

        const auto& input0 = opInfo.inputs[0];
        const auto& input1 = opInfo.inputs[1];

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

            ProcessGroups(dfc, pleAddr, kGroupsPerBlock);

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

            ProcessGroups(dfc, pleAddr, numEdgeGroups);

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
    if constexpr (!k_IsSigned)
    {
        ve_regrep_16<kRegZero>(0U);
    }

    PleState2 pleState;
    Main([&]() { pleState.WaitForEvent<Event::SETIRQ_EVENT>(); }, [&]() { return ProcessStripe(pleState); });
}
