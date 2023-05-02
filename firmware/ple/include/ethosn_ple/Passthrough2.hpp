//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../include/ethosn_ple/CExec.hpp"
#include "../include/ethosn_ple/Common.hpp"

#include <cstdint>

namespace
{
constexpr Xy kElementsPerWord = { ELEMENTS_PER_REGISTER / WORDS_PER_REGISTER, 1 };

struct XyInWpg : Xy
{
    constexpr XyInWpg(const Xy xy = {})
        : Xy{ xy }
    {}

    constexpr XyInWpg(const unsigned x, const unsigned y)
        : Xy{ x, y }
    {}

    constexpr Xy InWords() const
    {
        return static_cast<const Xy&>(*this) / kElementsPerWord;
    }

    constexpr Xy InPatches() const
    {
        return static_cast<const Xy&>(*this) / ELEMENTS_PER_PATCH_1D;
    }

    constexpr Xy InGroups() const
    {
        return static_cast<const Xy&>(*this) / Xy{
            std::min(x, ELEMENTS_PER_GROUP_1D),
            std::min(y, ELEMENTS_PER_GROUP_1D),
        };
    }

    friend constexpr bool operator==(const XyInWpg&, const XyInWpg&) noexcept = default;
};

struct PassthroughCfg
{
    XyInWpg mceBlockSize = Xy{ BLOCK_WIDTH_IN_ELEMENTS, BLOCK_HEIGHT_IN_ELEMENTS };
    XyInWpg inpBlockSize = Xy{ BLOCK_MULTIPLIER, 1 } * mceBlockSize;
    XyInWpg outBlockSize = inpBlockSize;
    XyInWpg inpGroupSize = Xy{
        std::min(inpBlockSize.x, ELEMENTS_PER_GROUP_1D),
        std::min(inpBlockSize.y, ELEMENTS_PER_GROUP_1D),
    };
    XyInWpg outGroupSize = Xy{
        std::min(outBlockSize.x, ELEMENTS_PER_GROUP_1D),
        std::min(outBlockSize.y, ELEMENTS_PER_GROUP_1D),
    };
};

struct PersistentState
{
    uint32_t inramAddrBlock;
    uint8_t numBlocksProcessed;
};

struct PassthroughState
{
    Xyz patchesInOutput;

    Xy outDfcGroupStride;

    udma::Params udmaParamsCentralBlock;
    udma::Params udmaParamsEdgeBlockX;
    udma::Params udmaParamsEdgeBlockY;
    udma::Params udmaParamsEdgeBlockXY;

    uint32_t inramAddrGroupY;
    uint32_t inramAddrGroup;

    uint32_t outramAddrBlock;
    uint32_t outramAddrGroup;

    uint32_t outDfcAddrZ;
    uint32_t outDfcAddrY;
    uint32_t outDfcAddrBlock;

    Xyz blockCountdown;
    Xy groupCountdown;

    uint32_t og;

#if (NUM_SRAMS != NUM_MCEIF)
    unsigned numActiveOgs;
#else
    static constexpr unsigned numActiveOgs = NUM_MCEIF;
#endif
};

template <typename T>
concept Operation = requires(T op, const StripeInfo info, PassthroughState ctx)
{
    op.Init(info);
    op.template ProcessGroup<Xy{ 2, 2 }>(ctx);
    op.template ProcessGroup<Xy{ 1, 2 }>(ctx);
    op.template ProcessGroup<Xy{ 2, 1 }>(ctx);
    op.template ProcessGroup<Xy{ 1, 1 }>(ctx);
};

template <Operation Op, PassthroughCfg cfg = PassthroughCfg{}>
class Passthrough
{
    static_assert((cfg.inpBlockSize.x % cfg.mceBlockSize.x) == 0);
    static_assert(cfg.inpBlockSize.y == cfg.mceBlockSize.y);

    static_assert((cfg.mceBlockSize % kElementsPerWord) == XyInWpg{});
    static_assert((cfg.inpBlockSize % kElementsPerWord) == XyInWpg{});
    static_assert((cfg.outBlockSize % kElementsPerWord) == XyInWpg{});
    static_assert((cfg.inpGroupSize % kElementsPerWord) == XyInWpg{});
    static_assert((cfg.outGroupSize % kElementsPerWord) == XyInWpg{});

    static_assert((cfg.mceBlockSize % ELEMENTS_PER_PATCH_1D) == XyInWpg{});
    static_assert((cfg.inpBlockSize % ELEMENTS_PER_PATCH_1D) == XyInWpg{});
    static_assert((cfg.outBlockSize % ELEMENTS_PER_PATCH_1D) == XyInWpg{});
    static_assert((cfg.inpGroupSize % ELEMENTS_PER_PATCH_1D) == XyInWpg{});
    static_assert((cfg.outGroupSize % ELEMENTS_PER_PATCH_1D) == XyInWpg{});

    static_assert((cfg.mceBlockSize % cfg.inpGroupSize) == XyInWpg{});
    static_assert((cfg.inpBlockSize % cfg.inpGroupSize) == XyInWpg{});
    static_assert((cfg.outBlockSize % cfg.outGroupSize) == XyInWpg{});

public:
    Passthrough(const PersistentState& persistent)
        : m_Persistent(persistent)
        , m_Ctx()
    {}

    PersistentState GetPersistent() const
    {
        return m_Persistent;
    }

    static __inline_always __attribute__((noreturn)) void Main()
    {
        PersistentState persistent = { 0 };

        ::Main([&]() { WaitForEvent<Event::SETIRQ_EVENT>(); },
               [&]() {
                   Passthrough p(persistent);
                   ncu_ple_interface::PleMsg::StripeDone result = p.ProcessStripe();
                   persistent                                   = p.GetPersistent();
                   return result;
               });
    }

private:
    struct Ctx : public PassthroughState
    {
        Op operation;
    };

    __inline_always udma::Params UdmaParamsForBlock(const Xy blockSize) const
    {
        const Xy blockSizeInGroups = DivRoundUp(blockSize, cfg.outGroupSize);

        return udma::Params{
            .colGrpStride = (m_Ctx.outDfcGroupStride.x - TotalSize(cfg.outGroupSize.InWords())) / WORDS_PER_REGISTER,
            .rowGrpStride =
                (m_Ctx.outDfcGroupStride.y - (blockSizeInGroups.x * m_Ctx.outDfcGroupStride.x)) / WORDS_PER_REGISTER,
            .colGrpCountMinusOne = blockSizeInGroups.x - 1,
            .rowGrpCountMinusOne = blockSizeInGroups.y - 1,
        };
    }

    template <Event E>
    static __inline_always void WaitForEvent()
    {
#pragma unroll 1
        while (!EnumBitset<Event>(read_reg(CE_PLE_STATUS))[E])
        {
            __WFE();
        }
    }

    __inline_always void WaitForBlock(const uint32_t patchesInOutputBlockX)
    {
        if constexpr (cfg.inpBlockSize.x == cfg.mceBlockSize.x)
        {
#pragma unroll 1
            while (read_reg(CE_PLE_COUNTERS) == m_Persistent.numBlocksProcessed)
            {
                __WFE();
            }
            ++m_Persistent.numBlocksProcessed;
        }
        else
        {
            const uint32_t mceBlocksWait = DivRoundUp(patchesInOutputBlockX, cfg.mceBlockSize.InPatches().x);
#pragma unroll 1
            while (static_cast<uint8_t>(read_reg(CE_PLE_COUNTERS) - m_Persistent.numBlocksProcessed) < mceBlocksWait)
            {
                __WFE();
            }
            m_Persistent.numBlocksProcessed += mceBlocksWait;
        }
    }

    __inline_always void InitStripe()
    {
        StripeInfo iface = ReadStripeInfo();

        m_Ctx.operation.Init(iface);

#if (NUM_SRAMS != NUM_MCEIF)
        const bool isDepthwise = iface.mceOp == MceOp::DEPTHWISE_CONVOLUTION;
        m_Ctx.numActiveOgs     = isDepthwise ? NUM_SRAMS : NUM_MCEIF;
#endif

        m_Ctx.outramAddrBlock = 0;

        m_Ctx.patchesInOutput.x = DivRoundUp(iface.stripeWidth, ELEMENTS_PER_PATCH_1D);
        m_Ctx.patchesInOutput.y = DivRoundUp(iface.stripeHeight, ELEMENTS_PER_PATCH_1D);
        m_Ctx.patchesInOutput.z = DivRoundUp(std::max<uint32_t>(iface.stripeDepth, g_CeId) - g_CeId, NUM_CES);

        m_Ctx.outDfcGroupStride.x =
            TotalSize(cfg.outGroupSize.InWords()) * DivRoundUp(iface.stripeDepth, TOTAL_NUM_SRAMS);
        m_Ctx.outDfcGroupStride.y =
            DivRoundUp(m_Ctx.patchesInOutput.x, cfg.outGroupSize.InPatches().x) * m_Ctx.outDfcGroupStride.x;

        {
            const Xy edgBlockSize = {
                iface.stripeWidth % cfg.outBlockSize.x,
                iface.stripeHeight % cfg.outBlockSize.y,
            };

            m_Ctx.udmaParamsCentralBlock = UdmaParamsForBlock(cfg.outBlockSize);
            m_Ctx.udmaParamsEdgeBlockX   = UdmaParamsForBlock({ edgBlockSize.x, cfg.outBlockSize.y });
            m_Ctx.udmaParamsEdgeBlockY   = UdmaParamsForBlock({ cfg.outBlockSize.x, edgBlockSize.y });
            m_Ctx.udmaParamsEdgeBlockXY  = UdmaParamsForBlock(edgBlockSize);
        }

        m_Ctx.outDfcAddrZ = WORDS_PER_REGISTER * iface.output.dfcAddr;

        // Initial dummy transfer to generate initial udma load event
        udma::SetStoreParams<1>({});
        udma::Transfer<udma::Direction::OUTRAM_DFC>(0, { m_Ctx.outDfcAddrZ, 0 });
    }

    __inline_always void SetUdmaParams() const
    {
        const bool isEdgeOutputBlockX = m_Ctx.blockCountdown.x == 0;
        const bool isEdgeOutputBlockY = m_Ctx.blockCountdown.y == 0;

        if (isEdgeOutputBlockX && isEdgeOutputBlockY)
        {
            udma::SetStoreParams<TotalSize(cfg.outGroupSize.InPatches())>(m_Ctx.udmaParamsEdgeBlockXY);
        }
        else if (isEdgeOutputBlockX)
        {
            udma::SetStoreParams<TotalSize(cfg.outGroupSize.InPatches())>(m_Ctx.udmaParamsEdgeBlockX);
        }
        else if (isEdgeOutputBlockY)
        {
            udma::SetStoreParams<TotalSize(cfg.outGroupSize.InPatches())>(m_Ctx.udmaParamsEdgeBlockY);
        }
        else
        {
            udma::SetStoreParams<TotalSize(cfg.outGroupSize.InPatches())>(m_Ctx.udmaParamsCentralBlock);
        }
    }

    template <Xy patchesInGroup>
    __inline_always void ProcessGroup()
    {
        m_Ctx.operation.template ProcessGroup<patchesInGroup>(m_Ctx);

        m_Ctx.inramAddrGroup += TotalSize(cfg.inpGroupSize.InWords());
        m_Ctx.outramAddrGroup += TotalSize(cfg.outGroupSize.InWords());
    }

    template <uint32_t PatchesInGroupY>
    __inline_always void ProcessGroupRow(const uint32_t patchesInOutputBlockX)
    {
        m_Ctx.inramAddrGroup = m_Ctx.inramAddrGroupY;
        m_Ctx.inramAddrGroupY += TotalSize(cfg.inpGroupSize.InWords()) * cfg.inpBlockSize.InGroups().x;

        for (m_Ctx.groupCountdown.x = patchesInOutputBlockX / cfg.outGroupSize.InPatches().x;
             m_Ctx.groupCountdown.x != 0; --m_Ctx.groupCountdown.x)
        {
            ProcessGroup<Xy{ cfg.inpGroupSize.InPatches().x, PatchesInGroupY }>();
        }

        if ((patchesInOutputBlockX % cfg.outGroupSize.InPatches().x) != 0)
        {
            ProcessGroup<Xy{ 1, PatchesInGroupY }>();
        }
    }

    __inline_always void ProcessBlock(const Xyz patchesInOutputBlock)
    {
        if (patchesInOutputBlock.x == 0)
        {
            return;
        }

        WaitForBlock(patchesInOutputBlock.x);

        for (m_Ctx.og = 0; m_Ctx.og < patchesInOutputBlock.z; m_Ctx.og += NUM_PLE_LANES)
        {
            m_Ctx.inramAddrGroupY = m_Persistent.inramAddrBlock;
            m_Ctx.outramAddrGroup = m_Ctx.outramAddrBlock;

            for (m_Ctx.groupCountdown.y = patchesInOutputBlock.y / cfg.outGroupSize.InPatches().y;
                 m_Ctx.groupCountdown.y != 0; --m_Ctx.groupCountdown.y)
            {
                ProcessGroupRow<cfg.inpGroupSize.InPatches().y>(patchesInOutputBlock.x);
            }

            if ((patchesInOutputBlock.y % cfg.outGroupSize.InPatches().y) != 0)
            {
                ProcessGroupRow<1>(patchesInOutputBlock.x);
            }

            WaitForEvent<Event::UDMA_STORE_DONE>();

            SetUdmaParams();
            const udma::Address udmaAddr = {
                m_Ctx.outDfcAddrBlock + (TotalSize(cfg.outGroupSize.InWords()) * (m_Ctx.og / NUM_SRAMS)),
                m_Ctx.outramAddrBlock,
            };
            udma::Transfer<udma::Direction::OUTRAM_DFC>(m_Ctx.og % NUM_SRAMS, udmaAddr);

            m_Ctx.outramAddrBlock ^= TotalSize(cfg.outBlockSize.InWords());
        }

        const uint32_t mceBlocksConsumed = (cfg.inpBlockSize.x == cfg.mceBlockSize.x)
                                               ? 1
                                               : DivRoundUp(patchesInOutputBlock.x, cfg.mceBlockSize.InPatches().x);

        SignalBufferFreed(mceBlocksConsumed);

        m_Persistent.inramAddrBlock += TotalSize(cfg.mceBlockSize.InWords()) * mceBlocksConsumed;
        m_Ctx.outDfcAddrBlock += cfg.outBlockSize.InGroups().x * m_Ctx.outDfcGroupStride.x;
    }

    __inline_always void ProcessBlockRow(const uint32_t patchesInOutputBlockY, const uint32_t patchesInOutputBlockZ)
    {
        if (patchesInOutputBlockY == 0)
        {
            return;
        }

        m_Ctx.outDfcAddrBlock = m_Ctx.outDfcAddrY;
        m_Ctx.outDfcAddrY += cfg.outBlockSize.InGroups().y * m_Ctx.outDfcGroupStride.y;

#pragma unroll 1
        for (m_Ctx.blockCountdown.x = m_Ctx.patchesInOutput.x / cfg.outBlockSize.InPatches().x;
             m_Ctx.blockCountdown.x != 0; --m_Ctx.blockCountdown.x)
        {
            ProcessBlock({ cfg.outBlockSize.InPatches().x, patchesInOutputBlockY, patchesInOutputBlockZ });
        }

        ProcessBlock(
            { m_Ctx.patchesInOutput.x % cfg.outBlockSize.InPatches().x, patchesInOutputBlockY, patchesInOutputBlockZ });
    }

    __inline_always void ProcessXyPlane(const uint32_t patchesInOutputBlockZ)
    {
        if (patchesInOutputBlockZ == 1)
        {
            SetPleLanesInUse(1);
        }

        m_Ctx.outDfcAddrY = m_Ctx.outDfcAddrZ;
        m_Ctx.outDfcAddrZ += TotalSize(cfg.outGroupSize.InWords()) * (m_Ctx.numActiveOgs / NUM_SRAMS);

#pragma unroll 1
        for (m_Ctx.blockCountdown.y = m_Ctx.patchesInOutput.y / cfg.outBlockSize.InPatches().y;
             m_Ctx.blockCountdown.y != 0; --m_Ctx.blockCountdown.y)
        {
            ProcessBlockRow(cfg.outBlockSize.InPatches().y, patchesInOutputBlockZ);
        }

        ProcessBlockRow(m_Ctx.patchesInOutput.y % cfg.outBlockSize.InPatches().y, patchesInOutputBlockZ);
    }

    __inline_always ncu_ple_interface::PleMsg::StripeDone ProcessStripe()
    {
        InitStripe();

#pragma unroll 1
        for (m_Ctx.blockCountdown.z = DivRoundUp(m_Ctx.patchesInOutput.z, m_Ctx.numActiveOgs);
             m_Ctx.blockCountdown.z != 0; --m_Ctx.blockCountdown.z)
        {
            ProcessXyPlane((m_Ctx.blockCountdown.z == 1) ? LastIter(m_Ctx.patchesInOutput.z, m_Ctx.numActiveOgs)
                                                         : m_Ctx.numActiveOgs);
        }

        WaitForEvent<Event::UDMA_STORE_DONE>();

        return ncu_ple_interface::PleMsg::StripeDone{};
    }

    PersistentState m_Persistent;
    Ctx m_Ctx;
};

template <Xy patchesInGroup>
auto LoadGroup(const unsigned inramId, const lsu::Address inramAddr)
{
    static_assert(patchesInGroup.x == 2);
    static_assert(patchesInGroup.y == 2);

    return std::tuple{
        cexec::LoadInramRf<0>{ inramId, inramAddr },
        cexec::LoadInramRf<2>{ inramId, inramAddr },
    };
}

template <>
auto LoadGroup<Xy{ 1, 2 }>(const unsigned inramId, const lsu::Address inramAddr)
{
    return std::tuple{
        cexec::LoadInramRf<0>{ inramId, inramAddr },
        cexec::Sub16<2, 2, 2>{},
    };
}

template <>
auto LoadGroup<Xy{ 2, 1 }>(const unsigned inramId, const lsu::Address inramAddr)
{
    return std::tuple{
        cexec::LoadHalfInramRf<0>{ inramId, inramAddr },
        cexec::Xor8<1, 1, 1>{},
        cexec::LoadHalfInramRf<2>{ inramId, inramAddr },
        cexec::Xor8<3, 3, 3>{},
    };
}

template <>
auto LoadGroup<Xy{ 1, 1 }>(const unsigned inramId, const lsu::Address inramAddr)
{
    return std::tuple{
        cexec::LoadHalfInramRf<0>{ inramId, inramAddr },
        cexec::Xor8<1, 1, 1>{},
        cexec::Sub16<2, 2, 2>{},
    };
}

template <Xy patchesInGroup>
auto LoadGroup(const unsigned inramId, const unsigned inramAddr)
{
    // By using reinterpret_cast, we generate a few instructions less
    return LoadGroup<patchesInGroup>(inramId, reinterpret_cast<const lsu::Address&>(inramAddr));
}

auto StoreGroup(const lsu::Address outramAddr)
{
    return std::tuple{
        cexec::StoreRfOutram<0>{ outramAddr },
        cexec::StoreRfOutram<2>{ outramAddr },
    };
}

auto StoreGroup(const unsigned outramAddr)
{
    // By using reinterpret_cast, we generate a few instructions less
    return StoreGroup(reinterpret_cast<const lsu::Address&>(outramAddr));
}
}    // namespace
