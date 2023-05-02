//
// Copyright © 2018-2020,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "DfcSramTraversal.hpp"
#include "Input.hpp"
#include "Output.hpp"
#include "PleState.hpp"
#include "Sizes.hpp"
#include "hw.h"
#include "udma.h"
#include "utils.h"
#include "xyz.h"

namespace
{
template <typename InBlockSize, typename OutBlockSize, typename Derived, bool IsTranspose = false>
class PassthroughBase
{
    static constexpr unsigned mk_OutBlockSizeInWords = WORDS_PER_REGISTER * TotalSize(OutBlockSize{});
    static constexpr unsigned mk_OutQueueSizeInWords = 2U * mk_OutBlockSizeInWords;

    static_assert((mk_OutQueueSizeInWords & (mk_OutQueueSizeInWords - 1U)) == 0,
                  "mk_OutQueueSizeInWords must be a power of 2");

public:
    using OutGroupSize = sizes::GroupSize<std::min(PATCHES_PER_GROUP_1D, OutBlockSize::X),
                                          std::min(PATCHES_PER_GROUP_1D, OutBlockSize::Y),
                                          OutBlockSize::Z>;

    PassthroughBase(EnumBitset<Event>& activeEvents, const Xyz& outSizeInElements, const uint16_t outDfcAddr)
        : m_Input()
        , m_Output()
        , m_OutputTraversal(outSizeInElements)
        , m_UdmaStorer(activeEvents)
        , m_OutramAddr(0)
        , m_OutDfcAddr(outDfcAddr)
        , m_OutDfcAddrY(outDfcAddr)
        , m_OutDfcAddrZ(outDfcAddr)
    {}

    ~PassthroughBase()
    {
        m_UdmaStorer.WaitForUdma();
    }

    void ProcessFullBlock(const unsigned firstOg, const unsigned lastOg, const unsigned inramAddr, const Xyz& pos)
    {
        for (unsigned og = firstOg; og < lastOg; og += NUM_PLE_LANES)
        {
            m_Input.LoadFullBlock(og, { inramAddr, 0U });
            nop<static_cast<int>(VE_TIMING::LOAD_INRAM_RF::WRITE_BACK - (TotalSize(InBlockSize{}) / 2))>();
            ProcessBlock();
            // there is no special function for full block in case of transpose, this handled by InputToOutputSize
            m_Output.StoreFullBlock({ m_OutramAddr, 0U });

            UdmaBlock(og, m_Output.mk_BlockSizeInGroups);
        }

        Advance(pos, m_Output.mk_BlockSizeInGroups);
    }

    void ProcessPartialWidthBlock(
        const unsigned firstOg, const unsigned lastOg, const unsigned inramAddr, const Xyz& pos, unsigned width)
    {
        const Xy outBlockSize      = InputToOutputSize(Xyz(width, InBlockSize::Y));
        const Xy blockSizeInGroups = DivRoundUp(outBlockSize, Xy(Xyz(OutGroupSize{})));

        for (unsigned og = firstOg; og < lastOg; og += NUM_PLE_LANES)
        {
            m_Input.LoadPartialWidthBlock(og, { inramAddr, 0U }, width);
            nop<static_cast<int>(VE_TIMING::LOAD_INRAM_RF::WRITE_BACK - (InBlockSize::Y / 2))>();
            ProcessBlock();
            // for transpose operations partial width input become partial height output
            if (IsTranspose)
            {
                m_Output.StorePartialHeightBlock({ m_OutramAddr, 0U }, outBlockSize.y);
            }
            else
            {
                m_Output.StorePartialWidthBlock({ m_OutramAddr, 0U }, outBlockSize.x);
            }

            UdmaBlock(og, blockSizeInGroups);
        }

        Advance(pos, blockSizeInGroups);
    }

    void ProcessPartialHeightBlock(
        const unsigned firstOg, const unsigned lastOg, const unsigned inramAddr, const Xyz& pos, unsigned height)
    {
        const Xy outBlockSize      = InputToOutputSize(Xyz(InBlockSize::X, height));
        const Xy blockSizeInGroups = DivRoundUp(outBlockSize, Xy(Xyz(OutGroupSize{})));

        for (unsigned og = firstOg; og < lastOg; og += NUM_PLE_LANES)
        {
            m_Input.LoadPartialHeightBlock(og, { inramAddr, 0U }, height);
            nop<static_cast<int>(VE_TIMING::LOAD_INRAM_RF::WRITE_BACK - InBlockSize::X)>();
            ProcessBlock();
            // for transpose operations partial height input become partial width output
            if (IsTranspose)
            {
                m_Output.StorePartialWidthBlock({ m_OutramAddr, 0U }, outBlockSize.x);
            }
            else
            {
                m_Output.StorePartialHeightBlock({ m_OutramAddr, 0U }, outBlockSize.y);
            }

            UdmaBlock(og, blockSizeInGroups);
        }

        Advance(pos, blockSizeInGroups);
    }

    void ProcessPartialBlock(
        const unsigned firstOg, const unsigned lastOg, const unsigned inramAddr, const Xyz& pos, const Xy& size)
    {
        const Xy outBlockSize      = InputToOutputSize(Xyz(size, 0));
        const Xy blockSizeInGroups = DivRoundUp(outBlockSize, Xy(Xyz(OutGroupSize{})));

        for (unsigned og = firstOg; og < lastOg; og += NUM_PLE_LANES)
        {
            m_Input.LoadPartialBlock(og, { inramAddr, 0U }, size);
            nop<VE_TIMING::LOAD_INRAM_RF::WRITE_BACK - 1>();
            ProcessBlock();
            // there is no special function for partial block in case of transpose, this handled by InputToOutputSize
            m_Output.StorePartialBlock({ m_OutramAddr, 0U }, outBlockSize);

            UdmaBlock(og, blockSizeInGroups);
        }

        Advance(pos, blockSizeInGroups);
    }

    void NextRow(const unsigned numActiveOgs, const unsigned posY)
    {
        // NextRow means next input row, but in transpose it implies next output column
        const Xyz adv = IsTranspose ? Xyz(m_Output.mk_BlockSizeInGroups.x) : Xyz(0, m_Output.mk_BlockSizeInGroups.y);

        m_OutDfcAddrY += m_OutputTraversal.Advance(Xyz(0, posY, 0), adv);
        m_OutDfcAddr = m_OutDfcAddrY;
    }

    void NextDepth(const unsigned numActiveOgs)
    {
        m_OutDfcAddrZ += m_OutputTraversal.Advance(Xyz{}, Xyz(0, 0, numActiveOgs));
        m_OutDfcAddr  = m_OutDfcAddrZ;
        m_OutDfcAddrY = m_OutDfcAddrZ;
    }

private:
    static constexpr Xyz InputToOutputSize(const Xyz& inSize)
    {
        if (IsTranspose)
        {
            return DivRoundUp(TransposeXY(inSize) * Xyz(OutBlockSize{}), TransposeXY(Xyz(InBlockSize{})));
        }
        else
        {
            return DivRoundUp(inSize * Xyz(OutBlockSize{}), Xyz(InBlockSize{}));
        }
    }

    void Advance(const Xyz& pos, const Xy& blockSizeInGroups)
    {
        // in normal operation the blocks advance in row-major order in the output,
        // whereas in transpose mode they advance in column-major order.
        const Xyz adv = IsTranspose ? Xyz(0, blockSizeInGroups.y) : Xyz(blockSizeInGroups.x);
        m_OutDfcAddr += m_OutputTraversal.Advance(pos, adv);
    }

    void UdmaBlock(const unsigned og, const Xy& blockSizeInGroups)
    {
        udma::Address udmaAddr = { m_OutDfcAddr, m_OutramAddr };

        auto& udmaAddrAsU32 = reinterpret_cast<unsigned&>(udmaAddr);
        udmaAddrAsU32 += m_OutputTraversal.Advance(Xyz{}, Xyz(0, 0, og));

        m_OutramAddr = (m_OutramAddr + mk_OutBlockSizeInWords) % mk_OutQueueSizeInWords;

        m_UdmaStorer.WaitForUdma();
        m_OutputTraversal.SetUdmaStoreParams(blockSizeInGroups);
        m_UdmaStorer.Store(og % NUM_SRAMS, udmaAddr);
    }

    void ProcessBlock()
    {
        static_cast<Derived*>(this)->ProcessBlock();
    }

private:
    const input::InramRfLoader<InBlockSize> m_Input;
    const output::RfOutramStorer<OutBlockSize, OutGroupSize> m_Output;
    const dfcsram::Traversal<OutGroupSize> m_OutputTraversal;

    udma::UdmaStorer m_UdmaStorer;

    uint16_t m_OutramAddr;
    uint16_t m_OutDfcAddr;
    uint16_t m_OutDfcAddrY;
    uint16_t m_OutDfcAddrZ;
};
}    // namespace
