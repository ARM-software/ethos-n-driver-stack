//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "hw.h"
#include "utils.h"

namespace    // Internal linkage
{
namespace udma
{

struct Address
{
    unsigned dfcAddrWords : 16;
    unsigned pleAddr : 16;
};

struct SrcDst
{
    unsigned src : 4;
    unsigned dst : 4;
};

struct Params
{
    unsigned colGrpStride : 12;
    unsigned rowGrpStride : 12;
    unsigned colGrpCountMinusOne : 4;
    unsigned rowGrpCountMinusOne : 4;
};

enum class Direction : unsigned
{
    DFC_OUTRAM  = 0b000,
    DFC_INRAM   = 0b010,
    OUTRAM_DFC  = 0b100,
    DFC_CODERAM = 0b001,
    CODERAM_DFC = 0b101
};

namespace impl
{

constexpr unsigned CP_NUM = 7;

constexpr unsigned SET_UDMA_LOAD_PARAMS_OPC1  = 0b011;
constexpr unsigned SET_UDMA_STORE_PARAMS_OPC1 = 0b111;

template <unsigned Opc1, unsigned GroupSize>
void SetParams(const Params params)
{
    mcr<CP_NUM, Opc1, 0, ((GroupSize - 1) >> 4), ((GroupSize - 1) & 0xf)>(params);
}

template <Direction Dir>
void Transfer(const Address addr, const SrcDst srcDst)
{
    mcrr<CP_NUM, static_cast<unsigned>(Dir), 0>(addr, srcDst);
}

}    // namespace impl

template <unsigned GroupSize>
void SetLoadParams(const Params params)
{
    impl::SetParams<impl::SET_UDMA_LOAD_PARAMS_OPC1, GroupSize>(params);
}

template <unsigned GroupSize>
void SetStoreParams(const Params params)
{
    impl::SetParams<impl::SET_UDMA_STORE_PARAMS_OPC1, GroupSize>(params);
}

template <Direction Dir>
void Transfer(const unsigned dfcId, const Address addr)
{
    switch (Dir)
    {
        case Direction::DFC_OUTRAM:
        {
            impl::Transfer<Dir>(addr, { .src = dfcId });
            break;
        }
        case Direction::DFC_INRAM:
        {
            impl::Transfer<Dir>(addr, { .src = dfcId, .dst = dfcId });
            break;
        }
        case Direction::OUTRAM_DFC:
        {
            impl::Transfer<Dir>(addr, { .dst = dfcId });
            break;
        }
        case Direction::DFC_CODERAM:
        {
            impl::Transfer<Dir>(addr, { .src = dfcId });
            break;
        }
        case Direction::CODERAM_DFC:
        {
            impl::Transfer<Dir>(addr, { .dst = dfcId });
            break;
        }
    }
}

class UdmaStorer
{
public:
    UdmaStorer(EnumBitset<Event>& activeEvents)
        : m_ActiveEvents(activeEvents)
        , m_UdmaBusy(false)
    {}

    template <Direction Dir>
    void Transfer(const unsigned dfcId, const udma::Address udmaAddr)
    {
        static_assert((Dir == Direction::OUTRAM_DFC) || (Dir == Direction::CODERAM_DFC), "");

        udma::Transfer<Dir>(dfcId, udmaAddr);
        m_UdmaBusy = true;
    }

    void Store(const unsigned dfcId, const udma::Address udmaAddr)
    {
        Transfer<Direction::OUTRAM_DFC>(dfcId, udmaAddr);
    }

    void WaitForUdma()
    {
        if (m_UdmaBusy)
        {
            WaitForEvent<Event::UDMA_STORE_DONE>(m_ActiveEvents);
            m_UdmaBusy = false;
        }
    }

private:
    EnumBitset<Event>& m_ActiveEvents;
    bool m_UdmaBusy;
};

class UdmaLoader
{
public:
    UdmaLoader(EnumBitset<Event>& activeEvents)
        : m_ActiveEvents(activeEvents)
        , m_UdmaBusy(false)
    {}

    template <Direction Dir>
    void Transfer(const unsigned dfcId, const udma::Address udmaAddr)
    {
        static_assert(
            (Dir == Direction::DFC_OUTRAM) || (Dir == Direction::DFC_INRAM) || (Dir == Direction::DFC_CODERAM), "");

        udma::Transfer<Dir>(dfcId, udmaAddr);
        m_UdmaBusy = true;
    }

    void Load(const unsigned dfcId, const udma::Address udmaAddr)
    {
        Transfer<Direction::DFC_INRAM>(dfcId, udmaAddr);
    }

    void WaitForUdma()
    {
        if (m_UdmaBusy)
        {
            WaitForEvent<Event::UDMA_LOAD_DONE>(m_ActiveEvents);
            m_UdmaBusy = false;
        }
    }

private:
    EnumBitset<Event>& m_ActiveEvents;
    bool m_UdmaBusy;
};

}    // namespace udma
}    // namespace
