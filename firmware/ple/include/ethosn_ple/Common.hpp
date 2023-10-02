//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "DfcSramTraversal.hpp"
#include "Output.hpp"
#include "Sizes.hpp"
#include "hw.h"
#include "udma.h"
#include "utils.h"
#include "xyz.h"

#include <array>

namespace
{

enum class Flags : size_t
{
    TOP,
    BOTTOM,
    LEFT,
    RIGHT,
};

struct InputInfo
{
    uint16_t dfcAddr;
    int16_t zeroPoint;
    uint16_t multiplier;
    uint16_t shift;
};

struct OutputInfo
{
    uint16_t dfcAddr;
    int16_t zeroPoint;
};

enum class MceOp : uint16_t
{
    CONVOLUTION,
    DEPTHWISE_CONVOLUTION
};

constexpr unsigned MAX_INPUTS = 2;

struct StripeInfo
{
    EnumBitset<Flags, uint32_t> flags;
    std::array<InputInfo, MAX_INPUTS> inputs;
    OutputInfo output;
    uint16_t stripeWidth;
    uint16_t stripeHeight;
    uint16_t stripeDepth;
    MceOp mceOp;
};

struct OperatorInfo
{
    EnumBitset<Flags> flags;
    std::array<InputInfo, MAX_INPUTS> inputs;
    OutputInfo output;
    /// The size of the input stripe.
    Xyz sizeInElements;
#if (NUM_SRAMS != NUM_MCEIF)
    unsigned numActiveOgs;
#else
    static constexpr unsigned numActiveOgs = NUM_MCEIF;
#endif
};

inline StripeInfo ReadStripeInfo()
{
    StripeInfo iface;

    static_assert((sizeof(iface) % 4) == 0, "");

    const auto src = reinterpret_cast<const volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_SCRATCH0));
    const auto dst = reinterpret_cast<uint32_t*>(&iface);

#pragma unroll
    for (unsigned i = 0; i < (sizeof(iface) / 4); ++i)
    {
        dst[i] = src[i];
    }

    return iface;
}

struct OutputToInputIdentity
{
    constexpr Xyz operator()(const Xyz out, const EnumBitset<Flags>) const
    {
        return out;
    }
};

template <typename OutputToInput = OutputToInputIdentity>
inline OperatorInfo GetOperatorInfo()
{
    const StripeInfo iface = ReadStripeInfo();

    OperatorInfo opInfo;

    opInfo.flags = iface.flags;

    opInfo.inputs = iface.inputs;

    for (auto& inp : opInfo.inputs)
    {
        inp.dfcAddr *= WORDS_PER_REGISTER;
    }

    opInfo.output = iface.output;
    opInfo.output.dfcAddr *= WORDS_PER_REGISTER;

    opInfo.sizeInElements =
        OutputToInput{}(Xyz{ iface.stripeWidth, iface.stripeHeight, iface.stripeDepth }, opInfo.flags);

#if (NUM_SRAMS != NUM_MCEIF)
    const bool isDepthwise = iface.mceOp == MceOp::DEPTHWISE_CONVOLUTION;
    opInfo.numActiveOgs    = isDepthwise ? NUM_SRAMS : NUM_MCEIF;
#endif

    return opInfo;
}

// Helper to call a function without inlining it.
template <typename TFunctor>
decltype(auto) __attribute__((noinline)) NoInline(TFunctor f)
{
    return f();
}

template <typename ProcessStripe, typename WaitForIrq>
void __attribute__((noreturn, __always_inline__)) Main(WaitForIrq waitForIrq, ProcessStripe processStripe)
{
    while (true)
    {
        // The PLE lane selection set for the previous stripe that was processed by either this
        // kernel or a previous one is still in effect here. This means that any coprocessor
        // instructions that are used will only affect the currently active PLEs. To ensure that all
        // PLEs are affected by coprocessor instructions that are used to process the next stripe,
        // before a new lane selection has been performed. The PLE lane selection is reset to its
        // default value (enable all lanes).
        SetPleLanesInUse(NUM_PLE_LANES);

        // Wait for the NCU MCU to instruct us to process a new stripe.
        waitForIrq();

        // Use NoInline to force the main body of the kernel into a separate stack frame. This is required so that
        // the stack usage when we enter WFE is quite small, as this is the time when the PLE will be reset
        // by the CU to load a new kernel. When the PLE gets reset via the interrupt handler, the PLE MCU
        // will automatically save some registers to the stack before jumping to the interrupt handler.
        // If the stack usage at that time is too high, then pushing the registers to the stack could cause 2 problems:
        //     1. It might breach the MSPLIM we set, leading to an exception
        //     2. Saving the registers could overwrite the newly loaded code from the new kernel, if
        //        the end of the new kernel's code is too close to the top of our current stack
        // Both these problems are solved if the stack usage is kept low when the reset occurs, AND both the old AND
        // new kernels have their max stack size set high enough that there is sufficient room in the stack for the
        // registers to be pushed. ple.scatter enforces the max stack size is sufficiently large.
        ncu_ple_interface::PleMsg::StripeDone stripeDoneMsg = NoInline([&]() { return processStripe(); });

        // Notify the NCU MCU that we have finished processing this stripe, and record the number
        // of blocks that we processed. The NCU MCU uses this information to inform its scheduling.
        auto& pleMsg = *reinterpret_cast<volatile ncu_ple_interface::PleMsg*>(PLE_REG(CE_RP, CE_PLE_SCRATCH0));
        WriteToRegisters(&pleMsg.type, ncu_ple_interface::PleMsg::StripeDone::type);
        WriteToRegisters(&pleMsg.stripeDone, stripeDoneMsg);
        SignalPleStripeDone();
    }
}

}    // namespace
