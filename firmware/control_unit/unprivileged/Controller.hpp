//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "DmaRdController.hpp"
#include "DmaWrController.hpp"
#include "MceController.hpp"
#include "PleController.hpp"

namespace ethosn::control_unit
{

/// This implements the high level controller logic that owns the HwAbstraction, the overall agent progress
/// to track interdependencies between the different low level controllers and the low level controllers themselves.
///
/// HwAbstractionT is left unspecified for testability of this logic.
/// (see https://en.wikipedia.org/wiki/Dependency_injection)
template <typename HwAbstractionT>
class Controller
{
public:
    Controller() = default;

    /// Create a Controller with a copy/reference of the given hwAbstraction to pass down to lower level controllers.
    explicit Controller(HwAbstractionT&& hwAbstraction, const command_stream::CommandStream& cmdStream);

    /// Run one iteration of the controller algorithm. This is intended to be called inside a top level control loop.
    bool Spin();

    /// Returns true if there are no pending actions in this controller instance
    /// (i.e. all commands have been executed and completed by the HW).
    bool IsDone() const;

    void WaitForEvents();

    decltype(auto) GetHwAbstraction()
    {
        return m_Ctrl.hwAbstraction;
    }

private:
    /// Common controller state to pass down to lower level controllers.
    struct Ctrl
    {
        HwAbstractionT hwAbstraction;

        const Agent* agents;

        uint32_t dmaRdCounter;
        uint32_t dmaWrCounter;
        uint32_t mceifCounter;
        uint32_t mceStripeCounter;
        uint32_t pleCodeLoadedIntoPleSramCounter;
        uint32_t pleStripeCounter;
    };

    void UpdateProgress(const CompletedTsuEvents& tsuEvents);

    bool HandleCommands();

    void LogProgress();

    const command_stream::CommandStream& m_CmdStream;

    Ctrl m_Ctrl;

    DmaRdController m_DmaRdCtrl;
    MceController m_MceCtrl;
    PleController m_PleCtrl;
    DmaWrController m_DmaWrCtrl;
};

/// Class template argument deduction guideline that allows for instantiation code to cleanly choose
/// whether the HwAbstraction should be captured by value or by reference.
///
/// Examples:
///     HwAbstraction hw;
///
///     Controller ctrl{ hw };                  // hw will be captured by reference.
///                                             // Note: decltype(ctrl) = Controller<HwAbstraction&>
///
///     Controller ctrl{ HwAbstraction{ hw } }; // hw will be captured by value.
///                                             // Note: decltype(ctrl) = Controller<HwAbstraction>
///
///     Controller ctrl{ HwAbstraction{} };     // An internal HwAbstraction will be copy-initialized
///                                             // from the expression HwAbstraction{}.
///                                             // Note: decltype(ctrl) = Controller<HwAbstraction>
///
///     Controller<HwAbstraction> ctrl{};       // An internal HwAbstraction will be default-initialized.
///                                             // Note: decltype(ctrl) = Controller<HwAbstraction>
template <typename HwAbstractionT>
Controller(HwAbstractionT&&, const command_stream::CommandStream& cmdStream)->Controller<HwAbstractionT>;

template <typename HwAbstractionT>
Controller<HwAbstractionT>::Controller(HwAbstractionT&& hwAbstraction, const command_stream::CommandStream& cmdStream)
    : m_CmdStream(cmdStream)
    , m_Ctrl{ std::forward<HwAbstractionT>(hwAbstraction), cmdStream.GetAgentsArray(), 0, 0, 0, 0, 0, 0 }
    , m_DmaRdCtrl(cmdStream.GetDmaRdCommandsBegin(), cmdStream.NumDmaRdCommands)
    , m_MceCtrl(cmdStream.GetMceCommandsBegin(), cmdStream.NumMceCommands)
    , m_PleCtrl(cmdStream.GetPleCommandsBegin(), cmdStream.NumPleCommands)
    , m_DmaWrCtrl(cmdStream.GetDmaWrCommandsBegin(), cmdStream.NumDmaWrCommands)
{}

template <typename HwAbstractionT>
bool Controller<HwAbstractionT>::Spin()
{
    CompletedTsuEvents tsuEvents = m_Ctrl.hwAbstraction.UpdateTsuEvents();

    LogProgress();

    UpdateProgress(tsuEvents);

    bool madeProgress = HandleCommands();

    return madeProgress && !tsuEvents.pleError;
}

template <typename HwAbstractionT>
bool Controller<HwAbstractionT>::IsDone() const
{
    return m_DmaRdCtrl.IsDone() && m_MceCtrl.IsDone() && m_PleCtrl.IsDone() && m_DmaWrCtrl.IsDone();
}

template <typename HwAbstractionT>
void Controller<HwAbstractionT>::WaitForEvents()
{
    m_Ctrl.hwAbstraction.WaitForEvents();
}

template <typename HwAbstractionT>
void Controller<HwAbstractionT>::UpdateProgress(const CompletedTsuEvents& tsuEvents)
{
    uint8_t updateProgressEventId = m_Ctrl.hwAbstraction.GetProfiling().RecordStart(TimelineEventType::UpdateProgress);

    m_MceCtrl.UpdateProgress(m_Ctrl);
    m_DmaRdCtrl.UpdateProgress(m_Ctrl);
    m_DmaWrCtrl.UpdateProgress(m_Ctrl);
    m_PleCtrl.UpdateProgress(m_Ctrl, tsuEvents.pleStripeDone, tsuEvents.pleCodeLoadedIntoPleSram);

    m_Ctrl.hwAbstraction.GetProfiling().RecordEnd(updateProgressEventId);
}

template <typename HwAbstractionT>
bool Controller<HwAbstractionT>::HandleCommands()
{
    // The MCE HandleCommands is called first to optimise the
    // execution of the inference by keeping the MCE as busy as
    // possible.
    bool madeProgress = false;
    madeProgress |= m_MceCtrl.HandleCommands(m_Ctrl);
    madeProgress |= m_DmaRdCtrl.HandleCommands(m_Ctrl);
    madeProgress |= m_PleCtrl.HandleCommands(m_Ctrl);
    madeProgress |= m_DmaWrCtrl.HandleCommands(m_Ctrl);

    return madeProgress;
}

template <typename HwAbstractionT>
void Controller<HwAbstractionT>::LogProgress()
{
    // This optional debugging feature will update the GP registers (GP0 - GP3) with the progress
    // counters as the command stream is executed. This is useful for
    // diagnosing hangs as you can dump the GP registers from the kernel (cat /sys/kernel/debug/ethosn0/core0/registers)
    // and see where it got stuck. This gives much less information than full logging or profiling, but
    // has much less effect on the timings and so is useful for hangs which are timing-sensitive.
    //
    // IMPORTANT: If enabling this, the MPU permissions must be changed to give unprivileged access to the GP regs (Mpu.cpp, change region 4 to ARM_MPU_AP_FULL)
    bool debugSaveProgressInGpRegs = false;
    if (debugSaveProgressInGpRegs)
    {
        m_Ctrl.hwAbstraction.StoreDebugGpRegister(0, m_Ctrl.dmaRdCounter);
        m_Ctrl.hwAbstraction.StoreDebugGpRegister(1, m_Ctrl.dmaWrCounter);
        m_Ctrl.hwAbstraction.StoreDebugGpRegister(2, m_Ctrl.mceifCounter);
        m_Ctrl.hwAbstraction.StoreDebugGpRegister(3, m_Ctrl.mceStripeCounter);
        m_Ctrl.hwAbstraction.StoreDebugGpRegister(4, m_Ctrl.pleCodeLoadedIntoPleSramCounter);
        m_Ctrl.hwAbstraction.StoreDebugGpRegister(5, m_Ctrl.pleStripeCounter);
    }

    // Unfortunately we can't rely on the compiler to optimise out all the logging code below
    // when logging is disabled, because GetNumCmdsInHwQueue called by GetStateString performs register reads
    if (g_LogCompileTimeMaxSeverity >= ethosn::utils::log::Severity::Debug)
    {
        m_Ctrl.hwAbstraction.GetLogger().Debug(
            "%s", m_DmaRdCtrl.GetStateString(m_Ctrl, m_CmdStream.NumDmaRdCommands).GetCString());
        m_Ctrl.hwAbstraction.GetLogger().Debug(
            "%s", m_MceCtrl.GetStateString(m_Ctrl, m_CmdStream.NumMceCommands).GetCString());
        m_Ctrl.hwAbstraction.GetLogger().Debug(
            "%s", m_PleCtrl.GetStateString(m_Ctrl, m_CmdStream.NumPleCommands).GetCString());
        m_Ctrl.hwAbstraction.GetLogger().Debug(
            "%s", m_DmaWrCtrl.GetStateString(m_Ctrl, m_CmdStream.NumDmaWrCommands).GetCString());
    }
}

}    // namespace ethosn::control_unit
