//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Mailbox.hpp"

#include "HardwareHelpers.hpp"

#include <common/FirmwareApi.hpp>
#include <common/Log.hpp>
#include <common/TaskSvc.hpp>
#include <common/hals/HardwareHal.hpp>
#include <unprivileged/Task.hpp>

#if CONTROL_UNIT_DEBUG_MONITOR
#include <mri.h>
#endif

#include <cinttypes>
#include <cstdio>

namespace ethosn
{
namespace control_unit
{

__attribute__((used)) void SvcHandler(unsigned number, unsigned* args)
{
    switch (number)
    {
        case TASK_SVC_LOG_MESSAGE:
        {
            Mailbox<HardwareHal>* mailbox = *reinterpret_cast<Mailbox<HardwareHal>**>(TOP_REG(DL1_RP, DL1_GP7));
            mailbox->Log(static_cast<ethosn_log_severity>(args[0]), reinterpret_cast<char*>(args[1]));
            break;
        }
        case TASK_SVC_GET_DWT_SLEEP_CYCLE_COUNT:
        {
            args[0] = Dwt::GetCycleCount();
            break;
        }
        case TASK_SVC_TASK_SWITCH:
        {
            Interrupt::SetPendSV();
            break;
        }
        case TASK_SVC_DCACHE_CLEAN_INVALIDATE:
        {
            Cache::DCleanInvalidate();
            break;
        }
        default:
        {
#if defined(ETHOSN_LOGGING)
            char buf[30] = { 0 };
            if (snprintf(buf, sizeof(buf), "Unknown SVC number: %u", number & 0xFF) < 0)
            {
                return;
            }
            Mailbox<HardwareHal>* mailbox = *reinterpret_cast<Mailbox<HardwareHal>**>(TOP_REG(DL1_RP, DL1_GP7));
            mailbox->Log(ETHOSN_LOG_WARNING, buf);
#endif
            break;
        }
    }
}

void PopulateTaskConfig(TaskConfig* config)
{
    uint32_t mailboxAddr = *reinterpret_cast<volatile uint32_t*>(TOP_REG(DL1_RP, GP_MAILBOX));
    config->logSeverity =
        static_cast<ethosn::utils::log::Severity>(reinterpret_cast<ethosn_mailbox*>(mailboxAddr)->severity);

    uint32_t addrExtend   = *reinterpret_cast<volatile uint32_t*>(TOP_REG(DL1_RP, DL1_STREAM0_ADDRESS_EXTEND));
    config->pleAddrExtend = dl1_stream0_address_extend_r(addrExtend).get_addrextend();
}

namespace
{

// Context switch to the unprivileged task so it can process the message
TaskMessageStatus SendTaskMessage(TaskMessage** message)
{
    register TaskMessage* msg asm("r0") = nullptr;
    asm volatile("svc %[svc_num]" : "=r"(msg) : [svc_num] "i"(TASK_SVC_TASK_SWITCH));
    *message = msg;
    return msg ? msg->status : TaskMessageStatus::FAILED;
}

}    // namespace

void Main()
{
    LoggerType logger;
    HardwareHal hardware(logger);

    hardware.ClearSram();

    // Read address of mailbox from GP_MAILBOX and store pointer to mailbox object in GP7
    uint32_t mailboxAddr = hardware.ReadReg(TOP_REG(DL1_RP, GP_MAILBOX));
    Mailbox<HardwareHal> mailbox(hardware, reinterpret_cast<ethosn_mailbox*>(mailboxAddr));
    hardware.WriteReg(TOP_REG(DL1_RP, DL1_GP7), reinterpret_cast<uint32_t>(&mailbox));

    // Hook up the logging framework to send messages using the mailbox
    logger.AddSink([](ethosn::utils::log::Severity severity, const char* msg) {
        Mailbox<HardwareHal>* mailbox = *reinterpret_cast<Mailbox<HardwareHal>**>(TOP_REG(DL1_RP, DL1_GP7));
        mailbox->Log(static_cast<ethosn_log_severity>(severity), msg);
    });
    logger.SetMaxSeverity(
        static_cast<ethosn::utils::log::Severity>(reinterpret_cast<ethosn_mailbox*>(mailboxAddr)->severity));

    // Delegate access to control registers in DL2 for the unprivileged part
    dl1_delegation_r delegate(hardware.ReadReg(TOP_REG(DL1_RP, DL1_DELEGATION)));
    delegate.set_pwrctlr(delegation_t::DELEGATED);
    delegate.set_intext(delegation_t::DELEGATED);
    hardware.WriteReg(TOP_REG(DL1_RP, DL1_DELEGATION), delegate.word);

#if CONTROL_UNIT_DEBUG_MONITOR
    logger.Debug("Initializing mri...");
    mriInit("");
    logger.Debug("Done initializing mri!");
#endif

    TaskMessage* message = nullptr;
    // Initial message to start the task
    logger.Debug("Starting unprivileged task");
    bool taskRunning = SendTaskMessage(&message) == TaskMessageStatus::OK;
    if (!taskRunning)
    {
        FATAL_MSG("Failed to start task");
    }
    logger.Debug("Finished starting task");

    // Signal to the kernel module that firmware has booted successfully.
    // Note that we do this as late as possible before entering the message processing loop,
    // to catch as many potential problems as possible.
    hardware.WriteReg(TOP_REG(DL1_RP, GP_BOOT_SUCCESS), ETHOSN_FIRMWARE_BOOT_SUCCESS_MAGIC);
    logger.Info("Ethos-N is running");

    while (true)
    {
        // Read the message header
        ethosn_message_header header;
        union
        {
            ethosn_message_inference_request inference;
            ethosn_firmware_profiling_configuration profilingConfig;
            uint32_t delay;
        } data;
        Mailbox<HardwareHal>::Status status =
            mailbox.ReadMessage(header, reinterpret_cast<uint8_t*>(&data), sizeof(data));
        if (status != Mailbox<HardwareHal>::Status::OK)
        {
            continue;
        }

        if (!taskRunning)
        {
            switch (header.type)
            {
                case ETHOSN_MESSAGE_FW_HW_CAPS_REQUEST:
                    // Fall-through
                case ETHOSN_MESSAGE_CONFIGURE_PROFILING:
                    // Fall-through
                case ETHOSN_MESSAGE_INFERENCE_REQUEST:
                    logger.Error("Unable to process message: task not running");
                    mailbox.SendErrorResponse(header.type, ETHOSN_ERROR_STATUS_INVALID_STATE);
                    continue;
                default:
                    break;
            }
        }

        switch (header.type)
        {
            case ETHOSN_MESSAGE_DELAY:
            {
                Tick::Delay(data.delay);
                break;
            }
            case ETHOSN_MESSAGE_FW_HW_CAPS_REQUEST:
            {
                message->type = TaskMessageType::CAPABILITIES;
                if (SendTaskMessage(&message) != TaskMessageStatus::OK)
                {
                    logger.Error("Failed to get FW & HW capabilities");
                    mailbox.SendErrorResponse(header.type, ETHOSN_ERROR_STATUS_FAILED);
                }
                else
                {
                    mailbox.SendFwAndHwCapabilitiesResponse(
                        { message->data.capabilities.data, message->data.capabilities.size });
                }

                break;
            }
            case ETHOSN_MESSAGE_CONFIGURE_PROFILING:
            {
#if defined(CONTROL_UNIT_PROFILING)
                if (data.profilingConfig.enable_profiling)
                {
                    if (data.profilingConfig.num_hw_counters > ETHOSN_PROFILING_MAX_HW_COUNTERS)
                    {
                        logger.Error("Invalid number of HW counters in profiling config: %u",
                                     data.profilingConfig.num_hw_counters);
                        mailbox.SendErrorResponse(header.type, ETHOSN_ERROR_STATUS_INVALID_MESSAGE);
                        break;
                    }

                    message->type                        = TaskMessageType::PROFILING_ENABLE;
                    message->data.profilingConfig.config = data.profilingConfig;

                    Dwt::Reset();
                    Dwt::Start();
                }
                else
                {
                    message->type = TaskMessageType::PROFILING_DISABLE;
                    Dwt::Stop();
                }

                if (SendTaskMessage(&message) != TaskMessageStatus::OK)
                {
                    logger.Error("Configure profiling request failed");
                    mailbox.SendErrorResponse(header.type, ETHOSN_ERROR_STATUS_FAILED);
                    break;
                }
#else
                if (data.profilingConfig.enable_profiling)
                {
                    logger.Error("Profiling cannot be turned on because the firmware has not been built with "
                                 "CONTROL_UNIT_PROFILING");
                }
#endif
                mailbox.SendConfigureProfilingAck();
                break;
            }
            case ETHOSN_MESSAGE_INFERENCE_REQUEST:
            {
#if CONTROL_UNIT_DEBUG_MONITOR
                logger.Debug("Example software breakpoint (please move/remove as appropriate):");
                __debugbreak();
                logger.Debug("After example software breakpoint");
#endif

                message->type                       = TaskMessageType::INFERENCE;
                message->data.inference.bufferArray = data.inference.buffer_array;

                // Make sure the buffer table and command stream which we are about to read are up-to-date,
                // as the host CPU will have just modified these.
                Cache::DCleanInvalidate();

                bool success = (SendTaskMessage(&message) == TaskMessageStatus::OK);

                // FIXME: We need to invalidate only what is needed
                Cache::DCleanInvalidate();

                // It can notify the host that the inference has finished after the profiling write pointer
                // has been flushed.
                mailbox.SendInferenceResponse(success ? ETHOSN_INFERENCE_STATUS_OK : ETHOSN_INFERENCE_STATUS_ERROR,
                                              data.inference.user_argument, message->data.inference.cycleCount);

                hardware.ClearSram();

                // Send message to unprivileged firmware to do any post-inference logging (e.g. profiling status)
                message->type = TaskMessageType::POST_INFERENCE_CLEANUP;
                SendTaskMessage(&message);

                break;
            }
            case ETHOSN_MESSAGE_PING:
            {
                mailbox.SendPong();
                break;
            }
            default:
            {
                // Error Invalid message
                logger.Error("Invalid message type. type=%u, length=%u", header.type, header.length);
                mailbox.SendErrorResponse(header.type, ETHOSN_ERROR_STATUS_INVALID_MESSAGE);
            }
        }
    }
}

}    // namespace control_unit
}    // namespace ethosn

extern "C" void main(void)
{
    // It seems that armclang doesn't store line number information in the debug info for extern "C" functions,
    // so in order to be able to debug main, we put all the code in a regular C++ function.
    ethosn::control_unit::Main();
}
