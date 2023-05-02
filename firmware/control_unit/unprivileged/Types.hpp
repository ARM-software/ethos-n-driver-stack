//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

#include <common/FixedString.hpp>
#include <common/Log.hpp>

#include <ethosn_command_stream/CommandStream.hpp>

namespace ethosn::control_unit
{

using Agent = command_stream::Agent;
using MceS  = command_stream::MceS;
using PleS  = command_stream::PleS;
using IfmS  = command_stream::IfmS;
using OfmS  = command_stream::OfmS;
using WgtS  = command_stream::WgtS;
using PleL  = command_stream::PleL;

using Command     = command_stream::Command;
using CommandType = command_stream::CommandType;

using CounterName                   = command_stream::CounterName;
using WaitForCounterCommand         = command_stream::WaitForCounterCommand;
using DmaCommand                    = command_stream::DmaCommand;
using ProgramMceStripeCommand       = command_stream::ProgramMceStripeCommand;
using ConfigMceifCommand            = command_stream::ConfigMceifCommand;
using StartMceStripeCommand         = command_stream::StartMceStripeCommand;
using LoadPleCodeIntoPleSramCommand = command_stream::LoadPleCodeIntoPleSramCommand;
using StartPleStripeCommand         = command_stream::StartPleStripeCommand;

using AgentId  = uint32_t;
using StripeId = uint32_t;

inline const char* ToString(ethosn::command_stream::AgentType t)
{
    using namespace ethosn::command_stream;
    switch (t)
    {
        case AgentType::IFM_STREAMER:
            return "IfmS";
        case AgentType::WGT_STREAMER:
            return "WgtS";
        case AgentType::MCE_SCHEDULER:
            return "MceS";
        case AgentType::PLE_LOADER:
            return "PleL";
        case AgentType::PLE_SCHEDULER:
            return "PleS";
        case AgentType::OFM_STREAMER:
            return "OfmS";
        default:
            return "<Unknown>";
    }
}

inline const char* ToString(CommandType c)
{
    switch (c)
    {
        case CommandType::WaitForCounter:
            return "WaitForCounter";
        case CommandType::LoadIfmStripe:
            return "LoadIfmStripe";
        case CommandType::LoadWgtStripe:
            return "LoadWgtStripe";
        case CommandType::ProgramMceStripe:
            return "ProgramMceStripe";
        case CommandType::ConfigMceif:
            return "ConfigMceif";
        case CommandType::StartMceStripe:
            return "StartMceStripe";
        case CommandType::LoadPleCodeIntoSram:
            return "LoadPleCodeIntoSram";
        case CommandType::LoadPleCodeIntoPleSram:
            return "LoadPleCodeIntoPleSram";
        case CommandType::StartPleStripe:
            return "StartPleStripe";
        case CommandType::StoreOfmStripe:
            return "StoreOfmStripe";
        default:
            return "<Unknown>";
    }
}

inline const char* ToString(CounterName c)
{
    switch (c)
    {
        case CounterName::DmaRd:
            return "DmaRd";
        case CounterName::DmaWr:
            return "DmaWr";
        case CounterName::Mceif:
            return "Mceif";
        case CounterName::MceStripe:
            return "MceStripe";
        case CounterName::PleCodeLoadedIntoPleSram:
            return "PleCodeLoadedIntoPleSram";
        case CounterName::PleStripe:
            return "PleStripe";
        default:
            return "<Unknown>";
    }
}

inline LoggingString ToString(const Command& c)
{
    LoggingString details;
    switch (c.type)
    {
        case CommandType::WaitForCounter:
            details =
                LoggingString::Format("%s, %u", ToString(static_cast<const WaitForCounterCommand&>(c).counterName),
                                      static_cast<const WaitForCounterCommand&>(c).counterValue);
            break;
        case CommandType::LoadIfmStripe:
            details = LoggingString::Format("%u", static_cast<const DmaCommand&>(c).agentId);
            break;
        case CommandType::LoadWgtStripe:
            details = LoggingString::Format("%u", static_cast<const DmaCommand&>(c).agentId);
            break;
        case CommandType::ProgramMceStripe:
            details = LoggingString::Format("%u", static_cast<const ProgramMceStripeCommand&>(c).agentId);
            break;
        case CommandType::ConfigMceif:
            details = LoggingString::Format("%u", static_cast<const ConfigMceifCommand&>(c).agentId);
            break;
        case CommandType::StartMceStripe:
            details = LoggingString::Format("%u", static_cast<const StartMceStripeCommand&>(c).agentId);
            break;
        case CommandType::LoadPleCodeIntoSram:
            details = LoggingString::Format("%u", static_cast<const DmaCommand&>(c).agentId);
            break;
        case CommandType::LoadPleCodeIntoPleSram:
            details = LoggingString::Format("%u", static_cast<const LoadPleCodeIntoPleSramCommand&>(c).agentId);
            break;
        case CommandType::StartPleStripe:
            details = LoggingString::Format("%u", static_cast<const StartPleStripeCommand&>(c).agentId);
            break;
        case CommandType::StoreOfmStripe:
            details = LoggingString::Format("%u", static_cast<const DmaCommand&>(c).agentId);
            break;
        default:
            details = LoggingString("<Unknown>");
            break;
    }
    return LoggingString::Format("%s { %s }", ToString(c.type), details.GetCString());
}

}    // namespace ethosn::control_unit
