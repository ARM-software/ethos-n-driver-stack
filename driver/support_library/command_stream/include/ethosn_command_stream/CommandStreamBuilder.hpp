//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "CommandStream.hpp"

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ethosn
{
namespace command_stream
{

template <typename T, typename Word>
void EmplaceBack(std::vector<Word>& data, const T& cmd)
{
    static_assert(alignof(T) <= alignof(Word), "Must not have a stronger alignment requirement");
    static_assert((sizeof(T) % sizeof(Word)) == 0, "Size must be a multiple of the word size");

    const size_t prevSize = data.size();
    data.resize(data.size() + (sizeof(T) / sizeof(Word)));
    new (&data[prevSize]) T(cmd);
}

/// A variant (tagged union) which holds one of the concrete Command subtypes.
/// This is used to store and build up vectors of commands, which isn't easy to do with the
/// Command type from the command stream, as each Command can be a different type that means
/// unique_ptrs (or similar) would be needed, which then needs virtual destructors, which we don't
/// want to add into the command stream types.
/// We don't want to use this type in the command stream itself, because the union will take up
/// as much space as the largest member, which in this case is quite large (ProgramMceStripeCommand
/// is way bigger than the others), and so would waste command stream space.
struct CommandVariant
{
    /// Even though all the Command subtypes have a `type` field too, we have no (safe) way to
    /// access them, unless we know which union member is active, so we need to duplicate this
    /// here.
    CommandType type;

    union
    {
        WaitForCounterCommand waitForCounter;
        DmaCommand dma;
        ProgramMceStripeCommand programMceStripe;
        ConfigMceifCommand configMceif;
        StartMceStripeCommand startMceStripe;
        LoadPleCodeIntoPleSramCommand loadPleCodeIntoPleSram;
        StartPleStripeCommand startPleStripe;
    };

    explicit CommandVariant(const WaitForCounterCommand& c)
        : type(c.type)
        , waitForCounter(c)
    {}
    explicit CommandVariant(const DmaCommand& c)
        : type(c.type)
        , dma(c)
    {}
    explicit CommandVariant(const ConfigMceifCommand& c)
        : type(c.type)
        , configMceif(c)
    {}
    explicit CommandVariant(const ProgramMceStripeCommand& c)
        : type(c.type)
        , programMceStripe(c)
    {}
    explicit CommandVariant(const StartMceStripeCommand& c)
        : type(c.type)
        , startMceStripe(c)
    {}
    explicit CommandVariant(const LoadPleCodeIntoPleSramCommand& c)
        : type(c.type)
        , loadPleCodeIntoPleSram(c)
    {}
    explicit CommandVariant(const StartPleStripeCommand& c)
        : type(c.type)
        , startPleStripe(c)
    {}

    /// All the Command subtypes inherit Command, and this provides a (safe) way to
    /// get access to that, without the caller having to know the actual Command subtype.
    const Command& AsBaseCommand() const
    {
        switch (type)
        {
            case CommandType::WaitForCounter:
                return waitForCounter;
            case CommandType::LoadIfmStripe:
                return dma;
            case CommandType::LoadWgtStripe:
                return dma;
            case CommandType::ProgramMceStripe:
                return programMceStripe;
            case CommandType::ConfigMceif:
                return configMceif;
            case CommandType::StartMceStripe:
                return startMceStripe;
            case CommandType::LoadPleCodeIntoSram:
                return dma;
            case CommandType::LoadPleCodeIntoPleSram:
                return loadPleCodeIntoPleSram;
            case CommandType::StartPleStripe:
                return startPleStripe;
            case CommandType::StoreOfmStripe:
                return dma;
            default:
                throw std::runtime_error("Invalid command type: " + std::to_string(static_cast<uint32_t>(type)));
        }
    }
};

/// Builds a command stream containing all of the agents and commands provided.
inline std::vector<uint32_t> BuildCommandStream(const std::vector<Agent>& agents,
                                                const std::vector<CommandVariant>& dmaRdCommands,
                                                const std::vector<CommandVariant>& dmaWrCommands,
                                                const std::vector<CommandVariant>& mceCommands,
                                                const std::vector<CommandVariant>& pleCommands)
{
    std::vector<uint32_t> raw;

    // Tag to identify the command stream data structure using "FourCC" style
    constexpr uint32_t fourcc = static_cast<uint32_t>('E') | (static_cast<uint32_t>('N') << 8) |
                                (static_cast<uint32_t>('C') << 16) | (static_cast<uint32_t>('S') << 24);

    std::array<uint32_t, 4> header = { fourcc, ETHOSN_COMMAND_STREAM_VERSION_MAJOR, ETHOSN_COMMAND_STREAM_VERSION_MINOR,
                                       ETHOSN_COMMAND_STREAM_VERSION_PATCH };

    for (uint32_t w : header)
    {
        raw.push_back(w);
    }

    CommandStream commandStream;
    uint32_t offset = sizeof(CommandStream);

    commandStream.AgentsOffset = offset;
    commandStream.NumAgents    = static_cast<uint32_t>(agents.size());
    offset += commandStream.NumAgents * static_cast<uint32_t>(sizeof(Agent));

    // Helper function to sum up the size of all commands in a list (as each Command may be
    // a different type, and so have a different size).
    auto sumCommandSize = [](const std::vector<CommandVariant>& commands) {
        return std::accumulate(commands.begin(), commands.end(), 0u, [](uint32_t sum, const auto& cmd) {
            return sum + static_cast<uint32_t>(cmd.AsBaseCommand().GetSize());
        });
    };

    commandStream.DmaRdCommandsOffset = offset;
    commandStream.NumDmaRdCommands    = static_cast<uint32_t>(dmaRdCommands.size());
    offset += sumCommandSize(dmaRdCommands);

    commandStream.DmaWrCommandsOffset = offset;
    commandStream.NumDmaWrCommands    = static_cast<uint32_t>(dmaWrCommands.size());
    offset += sumCommandSize(dmaWrCommands);

    commandStream.MceCommandsOffset = offset;
    commandStream.NumMceCommands    = static_cast<uint32_t>(mceCommands.size());
    offset += sumCommandSize(mceCommands);

    commandStream.PleCommandsOffset = offset;
    commandStream.NumPleCommands    = static_cast<uint32_t>(pleCommands.size());
    offset += sumCommandSize(pleCommands);

    commandStream.TotalSize = offset;

    // The command stream "header"
    EmplaceBack(raw, commandStream);

    // The agents array
    for (const Agent& agent : agents)
    {
        EmplaceBack(raw, agent);
    }

    // The four command lists
    auto appendCommandList = [&](const std::vector<CommandVariant>& commands) {
        for (uint32_t cmdIdx = 0; cmdIdx < commands.size(); ++cmdIdx)
        {
            const CommandVariant& c = commands[cmdIdx];
            // Convert to the concrete command type before appending to the command list,
            // otherwise only the base Command fields would be added.
            // Note that we don't add the CommandVariants themselves to the command stream itself,
            // because the union will take up as much space as the largest member, which in this
            // case is quite large (ProgramMceStripeCommand is way bigger than the others),
            // and so would waste command stream space.
            switch (c.type)
            {
                case CommandType::WaitForCounter:
                    EmplaceBack(raw, c.waitForCounter);
                    break;
                case CommandType::LoadIfmStripe:
                    EmplaceBack(raw, c.dma);
                    break;
                case CommandType::LoadWgtStripe:
                    EmplaceBack(raw, c.dma);
                    break;
                case CommandType::ProgramMceStripe:
                    EmplaceBack(raw, c.programMceStripe);
                    break;
                case CommandType::ConfigMceif:
                    EmplaceBack(raw, c.configMceif);
                    break;
                case CommandType::StartMceStripe:
                    EmplaceBack(raw, c.startMceStripe);
                    break;
                case CommandType::LoadPleCodeIntoSram:
                    EmplaceBack(raw, c.dma);
                    break;
                case CommandType::LoadPleCodeIntoPleSram:
                    EmplaceBack(raw, c.loadPleCodeIntoPleSram);
                    break;
                case CommandType::StartPleStripe:
                    EmplaceBack(raw, c.startPleStripe);
                    break;
                case CommandType::StoreOfmStripe:
                    EmplaceBack(raw, c.dma);
                    break;
                default:
                    throw std::runtime_error("Invalid command type: " + std::to_string(static_cast<uint32_t>(c.type)));
            }
        }
    };

    appendCommandList(dmaRdCommands);
    appendCommandList(dmaWrCommands);
    appendCommandList(mceCommands);
    appendCommandList(pleCommands);

    return raw;
}

}    // namespace command_stream
}    // namespace ethosn
