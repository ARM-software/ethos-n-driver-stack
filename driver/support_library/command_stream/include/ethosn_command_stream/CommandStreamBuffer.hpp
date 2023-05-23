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

template <typename Word, Opcode O>
void EmplaceBack(std::vector<Word>& data, const CommandData<O>& c)
{
    EmplaceBack<Command<O>>(data, Command<O>{ c });
}

class CommandStreamBuffer
{
public:
    using ConstIterator = CommandStreamConstIterator;

    static constexpr size_t VersionHeaderSizeWords = 4;

    CommandStreamBuffer()
        : m_Data()
        , m_Count(0)
    {
        // Tag to identify the command stream data structure using "FourCC" style
        constexpr uint32_t fourcc = static_cast<uint32_t>('E') | (static_cast<uint32_t>('N') << 8) |
                                    (static_cast<uint32_t>('C') << 16) | (static_cast<uint32_t>('S') << 24);

        std::array<uint32_t, VersionHeaderSizeWords> header = { fourcc, ETHOSN_COMMAND_STREAM_VERSION_MAJOR,
                                                                ETHOSN_COMMAND_STREAM_VERSION_MINOR,
                                                                ETHOSN_COMMAND_STREAM_VERSION_PATCH };

        for (uint32_t w : header)
        {
            m_Data.push_back(w);
        }
    }

    template <typename T>
    void EmplaceBack(const T& cmd)
    {
        command_stream::EmplaceBack(m_Data, cmd);
        ++m_Count;
    }

    ConstIterator begin() const
    {
        return ConstIterator(reinterpret_cast<const CommandHeader*>(m_Data.data() + VersionHeaderSizeWords));
    }

    ConstIterator end() const
    {
        return ConstIterator(reinterpret_cast<const CommandHeader*>(m_Data.data() + m_Data.size()));
    }

    const std::vector<uint32_t>& GetData() const
    {
        return m_Data;
    }

    uint32_t GetCount() const
    {
        return m_Count;
    }

private:
    std::vector<uint32_t> m_Data;
    uint32_t m_Count;
};

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
    cascading::CommandType type;

    union
    {
        cascading::WaitForCounterCommand waitForCounter;
        cascading::DmaCommand dma;
        cascading::ProgramMceStripeCommand programMceStripe;
        cascading::ConfigMceifCommand configMceif;
        cascading::StartMceStripeCommand startMceStripe;
        cascading::LoadPleCodeIntoPleSramCommand loadPleCodeIntoPleSram;
        cascading::StartPleStripeCommand startPleStripe;
    };

    explicit CommandVariant(const cascading::WaitForCounterCommand& c)
        : type(c.type)
        , waitForCounter(c)
    {}
    explicit CommandVariant(const cascading::DmaCommand& c)
        : type(c.type)
        , dma(c)
    {}
    explicit CommandVariant(const cascading::ConfigMceifCommand& c)
        : type(c.type)
        , configMceif(c)
    {}
    explicit CommandVariant(const cascading::ProgramMceStripeCommand& c)
        : type(c.type)
        , programMceStripe(c)
    {}
    explicit CommandVariant(const cascading::StartMceStripeCommand& c)
        : type(c.type)
        , startMceStripe(c)
    {}
    explicit CommandVariant(const cascading::LoadPleCodeIntoPleSramCommand& c)
        : type(c.type)
        , loadPleCodeIntoPleSram(c)
    {}
    explicit CommandVariant(const cascading::StartPleStripeCommand& c)
        : type(c.type)
        , startPleStripe(c)
    {}

    /// All the Command subtypes inherit Command, and this provides a (safe) way to
    /// get access to that, without the caller having to know the actual Command subtype.
    const cascading::Command& AsBaseCommand() const
    {
        switch (type)
        {
            case cascading::CommandType::WaitForCounter:
                return waitForCounter;
            case cascading::CommandType::LoadIfmStripe:
                return dma;
            case cascading::CommandType::LoadWgtStripe:
                return dma;
            case cascading::CommandType::ProgramMceStripe:
                return programMceStripe;
            case cascading::CommandType::ConfigMceif:
                return configMceif;
            case cascading::CommandType::StartMceStripe:
                return startMceStripe;
            case cascading::CommandType::LoadPleCodeIntoSram:
                return dma;
            case cascading::CommandType::LoadPleCodeIntoPleSram:
                return loadPleCodeIntoPleSram;
            case cascading::CommandType::StartPleStripe:
                return startPleStripe;
            case cascading::CommandType::StoreOfmStripe:
                return dma;
            default:
                throw std::runtime_error("Invalid cascading command type: " +
                                         std::to_string(static_cast<uint32_t>(type)));
        }
    }
};

/// Adds a new command of type CASCADE to the given `cmdStream`. The CASCADE command will contain
/// all of the agents and commands provided.
inline void AddCascade(ethosn::command_stream::CommandStreamBuffer& cmdStream,
                       const std::vector<cascading::Agent>& agents,
                       const std::vector<CommandVariant>& dmaRdCommands,
                       const std::vector<CommandVariant>& dmaWrCommands,
                       const std::vector<CommandVariant>& mceCommands,
                       const std::vector<CommandVariant>& pleCommands)
{
    using namespace ethosn::command_stream::cascading;

    command_stream::Cascade cascade;
    uint32_t offset = sizeof(command_stream::Cascade);

    cascade.AgentsOffset = offset;
    cascade.NumAgents    = static_cast<uint32_t>(agents.size());
    offset += cascade.NumAgents * static_cast<uint32_t>(sizeof(Agent));

    // Helper function to sum up the size of all commands in a list (as each Command may be
    // a different type, and so have a different size).
    auto sumCommandSize = [](const std::vector<CommandVariant>& commands) {
        return std::accumulate(commands.begin(), commands.end(), 0u, [](uint32_t sum, const auto& cmd) {
            return sum + static_cast<uint32_t>(cmd.AsBaseCommand().GetSize());
        });
    };

    cascade.DmaRdCommandsOffset = offset;
    cascade.NumDmaRdCommands    = static_cast<uint32_t>(dmaRdCommands.size());
    offset += sumCommandSize(dmaRdCommands);

    cascade.DmaWrCommandsOffset = offset;
    cascade.NumDmaWrCommands    = static_cast<uint32_t>(dmaWrCommands.size());
    offset += sumCommandSize(dmaWrCommands);

    cascade.MceCommandsOffset = offset;
    cascade.NumMceCommands    = static_cast<uint32_t>(mceCommands.size());
    offset += sumCommandSize(mceCommands);

    cascade.PleCommandsOffset = offset;
    cascade.NumPleCommands    = static_cast<uint32_t>(pleCommands.size());
    offset += sumCommandSize(pleCommands);

    cascade.TotalSize = offset;

    // The cascade command "header"
    cmdStream.EmplaceBack(cascade);

    // The agents array
    for (const Agent& agent : agents)
    {
        cmdStream.EmplaceBack<Agent>(agent);
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
                    cmdStream.EmplaceBack(c.waitForCounter);
                    break;
                case CommandType::LoadIfmStripe:
                    cmdStream.EmplaceBack(c.dma);
                    break;
                case CommandType::LoadWgtStripe:
                    cmdStream.EmplaceBack(c.dma);
                    break;
                case CommandType::ProgramMceStripe:
                    cmdStream.EmplaceBack(c.programMceStripe);
                    break;
                case CommandType::ConfigMceif:
                    cmdStream.EmplaceBack(c.configMceif);
                    break;
                case CommandType::StartMceStripe:
                    cmdStream.EmplaceBack(c.startMceStripe);
                    break;
                case CommandType::LoadPleCodeIntoSram:
                    cmdStream.EmplaceBack(c.dma);
                    break;
                case CommandType::LoadPleCodeIntoPleSram:
                    cmdStream.EmplaceBack(c.loadPleCodeIntoPleSram);
                    break;
                case CommandType::StartPleStripe:
                    cmdStream.EmplaceBack(c.startPleStripe);
                    break;
                case CommandType::StoreOfmStripe:
                    cmdStream.EmplaceBack(c.dma);
                    break;
                default:
                    throw std::runtime_error("Invalid cascading command type: " +
                                             std::to_string(static_cast<uint32_t>(c.type)));
            }
        }
    };

    appendCommandList(dmaRdCommands);
    appendCommandList(dmaWrCommands);
    appendCommandList(mceCommands);
    appendCommandList(pleCommands);
}

}    // namespace command_stream
}    // namespace ethosn
