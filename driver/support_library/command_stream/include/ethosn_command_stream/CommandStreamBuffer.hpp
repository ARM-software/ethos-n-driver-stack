//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "CommandStream.hpp"

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

/// Adds a new command of type CASCADE to the given `cmdStream`. The CASCADE command will contain
/// all of the agents, commands and extra data provided. The extra data is automatically associated
/// with commands based on the type of the commands, and is assumed to be in the same order.
inline void AddCascade(ethosn::command_stream::CommandStreamBuffer& cmdStream,
                       const std::vector<cascading::Agent>& agents,
                       const std::vector<cascading::Command>& dmaRdCommands,
                       const std::vector<cascading::Command>& dmaWrCommands,
                       const std::vector<cascading::Command>& mceCommands,
                       const std::vector<cascading::Command>& pleCommands,
                       const std::vector<cascading::DmaExtraData>& dmaExtraData,
                       const std::vector<cascading::ProgramMceExtraData>& programMceExtraData,
                       const std::vector<cascading::StartMceExtraData>& startMceExtraData,
                       const std::vector<cascading::StartPleExtraData>& startPleExtraData)
{
    using namespace ethosn::command_stream::cascading;
    using Command = ethosn::command_stream::cascading::Command;

    command_stream::Cascade cascade;
    uint32_t offset = sizeof(command_stream::Cascade);

    cascade.AgentsOffset = offset;
    cascade.NumAgents    = static_cast<uint32_t>(agents.size());
    offset += cascade.NumAgents * static_cast<uint32_t>(sizeof(Agent));

    cascade.DmaRdCommandsOffset = offset;
    cascade.NumDmaRdCommands    = static_cast<uint32_t>(dmaRdCommands.size());
    offset += cascade.NumDmaRdCommands * static_cast<uint32_t>(sizeof(Command));

    cascade.DmaWrCommandsOffset = offset;
    cascade.NumDmaWrCommands    = static_cast<uint32_t>(dmaWrCommands.size());
    offset += cascade.NumDmaWrCommands * static_cast<uint32_t>(sizeof(Command));

    cascade.MceCommandsOffset = offset;
    cascade.NumMceCommands    = static_cast<uint32_t>(mceCommands.size());
    offset += cascade.NumMceCommands * static_cast<uint32_t>(sizeof(Command));

    cascade.PleCommandsOffset = offset;
    cascade.NumPleCommands    = static_cast<uint32_t>(pleCommands.size());
    offset += cascade.NumPleCommands * static_cast<uint32_t>(sizeof(Command));

    cascade.DmaExtraDataOffset = offset;
    cascade.NumDmaExtraData    = static_cast<uint32_t>(dmaExtraData.size());
    offset += cascade.NumDmaExtraData * static_cast<uint32_t>(sizeof(DmaExtraData));

    cascade.ProgramMceExtraDataOffset = offset;
    cascade.NumProgramMceExtraData    = static_cast<uint32_t>(programMceExtraData.size());
    offset += cascade.NumProgramMceExtraData * static_cast<uint32_t>(sizeof(ProgramMceExtraData));

    cascade.StartMceExtraDataOffset = offset;
    cascade.NumStartMceExtraData    = static_cast<uint32_t>(startMceExtraData.size());
    offset += cascade.NumStartMceExtraData * static_cast<uint32_t>(sizeof(StartMceExtraData));

    cascade.StartPleExtraDataOffset = offset;
    cascade.NumStartPleExtraData    = static_cast<uint32_t>(startPleExtraData.size());
    offset += cascade.NumStartPleExtraData * static_cast<uint32_t>(sizeof(StartPleExtraData));

    cascade.TotalSize = offset;

    size_t dmaExtraDataOffset        = cascade.DmaExtraDataOffset;
    size_t programMceExtraDataOffset = cascade.ProgramMceExtraDataOffset;
    size_t startMceExtraDataOffset   = cascade.StartMceExtraDataOffset;
    size_t startPleExtraDataOffset   = cascade.StartPleExtraDataOffset;

    // The cascade command "header"
    cmdStream.EmplaceBack(cascade);

    // The agents array
    for (const Agent& agent : agents)
    {
        cmdStream.EmplaceBack<Agent>(agent);
    }

    // The four command arrays
    for (uint32_t cmdIdx = 0; cmdIdx < dmaRdCommands.size(); ++cmdIdx)
    {
        Command c = dmaRdCommands[cmdIdx];
        // Set offset to extra data (if appropriate), assuming that everything is in the same order
        if (c.type == CommandType::LoadIfmStripe || c.type == CommandType::LoadWgtStripe ||
            c.type == CommandType::LoadPleCode)
        {
            c.extraDataOffset =
                static_cast<uint32_t>(dmaExtraDataOffset - (cascade.DmaRdCommandsOffset + cmdIdx * sizeof(Command)));
            dmaExtraDataOffset += sizeof(DmaExtraData);
        }
        cmdStream.EmplaceBack(c);
    }
    for (uint32_t cmdIdx = 0; cmdIdx < dmaWrCommands.size(); ++cmdIdx)
    {
        Command c = dmaWrCommands[cmdIdx];
        // Set offset to extra data (if appropriate), assuming that everything is in the same order
        if (c.type == CommandType::StoreOfmStripe)
        {
            c.extraDataOffset =
                static_cast<uint32_t>(dmaExtraDataOffset - (cascade.DmaWrCommandsOffset + cmdIdx * sizeof(Command)));
            dmaExtraDataOffset += sizeof(DmaExtraData);
        }
        cmdStream.EmplaceBack(c);
    }
    for (uint32_t cmdIdx = 0; cmdIdx < mceCommands.size(); ++cmdIdx)
    {
        Command c = mceCommands[cmdIdx];
        // Set offset to extra data (if appropriate), assuming that everything is in the same order
        if (c.type == CommandType::ProgramMceStripe)
        {
            c.extraDataOffset = static_cast<uint32_t>(programMceExtraDataOffset -
                                                      (cascade.MceCommandsOffset + cmdIdx * sizeof(Command)));
            programMceExtraDataOffset += sizeof(ProgramMceExtraData);
        }
        else if (c.type == CommandType::StartMceStripe)
        {
            c.extraDataOffset =
                static_cast<uint32_t>(startMceExtraDataOffset - (cascade.MceCommandsOffset + cmdIdx * sizeof(Command)));
            startMceExtraDataOffset += sizeof(StartMceExtraData);
        }
        cmdStream.EmplaceBack(c);
    }
    for (uint32_t cmdIdx = 0; cmdIdx < pleCommands.size(); ++cmdIdx)
    {
        Command c = pleCommands[cmdIdx];
        // Set offset to extra data (if appropriate), assuming that everything is in the same order
        if (c.type == CommandType::StartPleStripe)
        {
            c.extraDataOffset =
                static_cast<uint32_t>(startPleExtraDataOffset - (cascade.PleCommandsOffset + cmdIdx * sizeof(Command)));
            startPleExtraDataOffset += sizeof(StartPleExtraData);
        }
        cmdStream.EmplaceBack(c);
    }

    // The four extra data arrays
    for (const DmaExtraData& d : dmaExtraData)
    {
        cmdStream.EmplaceBack(d);
    }
    for (const ProgramMceExtraData& d : programMceExtraData)
    {
        cmdStream.EmplaceBack(d);
    }
    for (const StartMceExtraData& d : startMceExtraData)
    {
        cmdStream.EmplaceBack(d);
    }
    for (const StartPleExtraData& d : startPleExtraData)
    {
        cmdStream.EmplaceBack(d);
    }
}

}    // namespace command_stream
}    // namespace ethosn
