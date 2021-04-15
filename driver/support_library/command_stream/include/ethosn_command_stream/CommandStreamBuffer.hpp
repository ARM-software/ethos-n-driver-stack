//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "CommandStream.hpp"

#include <vector>

namespace ethosn
{
namespace command_stream
{

template <typename T, typename Word, typename... Us>
void EmplaceBack(std::vector<Word>& data, Us&&... us)
{
    static_assert(alignof(T) <= alignof(Word), "Must not have a stronger alignment requirement");
    static_assert((sizeof(T) % sizeof(Word)) == 0, "Size must be a multiple of the word size");

    const size_t prevSize = data.size();
    data.resize(data.size() + (sizeof(T) / sizeof(Word)));
    new (&data[prevSize]) T(std::forward<Us>(us)...);
}

template <typename T, typename Word>
void EmplaceBack(std::vector<Word>& data, const T& t)
{
    EmplaceBack<T, Word, const T&>(data, t);
}

template <typename Word>
void EmplaceBackCommands(std::vector<Word>&)
{}

template <typename Word, Opcode O, Opcode... Os>
void EmplaceBackCommands(std::vector<Word>& data, const CommandData<O>& c, const CommandData<Os>&... cs)
{
    EmplaceBack<Command<O>>(data, c);
    EmplaceBackCommands(data, cs...);
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

    template <Opcode O>
    void EmplaceBack(const CommandData<O>& comData)
    {
        EmplaceBackCommands(m_Data, comData);
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

}    // namespace command_stream
}    // namespace ethosn
