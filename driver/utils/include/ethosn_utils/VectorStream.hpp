//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <vector>

namespace ethosn
{
namespace utils
{

/// A C++ I/O stream which uses a std::vector as its backing store.
class VectorStream : public std::ostream
{
    class VectorStreamBuf : public std::streambuf
    {
    public:
        VectorStreamBuf(std::vector<char>& buffer)
            : m_Buffer(buffer)
        {}

        int_type overflow(int_type c) override
        {
            if (c != EOF)
            {
                m_Buffer.push_back(static_cast<char>(c));
            }
            return c;
        }

    private:
        std::vector<char>& m_Buffer;
    };

public:
    VectorStream(std::vector<char>& buffer)
        : std::ostream(&m_StreamBuf)
        , m_StreamBuf(buffer)
    {}

private:
    VectorStreamBuf m_StreamBuf;
};

}    // namespace utils
}    // namespace ethosn
