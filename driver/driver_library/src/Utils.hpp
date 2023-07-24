//
// Copyright © 2018-2020,2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <ethosn_utils/Log.hpp>
#include <ethosn_utils/Macros.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace ethosn
{
namespace driver_library
{

#if defined(ETHOSN_LOGGING)
constexpr ethosn::utils::log::Severity g_LogCompileTimeMaxSeverity = ethosn::utils::log::Severity::Debug;
#else
constexpr ethosn::utils::log::Severity g_LogCompileTimeMaxSeverity = ethosn::utils::log::Severity::Info;
#endif
using LoggerType = ethosn::utils::log::Logger<g_LogCompileTimeMaxSeverity>;
extern LoggerType g_Logger;

/// Returns the first argument rounded UP to the nearest multiple of the second argument
constexpr uint64_t RoundUpToNearestMultiple(uint64_t num, uint64_t nearestMultiple)
{
    uint64_t remainder = num % nearestMultiple;

    if (remainder == 0)
    {
        return num;
    }

    return num + nearestMultiple - remainder;
}

constexpr uint32_t g_BrickWidth          = 4;
constexpr uint32_t g_BrickHeight         = 4;
constexpr uint32_t g_BrickDepth          = 16;
constexpr uint32_t g_BrickCountInGroup   = 4;
constexpr uint32_t g_BrickGroupSizeBytes = g_BrickWidth * g_BrickHeight * g_BrickDepth * g_BrickCountInGroup;

/// Calculates the quotient of numerator and denominator as an integer where the result is rounded up to the nearest
/// integer. i.e. ceil(numerator/denominator).
constexpr uint32_t DivRoundUp(uint32_t numerator, uint32_t denominator)
{
    return (numerator + denominator - 1) / denominator;
}

inline uint32_t GetTotalSizeNhwcb(uint32_t w, uint32_t h, uint32_t c)
{
    return DivRoundUp(w, 8) * DivRoundUp(h, 8) * DivRoundUp(c, 16) * g_BrickGroupSizeBytes;
}

// Helper class to read data from a tightly-packed multidimensional array
template <typename T, uint32_t D>
class MultiDimensionalArray
{
public:
    MultiDimensionalArray(T* data, const std::array<uint32_t, D>& dims)
        : m_Data(data)
        , m_Dims(dims)
    {}

    T GetElement(const std::array<uint32_t, D>& indexes) const
    {
        return m_Data[GetOffset(indexes)];
    }

    void SetElement(const std::array<uint32_t, D>& indexes, T value) const
    {
        m_Data[GetOffset(indexes)] = value;
    }

    uint32_t GetDimSize(uint32_t dim) const
    {
        return m_Dims[dim];
    }

    uint32_t GetSize() const
    {
        return std::accumulate(m_Dims.begin(), m_Dims.end(), 1, std::multiplies<uint32_t>());
    }

private:
    uint32_t GetOffset(const std::array<uint32_t, D>& indexes) const
    {
        uint32_t offset  = 0;
        uint32_t product = 1;
        for (int32_t d = indexes.size() - 1; d >= 0; d--)
        {
            assert(indexes[d] < m_Dims[d]);
            offset += product * indexes[d];
            product *= m_Dims[d];
        }

        return offset;
    }

    T* m_Data;
    std::array<uint32_t, D> m_Dims;
};

inline void WriteHex(std::ostream& os, const uint32_t startAddr, const uint8_t* const data, const uint32_t numBytes)
{
    const std::ios_base::fmtflags flags = os.flags();

    os << std::hex << std::setfill('0');

    // Loop over rows (16-bytes each)
    for (uint32_t addr = startAddr, i = 0; i < numBytes; i += 16, addr += 16)
    {
        os << std::setw(8) << addr << ": ";

        // Loop over columns (4-bytes each)
        for (uint32_t j = 0; j < 4; ++j)
        {
            // Loop over bytes within the column.
            // Hex files are little-endian so we loop over the bytes in reverse order
            for (int32_t k = 3; k >= 0; --k)
            {
                const uint32_t b     = i + j * 4 + static_cast<uint32_t>(k);
                const uint32_t value = (b < numBytes) ? data[b] : 0;
                os << std::setw(2) << value;
            }
            if (j < 3)
            {
                os << " ";
            }
        }

        os << std::endl;
    }

    os.flags(flags);
}

inline bool FileExists(const char* const pathname)
{
    struct stat info;
    return (pathname != NULL) && (stat(pathname, &info) == 0) && ((info.st_mode & S_IFMT) == S_IFREG);
}

// TBufferInfo can be either a BufferInfo, an InputBufferInfo or an OutputBufferInfo.
template <typename TBufferInfo>
uint32_t GetLastAddressedMemory(const std::vector<TBufferInfo>& buffers)
{
    if (buffers.size() == 0)
    {
        return 0;
    }
    auto BufferSize = [](const TBufferInfo& buf) { return buf.m_Offset + buf.m_Size; };
    auto maxBuffer  = std::max_element(buffers.begin(), buffers.end(), [&](const TBufferInfo& a, const TBufferInfo& b) {
        return BufferSize(a) < BufferSize(b);
    });
    return BufferSize(*maxBuffer);
}

}    // namespace driver_library
}    // namespace ethosn
