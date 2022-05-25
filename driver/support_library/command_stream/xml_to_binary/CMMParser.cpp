//
// Copyright © 2019-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CMMParser.hpp"

#include "BinaryParser.hpp"

#include <ethosn_command_stream/CommandStream.hpp>
#include <ethosn_firmware.h>

#include <cassert>
#include <inttypes.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

std::vector<uint32_t> GetBinaryDataFromHexFile(std::istream& input, uint32_t startAddress, uint32_t length)
{
    input.clear();
    input.seekg(0);
    std::vector<uint32_t> out;
    assert(startAddress % 4 == 0);
    assert(length % 4 == 0);
    out.reserve(length / 4);
    uint32_t endAddress = startAddress + length;

    // Get the addresses of the lines which contains the start and end addresses
    uint32_t startLine = startAddress & ~uint32_t{ 16 - 1 };
    uint32_t endLine   = endAddress & ~uint32_t{ 16 - 1 };

    for (std::string line; std::getline(input, line);)
    {
        uint32_t addr;
        std::array<uint32_t, 4> words;
        // Format of Combined Memory Map hex file lines
        const char* formatString = "%" SCNx32 ": %8" SCNx32 " %8" SCNx32 " %8" SCNx32 " %8" SCNx32;
        if (sscanf(line.c_str(), formatString, &addr, &words[0], &words[1], &words[2], &words[3]) != 5)
        {
            throw ParseException("Unable to parse data field in Memory Map file");
        }
        if (addr < startLine)
        {
            continue;
        }
        if (addr > endLine)
        {
            break;
        }
        for (uint32_t i = 0; i < 4; ++i)
        {
            uint32_t currentAddr = addr + i * static_cast<uint32_t>(sizeof(uint32_t));
            if (currentAddr < startAddress || currentAddr >= endAddress)
            {
                continue;
            }
            out.push_back(words[i]);
        }
    }
    return out;
}

template <typename T>
T GetBinaryDataFromHexFile(std::istream& input, uint32_t offset)
{
    std::vector<uint32_t> data = GetBinaryDataFromHexFile(input, offset, sizeof(T));
    return *reinterpret_cast<T*>(data.data());
}

}    // namespace

CMMParser::CMMParser(std::istream& input)
    : m_Input(input)
{}

// Skip to the mailbox address in the input stream and return the inference address
uint32_t CMMParser::GetInferenceAddress(std::istream& input)
{
    const uint32_t mailboxAddress = 0x60000000;    // Default hardcoded value
    uint32_t inferenceAddress     = 0;
    std::string line;

    while (std::getline(input, line))
    {
        uint32_t addr;
        if (sscanf(line.c_str(), "%8x:", &addr) != 1)
        {
            throw ParseException("Unable to parse address field in Memory Map file");
        }
        // Only begin parsing when we hit the mailbox address
        if (addr == mailboxAddress)
        {
            // The first uint32_t at mailbox address is the inference address
            if (sscanf(line.c_str(), "%8x: %8x", &addr, &inferenceAddress) != 2)
            {
                throw ParseException("Unable to parse data field in Memory Map file");
            }
            break;
        }
    }
    return inferenceAddress;
}

// Extract the Command Stream from a Combined Memory Map
void CMMParser::ExtractCSFromCMM(std::ostream& output, bool doXmlToBinary)
{
    uint32_t inferenceAddress = GetInferenceAddress(m_Input);

    ethosn_buffer_desc bufferInfo = GetBinaryDataFromHexFile<ethosn_buffer_desc>(
        m_Input, inferenceAddress + static_cast<uint32_t>(sizeof(ethosn_buffer_array)));

    std::vector<uint32_t> data =
        GetBinaryDataFromHexFile(m_Input, static_cast<uint32_t>(bufferInfo.address), bufferInfo.size);

    if (data.empty())
    {
        throw ParseException("Could not extract command stream from combined memory map");
    }
    if (doXmlToBinary)
    {
        output.write(reinterpret_cast<const char*>(data.data()), sizeof(data[0]) * data.size());
        if (!output.good())
        {
            throw IOException("IO error on binary write");
        }
    }
    else
    {
        std::stringstream tmp;
        tmp.write(reinterpret_cast<const char*>(data.data()), sizeof(data[0]) * data.size());
        tmp.seekg(0);
        BinaryParser(tmp).WriteXml(output);
    }
}

// Extract the Binding Table from a Combined Memory Map
void CMMParser::ExtractBTFromCMM(std::ostream& output)
{
    uint32_t inferenceAddress = GetInferenceAddress(m_Input);

    ethosn_buffer_array header = GetBinaryDataFromHexFile<ethosn_buffer_array>(m_Input, inferenceAddress);

    // Output data as XML
    output << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;
    output << "<BIND>" << std::endl;
    for (uint32_t i = 0; i < header.num_buffers; ++i)
    {
        ethosn_buffer_desc bufferInfo = GetBinaryDataFromHexFile<ethosn_buffer_desc>(
            m_Input, inferenceAddress + static_cast<uint32_t>(sizeof(ethosn_buffer_array)) +
                         i * static_cast<uint32_t>(sizeof(ethosn_buffer_desc)));

        std::string tType = "";
        switch (bufferInfo.type)
        {
            case ETHOSN_BUFFER_INPUT:
            {
                tType = "INPUT";
                break;
            }
            case ETHOSN_BUFFER_INTERMEDIATE:
            {
                tType = "INTERMEDIATE";
                break;
            }
            case ETHOSN_BUFFER_OUTPUT:
            {
                tType = "OUTPUT";
                break;
            }
            case ETHOSN_BUFFER_CONSTANT:
            {
                tType = "CONSTANT";
                break;
            }
            case ETHOSN_BUFFER_CMD_FW:
            {
                tType = "CMD_FW";
                break;
            }
            default:
            {
                tType = "UNKNOWN";
                break;
            }
        }
        output << "  <BUFFER>" << std::endl;
        output << "    <ID>" << i << "</ID>" << std::endl;
        output << "    <ADDRESS>0x" << std::hex << bufferInfo.address << "</ADDRESS>" << std::endl;
        output << "    <SIZE>" << std::dec << bufferInfo.size << "</SIZE>" << std::endl;
        output << "    <TYPE>" << tType << "</TYPE>" << std::endl;
        output << "  </BUFFER>" << std::endl;
    }
    output << "</BIND>" << std::endl;
}
