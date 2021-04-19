//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Common.hpp"

#include <ethosn_command_stream/CommandStreamBuffer.hpp>

#include <cstdio>
#include <queue>
#include <string>
#include <unordered_map>

class XmlParser
{
public:
    using XmlData = std::unordered_map<std::string, std::queue<std::string>>;

    XmlParser(std::istream& input);

    void WriteBinary(std::ostream& output);

    const XmlData& GetXmlData() const
    {
        return m_XmlData;
    }

    const ethosn::command_stream::CommandStreamBuffer& GetCommandStreamBuffer() const
    {
        return m_CSBuffer;
    }

private:
    static void SaxCallback(mxml_node_t* node, mxml_sax_event_t event, void* parserAsVoid);

    template <ethosn::command_stream::Opcode O>
    void Parse();

    void Push(std::string key, std::string value);
    std::string Pop(const std::string& key);

    void Pop(ethosn::command_stream::McePle& value);

    void Pop(ethosn::command_stream::PleOnly& value);

    void Pop(ethosn::command_stream::Softmax& value);

    void Pop(ethosn::command_stream::Convert& value);

    void Pop(ethosn::command_stream::SpaceToDepth& value);

    void Pop(const std::string& keyPrefix, ethosn::command_stream::TensorInfo& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::SramConfig& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::BlockConfig& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::MceOperation& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::MceAlgorithm& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::DataLocation& value);

    void Pop(ethosn::command_stream::MceData& value);

    void Pop(ethosn::command_stream::PleData& value);
    void Pop(ethosn::command_stream::Section& value);

    void Pop(ethosn::command_stream::Delay& value);

    void Pop(ethosn::command_stream::DumpDram& value);
    void Pop(ethosn::command_stream::DumpSram& value);

    void Pop(const std::string& key, bool& value);
    void Pop(const std::string& key, uint8_t& value);
    void Pop(const std::string& key, uint16_t& value);
    void Pop(const std::string& key, int16_t& value);
    void Pop(const std::string& key, int32_t& value);
    void Pop(const std::string& key, uint32_t& value);

    void Pop(const std::string& key, ethosn::command_stream::TensorShape& value);
    void Pop(const std::string& key, ethosn::command_stream::Filename& value);
    void Pop(const std::string& key, ethosn::command_stream::DataType& value);
    void Pop(const std::string& key, ethosn::command_stream::DataFormat& value);
    void Pop(const std::string& key, ethosn::command_stream::SramAllocationStrategy& value);
    void Pop(const std::string& key, ethosn::command_stream::UpsampleType& value);
    void Pop(const std::string& key, ethosn::command_stream::SectionType& value);
    void Pop(const std::string& key, ethosn::command_stream::PleOperation& value);

    XmlData m_XmlData;

    ethosn::command_stream::CommandStreamBuffer m_CSBuffer;
};
