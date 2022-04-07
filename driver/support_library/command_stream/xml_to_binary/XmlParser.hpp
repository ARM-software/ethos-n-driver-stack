//
// Copyright © 2018-2022 Arm Limited.
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

    template <typename T>
    void ParseKnownAgent();

    void ParseAgent();

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

    void Pop(ethosn::command_stream::Cascade& value);

    void Pop(ethosn::command_stream::DumpDram& value);
    void Pop(ethosn::command_stream::DumpSram& value);

    void Pop(const std::string& key, bool& value);
    void Pop(const std::string& key, uint8_t& value);
    void Pop(const std::string& key, int8_t& value);
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

    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::Dependency& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::Tile& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::BlockSize& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::MceSWorkSize<uint16_t>& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::StrideXy<uint8_t>& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::ReluActivation& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::FilterShape& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::Padding& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::IfmDelta& ifmDelta);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::IfmStripeShape& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::WgtSWorkSize<uint16_t>& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::PleIfmInfo& value);
    template <typename T>
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::TensorSize<T>& value);
    template <typename T>
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::SupertensorSize<T>& value);
    void Pop(const std::string& key, ethosn::command_stream::cascading::MceOperation& value);
    void Pop(const std::string& key, ethosn::command_stream::cascading::MceAlgorithm& value);
    void Pop(const std::string& key, ethosn::command_stream::cascading::PleInputMode& value);
    void Pop(const std::string& key, ethosn::command_stream::cascading::PleKernelId& value);
    void Pop(const std::string& key, ethosn::command_stream::cascading::FmsDataType& value);
    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::FcafInfo& value);

    void Pop(const std::string& keyPrefix, ethosn::command_stream::cascading::FmSData& value);

    void Pop(ethosn::command_stream::cascading::AgentDependencyInfo& info);
    void Pop(ethosn::command_stream::cascading::IfmS& value);
    void Pop(ethosn::command_stream::cascading::OfmS& value);
    void Pop(ethosn::command_stream::cascading::WgtS& value);
    void Pop(ethosn::command_stream::cascading::MceS& value);
    void Pop(ethosn::command_stream::cascading::PleL& value);
    void Pop(ethosn::command_stream::cascading::PleS& value);

    XmlData m_XmlData;

    ethosn::command_stream::CommandStreamBuffer m_CSBuffer;
    /// Cascading agents which have been parsed but not added to the command
    /// stream yet.
    std::vector<ethosn::command_stream::cascading::Agent> m_PendingAgents;
};
