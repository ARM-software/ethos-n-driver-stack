//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "../src/Utils.hpp"
#include "../src/cascading/Part.hpp"
#include "../src/cascading/Plan.hpp"

#include <ethosn_command_stream/CommandStreamBuffer.hpp>

namespace ethosn
{
namespace support_library
{

HardwareCapabilities GetEthosN78HwCapabilities();
HardwareCapabilities
    GetHwCapabilitiesWithFwOverrides(EthosNVariant variant,
                                     utils::Optional<uint32_t> sramSizeOverride,
                                     utils::Optional<uint32_t> ctrlAgentWindowSizeOverride,
                                     utils::Optional<uint32_t> maxMceStripesPerPleStripeOverride,
                                     utils::Optional<uint32_t> maxIfmAndWgtStripesPerPleStripeOverride);
HardwareCapabilities GetEthosN78HwCapabilities(EthosNVariant variant, uint32_t sramSizeOverride = 0);

std::vector<char> GetRawDefaultCapabilities();
std::vector<char> GetRawDefaultEthosN78Capabilities();
std::vector<char> GetRawEthosN78Capabilities(EthosNVariant variant, uint32_t sramSizeOverride = 0);
std::vector<char> GetRawCapabilitiesWithFwOverrides(EthosNVariant variant,
                                                    utils::Optional<uint32_t> sramSizeOverride,
                                                    utils::Optional<uint32_t> ctrlAgentWindowSizeOverride,
                                                    utils::Optional<uint32_t> maxMceStripesPerPleStripeOverride,
                                                    utils::Optional<uint32_t> maxIfmAndWgtStripesPerPleStripeOverride);

bool Contains(const char* string, const char* substring);

std::vector<uint8_t> GetCommandStreamData(const ethosn::command_stream::CommandStreamBuffer& cmdStream);

ethosn::command_stream::CommandStream GetCommandStream(const CompiledNetwork* compiledNetwork);

class MockPart : public BasePart
{
public:
    MockPart(PartId id, bool hasInput = true, bool hasOutput = true)
        : BasePart(id, "MockPart", estOpt, compOpt, GetEthosN78HwCapabilities())
        , m_HasInput(hasInput)
        , m_HasOutput(hasOutput)
    {}
    virtual Plans GetPlans(CascadeType, ethosn::command_stream::BlockConfig, Buffer*, uint32_t) const override;

    virtual utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const override
    {
        return {};
    }

protected:
    bool m_HasInput;
    bool m_HasOutput;

private:
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
};

/// Simple Node type for tests.
/// Includes a friendly name and ignores shape, quantisation info etc. so that tests
/// can focus on graph topology.
class NameOnlyNode : public Node
{
public:
    NameOnlyNode(NodeId id, std::string name)
        : Node(id,
               TensorShape(),
               DataType::UINT8_QUANTIZED,
               QuantizationInfo(),
               CompilerDataFormat::NONE,
               std::set<uint32_t>{ 0 })
        , m_Name(name)
    {}

    DotAttributes GetDotAttributes() override
    {
        return DotAttributes(std::to_string(m_Id), m_Name, "");
    }

    bool IsPrepared() override
    {
        return false;
    }

    NodeType GetNodeType() override
    {
        return NodeType::NameOnlyNode;
    }

    std::string m_Name;
};

bool IsEstimateOnlyOp(const Op* const op);
bool IsMceOp(const Op* const op);
bool IsPleOp(const Op* const op);

}    // namespace support_library
}    // namespace ethosn
