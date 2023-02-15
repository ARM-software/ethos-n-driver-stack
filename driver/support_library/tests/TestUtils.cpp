//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "TestUtils.hpp"

#include "../src/CapabilitiesInternal.hpp"
#include "../src/Compiler.hpp"
#include "../src/cascading/Plan.hpp"

#include <cstring>
#include <typeinfo>

namespace ethosn
{
namespace support_library
{

ethosn::support_library::Plans
    MockPart::GetPlans(CascadeType, ethosn::command_stream::BlockConfig, Buffer*, uint32_t) const
{
    Plans plans;

    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;

    OwnedOpGraph opGraph;

    if (m_HasInput)
    {
        DramBuffer* buffer         = opGraph.AddBuffer(std::make_unique<DramBuffer>());
        buffer->m_Format           = CascadingBufferFormat::NHWCB;
        buffer->m_TensorShape      = { 1, 16, 16, 16 };
        buffer->m_SizeInBytes      = 16 * 16 * 16;
        buffer->m_QuantizationInfo = { 0, 1.f };

        inputMappings[buffer] = PartInputSlot{ m_PartId, 0 };
    }
    if (m_HasOutput)
    {
        DramBuffer* buffer         = opGraph.AddBuffer(std::make_unique<DramBuffer>());
        buffer->m_Format           = CascadingBufferFormat::NHWCB;
        buffer->m_TensorShape      = { 1, 16, 16, 16 };
        buffer->m_SizeInBytes      = 16 * 16 * 16;
        buffer->m_QuantizationInfo = { 0, 1.f };

        outputMappings[buffer] = PartOutputSlot{ m_PartId, 0 };
    }
    if (m_HasInput && m_HasOutput)
    {
        opGraph.AddOp(std::make_unique<PleOp>(ethosn::command_stream::PleOperation::PASSTHROUGH,
                                              ethosn::command_stream::BlockConfig{ 8u, 8u }, 1,
                                              std::vector<TensorShape>{ TensorShape{ 1, 16, 16, 16 } },
                                              TensorShape{ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, true));

        opGraph.AddConsumer(opGraph.GetBuffers().front(), opGraph.GetOps()[0], 0);
        opGraph.SetProducer(opGraph.GetBuffers().back(), opGraph.GetOps()[0]);
    }

    Plan plan(std::move(inputMappings), std::move(outputMappings));
    plan.m_OpGraph = std::move(opGraph);
    plans.push_back(std::move(plan));

    return plans;
}

HardwareCapabilities GetEthosN78HwCapabilities()
{
    FirmwareAndHardwareCapabilities fwHwCapabilities =
        GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, 0);
    return HardwareCapabilities(fwHwCapabilities);
}

namespace
{

FirmwareAndHardwareCapabilities
    GetFwHwCapabilitiesWithFwOverrides(EthosNVariant variant,
                                       utils::Optional<uint32_t> sramSizeOverride,
                                       utils::Optional<uint32_t> ctrlAgentWindowSizeOverride,
                                       utils::Optional<uint32_t> maxMceStripesPerPleStripeOverride,
                                       utils::Optional<uint32_t> maxIfmAndWgtStripesPerPleStripeOverride)
{
    FirmwareAndHardwareCapabilities fwHwCapabilities =
        GetEthosN78FwHwCapabilities(variant, sramSizeOverride.has_value() ? sramSizeOverride.value() : 0);
    if (ctrlAgentWindowSizeOverride.has_value())
    {
        fwHwCapabilities.m_AgentWindowSize = ctrlAgentWindowSizeOverride.value();
    }
    if (maxMceStripesPerPleStripeOverride.has_value())
    {
        fwHwCapabilities.m_MaxMceStripesPerPleStripe = maxMceStripesPerPleStripeOverride.value();
    }
    if (maxIfmAndWgtStripesPerPleStripeOverride.has_value())
    {
        fwHwCapabilities.m_MaxIfmAndWgtStripesPerPleStripe = maxIfmAndWgtStripesPerPleStripeOverride.value();
    }
    return fwHwCapabilities;
}

}    // namespace

HardwareCapabilities GetHwCapabilitiesWithFwOverrides(EthosNVariant variant,
                                                      utils::Optional<uint32_t> sramSizeOverride,
                                                      utils::Optional<uint32_t> ctrlAgentWindowSizeOverride,
                                                      utils::Optional<uint32_t> maxMceStripesPerPleStripeOverride,
                                                      utils::Optional<uint32_t> maxIfmAndWgtStripesPerPleStripeOverride)
{
    FirmwareAndHardwareCapabilities fwHwCapabilities =
        GetFwHwCapabilitiesWithFwOverrides(variant, sramSizeOverride, ctrlAgentWindowSizeOverride,
                                           maxMceStripesPerPleStripeOverride, maxIfmAndWgtStripesPerPleStripeOverride);
    return HardwareCapabilities(fwHwCapabilities);
}

HardwareCapabilities GetEthosN78HwCapabilities(EthosNVariant variant, uint32_t sramSizeOverride)
{
    FirmwareAndHardwareCapabilities fwHwCapabilities = GetEthosN78FwHwCapabilities(variant, sramSizeOverride);
    return HardwareCapabilities(fwHwCapabilities);
}

namespace
{

std::vector<char> GetRawCapabilities(const FirmwareAndHardwareCapabilities& fwHwCapabilities)
{
    return std::vector<char>(reinterpret_cast<const char*>(&fwHwCapabilities),
                             reinterpret_cast<const char*>(&fwHwCapabilities) + sizeof(fwHwCapabilities));
}

}    // namespace

std::vector<char> GetRawDefaultCapabilities()
{
    FirmwareAndHardwareCapabilities fwHwCapabilities =
        GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 0);
    return GetRawCapabilities(fwHwCapabilities);
}

std::vector<char> GetRawDefaultEthosN78Capabilities()
{
    FirmwareAndHardwareCapabilities fwHwCapabilities =
        GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, 0);
    return GetRawCapabilities(fwHwCapabilities);
}

std::vector<char> GetRawEthosN78Capabilities(EthosNVariant variant, uint32_t sramSizeOverride)
{
    FirmwareAndHardwareCapabilities fwHwCapabilities = GetEthosN78FwHwCapabilities(variant, sramSizeOverride);
    return GetRawCapabilities(fwHwCapabilities);
}

std::vector<char> GetRawCapabilitiesWithFwOverrides(EthosNVariant variant,
                                                    utils::Optional<uint32_t> sramSizeOverride,
                                                    utils::Optional<uint32_t> ctrlAgentWindowSizeOverride,
                                                    utils::Optional<uint32_t> maxMceStripesPerPleStripeOverride,
                                                    utils::Optional<uint32_t> maxIfmAndWgtStripesPerPleStripeOverride)
{
    FirmwareAndHardwareCapabilities fwHwCapabilities =
        GetFwHwCapabilitiesWithFwOverrides(variant, sramSizeOverride, ctrlAgentWindowSizeOverride,
                                           maxMceStripesPerPleStripeOverride, maxIfmAndWgtStripesPerPleStripeOverride);
    return GetRawCapabilities(fwHwCapabilities);
}

bool Contains(const char* string, const char* substring)
{
    return strstr(string, substring) != nullptr;
}

std::vector<uint8_t> GetCommandStreamData(const ethosn::command_stream::CommandStreamBuffer& cmdStream)
{
    std::vector<uint8_t> data;
    const uint8_t* begin = reinterpret_cast<const uint8_t*>(cmdStream.GetData().data());
    const uint8_t* end   = begin + cmdStream.GetData().size() * sizeof(cmdStream.GetData()[0]);
    data.assign(begin, end);
    return data;
}

ethosn::command_stream::CommandStream GetCommandStream(const CompiledNetwork* compiledNetwork)
{
    const CompiledNetworkImpl* cnImpl = static_cast<const CompiledNetworkImpl*>(compiledNetwork);
    auto& cuBufferInfo                = cnImpl->GetConstantControlUnitDataBufferInfos();
    // The command stream buffer id is defined to be 0.
    auto cmdStreamBufferInfo =
        std::find_if(cuBufferInfo.begin(), cuBufferInfo.end(), [](const auto& b) { return b.m_Id == 0; });
    if (cmdStreamBufferInfo == cuBufferInfo.end())
    {
        throw std::exception();
    }

    const uint32_t* begin =
        reinterpret_cast<const uint32_t*>(cnImpl->GetConstantControlUnitData().data() + cmdStreamBufferInfo->m_Offset);
    const uint32_t* end = begin + cmdStreamBufferInfo->m_Size / sizeof(uint32_t);
    return ethosn::command_stream::CommandStream(begin, end);
}

bool IsEstimateOnlyOp(const Op* const op)
{
    return typeid(*op) == typeid(EstimateOnlyOp);
}

bool IsMceOp(const Op* const op)
{
    return typeid(*op) == typeid(MceOp);
}

bool IsPleOp(const Op* const op)
{
    return typeid(*op) == typeid(PleOp);
}

}    // namespace support_library
}    // namespace ethosn
