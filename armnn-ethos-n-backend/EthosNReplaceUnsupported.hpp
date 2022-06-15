//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <armnn/backends/SubgraphView.hpp>

namespace armnn
{

struct EthosNConfig;

namespace ethosnbackend
{

bool ReplaceConstantMultiplicationWithDepthwise(SubgraphView& graph,
                                                IConnectableLayer* layer,
                                                const EthosNConfig& config,
                                                const std::vector<char>& capabilities);

bool ReplaceConstantAdditionWithDepthwise(SubgraphView& graph, IConnectableLayer* layer);

bool ReplaceScalarMultiplicationWithReinterpretQuantization(SubgraphView& graph,
                                                            IConnectableLayer* layer,
                                                            const EthosNConfig& config,
                                                            const std::vector<char>& capabilities,
                                                            std::string& outFailureReason);

bool ReplaceMultiplication(SubgraphView& graph,
                           IConnectableLayer* layer,
                           const EthosNConfig& config,
                           const std::vector<char>& capabilities);
bool ReplaceConstantAdditionWithReinterpretQuantization(SubgraphView& graph,
                                                        IConnectableLayer* layer,
                                                        std::string& outFailureReason);

bool ReplaceAddition(SubgraphView& graph,
                     IConnectableLayer* layer,
                     const EthosNConfig& config,
                     const std::vector<char>& capabilities);

void ReplaceUnsupportedLayers(SubgraphView& graph, const EthosNConfig& config, const std::vector<char>& capabilities);

/// When replacing an addition-with-broadcasted-constant with a depthwise layer, there are various properties
/// of the depthwise layer that need to be set correctly for the replacement to be valid.
struct ConstantAddToDepthwiseReplacementConfig
{
    DepthwiseConvolution2dDescriptor m_Desc;
    TensorInfo m_WeightsInfo;
    uint8_t m_WeightsQuantizedValue;    ///< The quantized value to be used to fill the identity weights tensor.
    TensorInfo m_BiasInfo;
};

/// This information is needed in both the support checks (EthosNLayerSupport::IsAdditionSupported) and also
/// the graph conversion (ReplaceUnsupportedLayers) and so we have common logic here validate and
/// calculate the depthwise configuration.
Optional<ConstantAddToDepthwiseReplacementConfig>
    CalcConstantAddToDepthwiseReplacementConfig(const TensorInfo& inputInfo,
                                                const TensorInfo& constantInfo,
                                                const TensorInfo& outputInfo,
                                                std::string& outFailureReason);

}    // namespace ethosnbackend
}    // namespace armnn
