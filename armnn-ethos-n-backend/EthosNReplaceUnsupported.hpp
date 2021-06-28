//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNMapping.hpp"
#include <Graph.hpp>

namespace armnn
{

struct EthosNConfig;

namespace ethosnbackend
{

bool ReplaceConstantMultiplicationWithDepthwise(Graph& graph,
                                                Layer* layer,
                                                const EthosNConfig& config,
                                                const EthosNMappings& mappings,
                                                const std::vector<char>& capabilities);

bool ReplaceConstantAdditionWithDepthwise(Graph& graph,
                                          Layer* layer,
                                          const EthosNConfig& config,
                                          const EthosNMappings& mappings,
                                          const std::vector<char>& capabilities);

bool ReplaceScalarMultiplicationWithReinterpretQuantization(Graph& graph,
                                                            Layer* layer,
                                                            const EthosNConfig& config,
                                                            const EthosNMappings& mappings,
                                                            const std::vector<char>& capabilities,
                                                            std::string& outFailureReason);

bool ReplaceMultiplication(Graph& graph,
                           Layer* layer,
                           const EthosNConfig& config,
                           const EthosNMappings& mappings,
                           const std::vector<char>& capabilities);

void ReplaceUnsupportedLayers(Graph& graph,
                              const EthosNConfig& config,
                              const EthosNMappings& mappings,
                              const std::vector<char>& capabilities);

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
