//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNCaching.hpp"
#include "EthosNConfig.hpp"
#include <armnn/backends/SubgraphView.hpp>

#include <ethosn_support_library/Support.hpp>

#include <memory>
#include <unordered_map>

namespace armnn
{

namespace ethosn_lib = ethosn::support_library;

using EthosNCompiledNetworkPtr = std::unique_ptr<ethosn_lib::CompiledNetwork>;
using EthosNNetworkPtr         = std::shared_ptr<ethosn_lib::Network>;
using EthosNOperandPtr         = std::shared_ptr<ethosn_lib::Operand>;
using EthosNConstantPtr        = std::shared_ptr<ethosn_lib::Constant>;
/// The type returned by most of the Ethos-N's AddXXX() functions.
using EthosNAddOperationResult = ethosn_lib::TensorAndId<ethosn_lib::Operand>;
using EthosNOperationId        = uint32_t;
/// The Ethos-N identifies inputs and outputs from its networks as a pair of:
///  * ID of the operation which produces the output
///  * Index of the specific output from that operation (for the case of an operation with multiple outputs).
using EthosNInputOutputId = std::pair<EthosNOperationId, uint32_t>;

/// Expanded definition of an Ethos-N Operand (which is the return type of adding an operation to the Ethos-N network).
/// This adds the outputIndex field which is implicit as a result from the Ethos-N API but we prefer to store it explicitly.
struct EthosNOperand
{
    /// The unique ID of the operation that produces this operand.
    uint32_t operationId;
    /// The opaque operand object itself, used for Ethos-N APIs.
    std::shared_ptr<ethosn_lib::Operand> tensor;
    /// The index into the outputs of the operation that produces this operand.
    uint32_t outputIndex;
};

class EthosNSupportLibraryInterface
{
public:
    virtual ~EthosNSupportLibraryInterface()
    {}

    virtual std::vector<std::unique_ptr<ethosn_lib::CompiledNetwork>>
        Compile(const ethosn_lib::Network& network, const ethosn_lib::CompilationOptions& options)
    {
        return ethosn_lib::Compile(network, options);
    }
};

ARMNN_DLLEXPORT extern std::unique_ptr<EthosNSupportLibraryInterface> g_EthosNSupportLibraryInterface;

class EthosNSubgraphViewConverter
{
public:
    EthosNSubgraphViewConverter(const SubgraphView& subgraph,
                                ModelOptions modelOptions,
                                const EthosNConfig& config,
                                const std::vector<char>& capabilities);
    ~EthosNSubgraphViewConverter() = default;

    std::vector<PreCompiledObjectPtr> CompileNetwork();

    static void ResetNextInstanceId();

protected:
    // This is protected so it can used in unit tests.
    void CreateUncompiledNetwork();

private:
    std::vector<PreCompiledObjectPtr> Estimate();
    std::vector<PreCompiledObjectPtr> Compile();

    /// Adds operation(s) to the Ethos-N network that correspond to the given Arm NN layer.
    /// This will update m_ConvertedOutputSlots.
    /// @{
    void AddInput(uint32_t inputSlotIdx);
    void AddOutput(uint32_t outputSlotIdx);

    void AddActivationLayer(const IConnectableLayer* layer);
    void AddAdditionLayer(const IConnectableLayer* layer);
    void AddConstantLayer(const IConnectableLayer* layer);
    void AddConvolution2dLayer(const IConnectableLayer* layer);
    void AddDepthwiseConvolution2dLayer(const IConnectableLayer* layer);
    void AddTransposeConvolution2dLayer(const IConnectableLayer* layer);
    void AddFullyConnectedLayer(const IConnectableLayer* layer);
    void AddConcatLayer(const IConnectableLayer* layer);
    void AddPooling2dLayer(const IConnectableLayer* layer);
    void AddReshapeLayer(const IConnectableLayer* layer);
    void AddSoftmaxLayer(const IConnectableLayer* layer);
    void AddSplitterLayer(const IConnectableLayer* layer);
    void AddDepthToSpaceLayer(const IConnectableLayer* layer);
    void AddSpaceToDepthLayer(const IConnectableLayer* layer);
    void AddTransposeLayer(const IConnectableLayer* layer);
    void AddQuantizeLayer(const IConnectableLayer* layer);
    void AddResizeLayer(const IConnectableLayer* layer);
    void AddMeanXyLayer(const IConnectableLayer* layer);
    void AddStandInLayer(const IConnectableLayer* layer);
    /// @}

    /// When in PerfOnly mode, this function tries to add estimate only layer from Support Library if the layer in
    /// consideration is unknown.
    void HandleUnknownLayer(const IConnectableLayer* layer, const std::string& reason);

    void AddEstimateOnly(const IConnectableLayer* layer, const std::string& reason);
    /// Converts the layer that owns the given OutputSlot and adds it to the Ethos-N network.
    /// Returns the corresponding Ethos-N operand representing the same output as the given OutputSlot.
    /// If the layer has already been converted then this returns the existing corresponding Ethos-N operand and does
    /// not modify the Ethos-N network.
    EthosNOperand AddOrRetrieveEthosNOperand(const IOutputSlot* outputSlot);

    /// Converts biases
    // TODO This method  will only need the layer parameter once all relevant layers will have the
    // data layout in their respective descriptors. Currently Convolution2dLayer and
    // DepthwiseConvolution2dLayer have it, but we need to use this method with FullyConnectedLayer
    // as well.
    EthosNConstantPtr
        AddBiases(const Layer& layer, const ConstTensorHandle* bias, const TensorInfo& weightInfo, bool biasEnabled);

    /// Converts weights
    EthosNConstantPtr AddWeights(const Layer& layer, DataLayout dataLayout, const ConstTensorHandle& weights);
    EthosNConstantPtr AddWeights(const FullyConnectedLayer& layer, const ConstTensorHandle& weights);

    /// Helper function to insert a converted Arm NN layer in m_ConvertedOutputSlots, for layers with a single output.
    void InsertConvertedLayerSingleOutput(const IConnectableLayer* layer,
                                          EthosNAddOperationResult ethosnAddOperationResult);
    /// Helper function to insert a converted Arm NN layer in m_ConvertedOutputSlots, for layers with multiple outputs.
    void InsertConvertedLayerMultipleOutput(const IConnectableLayer* layer,
                                            ethosn_lib::TensorsAndId ethosnAddOperationResult);

private:
    /// ID number for next constructed instance
    static uint32_t ms_NextInstanceId;

    const uint32_t m_InstanceId;

    /// Original Arm NN sub-graph
    const SubgraphView& m_Subgraph;

    /// Ethos-N network resulting after converting the sub-graph
    EthosNNetworkPtr m_Network;

    /// Map used to store previously converted layers.
    /// Specifically, we map the OutputSlots of the Arm NN graph (rather than Layers) because a layer may have
    /// multiple outputs and each OutputSlots belonging to the same layer will map to a different Ethos-N operand.
    std::unordered_map<const IOutputSlot*, EthosNOperand> m_ConvertedOutputSlots;

    /// Contains the mapping from the Ethos-N's identifier of an input to Arm NN input slot indices
    /// (i.e. within m_Subgraph.GetInputSlots()).
    std::map<EthosNInputOutputId, uint32_t> m_EthosNInputIdToInputSlot;
    /// Contains the mapping from the Ethos-N's identifier of an output to Arm NN output slot indices
    /// (i.e. within m_Subgraph.GetOutputSlots()).
    std::map<EthosNInputOutputId, uint32_t> m_EthosNOutputIdToOutputSlot;

    EthosNConfig m_EthosNConfig;
    std::vector<char> m_Capabilities;

    /// Map from Ethos-N operation ID to the corresponding Arm NN layer name.
    std::map<uint32_t, std::string> m_EthosNOperationNameMapping;

    /// Options that are passed to the support library.
    ethosn_lib::CompilationOptions m_CompilationOptions;
};

/// Gets the compilation options to use based on the given EthosNConfig and ModelOptions.
ethosn_lib::CompilationOptions
    GetCompilationOptions(const EthosNConfig& config, const ModelOptions& modelOptions, uint32_t instanceId);

}    // namespace armnn
