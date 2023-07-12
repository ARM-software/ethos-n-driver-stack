//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "GgfParser.hpp"
#include "GlobalParameters.hpp"

#include <ethosn_driver_library/Network.hpp>

namespace ethosn
{
namespace system_tests
{

class EthosNParseRunner : public GgfParser
{
public:
    struct ActionsCallback
    {
        ethosn::support_library::utils::Optional<std::function<void(const InferenceResult& inferenceResult)>>
            afterScheduleCallback;
    };

    struct CreationOptions
    {
        CreationOptions(std::istream& ggfFile, LayerData& layerData)
            : m_GgfFile(ggfFile)
            , m_LayerData(layerData)
        {}

        /// Creates a CreationOptions with options determined by the global settings (GlobalParameters.hpp,
        /// typically configured via the command-line arguments to system_tests).
        static CreationOptions CreateWithGlobalOptions(std::istream& ggfFile, LayerData& layerData);

        std::istream& m_GgfFile;
        LayerData& m_LayerData;
        bool m_EstimationMode                                          = false;
        ethosn::support_library::EstimationOptions m_EstimationOptions = {};
        bool m_StrictPrecision                                         = false;
        bool m_DumpRam                                                 = false;
        ethosn::support_library::CompilationOptions::DebugLevel m_DumpDebugFiles =
            ethosn::support_library::CompilationOptions::DebugLevel::None;
        size_t m_NumberRuns   = 1;
        size_t m_RunBatchSize = 0;
    };

    EthosNParseRunner(const CreationOptions& creationOptions);

    void AddInput(const std::string& name, ethosn::support_library::TensorShape shape) override;

    void AddConstant(const std::string& name,
                     ethosn::support_library::TensorShape shape,
                     float constMin,
                     float constMax) override;

    void AddConvolution(const std::string& name,
                        const std::string& inputName,
                        uint32_t kernelWidth,
                        uint32_t kernelHeight,
                        uint32_t strideWidth,
                        uint32_t strideHeight,
                        uint32_t numOutput,
                        bool biasEnable,
                        const WeightParams& weightParams,
                        const OutputParams& outputParams,
                        PaddingInfo padInfo) override;

    void AddTransposeConvolution(const std::string& name,
                                 const std::string& inputName,
                                 uint32_t kernelWidth,
                                 uint32_t kernelHeight,
                                 uint32_t strideWidth,
                                 uint32_t strideHeight,
                                 uint32_t numOutput,
                                 bool biasEnable,
                                 const WeightParams& weightParams,
                                 const OutputParams& outputParams,
                                 PaddingInfo padInfo) override;

    void AddFullyConnected(const std::string& name,
                           const std::string& inputName,
                           uint32_t numOutput,
                           const WeightParams& weightParams,
                           const OutputParams& outputParams) override;

    void AddDepthwiseConvolution(const std::string& name,
                                 const std::string& inputName,
                                 uint32_t kernelWidth,
                                 uint32_t kernelHeight,
                                 uint32_t strideWidth,
                                 uint32_t strideHeight,
                                 uint32_t channelMultiplier,
                                 bool biasEnable,
                                 const WeightParams& weightParams,
                                 const OutputParams& outputParams,
                                 PaddingInfo padInfo) override;

    void AddStandalonePadding(const std::string& name, const std::string& inputName, PaddingInfo padInfo) override;

    void AddRelu(const std::string& name, const std::string& inputName) override;

    void AddLeakyRelu(const std::string& name, const std::string& inputName, const float alpha) override;

    void AddRequantize(const std::string& name,
                       const std::string& inputName,
                       ethosn::support_library::RequantizeInfo& requantizeInfo) override;

    void AddSigmoid(const std::string& name, const std::string& inputName) override;

    void AddTanh(const std::string& name, const std::string& inputName) override;

    void AddReshape(const std::string& name,
                    const std::string& inputName,
                    ethosn::support_library::TensorShape shape) override;

    void AddConcatenation(const std::string& name, const std::vector<std::string>& inputNames, uint32_t axis) override;

    void AddSplit(const std::string& name,
                  const std::string& inputName,
                  uint32_t axis,
                  std::vector<uint32_t> sizes) override;

    void AddAddition(const std::string& name,
                     const std::string& firstInputName,
                     const std::string& secondInputName) override;

    void AddMultiplication(const std::string& name,
                           const std::string& firstInputName,
                           const std::string& secondInputName) override;

    void AddMeanXy(const std::string& name, const std::string& inputName) override;

    void AddPooling(const std::string& name,
                    const std::string& inputName,
                    ethosn::support_library::PoolingInfo poolingInfo,
                    PaddingAlgorithm paddingAlgorithm) override;

    void AddDepthToSpace(const std::string& name, const std::string& inputName, uint32_t blockSize) override;

    void AddSpaceToDepth(const std::string& name, const std::string& inputName, uint32_t blockSize) override;

    void AddOutput(const std::string& name, const std::string& inputName) override;

    void AddTranspose(const std::string& name,
                      const std::string& inputName,
                      const std::array<uint32_t, 4>& permutation) override;

    void AddResize(const std::string& name, const std::string& inputName, const ResizeParams& params) override;

    void SetStrategies(const std::string& strategies);

    void SetBlockConfigs(const std::string& blockConfigs);

    void SetActionCallback(ActionsCallback callback);

    float GetComparisonTolerance();

    /// Read-only access to the underlying Ethos-N Network.
    const ethosn::support_library::Network* GetNetwork()
    {
        return m_Network.get();
    }

    const ethosn::driver_library::IntermediateBufferReq
        GetIntermediateBufferReq(const ethosn::system_tests::DmaBuffer* intermediateDmaBuf,
                                 uint32_t intermediateBufferSize);

    // Timeout in seconds. A timeout of 0 returns immediately,
    // negative timeout blocks until network is done or call
    // is interrupted.
    InferenceOutputs RunNetwork(int timeoutSeconds = g_EthosNTimeoutSeconds);

    // Estimate the network performance
    ethosn::support_library::NetworkPerformanceData EstimateNetwork();

    std::vector<std::unique_ptr<ethosn::support_library::CompiledNetwork>> GetCompiledNetworks();

    ethosn::support_library::TensorShape GetLayerOutputShape(const std::string& layerName);

    int GetEthosNIndex(std::vector<ethosn::support_library::OutputBufferInfo> outputBufferInfos,
                       std::pair<uint32_t, uint32_t> operand);

    const ethosn::support_library::EstimationOptions& GetEstimationOptions() const;

private:
    void AddConvolution(const std::string& name,
                        const std::string& inputName,
                        uint32_t kernelWidth,
                        uint32_t kernelHeight,
                        uint32_t strideWidth,
                        uint32_t strideHeight,
                        uint32_t outputChannels,
                        bool biasEnable,
                        const WeightParams& weightParams,
                        const OutputParams& outputParams,
                        PaddingInfo padInfo,
                        decltype(ethosn::support_library::AddConvolution)& addConv);

    /// Records added Ethos-N operations (with single- and multiple-output versions)
    /// Call this after adding an operation to the Ethos-N network so that its outputs are recorded and can be connected
    /// to future layers.
    /// @{
    void RecordAddedLayerSingleOutput(
        const std::string& name, ethosn::support_library::TensorAndId<ethosn::support_library::Operand> ethosnOutput);
    void RecordAddedLayerSingleOutput(
        const std::string& name, ethosn::support_library::TensorAndId<ethosn::support_library::Constant> ethosnOutput);

    void RecordAddedLayerMultipleOutput(const std::string& name, ethosn::support_library::TensorsAndId ethosnOutput);
    void RecordAddedLayerSingleOutput(const std::string& name,
                                      std::shared_ptr<ethosn::support_library::Operand> operand,
                                      uint32_t operationid);
    /// @}

    std::shared_ptr<ethosn::support_library::Network> m_Network;
    /// For each GGF layer we have parsed so far, this contains the Ethos-N operand for each output of those layers.
    /// For most layers the output name (the key) will be the same as the layer itself (e.g. conv1), but for
    /// multiple-output layers these will be different in order to distinguish them (e.g. split1_0, split1_1).
    std::map<std::string, std::shared_ptr<ethosn::support_library::Operand>> m_OutputToOperand;
    /// For each Ethos-N operand we have added to the Network, this contains the corresponding operation ID and output index
    /// from the producing operation.
    std::map<std::shared_ptr<ethosn::support_library::Operand>, std::pair<uint32_t, uint32_t>>
        m_OperandToOperationIdAndIndex;
    /// For each output GGF layer we have parsed, this contains the Ethos-N's operand that is exposed by that output.
    /// Potentially multiple output GGF layers may refer to the same Ethos-N operand.
    /// The operand is defined by a pair of "operation ID" and output index, to match the compiled network's queries.
    std::map<std::string, std::pair<uint32_t, uint32_t>> m_OutputNameToOperationIdAndIndex;
    /// For each input GGF layer we have parsed, this contains the Ethos-N's operand that is provided by that input.
    /// The operand is defined by a pair of "operation ID" and output index, to match the compiled network's queries.
    std::map<std::pair<uint32_t, uint32_t>, std::string> m_OperationIdAndIndexToInputName;
    /// For each output GGF layer we have parsed, this contains the Ethos-N's operand that is exposed by that output.
    std::map<std::string, std::shared_ptr<ethosn::support_library::Operand>> m_OutputLayerToOperand;

    ethosn::support_library::CompilationOptions m_Options;
    ethosn::support_library::EstimationOptions m_EstimationOptions;

    ActionsCallback m_Callbacks;

    size_t m_NumberRuns;
    size_t m_RunBatchSize;
};

}    // namespace system_tests
}    // namespace ethosn
