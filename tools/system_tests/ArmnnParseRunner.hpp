//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "GgfParser.hpp"

#include <armnn/ArmNN.hpp>

namespace ethosn
{
namespace system_tests
{

class ArmnnParseRunner : public GgfParser
{
public:
    ArmnnParseRunner(std::istream& ggfFile, LayerData& layerData);

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

    void AddPooling(const std::string& name,
                    const std::string& inputName,
                    ethosn::support_library::PoolingInfo poolInfo,
                    PaddingAlgorithm paddingAlgorithm) override;

    void AddDepthToSpace(const std::string& name, const std::string& inputName, uint32_t blockSize) override;

    void AddSpaceToDepth(const std::string& name, const std::string& inputName, uint32_t blockSize) override;

    void AddOutput(const std::string& name, const std::string& inputName) override;

    void AddTranspose(const std::string& name,
                      const std::string& inputName,
                      const std::array<uint32_t, 4>& permutation) override;

    void AddResize(const std::string& name, const std::string& inputName, const ResizeParams& params) override;

    void AddMeanXy(const std::string& name, const std::string& inputName) override;

    /// Read-only access to the underlying Arm NN INetwork.
    const armnn::INetwork* GetNetwork()
    {
        return m_Network.get();
    }

    InferenceOutputs RunNetwork(const std::vector<armnn::BackendId>& backends);

private:
    void AddActivation(const std::string& name,
                       const std::string& inputName,
                       const armnn::ActivationDescriptor& desc,
                       armnn::TensorInfo outputTensorInfo);

    template <typename ArmnnDescriptor>
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
                        PaddingInfo padInfo);

    std::shared_ptr<armnn::INetwork> m_Network;
    /// Map from GGF output name (e.g. conv1 or split1_0) to the corresponding Arm NN output slot.
    std::map<std::string, armnn::IOutputSlot*> m_OutputMap;
};

inline armnn::TensorShape CalcTensorShapeForMeanXy(const armnn::TensorShape& inputTensor)
{
    // The width and height is always 1 for output tensor
    return { inputTensor[0], 1, 1, inputTensor[3] };
}

}    // namespace system_tests
}    // namespace ethosn
