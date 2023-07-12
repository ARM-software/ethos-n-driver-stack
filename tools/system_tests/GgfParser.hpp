//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "LayerData.hpp"

#include <ethosn_support_library/Support.hpp>

#include <map>
#include <string>
#include <vector>

namespace ethosn
{
namespace system_tests
{

// Padding algorithms
enum class PaddingAlgorithm
{
    VALID,
    SAME,
    EXPLICIT
};

struct PaddingInfo
{
    PaddingAlgorithm alg;
    struct PadInfo
    {
        int32_t padLeft;
        int32_t padRight;
        int32_t padTop;
        int32_t padBottom;
    } info;
};

class GgfParser
{
public:
    GgfParser(std::istream& ggfFile, LayerData& layerData);
    virtual ~GgfParser() = default;

    const std::vector<std::string> GetInputLayerNames()
    {
        return m_InputLayerNames;
    }

    const std::vector<ethosn::support_library::TensorShape> GetInputLayerShapes()
    {
        return m_InputLayerShapes;
    }

    const std::vector<std::string> GetOutputLayerNames()
    {
        return m_OutputLayerNames;
    }

    int GetInputLayerIndex(const std::string& name)
    {
        std::vector<std::string>::iterator it = std::find(m_InputLayerNames.begin(), m_InputLayerNames.end(), name);
        return static_cast<int>(std::distance(m_InputLayerNames.begin(), it));
    }

    int GetOutputLayerIndex(const std::string& name)
    {
        std::vector<std::string>::iterator it = std::find(m_OutputLayerNames.begin(), m_OutputLayerNames.end(), name);
        return static_cast<int>(std::distance(m_OutputLayerNames.begin(), it));
    }

    const std::string GetGgfOutputLayerName(size_t ggfIdx)
    {
        return m_OutputLayerNames[ggfIdx];
    }

protected:
    void ParseNetwork(void);

    virtual void AddInput(const std::string& name, ethosn::support_library::TensorShape shape);

    virtual void AddConstant(const std::string& name,
                             ethosn::support_library::TensorShape shape,
                             float constMin,
                             float constMax) = 0;

    virtual void AddConvolution(const std::string& name,
                                const std::string& inputName,
                                uint32_t kernelWidth,
                                uint32_t kernelHeight,
                                uint32_t strideWidth,
                                uint32_t strideHeight,
                                uint32_t numOutput,
                                bool biasEnable,
                                const WeightParams& weightParams,
                                const OutputParams& outputParams,
                                PaddingInfo padInfo) = 0;

    virtual void AddTransposeConvolution(const std::string& name,
                                         const std::string& inputName,
                                         uint32_t kernelWidth,
                                         uint32_t kernelHeight,
                                         uint32_t strideWidth,
                                         uint32_t strideHeight,
                                         uint32_t numOutput,
                                         bool biasEnable,
                                         const WeightParams& weightParams,
                                         const OutputParams& outputParams,
                                         PaddingInfo padInfo) = 0;

    virtual void AddFullyConnected(const std::string& name,
                                   const std::string& inputName,
                                   uint32_t numOutput,
                                   const WeightParams& weightParams,
                                   const OutputParams& outputParams) = 0;

    virtual void AddDepthwiseConvolution(const std::string& name,
                                         const std::string& inputName,
                                         uint32_t kernelWidth,
                                         uint32_t kernelHeight,
                                         uint32_t strideWidth,
                                         uint32_t strideHeight,
                                         uint32_t channelMultiplier,
                                         bool biasEnable,
                                         const WeightParams& weightParams,
                                         const OutputParams& outputParams,
                                         PaddingInfo padInfo) = 0;

    virtual void AddStandalonePadding(const std::string& name, const std::string& inputName, PaddingInfo padInfo) = 0;

    virtual void AddRelu(const std::string& name, const std::string& inputName) = 0;

    virtual void AddLeakyRelu(const std::string& name, const std::string& inputName, const float alpha) = 0;

    virtual void AddRequantize(const std::string& name,
                               const std::string& inputName,
                               ethosn::support_library::RequantizeInfo& requantizeInfo) = 0;

    virtual void AddMeanXy(const std::string& name, const std::string& inputName) = 0;

    virtual void AddSigmoid(const std::string& name, const std::string& inputName) = 0;

    virtual void AddTanh(const std::string& name, const std::string& inputName) = 0;

    virtual void AddReshape(const std::string& name,
                            const std::string& inputName,
                            ethosn::support_library::TensorShape shape) = 0;

    virtual void
        AddConcatenation(const std::string& name, const std::vector<std::string>& inputNames, uint32_t axis) = 0;

    virtual void
        AddSplit(const std::string& name, const std::string& inputName, uint32_t axis, std::vector<uint32_t> sizes) = 0;

    virtual void
        AddAddition(const std::string& name, const std::string& firstInputName, const std::string& secondInputName) = 0;

    virtual void AddMultiplication(const std::string& name,
                                   const std::string& firstInputName,
                                   const std::string& secondInputName) = 0;

    virtual void AddPooling(const std::string& name,
                            const std::string& inputName,
                            ethosn::support_library::PoolingInfo poolInfo,
                            PaddingAlgorithm algo) = 0;

    virtual void AddDepthToSpace(const std::string& name, const std::string& inputName, uint32_t blockSize) = 0;

    virtual void AddSpaceToDepth(const std::string& name, const std::string& inputName, uint32_t blockSize) = 0;

    virtual void AddOutput(const std::string& name, const std::string& inputName);

    virtual void AddTranspose(const std::string& name,
                              const std::string& inputName,
                              const std::array<uint32_t, 4>& permutation) = 0;

    virtual void AddResize(const std::string& name, const std::string& inputName, const ResizeParams& params) = 0;

    LayerData& m_LayerData;

private:
    bool IsValidNoEntries(const int32_t noEntries);
    void OnMetadata(std::string& line);
    std::istream& m_GgfFile;
    std::vector<std::string> m_InputLayerNames;
    std::vector<ethosn::support_library::TensorShape> m_InputLayerShapes;
    std::vector<std::string> m_OutputLayerNames;
};

}    // namespace system_tests
}    // namespace ethosn
