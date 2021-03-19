//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "Operation.hpp"

#include <ethosn_command_stream/CommandData.hpp>

namespace ethosn
{
namespace support_library
{

class HardwareCapabilities;

// Network input
class Input : public VisitableOperation<Input>
{
public:
    Input(const detail::PosInNetwork pos, uint32_t id, const TensorInfo& info);

    const TensorInfo& GetTensorInfo() const
    {
        return m_Info;
    }

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Input";
    }

private:
    TensorInfo m_Info;
};

// Network output
class Output : public VisitableOperation<Output>
{
public:
    Output(const detail::PosInNetwork pos, uint32_t id, Operand& operand, const DataFormat format);

    TensorInfo GetTensorInfo() const;

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Output";
    }

private:
    DataFormat m_OutputFormat;
};

// Constant data (e.g. weights, biases)
class Constant : public VisitableOperation<Constant>
{
public:
    Constant(const detail::PosInNetwork pos, uint32_t id, const TensorInfo& info, const void* data);

    const TensorInfo& GetTensorInfo() const;

    const void* GetData() const
    {
        return m_Data.data();
    }

    /// Gets the internal data.
    const std::vector<uint8_t>& GetDataVector() const;

    /// Gets the internal data, reinterpreted as an array of the given type.
    /// Note this incurs a full copy of the data.
    template <typename T>
    std::vector<T> GetDataVectorAs() const;

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Constant";
    }

private:
    std::vector<uint8_t> m_Data;
};

// Convolution operation
class Convolution : public VisitableOperation<Convolution>
{
public:
    Convolution(const detail::PosInNetwork pos,
                uint32_t id,
                Operand& input,
                Constant& bias,
                Constant& weights,
                const ConvolutionInfo& convInfo);

    const Constant& GetBias() const
    {
        return m_Bias;
    }

    const Constant& GetWeights() const
    {
        return m_Weights;
    }

    const ConvolutionInfo& GetConvolutionInfo() const
    {
        return m_ConvInfo;
    }

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo,
                                                const TensorInfo& weightsInfo,
                                                const ConvolutionInfo& convInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Convolution";
    }

private:
    Constant& m_Bias;
    Constant& m_Weights;
    ConvolutionInfo m_ConvInfo;
};

// DepthwiseConvolution operation
class DepthwiseConvolution : public VisitableOperation<DepthwiseConvolution>
{
public:
    DepthwiseConvolution(const detail::PosInNetwork pos,
                         uint32_t id,
                         Operand& input,
                         Constant& bias,
                         Constant& weights,
                         const ConvolutionInfo& convInfo);

    const Constant& GetBias() const
    {
        return m_Bias;
    }

    const Constant& GetWeights() const
    {
        return m_Weights;
    }

    const ConvolutionInfo& GetConvolutionInfo() const
    {
        return m_ConvInfo;
    }

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo,
                                                const TensorInfo& weightsInfo,
                                                const ConvolutionInfo& convInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "DepthwiseConvolution";
    }

private:
    Constant& m_Bias;
    Constant& m_Weights;
    ConvolutionInfo m_ConvInfo;
};

// TransposeConvolution operation
class TransposeConvolution : public VisitableOperation<TransposeConvolution>
{
public:
    TransposeConvolution(const detail::PosInNetwork pos,
                         uint32_t id,
                         Operand& input,
                         Constant& bias,
                         Constant& weights,
                         const ConvolutionInfo& convInfo);

    const Constant& GetBias() const
    {
        return m_Bias;
    }

    const Constant& GetWeights() const
    {
        return m_Weights;
    }

    const ConvolutionInfo& GetConvolutionInfo() const
    {
        return m_ConvInfo;
    }

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo,
                                                const TensorInfo& weightsInfo,
                                                const ConvolutionInfo& convInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "TransposeConvolution";
    }

private:
    Constant& m_Bias;
    Constant& m_Weights;
    ConvolutionInfo m_ConvInfo;
};

class Concatenation : public VisitableOperation<Concatenation>
{
public:
    Concatenation(const detail::PosInNetwork pos,
                  uint32_t id,
                  const std::vector<Operand*>& inputs,
                  const ConcatenationInfo& concatInfo);

    const ConcatenationInfo& GetConcatenationInfo() const
    {
        return m_ConcatInfo;
    }
    static TensorInfo CalculateOutputTensorInfo(const std::vector<TensorInfo>& inputInfos,
                                                const ConcatenationInfo& concatInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Concatenation";
    }

private:
    ConcatenationInfo m_ConcatInfo;
};

class Split : public VisitableOperation<Split>
{
public:
    Split(const detail::PosInNetwork pos, uint32_t id, Operand& input, const SplitInfo& splitInfo);

    const SplitInfo& GetSplitInfo() const
    {
        return m_SplitInfo;
    }
    static std::vector<TensorInfo> CalculateOutputTensorInfos(const TensorInfo& inputInfo, const SplitInfo& splitInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Split";
    }

private:
    SplitInfo m_SplitInfo;
};

class Addition : public VisitableOperation<Addition>
{
public:
    Addition(const detail::PosInNetwork pos,
             uint32_t id,
             Operand& layer1,
             Operand& layer2,
             const QuantizationInfo& outputQuantizationInfo);

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo0,
                                                const TensorInfo& inputInfo1,
                                                const QuantizationInfo& outputQuantizationInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Addition";
    }
};

// Fully connected operation
class FullyConnected : public VisitableOperation<FullyConnected>
{
public:
    FullyConnected(const detail::PosInNetwork pos,
                   uint32_t id,
                   Operand& input,
                   Constant& bias,
                   Constant& weights,
                   const FullyConnectedInfo& fullyConnectedInfo);

    const Constant& GetBias() const
    {
        return m_Bias;
    }

    const Constant& GetWeights() const
    {
        return m_Weights;
    }

    const FullyConnectedInfo GetFullyConnectedInfo() const
    {
        return m_FullyConnectedInfo;
    }

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo,
                                                const TensorInfo& weightsInfo,
                                                const FullyConnectedInfo& fullyConnectedInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "FullyConnected";
    }

private:
    Constant& m_Bias;
    Constant& m_Weights;
    FullyConnectedInfo m_FullyConnectedInfo;
};

class Relu : public VisitableOperation<Relu>
{
public:
    Relu(const detail::PosInNetwork pos, uint32_t id, Operand& input, const ReluInfo& reluInfo);

    const ReluInfo& GetReluInfo() const
    {
        return m_ReluInfo;
    }

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Relu";
    }

private:
    ReluInfo m_ReluInfo;
};

class LeakyRelu : public VisitableOperation<LeakyRelu>
{
public:
    LeakyRelu(const detail::PosInNetwork pos, uint32_t id, Operand& input, const LeakyReluInfo& leakyReluInfo);

    const LeakyReluInfo& GetLeakyReluInfo() const
    {
        return m_LeakyReluInfo;
    }

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo, const LeakyReluInfo& leakyReluInfo);
    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "LeakyRelu";
    }

private:
    LeakyReluInfo m_LeakyReluInfo;
};

class Requantize : public VisitableOperation<Requantize>
{
public:
    Requantize(const detail::PosInNetwork pos, uint32_t id, Operand& input, const RequantizeInfo& requantizeInfo);

    const RequantizeInfo& GetRequantizeInfo() const
    {
        return m_RequantizeInfo;
    }

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo, const RequantizeInfo& requantizeInfo);
    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Requantize";
    }

private:
    RequantizeInfo m_RequantizeInfo;
};

class Softmax : public VisitableOperation<Softmax>
{
public:
    Softmax(const detail::PosInNetwork pos, uint32_t id, Operand& input);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Softmax";
    }
};

class Sigmoid : public VisitableOperation<Sigmoid>
{
public:
    Sigmoid(const detail::PosInNetwork pos, uint32_t id, Operand& input);

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Sigmoid";
    }
};

class Pooling : public VisitableOperation<Pooling>
{
public:
    Pooling(const detail::PosInNetwork pos, uint32_t id, Operand& input, const PoolingInfo& poolingInfo);

    const PoolingInfo& GetPoolingInfo() const
    {
        return m_PoolingInfo;
    }

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo, const PoolingInfo& poolingInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Pooling";
    }

private:
    PoolingInfo m_PoolingInfo;
};

class Reshape : public VisitableOperation<Reshape>
{
public:
    Reshape(const detail::PosInNetwork pos, uint32_t id, Operand& input, const TensorShape& newDimensions);

    const TensorShape& GetReshapeInfo() const
    {
        return m_NewDimensions;
    }

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo, const TensorShape& newDimensions);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Reshape";
    }

private:
    TensorShape m_NewDimensions;
};

class DepthToSpace : public VisitableOperation<DepthToSpace>
{
public:
    DepthToSpace(const detail::PosInNetwork pos, uint32_t id, Operand& input, const DepthToSpaceInfo& depthToSpaceInfo);

    const DepthToSpaceInfo& GetDepthToSpaceInfo() const
    {
        return m_DepthToSpaceInfo;
    }

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo, const DepthToSpaceInfo& depthToSpaceInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "DepthToSpace";
    }

private:
    DepthToSpaceInfo m_DepthToSpaceInfo;
};

class SpaceToDepth : public VisitableOperation<SpaceToDepth>
{
public:
    SpaceToDepth(const detail::PosInNetwork pos, uint32_t id, Operand& input, const SpaceToDepthInfo& spaceToDepthInfo);

    const SpaceToDepthInfo& GetSpaceToDepthInfo() const
    {
        return m_SpaceToDepthInfo;
    }

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo, const SpaceToDepthInfo& spaceToDepthInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "SpaceToDepth";
    }

private:
    SpaceToDepthInfo m_SpaceToDepthInfo;
};

class Transpose : public VisitableOperation<Transpose>
{
public:
    Transpose(const detail::PosInNetwork pos, uint32_t id, Operand& input, const TransposeInfo& transposeInfo);

    const TransposeInfo& GetTransposeInfo() const
    {
        return m_TransposeInfo;
    }

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo, const TransposeInfo& transposeInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Transpose";
    }

private:
    TransposeInfo m_TransposeInfo;
};

class Resize : public VisitableOperation<Resize>
{
public:
    Resize(const detail::PosInNetwork pos, uint32_t id, Operand& input, const ResizeInfo& resizeInfo);

    const ResizeInfo& GetResizeInfo() const
    {
        return m_ResizeInfo;
    }

    static TensorInfo CalculateOutputTensorInfo(const TensorInfo& inputInfo, const ResizeInfo& transposeInfo);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "Resize";
    }

private:
    ResizeInfo m_ResizeInfo;
};

class EstimateOnly : public VisitableOperation<EstimateOnly>
{
public:
    EstimateOnly(const detail::PosInNetwork pos,
                 uint32_t id,
                 const std::vector<Operand*>& inputs,
                 const EstimateOnlyInfo& info);

    void Print(std::ostream& os) final;
    const char* GetTypeName() final
    {
        return "EstimateOnly";
    }

private:
};

}    // namespace support_library
}    // namespace ethosn
