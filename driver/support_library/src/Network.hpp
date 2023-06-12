//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "Operation.hpp"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <list>

namespace ethosn
{
namespace support_library
{

class Operation;
class Input;
class Output;
class Constant;
class Convolution;
class DepthwiseConvolution;
class TransposeConvolution;
class Concatenation;
class Split;
class Addition;
class FullyConnected;
class ReinterpretQuantization;
class Relu;
class LeakyRelu;
class Requantize;
class Sigmoid;
class Tanh;
class MeanXy;
class Pooling;
class Reshape;
class DepthToSpace;
class SpaceToDepth;
class Transpose;
class Resize;
class EstimateOnly;

// Represents a data object. It's owned by its only producer (writer)
class Operand
{
public:
    // Link to the input (by index) of a consumer (reader) Operation
    struct Consumer
    {
        constexpr Consumer(Operation& operation, const size_t inputIndex) noexcept
            : m_Operation(operation)
            , m_InputIndex(inputIndex)
        {}

        Operation& m_Operation;
        size_t m_InputIndex;
    };

    Operand(Operation& producer, uint32_t producerOutputIndex, const TensorInfo& tensorInfo)
        : m_Producer(producer)
        , m_ProducerOutputIndex(producerOutputIndex)
        , m_Consumers()
        , m_TensorInfo(tensorInfo)
    {}

    Operand& AddConsumer(Operation& operation, const size_t index)
    {
        m_Consumers.emplace_back(operation, index);
        return *this;
    }

    const Operation& GetProducer() const
    {
        return m_Producer;
    }

    uint32_t GetProducerOutputIndex() const
    {
        return m_ProducerOutputIndex;
    }

    const std::vector<Consumer>& GetConsumers() const
    {
        return m_Consumers;
    }

    const TensorInfo& GetTensorInfo() const
    {
        return m_TensorInfo;
    }

private:
    Operation& m_Producer;
    uint32_t m_ProducerOutputIndex;
    std::vector<Consumer> m_Consumers;
    TensorInfo m_TensorInfo;
};

// A directed graph of inputs, outputs, constants, operations and operands
class Network
{
public:
    Network(const std::vector<char>& caps, bool estimatePerformance = false)
        : m_Operations()
        , m_NextOperationId(0)
        , m_OperationIds()
        , m_EstimatePerformanceMode(estimatePerformance)
        , m_Queries(caps)
    {}
    Network(const std::vector<char>& caps, bool estimatePerformance, bool forceExperimentalCompiler)
        : m_Operations()
        , m_NextOperationId(0)
        , m_OperationIds()
        , m_EstimatePerformanceMode(estimatePerformance)
        , m_Queries(caps, forceExperimentalCompiler)
    {}

    Input& AddInput(const TensorInfo& info);

    Output& AddOutput(Operand& operand, const DataFormat format);

    Constant& AddConstant(const TensorInfo& info, const void* data);

    Convolution& AddConvolution(Operand& input, Constant& bias, Constant& weights, const ConvolutionInfo& convInfo);

    DepthwiseConvolution&
        AddDepthwiseConvolution(Operand& input, Constant& bias, Constant& weights, const ConvolutionInfo& convInfo);

    StandalonePadding& AddStandalonePadding(Operand& input, const Padding& padding);

    TransposeConvolution&
        AddTransposeConvolution(Operand& input, Constant& bias, Constant& weights, const ConvolutionInfo& convInfo);

    Concatenation& AddConcatenation(const std::vector<Operand*>& inputs, const ConcatenationInfo& concatInfo);

    Split& AddSplit(Operand& input, const SplitInfo& splitInfo);

    Addition& AddAddition(Operand& layer1, Operand& layer2, const QuantizationInfo& outputQuantizationInfo);

    FullyConnected& AddFullyConnected(Operand& input,
                                      Constant& bias,
                                      Constant& weights,
                                      const FullyConnectedInfo fullyConnectedInfo);

    ReinterpretQuantization& AddReinterpretQuantization(Operand& input,
                                                        const ReinterpretQuantizationInfo& reinterpretQuantizationInfo);

    Relu& AddRelu(Operand& input, const ReluInfo& reluInfo);

    LeakyRelu& AddLeakyRelu(Operand& input, const LeakyReluInfo& leakyReluInfo);

    Requantize& AddRequantize(Operand& input, const RequantizeInfo& requantizeInfo);

    Sigmoid& AddSigmoid(Operand& input);

    Tanh& AddTanh(Operand& input);

    MeanXy& AddMeanXy(Operand& input);

    Pooling& AddPooling(Operand& input, const PoolingInfo& poolingInfo);

    Reshape& AddReshape(Operand& input, const TensorShape& newDimensions);

    DepthToSpace& AddDepthToSpace(Operand& input, const DepthToSpaceInfo& depthToSpaceInfo);

    SpaceToDepth& AddSpaceToDepth(Operand& input, const SpaceToDepthInfo& spaceToDepthInfo);

    Transpose& AddTranspose(Operand& input, const TransposeInfo& transposeInfo);

    Resize& AddResize(Operand& input, const ResizeInfo& resizeInfo);

    EstimateOnly& AddEstimateOnly(const std::vector<Operand*>& inputs, const EstimateOnlyInfo& estimateOnly);

    /// STL-style begin/end accessors for range-based for-loops.
    /// @{
    detail::OperationList::const_iterator begin() const
    {
        return m_Operations.begin();
    }

    detail::OperationList::const_iterator end() const
    {
        return m_Operations.end();
    }
    /// @}

    // Visit existing operations in topological order
    void Accept(INetworkVisitor& visitor) const
    {
        for (auto&& op : m_Operations)
        {
            op->Accept(visitor);
        }
    }

    // Overload for rvalue reference
    void Accept(INetworkVisitor&& visitor) const
    {
        Accept(visitor);
    }

    const std::set<uint32_t>& GetOperationIds() const
    {
        return m_OperationIds;
    }

    bool IsEstimationMode() const
    {
        return m_EstimatePerformanceMode;
    }

    const std::vector<char>& GetCapabilities() const
    {
        return m_Queries.GetCapabilities();
    }

private:
    // Add Operation of derived class Op to the Network
    template <typename Op, typename... Args>
    Op& AddOperation(Args&&... args)
    {
        const detail::OperationList::iterator pos = m_Operations.emplace(m_Operations.end());
        uint32_t newOpId                          = GetNextOperationId();
        m_OperationIds.insert(newOpId);

        auto operation = std::make_unique<Op>(detail::PosInNetwork(pos), newOpId, std::forward<Args>(args)...);
        Op* ptr        = operation.get();
        *pos           = std::move(operation);

        return *ptr;
    }

    uint32_t GetNextOperationId()
    {
        return m_NextOperationId++;
    }

    template <typename Op, typename... Args>
    Op& AddOperationWithId(Args&&... args)
    {
        return AddOperation<Op>(std::forward<Args>(args)...);
    }

    /// Checks if the supported level is good enough for the "network type"
    /// Estimation networks can support EstimationOnly and Supported.
    /// "Normal" networks can only support Supported.
    bool CheckSupportedLevel(SupportedLevel level)
    {
        if (level == SupportedLevel::Supported)
        {
            return true;
        }
        else if (level == SupportedLevel::EstimateOnly && m_EstimatePerformanceMode)
        {
            return true;
        }
        return false;
    }

    // Operations in topological order
    detail::OperationList m_Operations;
    uint32_t m_NextOperationId;
    std::set<uint32_t> m_OperationIds;
    const bool m_EstimatePerformanceMode;
    SupportQueries m_Queries;
};

}    // namespace support_library
}    // namespace ethosn
