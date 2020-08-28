//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"

#include <cassert>
#include <list>
#include <memory>
#include <vector>

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
class Relu;
class LeakyRelu;
class Requantize;
class Softmax;
class Sigmoid;
class Pooling;
class Reshape;
class DepthToSpace;
class SpaceToDepth;
class Transpose;
class Resize;
class EstimateOnly;

namespace detail
{

using OperationList = std::list<std::unique_ptr<Operation>>;

// Position of an Operation in a Network
class PosInNetwork
{
private:
    // Only class Network can create objects of this class
    friend class ethosn::support_library::Network;

    using Type = OperationList::const_iterator;

    explicit PosInNetwork(const Type pos)
        : m_Value(pos)
    {}

    Type m_Value;
};
}    // namespace detail

// Base abstract class for algorithms that visit Operations in a Network
// See Visitor Pattern: https://en.wikipedia.org/wiki/Visitor_pattern
class INetworkVisitor
{
public:
    virtual ~INetworkVisitor()
    {}
    virtual void Visit(Input& input)                               = 0;
    virtual void Visit(Output& output)                             = 0;
    virtual void Visit(Constant& constant)                         = 0;
    virtual void Visit(Convolution& convolution)                   = 0;
    virtual void Visit(DepthwiseConvolution& depthwiseConvolution) = 0;
    virtual void Visit(TransposeConvolution& transposeConvolution) = 0;
    virtual void Visit(Concatenation& concatenation)               = 0;
    virtual void Visit(Split& split)                               = 0;
    virtual void Visit(Addition& addition)                         = 0;
    virtual void Visit(FullyConnected& fullyConnected)             = 0;
    virtual void Visit(Relu& relu)                                 = 0;
    virtual void Visit(LeakyRelu& leakyRelu)                       = 0;
    virtual void Visit(Requantize& requantize)                     = 0;
    virtual void Visit(Softmax& softmax)                           = 0;
    virtual void Visit(Sigmoid& sigmoid)                           = 0;
    virtual void Visit(Pooling& pooling)                           = 0;
    virtual void Visit(Reshape& reshape)                           = 0;
    virtual void Visit(DepthToSpace& depthToSpace)                 = 0;
    virtual void Visit(SpaceToDepth& spaceToDepth)                 = 0;
    virtual void Visit(Transpose& transpose)                       = 0;
    virtual void Visit(Resize& resize)                             = 0;
    virtual void Visit(EstimateOnly& estimateOnly)                 = 0;
};

/// Implementation of INetworkVisitor with default no-op implementations.
/// This is useful when you only care about overriding some operations.
class NetworkVisitor : public INetworkVisitor
{
public:
    using INetworkVisitor::Visit;

    void Visit(Input&) override
    {}
    void Visit(Output&) override
    {}
    void Visit(Constant&) override
    {}
    void Visit(Convolution&) override
    {}
    void Visit(DepthwiseConvolution&) override
    {}
    void Visit(TransposeConvolution&) override
    {}
    void Visit(Concatenation&) override
    {}
    void Visit(Split&) override
    {}
    void Visit(Addition&) override
    {}
    void Visit(FullyConnected&) override
    {}
    void Visit(Relu&) override
    {}
    void Visit(LeakyRelu&) override
    {}
    void Visit(Requantize&) override
    {}
    void Visit(Softmax&) override
    {}
    void Visit(Sigmoid&) override
    {}
    void Visit(Pooling&) override
    {}
    void Visit(Reshape&) override
    {}
    void Visit(DepthToSpace&) override
    {}
    void Visit(SpaceToDepth&) override
    {}
    void Visit(Transpose&) override
    {}
    void Visit(Resize&) override
    {}
    void Visit(EstimateOnly&) override
    {}
};

// Base abstract class for operations in a Network
class Operation
{
public:
    Operation(const detail::PosInNetwork pos,
              uint32_t opId,
              const std::vector<Operand*>& inputs,
              const std::vector<TensorInfo>& outputTensorInfos);

    virtual ~Operation()
    {}

    std::vector<const Operand*> GetInputs() const
    {
        return std::vector<const Operand*>(m_Inputs.begin(), m_Inputs.end());
    }

    std::vector<Operand>& GetOutputs()
    {
        return m_Outputs;
    }

    const std::vector<Operand>& GetOutputs() const
    {
        return m_Outputs;
    }

    Operand& GetInput(size_t index)
    {
        return *m_Inputs.at(index);
    }

    const Operand& GetInput(size_t index) const
    {
        return *m_Inputs.at(index);
    }

    Operand& GetOutput(size_t index)
    {
        return m_Outputs.at(index);
    }

    const Operand& GetOutput(size_t index) const
    {
        return m_Outputs.at(index);
    }

    uint32_t GetId() const
    {
        return m_OperationId;
    }

    // Accept a visiting NetworkVisitor
    // See Visitor Pattern: https://en.wikipedia.org/wiki/Visitor_pattern
    virtual void Accept(INetworkVisitor& visitor) = 0;

    virtual void Print(std::ostream& os) = 0;

    // Position in container Network
    const detail::PosInNetwork m_Pos;

private:
    // Id of the operation - uniquely identifies this network layer
    uint32_t m_OperationId;
    std::vector<Operand*> m_Inputs;
    std::vector<Operand> m_Outputs;
};

// CRTP trick so Derived classes override the virtual function Operation::Accept()
// with a call to the corresponding overload of NetworkVisitor::VirtualVisit().
template <typename Derived, typename Base = Operation>
class VisitableOperation : public Base
{
public:
    static_assert(std::is_base_of<Operation, Base>::value,
                  "The 'Base' template parameter must be derived from Operation");

    using Base::Base;

    // So VisitableOperation is an abstract class
    virtual ~VisitableOperation() = 0;

    void Accept(INetworkVisitor& visitor) final
    {
        // Note this static_assert is here just because it needs to be in the body of a method, it doesn't
        // have to be this method.
        static_assert(std::is_base_of<VisitableOperation, Derived>::value,
                      "The 'Derived' template parameter must be derived from VisitableOperation<Derived>");

        visitor.Visit(static_cast<Derived&>(*this));
    }
};

// Destructor needs a definition even if declared pure virtual
template <typename Derived, typename Base>
VisitableOperation<Derived, Base>::~VisitableOperation()
{}

}    // namespace support_library
}    // namespace ethosn
