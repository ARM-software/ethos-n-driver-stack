//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Optional.hpp"
#include "GraphOfParts.hpp"
#include "MceEstimationUtils.hpp"
#include "Network.hpp"

#include <unordered_map>

namespace ethosn
{
namespace support_library
{

class McePart;
struct DebuggingContext;

class NetworkToGraphOfPartsConverter : public NetworkVisitor
{
public:
    using NetworkVisitor::Visit;
    NetworkToGraphOfPartsConverter(const Network& network,
                                   const HardwareCapabilities& capabilities,
                                   const EstimationOptions& estimationOptions,
                                   const CompilationOptions& compilationOptions,
                                   DebuggingContext& debuggingContext,
                                   ThreadPool& threadPool);
    ~NetworkToGraphOfPartsConverter();

    // Visitor functions for supported operations in the Network
    void Visit(Input& input) final;
    void Visit(Output& output) final;
    void Visit(Convolution& convolution) final;
    void Visit(Constant& constant) final;
    void Visit(DepthwiseConvolution& convolution) final;
    void Visit(StandalonePadding& padding) final;
    void Visit(FullyConnected& fullyConnected) final;
    void Visit(Pooling& pooling) final;
    void Visit(Reshape& reshape) final;
    void Visit(Concatenation& concatenation) final;
    void Visit(LeakyRelu& leakyRelu) final;
    void Visit(Sigmoid& sigmoid) final;
    void Visit(Tanh& tanh) final;
    void Visit(MeanXy& meanxy) final;
    void Visit(EstimateOnly& estimateOnly) final;
    void Visit(Addition& addition) final;
    void Visit(Multiplication& multiplication) final;
    void Visit(Resize& resize) final;
    void Visit(Relu& relu) final;
    void Visit(TransposeConvolution& transposeConvolution) final;
    void Visit(ReinterpretQuantization& reinterpretQuantization) final;
    void Visit(DepthToSpace& depthToSpace) final;
    void Visit(Split& split) final;
    void Visit(Transpose& split) final;
    void Visit(SpaceToDepth& split) final;
    void Visit(Requantize& split) final;

    void ConnectParts(Operation& operation, std::vector<BasePart*>& m_Part);
    void ConnectNoOp(Operation& operation);

    std::vector<uint8_t> OverrideWeights(const std::vector<uint8_t>& userWeights, const TensorInfo& weightsInfo) const;

    // Function used to release the GraphOfParts object. Caller should store the object locally, since
    // the function performs an std::move().
    GraphOfParts ReleaseGraphOfParts();

private:
    std::unique_ptr<McePart> CreateIdentityMcePart(const TensorShape& shape,
                                                   const QuantizationInfo& inputQuantInfo,
                                                   const QuantizationInfo& outputQuantInfo,
                                                   uint32_t operationId,
                                                   DataType inputDataType,
                                                   DataType outputDataType,
                                                   const EstimationOptions& estOpt,
                                                   const CompilationOptions& compOpt,
                                                   const HardwareCapabilities& capabilities);

    std::vector<BasePart*> CreateTransposeConv(const Stride& stride,
                                               const TensorInfo& weightsInfo,
                                               const std::vector<uint8_t>& weightsData,
                                               const TensorInfo& biasInfo,
                                               std::vector<int32_t> biasData,
                                               const Padding& padding,
                                               const TensorInfo& inputInfo,
                                               const TensorInfo& outputInfo,
                                               const std::set<uint32_t>& operationIds);

    const HardwareCapabilities& m_Capabilities;
    utils::Optional<const EstimationOptions&> m_EstimationOptions;
    const CompilationOptions& m_CompilationOptions;
    DebuggingContext& m_DebuggingContext;
    SupportQueries m_Queries;
    std::map<const Operand*, BasePart*> m_OperandToPart;
    GraphOfParts m_GraphOfParts;
    ThreadPool& m_ThreadPool;
};

/// Creates an McePart that passes through its input mostly unchanged, except it inserts "padding channels"
/// into the output tensor. These channels will contain entirely zeroes.
/// The `padAmounts` argument defines where and how many padding channels are added. Each entry in the array
/// describes one insertion of padding channels, with .first defining the location in the _original_ channels
/// to start adding padding channels, and the .second defining how many channels to add.
/// An example (ignoring XY):
///     Input: a, b, c, d
///     padAmounts: { { 0, 2 }, { 2, 3 } }
///     Output: 0, 0, a, b, 0, 0, 0, c, d
std::unique_ptr<McePart>
    CreateIdentityMcePartWithPaddedOutputChannels(PartId partId,
                                                  const TensorShape& shape,
                                                  const QuantizationInfo& inputQuantInfo,
                                                  const QuantizationInfo& outputQuantInfo,
                                                  uint32_t operationId,
                                                  DataType inputDataType,
                                                  DataType outputDataType,
                                                  const EstimationOptions& estOpt,
                                                  const CompilationOptions& compOpt,
                                                  const HardwareCapabilities& capabilities,
                                                  const std::vector<std::pair<uint32_t, uint32_t>>& padAmounts,
                                                  DebuggingContext& debuggingContext,
                                                  ThreadPool& threadPool);

/// Creates an McePart that passes through its input mostly unchanged, except it removes specified channels
/// from the output tensor.
/// The `removeAmounts` argument defines where and how many channels are removed. Each entry in the array
/// describes one removal of channels, with .first defining the location in the _original_ channels
/// to start adding removing, and the .second defining how many channels to remove.
/// An example (ignoring XY):
///     Input: a, b, c, d, e, f, g, h
///     removeAmounts: { { 0, 2 }, { 4, 3 } }
///     Output: c, d, h
std::unique_ptr<McePart>
    CreateIdentityMcePartWithRemovedInputChannels(PartId partId,
                                                  const TensorShape& shape,
                                                  const QuantizationInfo& inputQuantInfo,
                                                  const QuantizationInfo& outputQuantInfo,
                                                  uint32_t operationId,
                                                  DataType inputDataType,
                                                  DataType outputDataType,
                                                  const EstimationOptions& estOpt,
                                                  const CompilationOptions& compOpt,
                                                  const HardwareCapabilities& capabilities,
                                                  const std::vector<std::pair<uint32_t, uint32_t>>& removeAmounts,
                                                  DebuggingContext& debuggingContext,
                                                  ThreadPool& threadPool);

}    // namespace support_library
}    // namespace ethosn
