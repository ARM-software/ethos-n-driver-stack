//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Graph.hpp"
#include "Utils.hpp"

#include <ethosn_command_stream/PleOperation.hpp>

namespace ethosn
{
namespace support_library
{

/// A hint to describe what algorithm can be used.
enum class AlgorithmHint
{
    None,
    AllowWinograd,
    RequireDirect
};

class InputNode : public Node
{
public:
    InputNode(NodeId id, const TensorInfo& outputTensorInfo, std::set<uint32_t> correspondingOperationIds);

    bool IsPrepared() override;
    void Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam) override;
    DotAttributes GetDotAttributes() override;
    void Reset() override;
};

class OutputNode : public Node
{
public:
    OutputNode(NodeId id,
               DataType dataType,
               std::set<uint32_t> correspondingOperationIds,
               uint32_t sourceOperationOutputIndex)
        : Node(id, TensorShape(), dataType, QuantizationInfo(), CompilerDataFormat::NONE, correspondingOperationIds)
        , m_SourceOperationOutputIndex(sourceOperationOutputIndex)
    {}    // OutputNode doesn't really have an output...

    bool IsPrepared() override;
    void Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam) override;
    DotAttributes GetDotAttributes() override;
    bool FixGraph(Graph& graph, FixGraphSeverity severity) override;

private:
    uint32_t m_SourceOperationOutputIndex;
};

class ConstantNode : public Node
{
public:
    ConstantNode(NodeId id,
                 const TensorInfo& constantInfo,
                 std::vector<uint8_t> constantData,
                 std::set<uint32_t> correspondingOperationIds)
        : Node(id,
               constantInfo.m_Dimensions,
               constantInfo.m_DataType,
               constantInfo.m_QuantizationInfo,
               ConvertExternalToCompilerDataFormat(constantInfo.m_DataFormat),
               correspondingOperationIds)
        , m_ConstantDataType(constantInfo.m_DataType)
        , m_ConstantData(constantData)
    {}

    const std::vector<uint8_t>& GetConstantData() const;
    const DataType& GetConstantDataType() const;

    void PrepareAfterPassAssignment(SramAllocator& sramAllocator) override;
    bool IsPrepared() override;
    void Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam) override;

    DotAttributes GetDotAttributes() override;

private:
    DataType m_ConstantDataType;
    std::vector<uint8_t> m_ConstantData;
};

class MceOperationNode : public Node
{
public:
    MceOperationNode(NodeId id,
                     const TensorShape& uninterleavedInputTensorShape,
                     const TensorShape& outputTensorShape,
                     DataType dataType,
                     const QuantizationInfo& outputQuantizationInfo,
                     const TensorInfo& weightsInfo,
                     std::vector<uint8_t> weightsData,
                     const TensorInfo& biasInfo,
                     std::vector<int32_t> biasData,
                     Stride stride,
                     uint32_t padTop,
                     uint32_t padLeft,
                     command_stream::MceOperation op,
                     CompilerDataFormat format,
                     std::set<uint32_t> correspondingOperationIds);

    const TensorShape& GetUninterleavedInputShape() const;

    const TensorInfo& GetWeightsInfo() const;
    std::shared_ptr<const std::vector<uint8_t>> GetWeightsData() const;

    const TensorInfo& GetBiasInfo() const;
    const std::vector<int32_t>& GetBiasData() const;

    uint32_t GetPadTop() const;
    uint32_t GetPadLeft() const;

    Stride GetStride() const;
    void SetStride(Stride s);

    uint32_t GetUpscaleFactor() const;
    ethosn::command_stream::UpsampleType GetUpsampleType() const;
    void SetUpsampleParams(const uint32_t upscaleFactor, const ethosn::command_stream::UpsampleType upsampleType);

    ethosn::command_stream::MceOperation GetOperation() const;
    void SetOperation(ethosn::command_stream::MceOperation op);

    ethosn::command_stream::MceData GetMceData() const;

    CompilerMceAlgorithm GetAlgorithm() const;
    CompilerMceAlgorithm GetEffectiveAlgorithm(HardwareCapabilities capabilities, bool isWinogradEnabled) const;
    void SetAlgorithm(CompilerMceAlgorithm a);

    AlgorithmHint GetAlgorithmHint() const;
    void SetAlgorithmHint(AlgorithmHint a);

    AlgorithmHint GetFixGraphAlgorithmHint() const;
    void SetFixGraphAlgorithmHint(AlgorithmHint a);

    bool IsPrepared() override;
    DotAttributes GetDotAttributes() override;

    bool FixGraph(Graph& graph, FixGraphSeverity severity) override;

    void Reset() override;

    utils::ShapeMultiplier GetShapeMultiplier() const;

private:
    TensorShape m_UninterleavedInputShape;
    TensorInfo m_WeightsInfo;
    std::shared_ptr<const std::vector<uint8_t>> m_WeightsData;
    TensorInfo m_BiasInfo;
    std::vector<int32_t> m_BiasData;
    Stride m_Stride;
    uint32_t m_UpscaleFactor;
    ethosn::command_stream::UpsampleType m_UpsampleType;
    uint32_t m_PadTop;
    uint32_t m_PadLeft;
    command_stream::MceOperation m_Operation;
    CompilerMceAlgorithm m_Algorithm;

    AlgorithmHint m_AlgorithmHint;
    AlgorithmHint m_FixGraphAlgorithmHint;
};

class FuseOnlyPleOperationNode : public Node
{
public:
    FuseOnlyPleOperationNode(NodeId id,
                             const TensorShape& outputTensorShape,
                             DataType dataType,
                             const QuantizationInfo& outputQuantizationInfo,
                             command_stream::PleOperation k,
                             CompilerDataFormat format,
                             utils::ShapeMultiplier shapeMultiplier,
                             std::set<uint32_t> correspondingOperationIds);

    command_stream::PleOperation GetKernelOperation() const;

    utils::ShapeMultiplier GetShapeMultiplier() const
    {
        return m_ShapeMultiplier;
    }

    bool IsAgnosticToRequantisation() const;

    bool IsPrepared() override;
    DotAttributes GetDotAttributes() override;

    bool FixGraph(Graph& graph, FixGraphSeverity severity) override;

    virtual void SetOperationSpecificData(command_stream::McePle& data) const;

    void SetFixGraphInsertIdentityNodeHint(bool isIdentityNode);

    bool GetFixGraphInsertIdentityNodeHint() const
    {
        return m_InsertIdentityNodeHint;
    }

private:
    command_stream::PleOperation m_KernelOperation;

    bool m_InsertIdentityNodeHint;

    /// The effect this ple node has on the shape of the output
    utils::ShapeMultiplier m_ShapeMultiplier;
};

class LeakyReluNode : public FuseOnlyPleOperationNode
{
public:
    LeakyReluNode(NodeId id,
                  const TensorShape& outputTensorShape,
                  DataType dataType,
                  const QuantizationInfo& outputQuantizationInfo,
                  command_stream::PleOperation k,
                  CompilerDataFormat format,
                  utils::ShapeMultiplier shapeMultiplier,
                  std::set<uint32_t> correspondingOperationIds,
                  float alpha);

    float GetAlpha() const;

    void SetOperationSpecificData(command_stream::McePle&) const override;

private:
    const float m_Alpha;
};

class StandalonePleOperationNode : public Node
{
public:
    StandalonePleOperationNode(NodeId id,
                               const TensorShape& outputTensorShape,
                               DataType dataType,
                               const QuantizationInfo& outputQuantizationInfo,
                               command_stream::PleOperation k,
                               CompilerDataFormat format,
                               std::set<uint32_t> correspondingOperationIds);

    command_stream::PleOperation GetKernelOperation() const;

    bool IsPrepared() override;
    DotAttributes GetDotAttributes() override;
    bool FixGraph(Graph& graph, FixGraphSeverity severity) override;

private:
    command_stream::PleOperation m_KernelOperation;
};

class McePostProcessOperationNode : public Node
{
public:
    McePostProcessOperationNode(NodeId id,
                                const TensorShape& outputTensorShape,
                                DataType dataType,
                                const QuantizationInfo& outputQuantizationInfo,
                                int16_t lowerBound,
                                int16_t upperBound,
                                CompilerDataFormat format,
                                std::set<uint32_t> correspondingOperationIds);

    void Apply(ethosn::command_stream::MceData& mceData) const;

    bool IsPrepared() override;
    DotAttributes GetDotAttributes() override;
    bool FixGraph(Graph& graph, FixGraphSeverity severity) override;

private:
    int16_t m_LowerBound;
    int16_t m_UpperBound;
};

class SoftmaxNode : public Node
{
public:
    SoftmaxNode(NodeId id,
                const TensorShape& outputTensorShape,
                DataType dataType,
                const QuantizationInfo& outputQuantizationInfo,
                CompilerDataFormat format,
                std::set<uint32_t> correspondingOperationIds);

    bool IsPrepared() override;
};

class RequantizeNode : public Node
{
public:
    RequantizeNode(NodeId id,
                   const TensorShape& outputTensorShape,
                   DataType dataType,
                   const QuantizationInfo& outputQuantizationInfo,
                   CompilerDataFormat format,
                   std::set<uint32_t> correspondingOperationIds);

    bool IsPrepared() override;
    bool FixGraph(Graph& graph, FixGraphSeverity severity) override;
    DotAttributes GetDotAttributes() override;

    // Apply the Requantize node to change the activation min and max of a previous MceOperation
    void Apply(ethosn::command_stream::MceData& mceData, const QuantizationInfo& inputQuantizationInfo) const;
};

class CopyNode : public Node
{
public:
    CopyNode(NodeId id,
             const TensorShape& outputTensorShape,
             DataType dataType,
             const QuantizationInfo& outputQuantizationInfo,
             CompilerDataFormat format,
             std::set<uint32_t> correspondingOperationIds);

    bool IsPrepared() override;
    bool FixGraph(Graph& graph, FixGraphSeverity severity) override;
    DotAttributes GetDotAttributes() override;
};

class FormatConversionNode : public Node
{
public:
    FormatConversionNode(NodeId id,
                         const TensorShape& outputTensorShape,
                         DataType dataType,
                         const QuantizationInfo& outputQuantizationInfo,
                         CompilerDataFormat format,
                         std::set<uint32_t> correspondingOperationIds);

    bool IsPrepared() override;
    DotAttributes GetDotAttributes() override;
    bool FixGraph(Graph& graph, FixGraphSeverity severity) override;
};

class SpaceToDepthNode : public Node
{
public:
    SpaceToDepthNode(NodeId id,
                     const TensorShape& outputTensorShape,
                     DataType dataType,
                     const QuantizationInfo& outputQuantizationInfo,
                     CompilerDataFormat format,
                     std::set<uint32_t> correspondingOperationIds);

    bool IsPrepared() override;
};

class ReinterpretNode : public Node
{
public:
    ReinterpretNode(NodeId id,
                    const TensorShape& outputTensorShape,
                    DataType dataType,
                    const QuantizationInfo& outputQuantizationInfo,
                    CompilerDataFormat format,
                    std::set<uint32_t> correspondingOperationIds);

    bool IsPrepared() override;
    void Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam) override;
    DotAttributes GetDotAttributes() override;

    void PrepareAfterPassAssignment(SramAllocator& sramAllocator) override;
};

class ConcatNode : public Node
{
public:
    ConcatNode(NodeId id,
               const TensorShape& outputTensorShape,
               DataType dataType,
               const QuantizationInfo& outputQuantizationInfo,
               CompilerDataFormat format,
               uint32_t axis,
               std::set<uint32_t> correspondingOperationIds);

    bool IsPrepared() override;
    void Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam) override;
    DotAttributes GetDotAttributes() override;
    bool FixGraph(Graph& graph, FixGraphSeverity severity) override;
    void PrepareAfterPassAssignment(SramAllocator& sramAllocator) override;

    /// Gets the axis along which the concatenation occurs.
    uint32_t GetAxis() const
    {
        return m_Axis;
    }

private:
    uint32_t m_Axis;
};

class ExtractSubtensorNode : public Node
{
public:
    ExtractSubtensorNode(NodeId id,
                         const TensorShape& supertensorOffset,
                         const TensorShape& outputTensorShape,
                         DataType dataType,
                         const QuantizationInfo& outputQuantizationInfo,
                         CompilerDataFormat format,
                         std::set<uint32_t> correspondingOperationIds);

    DotAttributes GetDotAttributes() override;
    bool IsPrepared() override;
    bool FixGraph(Graph& graph, FixGraphSeverity severity) override;

    TensorShape GetSupertensorOffset();

private:
    TensorShape m_SupertensorOffset;
};

class EstimateOnlyNode : public Node
{
public:
    EstimateOnlyNode(NodeId id,
                     const TensorShape& outputTensorShape,
                     DataType dataType,
                     const QuantizationInfo& outputQuantizationInfo,
                     CompilerDataFormat format,
                     std::set<uint32_t> correspondingOperationIds,
                     const char* reasons);

    bool IsPrepared() override;

    void Estimate(NetworkPerformanceData& perfStream, const EstimationOptions& estimationOptions) override;

    DotAttributes GetDotAttributes() override;

private:
    std::string m_ReasonForEstimateOnly;
};

MceOperationNode* CreateIdentityMceOpNode(Graph& graph, Node* previousNode);

void InsertIdentityNode(Graph& graph, Edge* edge);

}    // namespace support_library
}    // namespace ethosn
