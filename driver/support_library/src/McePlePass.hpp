//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "GraphNodes.hpp"
#include "Pass.hpp"

#include <ethosn_command_stream/PleOperation.hpp>

namespace ethosn
{
namespace support_library
{

class FuseOnlyPleOperationNode;
class FormatConversionNode;
class McePostProcessOperationNode;
class RequantizeNode;

struct LinearNodesOutput
{
    // Keep track of the last set of nodes which can create a pass.
    // This is to prevent the case where we are able to create a pass then try and add an additional node
    // This then fails to create a pass which fails to prepare all the nodes. It should use the previously sucessful pass.
    std::vector<Node*> m_WorkingNodes;
    MceOperationNode* m_MceOperation = nullptr;
    bool m_StrategySelected          = false;
    TensorConfig m_TensorConfig;
    CompilerDataFormat m_RequiredOutputFormat = CompilerDataFormat::NONE;
    BufferLocation m_OutputLocation           = BufferLocation::None;
    SramAllocator m_SramAllocator;
    CompilerMceAlgorithm m_Algorithm = CompilerMceAlgorithm::None;
    std::vector<command_stream::BlockConfig> m_ValidBlockConfigs;
};

/// A set of operations which are evaluated by Ethos-N in a single "pass" through the MCE and PLE.
/// Consists of a single MCE operation (e.g. Convolution), 0 or more MCE post-process operations (e.g. Relu)
/// and optionally a PLE operation (e.g. Pooling).
/// All the operations in this pass are compiled to a single individual command in the command stream.
class McePlePass : public Pass
{
public:
    static std::unique_ptr<McePlePass> CreateGreedily(const HardwareCapabilities& capabilities,
                                                      size_t id,
                                                      std::vector<IStrategy*> allowedStrategies,
                                                      std::vector<command_stream::BlockConfig> allowedBlockConfigs,
                                                      bool enableIntermediateCompression,
                                                      bool enableWinograd,
                                                      Node* firstNode,
                                                      SramAllocator& sramAllocator,
                                                      bool forwardEst);

    McePlePass(const HardwareCapabilities& capabilities,
               size_t id,
               std::vector<Node*> nodes,
               const TensorConfig& tensorConfig,
               BufferLocation outputLocation,
               CompilerDataCompressedFormat intermediateCompressedFormat,
               CompilerMceAlgorithm algorithm,
               uint32_t sramOffset);

    /// Generates this Pass by adding appropriate entries to the given command stream, memory map and buffer table.
    void Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam) override;

    DotAttributes GetDotAttributes() override;

    static bool ChooseAndSetupStrategy(const HardwareCapabilities& capabilities,
                                       SramAllocator& sramAllocator,
                                       std::vector<IStrategy*> allowedStrategies,
                                       std::vector<command_stream::BlockConfig> allowedBlockConfigs,
                                       TensorConfig& tensorConfig,
                                       const TensorShape& inputShape,
                                       const TensorShape& outputShape,
                                       DataFormat weightsFormat,
                                       const TensorShape& weightsShape,
                                       const utils::ShapeMultiplier& shapeMultiplier,
                                       std::pair<bool, uint32_t> inputStaticAndOffset,
                                       CompilerMceAlgorithm algorithm,
                                       const uint32_t depthMax = UINT32_MAX);

private:
    static LinearNodesOutput FindLinearWorkingNodes(Node* firstNode,
                                                    const SramAllocator& sramAllocator,
                                                    const HardwareCapabilities& capabilities,
                                                    std::vector<IStrategy*> allowedStrategies,
                                                    std::vector<command_stream::BlockConfig> allowedBlockConfigs,
                                                    bool enableWinograd);
    // Update the set of block configs to those that are valid for the selected Mce operation or algorithm,
    // e.g.Winograd, FullyConnected
    static std::vector<command_stream::BlockConfig>
        FilterValidAndSortBlockConfigs(MceOperationNode* mceOperation,
                                       FuseOnlyPleOperationNode* pleOperation,
                                       const std::vector<command_stream::BlockConfig>& allowedBlockConfigs,
                                       const HardwareCapabilities& capabilities,
                                       const TensorShape& outputShape,
                                       CompilerMceAlgorithm algorithm);

    // Update the set of strategies to those that are valid for the selected Mce operation or algorithm.
    static std::vector<IStrategy*> GetValidStrategies(MceOperationNode* mceOperation,
                                                      std::vector<IStrategy*> allowedStrategies);

    PassStats GetStats(const EstimationOptions& estimationOptions) override;

    command_stream::PleOperation GetPleOperation() const;

    std::pair<uint32_t, uint32_t> GetWeightStripeSizeAndDepth();

    std::vector<FormatConversionNode*> m_PreConversionNodes;
    ExtractSubtensorNode* m_ExtractSubtensorNode;
    MceOperationNode* m_MceOperation;
    std::vector<McePostProcessOperationNode*> m_McePostProcessOperations;
    FuseOnlyPleOperationNode* m_PleOperation;
    std::vector<FormatConversionNode*> m_PostConversionNodes;
    std::vector<RequantizeNode*> m_RequantizeNodes;

    std::unique_ptr<WeightEncoder> m_WeightEncoder;

    /// Tensor sram allocation information
    TensorConfig m_TensorConfig;
};

}    // namespace support_library
}    // namespace ethosn
