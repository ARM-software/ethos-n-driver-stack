//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../Utils.hpp"
#include "PartUtils.hpp"

namespace ethosn
{
namespace support_library
{

bool NumStripes::operator<(const NumStripes& rhs) const
{
    if (m_Min < rhs.m_Min)
        return true;
    if (rhs.m_Min < m_Min)
        return false;
    if (m_Max < rhs.m_Max)
        return true;
    if (rhs.m_Max < m_Max)
        return false;
    return false;
}

bool MemoryStripeInfo::operator<(const MemoryStripeInfo& rhs) const
{
    if (m_Range < rhs.m_Range)
        return true;
    if (rhs.m_Range < m_Range)
        return false;
    if (m_Shape < rhs.m_Shape)
        return true;
    if (rhs.m_Shape < m_Shape)
        return false;
    return false;
}

bool MemoryStripesInfo::operator<(const MemoryStripesInfo& rhs) const
{
    if (m_Input < rhs.m_Input)
        return true;
    if (rhs.m_Input < m_Input)
        return false;
    if (m_Output < rhs.m_Output)
        return true;
    if (rhs.m_Output < m_Output)
        return false;
    if (m_Weight < rhs.m_Weight)
        return true;
    if (rhs.m_Weight < m_Weight)
        return false;
    if (m_PleInput < rhs.m_PleInput)
        return true;
    if (rhs.m_PleInput < m_PleInput)
        return false;
    return false;
}

bool NumMemoryStripes::operator<(const NumMemoryStripes& rhs) const
{
    if (m_Input < rhs.m_Input)
        return true;
    if (rhs.m_Input < m_Input)
        return false;
    if (m_Output < rhs.m_Output)
        return true;
    if (rhs.m_Output < m_Output)
        return false;
    if (m_Weight < rhs.m_Weight)
        return true;
    if (rhs.m_Weight < m_Weight)
        return false;
    if (m_PleInput < rhs.m_PleInput)
        return true;
    if (rhs.m_PleInput < m_PleInput)
        return false;
    return false;
}

bool DmaOnlyInfo::operator<(const DmaOnlyInfo& rhs) const
{
    if (m_Input < rhs.m_Input)
        return true;
    if (rhs.m_Input < m_Input)
        return false;
    if (m_Output < rhs.m_Output)
        return true;
    if (rhs.m_Output < m_Output)
        return false;
    return false;
}

CascadingBufferFormat PartUtils::GetFormat(Location location)
{
    switch (location)
    {
        case Location::Dram:
            return CascadingBufferFormat::NHWC;
        case Location::PleInputSram:
        case Location::Sram:
            return CascadingBufferFormat::NHWCB;
        case Location::VirtualSram:
            return CascadingBufferFormat::NHWC;
        default:
            throw NotSupportedException("Unkwnown location");
    }
}

CascadingBufferFormat PartUtils::GetCascadingBufferFormatFromCompilerDataFormat(const CompilerDataFormat& format)
{
    switch (format)
    {
        case (CompilerDataFormat::NHWC):
            return CascadingBufferFormat::NHWC;
        case (CompilerDataFormat::NCHW):
            return CascadingBufferFormat::NCHW;
        case (CompilerDataFormat::NHWCB):
            return CascadingBufferFormat::NHWCB;
        case (CompilerDataFormat::WEIGHT):
            return CascadingBufferFormat::WEIGHT;
        default:
        {
            std::string error = "In " + std::string(ETHOSN_FUNCTION_SIGNATURE) + ": value " +
                                std::to_string(static_cast<uint32_t>(format)) + " is not valid";
            throw NotSupportedException(error.c_str());
        }
    }
}

uint32_t PartUtils::CalculateBufferSize(const TensorShape& shape, CascadingBufferFormat f)
{
    switch (f)
    {
        case CascadingBufferFormat::NHWCB:
            return utils::TotalSizeBytesNHWCB(shape);
        case CascadingBufferFormat::NHWC:
            return utils::TotalSizeBytes(shape);
        default:
            assert(false);
            return 0;
    }
}

uint32_t PartUtils::CalculateSizeInBytes(const TensorShape& shape)
{
    return utils::TotalSizeBytesNHWCB(shape);
}

uint32_t PartUtils::CalculateTileSize(const HardwareCapabilities& caps,
                                      const TensorShape& tensorShape,
                                      const TensorShape& stripeShape,
                                      uint32_t numStripes)
{
    // Restrict the tile max size to be the full tensor so we don't waste space when we have partial stripes
    const uint32_t inputFullStripeSize = numStripes * utils::TotalSizeBytesNHWCB(stripeShape);
    const uint32_t inputTileSize       = utils::MaxTileSize(tensorShape, caps);

    return std::min(inputTileSize, inputFullStripeSize);
}

uint32_t PartUtils::CalculateTileSize(Node* node,
                                      const HardwareCapabilities& caps,
                                      const TensorShape& inputTensorShape,
                                      const TensorShape& inputStripeShape,
                                      const TensorShape& outputStripeShape,
                                      uint32_t numStripes)
{
    using namespace ethosn::support_library::utils;

    uint32_t inputFullStripeSize;

    if (IsObjectOfType<MceOperationNode>(node))
    {
        auto mceNode                    = GetObjectAs<MceOperationNode>(node);
        auto kernelHeight               = mceNode->GetWeightsInfo().m_Dimensions[0];
        auto padTop                     = mceNode->GetPadTop();
        const uint32_t brickGroupHeight = GetHeight(caps.GetBrickGroupShape());

        // Work out the tile sizes by deciding how many stripes we want in each tile
        const NeedBoundary needBoundaryY = ethosn::support_library::utils::GetBoundaryRequirements(
            padTop, GetHeight(inputTensorShape), GetHeight(inputStripeShape), GetHeight(outputStripeShape),
            kernelHeight);

        const bool isStreamingWidth = GetWidth(inputStripeShape) < GetWidth(inputTensorShape);

        const bool needsBoundarySlots = (needBoundaryY.m_Before || needBoundaryY.m_After) && (isStreamingWidth);
        const uint32_t inputStripeXZ  = GetWidth(inputStripeShape) * GetChannels(inputStripeShape);

        const uint32_t boundarySlotSize = needsBoundarySlots ? (brickGroupHeight * inputStripeXZ) : 0U;
        const uint32_t defaultSlotSize  = TotalSizeBytes(inputStripeShape);

        // We need the boundary slots both on the top and bottom of the stripe
        const uint32_t totalSlotSize = (2U * boundarySlotSize) + defaultSlotSize;

        inputFullStripeSize = totalSlotSize * numStripes;
    }
    else
    {
        // Restrict the tile max size to be the full tensor so we don't waste space when we have partial stripes
        inputFullStripeSize = numStripes * PartUtils::CalculateSizeInBytes(inputStripeShape);
    }
    const uint32_t inputTileSize = utils::MaxTileSize(inputTensorShape, caps);

    return std::min(inputTileSize, inputFullStripeSize);
}

void PartUtils::AddOpToOpGraphWithInputOutputBuffers(const PartId partId,
                                                     const HardwareCapabilities& capabilities,
                                                     OwnedOpGraph& opGraph,
                                                     Node* node,
                                                     Node* outputNode,
                                                     TraversalOrder order,
                                                     DmaOnlyInfo& info,
                                                     NumMemoryStripes& numMemoryStripes,
                                                     Location inputBufferLocation,
                                                     Location outputBufferLocation,
                                                     PartInputMapping& inputMappings,
                                                     PartOutputMapping& outputMappings)
{
    (void)outputMappings;    //Currently unused but expected to be used whenever multi output will be supported
    auto lifetime = info.m_Lifetime;

    assert(IsObjectOfType<ReinterpretNode>(node) || IsObjectOfType<FormatConversionNode>(node));

    if (IsObjectOfType<ReinterpretNode>(node))
    {
        opGraph.AddOp(std::make_unique<DummyOp>());
    }
    else if (IsObjectOfType<FormatConversionNode>(node))
    {
        opGraph.AddOp(std::make_unique<DmaOp>());
    }

    const OpGraph::BufferList& buffers = opGraph.GetBuffers();
    const OpGraph::OpList& ops         = opGraph.GetOps();
    Op* op                             = ops.back();
    op->m_Lifetime                     = lifetime;
    uint32_t inputIndex                = 0;
    for (Edge* edge : node->GetInputs())
    {
        opGraph.AddBuffer(
            std::make_unique<Buffer>(lifetime, inputBufferLocation, PartUtils::GetFormat(inputBufferLocation), order));
        Buffer* inBuffer        = buffers.back();
        const Node* inputNode   = edge->GetSource();
        inBuffer->m_TensorShape = inputNode->GetShape();
        inBuffer->m_StripeShape = info.m_Input.m_Shape;
        inBuffer->m_NumStripes  = numMemoryStripes.m_Input;
        inBuffer->m_SizeInBytes =
            inputBufferLocation == Location::Sram
                ? CalculateTileSize(node, capabilities, inBuffer->m_TensorShape, info.m_Input.m_Shape,
                                    info.m_Output.m_Shape, numMemoryStripes.m_Input)
                : PartUtils::CalculateBufferSize(inBuffer->m_TensorShape, inBuffer->m_Format);

        inBuffer->m_QuantizationInfo = inputNode->GetQuantizationInfo();
        inputMappings[inBuffer]      = PartInputSlot{ partId, inputIndex };
        opGraph.AddConsumer(inBuffer, op, 0);

        PleOp* pleOp = dynamic_cast<PleOp*>(op);
        if (pleOp)
        {
            pleOp->m_InputStripeShapes.push_back(inBuffer->m_StripeShape);
        }
        inputIndex++;
    }

    if (IsObjectOfType<FormatConversionNode>(node) &&
        (inputBufferLocation == Location::VirtualSram || outputBufferLocation == Location::VirtualSram))
    {
        GetObjectAs<DmaOp>(op)->m_Location = Location::VirtualSram;
    }

    opGraph.AddBuffer(
        std::make_unique<Buffer>(lifetime, outputBufferLocation, PartUtils::GetFormat(outputBufferLocation), order));
    auto outBuffer = buffers.back();
    opGraph.SetProducer(outBuffer, op);

    outBuffer->m_TensorShape = outputNode->GetShape();
    outBuffer->m_StripeShape = info.m_Output.m_Shape;
    outBuffer->m_NumStripes  = numMemoryStripes.m_Output;
    outBuffer->m_SizeInBytes = outputBufferLocation == Location::Sram
                                   ? PartUtils::CalculateTileSize(capabilities, outBuffer->m_TensorShape,
                                                                  outBuffer->m_StripeShape, numMemoryStripes.m_Output)
                                   : PartUtils::CalculateBufferSize(outBuffer->m_TensorShape, outBuffer->m_Format);

    outBuffer->m_QuantizationInfo = outputNode->GetQuantizationInfo();
}

}    // namespace support_library
}    // namespace ethosn
