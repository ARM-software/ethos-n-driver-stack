//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "PleKernelDatabase.hpp"

using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{
namespace plelib
{
namespace impl
{

const PleBlkSizeMap g_PleBlkSizeMap = { { { 8, 8 }, _8X8 },   { { 8, 16 }, _8X16 },   { { 8, 32 }, _8X32 },
                                        { { 16, 8 }, _16X8 }, { { 16, 16 }, _16X16 }, { { 32, 8 }, _8X32 } };

const PleKernelDataTypeMap g_PlekernelDataTypeMap = { { false, U8 }, { true, S8 } };

const PlekernelBlkMulMap plekernelBlkMulMap = { { 1, _1 }, { 2, _2 }, { 4, _4 } };

constexpr PleKernelIdDatabase GeneratePleKernelIdDatabase()
{
    PleKernelIdDatabase database{};

    database.data[PleOpIndex(PleOperation::ADDITION)][U8][_16X16][_1] = PleKernelId::ADDITION_16X16_1;
    database.data[PleOpIndex(PleOperation::ADDITION)][S8][_16X16][_1] = PleKernelId::ADDITION_16X16_1_S;

    database.data[PleOpIndex(PleOperation::ADDITION_RESCALE)][U8][_16X16][_1] = PleKernelId::ADDITION_RESCALE_16X16_1;
    database.data[PleOpIndex(PleOperation::ADDITION_RESCALE)][S8][_16X16][_1] = PleKernelId::ADDITION_RESCALE_16X16_1_S;

    database.data[PleOpIndex(PleOperation::AVGPOOL_3X3_1_1_UDMA)][U8][_16X16][_1] =
        PleKernelId::AVGPOOL_3X3_1_1_UDMA_16X16_1;
    database.data[PleOpIndex(PleOperation::AVGPOOL_3X3_1_1_UDMA)][S8][_16X16][_1] =
        PleKernelId::AVGPOOL_3X3_1_1_UDMA_16X16_1_S;

    database.data[PleOpIndex(PleOperation::INTERLEAVE_2X2_2_2)][U8][_16X16][_1] =
        PleKernelId::INTERLEAVE_2X2_2_2_16X16_1;

    database.data[PleOpIndex(PleOperation::MAXPOOL_2X2_2_2)][U8][_8X8][_4]   = PleKernelId::MAXPOOL_2X2_2_2_8X8_4;
    database.data[PleOpIndex(PleOperation::MAXPOOL_2X2_2_2)][U8][_8X16][_2]  = PleKernelId::MAXPOOL_2X2_2_2_8X16_2;
    database.data[PleOpIndex(PleOperation::MAXPOOL_2X2_2_2)][U8][_16X16][_1] = PleKernelId::MAXPOOL_2X2_2_2_16X16_1;
    database.data[PleOpIndex(PleOperation::MAXPOOL_2X2_2_2)][U8][_8X32][_1]  = PleKernelId::MAXPOOL_2X2_2_2_8X32_1;
    database.data[PleOpIndex(PleOperation::MAXPOOL_2X2_2_2)][S8][_8X8][_4]   = PleKernelId::MAXPOOL_2X2_2_2_8X8_4_S;
    database.data[PleOpIndex(PleOperation::MAXPOOL_2X2_2_2)][S8][_8X16][_2]  = PleKernelId::MAXPOOL_2X2_2_2_8X16_2_S;
    database.data[PleOpIndex(PleOperation::MAXPOOL_2X2_2_2)][S8][_16X16][_1] = PleKernelId::MAXPOOL_2X2_2_2_16X16_1_S;
    database.data[PleOpIndex(PleOperation::MAXPOOL_2X2_2_2)][S8][_8X32][_1]  = PleKernelId::MAXPOOL_2X2_2_2_8X32_1_S;

    database.data[PleOpIndex(PleOperation::MAXPOOL_3X3_2_2_EVEN)][U8][_8X8][_4] =
        PleKernelId::MAXPOOL_3X3_2_2_EVEN_8X8_4;
    database.data[PleOpIndex(PleOperation::MAXPOOL_3X3_2_2_EVEN)][U8][_8X16][_2] =
        PleKernelId::MAXPOOL_3X3_2_2_EVEN_8X16_2;
    database.data[PleOpIndex(PleOperation::MAXPOOL_3X3_2_2_EVEN)][U8][_8X32][_1] =
        PleKernelId::MAXPOOL_3X3_2_2_EVEN_8X32_1;
    database.data[PleOpIndex(PleOperation::MAXPOOL_3X3_2_2_EVEN)][S8][_8X8][_4] =
        PleKernelId::MAXPOOL_3X3_2_2_EVEN_8X8_4_S;
    database.data[PleOpIndex(PleOperation::MAXPOOL_3X3_2_2_EVEN)][S8][_8X16][_2] =
        PleKernelId::MAXPOOL_3X3_2_2_EVEN_8X16_2_S;
    database.data[PleOpIndex(PleOperation::MAXPOOL_3X3_2_2_EVEN)][S8][_8X32][_1] =
        PleKernelId::MAXPOOL_3X3_2_2_EVEN_8X32_1_S;

    database.data[PleOpIndex(PleOperation::MAXPOOL_3X3_2_2_ODD)][U8][_8X8][_4] = PleKernelId::MAXPOOL_3X3_2_2_ODD_8X8_4;
    database.data[PleOpIndex(PleOperation::MAXPOOL_3X3_2_2_ODD)][U8][_8X16][_2] =
        PleKernelId::MAXPOOL_3X3_2_2_ODD_8X16_2;
    database.data[PleOpIndex(PleOperation::MAXPOOL_3X3_2_2_ODD)][U8][_8X32][_1] =
        PleKernelId::MAXPOOL_3X3_2_2_ODD_8X32_1;
    database.data[PleOpIndex(PleOperation::MAXPOOL_3X3_2_2_ODD)][S8][_8X8][_4] =
        PleKernelId::MAXPOOL_3X3_2_2_ODD_8X8_4_S;
    database.data[PleOpIndex(PleOperation::MAXPOOL_3X3_2_2_ODD)][S8][_8X16][_2] =
        PleKernelId::MAXPOOL_3X3_2_2_ODD_8X16_2_S;
    database.data[PleOpIndex(PleOperation::MAXPOOL_3X3_2_2_ODD)][S8][_8X32][_1] =
        PleKernelId::MAXPOOL_3X3_2_2_ODD_8X32_1_S;

    database.data[PleOpIndex(PleOperation::MEAN_XY_7X7)][U8][_8X8][_1] = PleKernelId::MEAN_XY_7X7_8X8_1;
    database.data[PleOpIndex(PleOperation::MEAN_XY_7X7)][S8][_8X8][_1] = PleKernelId::MEAN_XY_7X7_8X8_1_S;
    database.data[PleOpIndex(PleOperation::MEAN_XY_8X8)][U8][_8X8][_1] = PleKernelId::MEAN_XY_8X8_8X8_1;
    database.data[PleOpIndex(PleOperation::MEAN_XY_8X8)][S8][_8X8][_1] = PleKernelId::MEAN_XY_8X8_8X8_1_S;

    database.data[PleOpIndex(PleOperation::PASSTHROUGH)][U8][_8X8][_1]   = PleKernelId::PASSTHROUGH_8X8_1;
    database.data[PleOpIndex(PleOperation::PASSTHROUGH)][U8][_8X8][_2]   = PleKernelId::PASSTHROUGH_8X8_2;
    database.data[PleOpIndex(PleOperation::PASSTHROUGH)][U8][_8X8][_4]   = PleKernelId::PASSTHROUGH_8X8_4;
    database.data[PleOpIndex(PleOperation::PASSTHROUGH)][U8][_16X8][_1]  = PleKernelId::PASSTHROUGH_16X8_1;
    database.data[PleOpIndex(PleOperation::PASSTHROUGH)][U8][_32X8][_1]  = PleKernelId::PASSTHROUGH_32X8_1;
    database.data[PleOpIndex(PleOperation::PASSTHROUGH)][U8][_8X16][_1]  = PleKernelId::PASSTHROUGH_8X16_1;
    database.data[PleOpIndex(PleOperation::PASSTHROUGH)][U8][_8X16][_2]  = PleKernelId::PASSTHROUGH_8X16_2;
    database.data[PleOpIndex(PleOperation::PASSTHROUGH)][U8][_16X16][_1] = PleKernelId::PASSTHROUGH_16X16_1;
    database.data[PleOpIndex(PleOperation::PASSTHROUGH)][U8][_8X32][_1]  = PleKernelId::PASSTHROUGH_8X32_1;

    database.data[PleOpIndex(PleOperation::SIGMOID)][U8][_8X8][_1]   = PleKernelId::SIGMOID_8X8_1;
    database.data[PleOpIndex(PleOperation::SIGMOID)][U8][_8X8][_2]   = PleKernelId::SIGMOID_8X8_2;
    database.data[PleOpIndex(PleOperation::SIGMOID)][U8][_8X8][_4]   = PleKernelId::SIGMOID_8X8_4;
    database.data[PleOpIndex(PleOperation::SIGMOID)][U8][_16X8][_1]  = PleKernelId::SIGMOID_16X8_1;
    database.data[PleOpIndex(PleOperation::SIGMOID)][U8][_32X8][_1]  = PleKernelId::SIGMOID_32X8_1;
    database.data[PleOpIndex(PleOperation::SIGMOID)][U8][_8X16][_1]  = PleKernelId::SIGMOID_8X16_1;
    database.data[PleOpIndex(PleOperation::SIGMOID)][U8][_8X16][_2]  = PleKernelId::SIGMOID_8X16_2;
    database.data[PleOpIndex(PleOperation::SIGMOID)][U8][_16X16][_1] = PleKernelId::SIGMOID_16X16_1;
    database.data[PleOpIndex(PleOperation::SIGMOID)][U8][_8X32][_1]  = PleKernelId::SIGMOID_8X32_1;
    database.data[PleOpIndex(PleOperation::SIGMOID)][S8][_8X8][_1]   = PleKernelId::SIGMOID_8X8_1_S;
    database.data[PleOpIndex(PleOperation::SIGMOID)][S8][_8X8][_2]   = PleKernelId::SIGMOID_8X8_2_S;
    database.data[PleOpIndex(PleOperation::SIGMOID)][S8][_8X8][_4]   = PleKernelId::SIGMOID_8X8_4_S;
    database.data[PleOpIndex(PleOperation::SIGMOID)][S8][_16X8][_1]  = PleKernelId::SIGMOID_16X8_1_S;
    database.data[PleOpIndex(PleOperation::SIGMOID)][S8][_32X8][_1]  = PleKernelId::SIGMOID_32X8_1_S;
    database.data[PleOpIndex(PleOperation::SIGMOID)][S8][_8X16][_1]  = PleKernelId::SIGMOID_8X16_1_S;
    database.data[PleOpIndex(PleOperation::SIGMOID)][S8][_8X16][_2]  = PleKernelId::SIGMOID_8X16_2_S;
    database.data[PleOpIndex(PleOperation::SIGMOID)][S8][_16X16][_1] = PleKernelId::SIGMOID_16X16_1_S;
    database.data[PleOpIndex(PleOperation::SIGMOID)][S8][_8X32][_1]  = PleKernelId::SIGMOID_8X32_1_S;

    database.data[PleOpIndex(PleOperation::TRANSPOSE_XY)][U8][_8X8][_1]   = PleKernelId::TRANSPOSE_XY_8X8_1;
    database.data[PleOpIndex(PleOperation::TRANSPOSE_XY)][U8][_8X8][_2]   = PleKernelId::TRANSPOSE_XY_8X8_2;
    database.data[PleOpIndex(PleOperation::TRANSPOSE_XY)][U8][_8X8][_4]   = PleKernelId::TRANSPOSE_XY_8X8_4;
    database.data[PleOpIndex(PleOperation::TRANSPOSE_XY)][U8][_16X8][_1]  = PleKernelId::TRANSPOSE_XY_16X8_1;
    database.data[PleOpIndex(PleOperation::TRANSPOSE_XY)][U8][_32X8][_1]  = PleKernelId::TRANSPOSE_XY_32X8_1;
    database.data[PleOpIndex(PleOperation::TRANSPOSE_XY)][U8][_8X16][_1]  = PleKernelId::TRANSPOSE_XY_8X16_1;
    database.data[PleOpIndex(PleOperation::TRANSPOSE_XY)][U8][_8X16][_2]  = PleKernelId::TRANSPOSE_XY_8X16_2;
    database.data[PleOpIndex(PleOperation::TRANSPOSE_XY)][U8][_16X16][_1] = PleKernelId::TRANSPOSE_XY_16X16_1;
    database.data[PleOpIndex(PleOperation::TRANSPOSE_XY)][U8][_8X32][_1]  = PleKernelId::TRANSPOSE_XY_8X32_1;

    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][U8][_8X8][_1]   = PleKernelId::LEAKY_RELU_8X8_1;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][U8][_8X8][_2]   = PleKernelId::LEAKY_RELU_8X8_2;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][U8][_8X8][_4]   = PleKernelId::LEAKY_RELU_8X8_4;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][U8][_16X8][_1]  = PleKernelId::LEAKY_RELU_16X8_1;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][U8][_32X8][_1]  = PleKernelId::LEAKY_RELU_32X8_1;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][U8][_8X16][_1]  = PleKernelId::LEAKY_RELU_8X16_1;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][U8][_8X16][_2]  = PleKernelId::LEAKY_RELU_8X16_2;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][U8][_16X16][_1] = PleKernelId::LEAKY_RELU_16X16_1;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][U8][_8X32][_1]  = PleKernelId::LEAKY_RELU_8X32_1;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][S8][_8X8][_1]   = PleKernelId::LEAKY_RELU_8X8_1_S;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][S8][_8X8][_2]   = PleKernelId::LEAKY_RELU_8X8_2_S;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][S8][_8X8][_4]   = PleKernelId::LEAKY_RELU_8X8_4_S;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][S8][_16X8][_1]  = PleKernelId::LEAKY_RELU_16X8_1_S;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][S8][_32X8][_1]  = PleKernelId::LEAKY_RELU_32X8_1_S;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][S8][_8X16][_1]  = PleKernelId::LEAKY_RELU_8X16_1_S;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][S8][_8X16][_2]  = PleKernelId::LEAKY_RELU_8X16_2_S;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][S8][_16X16][_1] = PleKernelId::LEAKY_RELU_16X16_1_S;
    database.data[PleOpIndex(PleOperation::LEAKY_RELU)][S8][_8X32][_1]  = PleKernelId::LEAKY_RELU_8X32_1_S;

    database.data[PleOpIndex(PleOperation::DOWNSAMPLE_2X2)][U8][_8X8][_2]   = PleKernelId::DOWNSAMPLE_2X2_8X8_2;
    database.data[PleOpIndex(PleOperation::DOWNSAMPLE_2X2)][U8][_8X8][_4]   = PleKernelId::DOWNSAMPLE_2X2_8X8_4;
    database.data[PleOpIndex(PleOperation::DOWNSAMPLE_2X2)][U8][_16X8][_1]  = PleKernelId::DOWNSAMPLE_2X2_16X8_1;
    database.data[PleOpIndex(PleOperation::DOWNSAMPLE_2X2)][U8][_32X8][_1]  = PleKernelId::DOWNSAMPLE_2X2_32X8_1;
    database.data[PleOpIndex(PleOperation::DOWNSAMPLE_2X2)][U8][_8X16][_1]  = PleKernelId::DOWNSAMPLE_2X2_8X16_1;
    database.data[PleOpIndex(PleOperation::DOWNSAMPLE_2X2)][U8][_8X16][_2]  = PleKernelId::DOWNSAMPLE_2X2_8X16_2;
    database.data[PleOpIndex(PleOperation::DOWNSAMPLE_2X2)][U8][_16X16][_1] = PleKernelId::DOWNSAMPLE_2X2_16X16_1;
    database.data[PleOpIndex(PleOperation::DOWNSAMPLE_2X2)][U8][_8X32][_1]  = PleKernelId::DOWNSAMPLE_2X2_8X32_1;

    return database;
}

const PleKernelIdDatabase* GetPleKernelIdDatabase()
{
    static const PleKernelIdDatabase database = GeneratePleKernelIdDatabase();
    return &database;
}

}    // namespace impl

using namespace impl;

PleKernelId FindPleKernelIdFromDatabase(BlockConfig blockConfig,
                                        uint32_t stripeWidth,
                                        ethosn::command_stream::DataType outputDataType,
                                        PleOperation op)
{
    PleKernelId id;

    const bool isSignAgnostic = (op == PleOperation::DOWNSAMPLE_2X2) || (op == PleOperation::FAULT) ||
                                (op == PleOperation::INTERLEAVE_2X2_2_2) || (op == PleOperation::PASSTHROUGH) ||
                                (op == PleOperation::TRANSPOSE_XY);

    bool isSigned = (outputDataType == ethosn::command_stream::DataType::S8) && !isSignAgnostic;

    uint8_t blkWidth, blkHeight;

    if (op == PleOperation::ADDITION || op == PleOperation::ADDITION_RESCALE ||
        op == ethosn::command_stream::PleOperation::AVGPOOL_3X3_1_1_UDMA)
    {
        // stand alone PLE kernels are blk size "agnostic"
        // hence block size is fixed to (16, 16)
        blkWidth  = 16;
        blkHeight = 16;
    }
    else
    {
        blkWidth  = static_cast<uint8_t>(blockConfig.m_BlockWidth());
        blkHeight = static_cast<uint8_t>(blockConfig.m_BlockHeight());
    }

    const PleKernelIdDatabase* pleKernelIdDatabase = GetPleKernelIdDatabase();

    // Convert from block size to PleKernelIdBlockSize
    // Block size must be valid
    PleBlkSizeKey blkSizeKey;
    blkSizeKey.blockHeight = blkHeight;
    blkSizeKey.blockWidth  = blkWidth;
    auto blkSizeIt         = g_PleBlkSizeMap.find(blkSizeKey);

    if (blkSizeIt == g_PleBlkSizeMap.end())
    {
        throw InternalErrorException("PleKernelID database: invalid block size");
    }
    PleKernelIdBlockSize blkSize = blkSizeIt->second;

    PleKernelIdDataType pleKernelIdDataType;
    auto dataTypeIt = g_PlekernelDataTypeMap.find(isSigned);

    if (dataTypeIt == g_PlekernelDataTypeMap.end())
    {
        throw InternalErrorException("PleKernelID database: invalid output data type");
    }
    pleKernelIdDataType = dataTypeIt->second;

    PleKernelIdBlockMultiplier blkMultiplier;
    uint32_t bestValue          = 1;
    uint32_t blkMultiplierValue = 1;

    const PleKernelId* pleBlkMulCandidates =
        &(pleKernelIdDatabase->data[PleOpIndex(op)][pleKernelIdDataType][blkSize][_1]);

    bool firstEntry = true;

    // Looks for the best blkMultiplier. The first available value that meets the condition:
    // blkMultiplier * blockWidth >= input stripe width
    // or the one that is the closest to meet this condition.
    for (blkMultiplierValue = 1; blkMultiplierValue <= 4; blkMultiplierValue *= 2, pleBlkMulCandidates++)
    {
        if (*pleBlkMulCandidates == PleKernelId::NOT_FOUND)
        {
            if (firstEntry)
            {
                continue;
            }
            else
            {
                break;
            }
        }

        bestValue = blkMultiplierValue;

        firstEntry = false;

        assert(blkMultiplierValue <= 4);
        if (blkMultiplierValue * blkWidth >= stripeWidth)
        {
            break;
        }
    }

    assert(bestValue <= 4);

    auto blkMulIt = plekernelBlkMulMap.find(bestValue);
    assert(blkMulIt != plekernelBlkMulMap.end());

    blkMultiplier = blkMulIt->second;
    assert(blkMultiplier < NUM_BLOCK_MS);

    id = pleKernelIdDatabase->data[PleOpIndex(op)][pleKernelIdDataType][blkSize][blkMultiplier];

    if (id == PleKernelId::NOT_FOUND)
    {
        throw InternalErrorException("PleKernelID database: invalid PleKernelId");
    }

    return id;
}

}    // namespace plelib
}    // namespace support_library
}    // namespace ethosn
