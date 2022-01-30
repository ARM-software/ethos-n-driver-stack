//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstring>

namespace ethosn
{
namespace command_stream
{
namespace cascading
{

/// This list uses X macro technique.
/// See https://en.wikipedia.org/wiki/X_Macro for more info
#define ETHOSN_PLE_KERNEL_ID_LIST                                                                                      \
    X(NOT_FOUND)                                                                                                       \
    X(ADDITION_16X16_1)                                                                                                \
    X(ADDITION_16X16_1_S)                                                                                              \
    X(ADDITION_RESCALE_16X16_1)                                                                                        \
    X(ADDITION_RESCALE_16X16_1_S)                                                                                      \
    X(AVGPOOL_3X3_1_1_UDMA_16X16_1)                                                                                    \
    X(AVGPOOL_3X3_1_1_UDMA_16X16_1_S)                                                                                  \
    X(INTERLEAVE_2X2_2_2_16X16_1)                                                                                      \
    X(MAXPOOL_2X2_2_2_8X8_4)                                                                                           \
    X(MAXPOOL_2X2_2_2_8X16_2)                                                                                          \
    X(MAXPOOL_2X2_2_2_16X16_1)                                                                                         \
    X(MAXPOOL_2X2_2_2_8X32_1)                                                                                          \
    X(MAXPOOL_2X2_2_2_8X8_4_S)                                                                                         \
    X(MAXPOOL_2X2_2_2_8X16_2_S)                                                                                        \
    X(MAXPOOL_2X2_2_2_16X16_1_S)                                                                                       \
    X(MAXPOOL_2X2_2_2_8X32_1_S)                                                                                        \
    X(MAXPOOL_3X3_2_2_EVEN_8X8_4)                                                                                      \
    X(MAXPOOL_3X3_2_2_EVEN_8X16_2)                                                                                     \
    X(MAXPOOL_3X3_2_2_EVEN_8X32_1)                                                                                     \
    X(MAXPOOL_3X3_2_2_EVEN_8X8_4_S)                                                                                    \
    X(MAXPOOL_3X3_2_2_EVEN_8X16_2_S)                                                                                   \
    X(MAXPOOL_3X3_2_2_EVEN_8X32_1_S)                                                                                   \
    X(MAXPOOL_3X3_2_2_ODD_8X8_4)                                                                                       \
    X(MAXPOOL_3X3_2_2_ODD_8X16_2)                                                                                      \
    X(MAXPOOL_3X3_2_2_ODD_8X32_1)                                                                                      \
    X(MAXPOOL_3X3_2_2_ODD_8X8_4_S)                                                                                     \
    X(MAXPOOL_3X3_2_2_ODD_8X16_2_S)                                                                                    \
    X(MAXPOOL_3X3_2_2_ODD_8X32_1_S)                                                                                    \
    X(MEAN_XY_7X7_8X8_1)                                                                                               \
    X(MEAN_XY_7X7_8X8_1_S)                                                                                             \
    X(MEAN_XY_8X8_8X8_1)                                                                                               \
    X(MEAN_XY_8X8_8X8_1_S)                                                                                             \
    X(PASSTHROUGH_8X8_1)                                                                                               \
    X(PASSTHROUGH_8X8_2)                                                                                               \
    X(PASSTHROUGH_8X8_4)                                                                                               \
    X(PASSTHROUGH_16X8_1)                                                                                              \
    X(PASSTHROUGH_32X8_1)                                                                                              \
    X(PASSTHROUGH_8X16_1)                                                                                              \
    X(PASSTHROUGH_8X16_2)                                                                                              \
    X(PASSTHROUGH_16X16_1)                                                                                             \
    X(PASSTHROUGH_8X32_1)                                                                                              \
    X(SIGMOID_8X8_1)                                                                                                   \
    X(SIGMOID_8X8_2)                                                                                                   \
    X(SIGMOID_8X8_4)                                                                                                   \
    X(SIGMOID_16X8_1)                                                                                                  \
    X(SIGMOID_32X8_1)                                                                                                  \
    X(SIGMOID_8X16_1)                                                                                                  \
    X(SIGMOID_8X16_2)                                                                                                  \
    X(SIGMOID_16X16_1)                                                                                                 \
    X(SIGMOID_8X32_1)                                                                                                  \
    X(SIGMOID_8X8_1_S)                                                                                                 \
    X(SIGMOID_8X8_2_S)                                                                                                 \
    X(SIGMOID_8X8_4_S)                                                                                                 \
    X(SIGMOID_16X8_1_S)                                                                                                \
    X(SIGMOID_32X8_1_S)                                                                                                \
    X(SIGMOID_8X16_1_S)                                                                                                \
    X(SIGMOID_8X16_2_S)                                                                                                \
    X(SIGMOID_16X16_1_S)                                                                                               \
    X(SIGMOID_8X32_1_S)                                                                                                \
    X(TRANSPOSE_XY_8X8_1)                                                                                              \
    X(TRANSPOSE_XY_8X8_2)                                                                                              \
    X(TRANSPOSE_XY_8X8_4)                                                                                              \
    X(TRANSPOSE_XY_16X8_1)                                                                                             \
    X(TRANSPOSE_XY_32X8_1)                                                                                             \
    X(TRANSPOSE_XY_8X16_1)                                                                                             \
    X(TRANSPOSE_XY_8X16_2)                                                                                             \
    X(TRANSPOSE_XY_16X16_1)                                                                                            \
    X(TRANSPOSE_XY_8X32_1)                                                                                             \
    X(LEAKY_RELU_8X8_1)                                                                                                \
    X(LEAKY_RELU_8X8_2)                                                                                                \
    X(LEAKY_RELU_8X8_4)                                                                                                \
    X(LEAKY_RELU_16X8_1)                                                                                               \
    X(LEAKY_RELU_32X8_1)                                                                                               \
    X(LEAKY_RELU_8X16_1)                                                                                               \
    X(LEAKY_RELU_8X16_2)                                                                                               \
    X(LEAKY_RELU_16X16_1)                                                                                              \
    X(LEAKY_RELU_8X32_1)                                                                                               \
    X(LEAKY_RELU_8X8_1_S)                                                                                              \
    X(LEAKY_RELU_8X8_2_S)                                                                                              \
    X(LEAKY_RELU_8X8_4_S)                                                                                              \
    X(LEAKY_RELU_16X8_1_S)                                                                                             \
    X(LEAKY_RELU_32X8_1_S)                                                                                             \
    X(LEAKY_RELU_8X16_1_S)                                                                                             \
    X(LEAKY_RELU_8X16_2_S)                                                                                             \
    X(LEAKY_RELU_16X16_1_S)                                                                                            \
    X(LEAKY_RELU_8X32_1_S)                                                                                             \
    X(DOWNSAMPLE_2X2_8X8_2)                                                                                            \
    X(DOWNSAMPLE_2X2_8X8_4)                                                                                            \
    X(DOWNSAMPLE_2X2_16X8_1)                                                                                           \
    X(DOWNSAMPLE_2X2_32X8_1)                                                                                           \
    X(DOWNSAMPLE_2X2_8X16_1)                                                                                           \
    X(DOWNSAMPLE_2X2_8X16_2)                                                                                           \
    X(DOWNSAMPLE_2X2_16X16_1)                                                                                          \
    X(DOWNSAMPLE_2X2_8X32_1)

// define actual enum
enum class PleKernelId : uint16_t
{
#define X(a) a,
    ETHOSN_PLE_KERNEL_ID_LIST
#undef X
};

namespace ple_id_detail
{
static constexpr const char* g_PleKernelNames[] = {
#define X(a) #a,
    ETHOSN_PLE_KERNEL_ID_LIST
#undef X
};
static constexpr uint16_t g_PleKernelNamesSize = sizeof(g_PleKernelNames) / sizeof(g_PleKernelNames[0]);
}    // namespace ple_id_detail

inline PleKernelId String2PleKernelId(const char* const str)
{
    for (uint16_t i = 0; i < ple_id_detail::g_PleKernelNamesSize; ++i)
    {
        if (strcmp(str, ple_id_detail::g_PleKernelNames[i]) == 0)
        {
            return static_cast<PleKernelId>(i);
        }
    }
    return PleKernelId::NOT_FOUND;
}

inline const char* PleKernelId2String(const PleKernelId id)
{
    const auto idU32 = static_cast<uint32_t>(id);

    if (idU32 >= ple_id_detail::g_PleKernelNamesSize)
    {
        return "NOT_FOUND";
    }

    return ple_id_detail::g_PleKernelNames[idU32];
}

}    // namespace cascading
}    // namespace command_stream
}    // namespace ethosn
