//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ethosn
{
namespace command_stream
{

// These enum values must match those in the PLE build system,
// and those in the SPA notebook.
enum class PleOperation : uint8_t
{
    ADDITION,
    ADDITION_RESCALE,
    AVGPOOL_3X3_1_1_UDMA,
    FAULT,
    INTERLEAVE_2X2_2_2,
    MAXPOOL_2X2_2_2,
    MAXPOOL_3X3_2_2_EVEN,
    MAXPOOL_3X3_2_2_ODD,
    MEAN_XY_7X7,
    MEAN_XY_8X8,
    PASSTHROUGH,
    SIGMOID,
    TRANSPOSE_XY,
    LEAKY_RELU,
    DOWNSAMPLE_2X2,
    NUM_OPS,
};

}    // namespace command_stream
}    // namespace ethosn
