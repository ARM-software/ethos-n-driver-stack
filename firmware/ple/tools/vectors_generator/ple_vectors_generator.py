#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2018-2021,2023 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import collections
import sys
import textwrap
import numpy as np

# Constants
PATCH = (1, 4, 4, 1)
GROUP = (1, 8, 8, 1)

# Size in registers of 16 bytes
DFC_SRAM_SIZE = 4096
# Pseudo-random number generator seed
SEED = 42
# Operators
PLE_ONLY_OPERATION = ["ADDITION", "ADDITION_RESCALE", "AVGPOOL_3X3_1_1_UDMA"]
OP_SUPPORTED = [
    "DOWNSAMPLE_2X2",
    "INTERLEAVE_2X2_2_2",
    "LEAKY_RELU",
    "MAXPOOL_2X2_2_2",
    "MAXPOOL_3X3_2_2_EVEN",
    "MAXPOOL_3X3_2_2_ODD",
    "MEAN_XY_7X7",
    "MEAN_XY_8X8",
    "PASSTHROUGH",
    "SIGMOID",
    "TRANSPOSE_XY",
]
OP_SUPPORTED.extend(PLE_ONLY_OPERATION)
# For depthwise convolutions N57 uses only one mce interface at the time
# 1: depthwise operation
# 0: any other operation
MCE_OPERATION_SUPPORTED = [0, 1]
# Number of bytes in a line of 16 hex with '\n'
LINE_LEN = 48

# Hardcoded values for Sigmoid test
sigmoid_multiplier = 44203
sigmoid_shift = 11

# Leaky relu values calculated from the following:
# input quant info as 0.0784313753247261  * (q - 128)
# Alpha = 0.1
# output quant info as 0.04347826086 * (q - 23)
# Calculate the multiplier and shift using CalculateRescaleMultiplierAndShift
# from the support library with the following ratios:
# ratio for input = input scale / output scale
# ratio for alpha = input scale * alpha / output scale
unsigned_leaky_relu_input_multiplier = 59110
unsigned_leaky_relu_input_zero_point = 128
unsigned_leaky_relu_input_shift = 15

unsigned_leaky_relu_alpha_multiplier = 47288
unsigned_leaky_relu_alpha_zero_point = 128
unsigned_leaky_relu_alpha_shift = 18

unsigned_leaky_relu_output_zero_point = 23

unsigned_zero_point = 132
signed_zero_point = -4

# input quant info as  0.21912626922130585 * (q - -8)
# Alpha = 0.1
# output quant info as 0.12603935599327087 * (q - -107)
# Calculate the multiplier and shift using CalculateRescaleMultiplierAndShift
# from the support library with the following ratios:
# ratio for input = input scale / output scale
# ratio for alpha = input scale * alpha / output scale
signed_leaky_relu_input_multiplier = 56968
signed_leaky_relu_input_zero_point = -8
signed_leaky_relu_input_shift = 15

signed_leaky_relu_alpha_multiplier = 45575
signed_leaky_relu_alpha_zero_point = -8
signed_leaky_relu_alpha_shift = 18

signed_leaky_relu_output_zero_point = -107

# Addition Rescale
# Input 0 quant info:
# 0.07028387486934662 * (q - -102)
# Input 1 quant info:
# 0.05844694748520851 * (q - -103)

# Output quant info:
# 0.06924975663423538 * (q - -101)
addition_rescale_zero_point0 = -102
addition_rescale_multiplier0 = 33257
addition_rescale_shift0 = 15


addition_rescale_zero_point1 = -103
addition_rescale_multiplier1 = 55312
addition_rescale_shift1 = 16

addition_rescale_output_zero_point = -101

# Tuple
variant_tuple = collections.namedtuple("VARIANT_TUPLE", ["mceif", "ces", "num_srams"])


# Divide round up
def div_round_up(x, y):
    return (x + y - 1) // y


# Round up to multiple
def round_up_to_multiple(val, mul):
    return div_round_up(val, mul) * mul


def get_nhwcb_shape(config, nhwc_shape):
    num_total_srams = config.get_num_sram() * config.get_ces()
    return tuple(round_up_to_multiple(nhwc_shape, np.array(GROUP[:-1] + (num_total_srams,))))


def get_operator_stride(config):
    if config.op_type in (
        "DOWNSAMPLE_2X2",
        "INTERLEAVE_2X2_2_2",
        "MAXPOOL_2X2_2_2",
        "MAXPOOL_3X3_2_2_EVEN",
        "MAXPOOL_3X3_2_2_ODD",
    ):
        stride = (2, 2)
    else:
        stride = (1, 1)
    return stride


# Calculate output size
def calc_out_size(config, in_size):
    stride = get_operator_stride(config)
    out_size = in_size // np.array((1, stride[1], stride[0], 1))

    if config.op_type == "INTERLEAVE_2X2_2_2":
        num_total_srams = config.get_num_sram() * config.get_ces()
        num_full_srams = ((in_size[3] - 1) % num_total_srams) + 1
        num_partial_srams = num_total_srams - num_full_srams
        out_size[3] = (in_size[3] * 4) + (num_partial_srams * 3)
    elif config.op_type in ("MEAN_XY_7X7", "MEAN_XY_8X8"):
        out_size[1:3] = 1
    elif config.op_type == "TRANSPOSE_XY":
        out_size[1], out_size[2] = out_size[2], out_size[1]

    return out_size


# Sizes class
class Sizes(object):
    def __init__(self, config, in_size, in_stripe_size, mce_block_size):
        self.in_size = in_size
        self.in_stripe_size = get_nhwcb_shape(config, np.minimum(in_stripe_size, in_size))
        self.mce_block_size = mce_block_size
        self.config = config

        self.out_size = tuple(calc_out_size(config, in_size))

        stride = get_operator_stride(config)
        # pylint: disable=assignment-from-no-return
        num_total_srams = config.get_num_sram() * config.get_ces()
        groupSize = (
            (1,)
            + tuple(np.minimum(GROUP[1:3], mce_block_size[1:3] // np.array(stride)))
            + (num_total_srams,)
        )

        self.out_stripe_size = tuple(
            round_up_to_multiple(calc_out_size(config, in_stripe_size), groupSize)
        )


# Config class
class Configuration(object):
    def __init__(
        self,
        ce_id,
        sram_id,
        mce_op,
        op_type,
        is_signed,
        hw_config=variant_tuple(4, 2, 4),
    ):
        self.ce_id = ce_id
        self.hw_config = hw_config
        self.sram_id = sram_id
        self.mce_op = mce_op
        self.op_type = op_type
        self.dtype = "int8" if is_signed else "uint8"
        self.zero_point = signed_zero_point if is_signed else unsigned_zero_point

    def get_num_sram(self):
        return self.hw_config.num_srams

    def get_mceif(self):
        return self.hw_config.mceif

    def get_ces(self):
        return self.hw_config.ces

    def validate_num_ce(self):
        return self.ce_id < self.get_ces()


def right_pad(nd_data, padded_shape, pad_value=0):
    pad_width = [(0, p) for p in padded_shape - np.array(nd_data.shape)]
    return np.pad(nd_data, pad_width, "constant", constant_values=pad_value)


# Write to file an amount of data in lines of 16 hex values
def write_to_file(outf, data):
    assert np.iinfo(data.dtype).bits == 8
    lines = [" ".join(["{:02X}"] * 16).format(*e) for e in data.astype("u1").reshape((-1, 16))]
    print(*lines, sep="\n", file=outf)


def get_ce_data(config, nd_data_nhwc):
    # Pad to sram brick size
    nd_data = right_pad(nd_data_nhwc, get_nhwcb_shape(config, nd_data_nhwc.shape))
    # Slice data corresponding to selected ce_id
    nd_data = nd_data[..., config.ce_id :: config.get_ces()]
    # Swizzling NHWC to NHWCB format is equivalent to transposing a padded and reshaped ND array
    #
    # An NHWC ND array is indexed:        a[n][y][x][z]
    # After padding and reshape (NHWCpr): a[n][y // 8][(y % 8) // 4][y % 4][x // 8][(x % 8) // 4][x % 4][z]
    # An NHWCB ND array is indexed:       a[n][y // 8][x // 8][z][(x % 8) // 4][(y % 8) // 4][y % 4][x % 4]
    #
    # That means NHWC padded and reshaped (NHWCpr) to NHWCB is equivalent to a transpose where axes
    # (0,1,2,3,4,5,6,7) in NHWCB correspond to axes (0, 1, 4, 7, 5, 2, 3, 6) in NHWCpr
    nd_data = nd_data.reshape(
        (
            nd_data.shape[0],
            nd_data.shape[1] // 8,
            2,
            4,
            nd_data.shape[2] // 8,
            2,
            4,
            nd_data.shape[3],
        )
    )
    nd_data = nd_data.transpose((0, 1, 4, 7, 5, 2, 3, 6))
    return nd_data


def get_sram_data(config, sram, nd_data_nhwc):
    # Get data corresponding to selected ce_id
    ce_data = get_ce_data(config, nd_data_nhwc)
    # Slice data corresponding to selected sram
    sram_data = ce_data.take(range(sram, ce_data.shape[3], config.get_num_sram()), axis=3)
    return sram_data


def write_sram_data(config, outf, sram, nd_data_nhwc, stripe_shape, tile_start=0, tile_size=None):
    num_srams = config.get_num_sram()
    num_total_srams = num_srams * config.get_ces()

    tile_end = tile_start + (
        tile_size or (np.prod(get_nhwcb_shape(config, nd_data_nhwc.shape)) // 16)
    )

    stripe_offset = tile_start

    batches, height, width, depth = nd_data_nhwc.shape
    sh, sw, sd = stripe_shape[1:]

    # Size of full (as opposed to partial) stripe in patches
    total_stripe_size = np.prod(stripe_shape) // num_total_srams // 16

    for n in range(batches):
        for sz in range(0, depth, sd):
            for sy in range(0, height, sh):
                for sx in range(0, width, sw):

                    stripe_data_nhwc = nd_data_nhwc[
                        n : (n + 1), sy : (sy + sh), sx : (sx + sw), sz : (sz + sd)
                    ]
                    stripe_data = get_sram_data(config, sram, stripe_data_nhwc)

                    outf.seek(LINE_LEN * stripe_offset)
                    write_to_file(outf, stripe_data)

                    stripe_offset += total_stripe_size
                    if (stripe_offset + total_stripe_size) > tile_end:
                        stripe_offset = tile_start


# Helper function for termination
def print_error(in_string):
    print("ERROR: {}".format(in_string), file=sys.stderr)


# Generate configuration and data information
def setup(args):
    config = Configuration(
        args.engine_number,
        0,
        args.mce_operation,
        args.operator_type,
        args.is_signed,
        variant_tuple(args.num_mceifs, args.num_engines, args.num_srams),
    )

    size = Sizes(config, get_in_size(args), get_stripe_size(args), get_mce_block_size(args))

    return [config, size]


# Generate vectors
def gen_vectors(config, size):
    input_data = gen_input_data(config, size)
    dump_inputs(input_data)

    gen_input_vector_file(config, input_data, size)

    out_data = execute_operator(config, input_data, size)
    dump_output(out_data)

    gen_test_plan(config, size)

    gen_ref_vector_file(config, input_data, out_data, size)


# Create input data of the given size
def gen_input_data(config, size):
    np.random.seed(SEED)

    in_size = size.in_size

    if config.op_type in PLE_ONLY_OPERATION:
        in_size = round_up_to_multiple(in_size, np.array(GROUP))
    if config.op_type.startswith("ADDITION"):
        in_size[0] = 2

    tinfo = np.iinfo(config.dtype)

    return np.random.randint(tinfo.min, tinfo.max + 1, size=in_size, dtype=config.dtype)


# Generate a file containing the input data
def dump_file(filename, nhwc_data):
    nhwc_patch_shape = round_up_to_multiple(nhwc_data.shape, np.array(PATCH))
    nchw_data = right_pad(nhwc_data, nhwc_patch_shape).transpose((0, 3, 1, 2))
    with open(filename, "w") as f:
        write_to_file(f, nchw_data)


# Dump input vectors
def dump_inputs(nhwc_inputs):
    for data_id in range(nhwc_inputs.shape[0]):
        filename = "dump_in_{}.hex".format(data_id)
        dump_file(filename, nhwc_inputs[data_id : data_id + 1])


# Dump output vector
def dump_output(nhwc_output):
    filename = "dump_out.hex"
    dump_file(filename, nhwc_output)


# Generate vector file
def gen_input_vector_file(config, nhwc_inputs, size):
    filename = "in.hex"
    with open(filename, "w") as outf:
        if config.op_type in PLE_ONLY_OPERATION:
            gen_input_vector_file_ple_only(config, outf, nhwc_inputs, size.in_stripe_size)
        else:
            gen_input_vector_file_mce(
                config, outf, nhwc_inputs, size.in_stripe_size, size.mce_block_size
            )


# Generate vector file
def gen_input_vector_file_mce(config, outf, nd_data_nhwc, stripe_shape, block_size):
    num_srams = config.get_num_sram()
    num_active_ogs = config.get_mceif() if config.mce_op == 0 else num_srams
    num_total_active_ogs = num_active_ogs * config.get_ces()

    nhwc_patch_shape = round_up_to_multiple(nd_data_nhwc.shape, np.array(PATCH))
    nd_data_nhwc = right_pad(nd_data_nhwc, nhwc_patch_shape, 0)

    height, width, depth = nd_data_nhwc.shape[1:]
    full_sh, full_sw, full_sd = stripe_shape[1:]
    bh, bw = block_size[1:3]

    # pylint: disable=too-many-nested-blocks
    for sz in range(0, depth, full_sd):
        for sy in range(0, height, full_sh):
            for sx in range(0, width, full_sw):
                sw = min(full_sw, width - sx)
                sh = min(full_sh, height - sy)
                sd = min(full_sd, depth - sz)
                stripe_data_nhwc = nd_data_nhwc[:, sy : (sy + sh), sx : (sx + sw), sz : (sz + sd)]
                for z in range(0, sd, num_total_active_ogs):
                    for y in range(0, sh, bh):
                        for x in range(0, sw, bw):
                            block_data_nhwc = stripe_data_nhwc[
                                :,
                                y : (y + bh),
                                x : (x + bw),
                                z : (z + num_total_active_ogs),
                            ]
                            padded_shape = block_data_nhwc.shape[:-3] + (
                                bh,
                                bw,
                                num_total_active_ogs,
                            )
                            block_data_nhwc = right_pad(block_data_nhwc, padded_shape, 0xFF)
                            block_data = get_ce_data(config, block_data_nhwc)
                            for og in range(num_active_ogs):
                                write_to_file(outf, block_data.take(og, axis=3))


# Generate vector file
def gen_input_vector_file_ple_only(config, outf, nd_data_nhwc, stripe_shape):
    for sram in range(config.get_num_sram()):
        write_sram_data(
            config,
            outf,
            sram,
            nd_data_nhwc,
            stripe_shape,
            tile_start=outf.tell() // LINE_LEN,
        )


# Generate vector file
def gen_ref_vector_file(config, in_data, out_data, size):
    data = out_data
    if config.op_type in PLE_ONLY_OPERATION:
        data = np.concatenate(
            [
                right_pad(d, round_up_to_multiple(d.shape, np.array(GROUP)))
                for d in (in_data, out_data)
            ]
        )
    stripe_shape = tuple(round_up_to_multiple(size.out_stripe_size, np.array(GROUP)))
    for sram in range(config.get_num_sram()):
        filename = "ple_reference_sram{}.hex".format(sram)
        file_zero_init(filename, DFC_SRAM_SIZE)
        with open(filename, "r+") as outf:
            write_sram_data(config, outf, sram, data, stripe_shape, tile_size=DFC_SRAM_SIZE)


# Zero padding
def file_zero_init(filename, size):
    with open(filename, "wt") as f:
        write_to_file(f, np.zeros(size * np.prod(PATCH), "u1"))


def gen_test_plan(config, size):
    # == Plan Commands ==
    #
    # rams: input|output|coderam|ce_sram|in_ram0|in_ram1|out_ram
    #
    # wait - wait for WFE event to occur
    #
    # copy <src ram>:# <dst ram>:# length:#  - copy 'length' bytes from src to dest
    #
    # signal <block> - raise this model signal
    #
    # waitcycles # - step model for '#' cycles
    #
    # printregs - display registers
    #
    # break # - Break at address '#', address must occur before next WFE
    #
    # resume - Continue stepping the PLE after a break
    #
    # cfg_regwrite <regname> <value> - Write to named register via NPU Model
    #
    # dump <ram>
    #
    # end - terminate plan
    if config.op_type in PLE_ONLY_OPERATION:
        gen_test_plan_ple_only(config, size)
    else:
        gen_test_plan_mce(config, size)


def set_ple_scratch_registers(**kwargs):
    return textwrap.dedent(
        """\
    cfg_regwrite CE{ce_id}.CE.PLE_SCRATCH0 0x{tblr:08X}
    cfg_regwrite CE{ce_id}.CE.PLE_SCRATCH1 0x{zero_point_in0:04X}{addr_in0:04X}
    cfg_regwrite CE{ce_id}.CE.PLE_SCRATCH2 0x{shift_in0:04X}{multiplier_in0:04X}
    cfg_regwrite CE{ce_id}.CE.PLE_SCRATCH3 0x{zero_point_in1:04X}{addr_in1:04X}
    cfg_regwrite CE{ce_id}.CE.PLE_SCRATCH4 0x{shift_in1:04X}{multiplier_in1:04X}
    cfg_regwrite CE{ce_id}.CE.PLE_SCRATCH5 0x{zero_point_out:04X}{addr_out:04X}
    cfg_regwrite CE{ce_id}.CE.PLE_SCRATCH6 0x{height:04X}{width:04X}
    cfg_regwrite CE{ce_id}.CE.PLE_SCRATCH7 0x{mce_op:04X}{depth:04X}
    """
    ).format(
        ce_id=kwargs["ce_id"],
        tblr=kwargs.get("tblr", 0),
        addr_in0=kwargs.get("addr_in0", 0),
        zero_point_in0=np.uint16(kwargs.get("zero_point_in0", 0)),
        shift_in0=kwargs.get("shift_in0", 0),
        multiplier_in0=kwargs.get("multiplier_in0", 1),
        addr_in1=kwargs.get("addr_in1", 0),
        zero_point_in1=np.uint16(kwargs.get("zero_point_in1", 0)),
        shift_in1=kwargs.get("shift_in1", 0),
        multiplier_in1=kwargs.get("multiplier_in1", 1),
        addr_out=kwargs["addr_out"],
        zero_point_out=np.uint16(kwargs.get("zero_point_out", 0)),
        width=kwargs["width"],
        height=kwargs["height"],
        depth=kwargs["depth"],
        mce_op=kwargs.get("mce_op", 0),
    )


# Generate test plan when MCE is operating


def gen_test_plan_mce(config, size):
    num_ces = config.get_ces()
    num_srams = config.get_num_sram()
    num_active_ogs = config.get_mceif() if config.mce_op == 0 else num_srams

    num_total_srams = num_srams * num_ces
    num_total_active_ogs = num_active_ogs * num_ces

    stride = get_operator_stride(config)

    total_out_stripe_size = np.prod(round_up_to_multiple(size.out_stripe_size, np.array(GROUP)))
    total_out_stripe_size //= num_total_srams * 16

    iheight, iwidth, idepth = size.in_size[1:]
    full_ish, full_isw, full_isd = size.in_stripe_size[1:]
    bh, bw = size.mce_block_size[1:3]

    oheight, owidth, odepth = size.out_size[1:]
    full_osh, full_osw, full_osd = size.out_stripe_size[1:]

    inram_size = 1024
    total_block_size = bw * bh
    assert (inram_size % total_block_size) == 0
    num_mceif_buffers = inram_size // total_block_size

    test_plan = textwrap.dedent(
        """\
    [plan]
    ; Set mask to enable setirq
    cfg_regwrite CE{ce_id}.CE.PLE_CONTROL_1 0x00410110

    ; Start PLE
    cfg_regwrite CE{ce_id}.CE.PLE_CONTROL_0 0x00000000
    wait

    ; Config MCE interface
    cfg_regwrite CE{ce_id}.GLOBAL.PLE_MCEIF_CONFIG 0x00000{mceif_buf_size:02X}{mceif_num_buf:01X}
    """
    ).format(
        ce_id=config.ce_id,
        mceif_buf_size=(total_block_size // 16) - 1,
        mceif_num_buf=num_mceif_buffers - 1,
    )

    input_offset = 0
    inram_offset = 0
    out_stripe_offset = 0
    osz = 0

    # pylint: disable=too-many-nested-blocks
    for isz in range(0, idepth, full_isd):
        osd = min(full_osd, odepth - osz)
        osy = 0
        for isy in range(0, iheight, full_ish):
            osh = min(full_osh, oheight - osy)
            osx = 0
            for isx in range(0, iwidth, full_isw):
                isw = min(full_isw, iwidth - isx)
                ish = min(full_ish, iheight - isy)
                isd = min(full_isd, idepth - isz)

                osw = min(full_osw, owidth - osx)

                top = isy == 0
                bottom = (isy + ish) == iheight
                left = isx == 0
                right = (isx + isw) == iwidth

                scratch_args = {
                    "ce_id": config.ce_id,
                    "tblr": (top << 0) | (bottom << 1) | (left << 2) | (right << 3),
                    "addr_out": out_stripe_offset,
                    "width": osw,
                    "height": osh,
                    "depth": osd,
                    "mce_op": config.mce_op,
                }

                if config.op_type == "SIGMOID":
                    scratch_args["multiplier_in0"] = sigmoid_multiplier
                    scratch_args["zero_point_in0"] = config.zero_point
                    scratch_args["shift_in0"] = sigmoid_shift
                elif config.op_type == "LEAKY_RELU":
                    if config.dtype == "int8":
                        leaky_relu_input_multiplier = signed_leaky_relu_input_multiplier
                        leaky_relu_input_zero_point = signed_leaky_relu_input_zero_point
                        leaky_relu_input_shift = signed_leaky_relu_input_shift
                        leaky_relu_alpha_multiplier = signed_leaky_relu_alpha_multiplier
                        leaky_relu_alpha_zero_point = signed_leaky_relu_alpha_zero_point
                        leaky_relu_alpha_shift = signed_leaky_relu_alpha_shift
                        leaky_relu_output_zero_point = signed_leaky_relu_output_zero_point
                    else:
                        leaky_relu_input_multiplier = unsigned_leaky_relu_input_multiplier
                        leaky_relu_input_zero_point = unsigned_leaky_relu_input_zero_point
                        leaky_relu_input_shift = unsigned_leaky_relu_input_shift
                        leaky_relu_alpha_multiplier = unsigned_leaky_relu_alpha_multiplier
                        leaky_relu_alpha_zero_point = unsigned_leaky_relu_alpha_zero_point
                        leaky_relu_alpha_shift = unsigned_leaky_relu_alpha_shift
                        leaky_relu_output_zero_point = unsigned_leaky_relu_output_zero_point

                    scratch_args["multiplier_in0"] = leaky_relu_input_multiplier
                    # Convert from signed numbers to their 16 bit unsigned equivalent
                    # e.g. -1 -> 65535
                    scratch_args["zero_point_in0"] = leaky_relu_input_zero_point & (2**16 - 1)
                    scratch_args["shift_in0"] = leaky_relu_input_shift

                    scratch_args["multiplier_in1"] = leaky_relu_alpha_multiplier
                    scratch_args["zero_point_in1"] = leaky_relu_alpha_zero_point & (2**16 - 1)
                    scratch_args["shift_in1"] = leaky_relu_alpha_shift

                    scratch_args["zero_point_out"] = leaky_relu_output_zero_point & (2**16 - 1)

                test_plan += textwrap.dedent(
                    """
                ; Stripe Start (input:\tsx={isx},\tsy={isy},\tsz={isz})
                ;              (output:\tsx={osx},\tsy={osy},\tsz={osz})
                {scratch}

                ; Raise setirq event and wait for completion
                cfg_regwrite CE{ce_id}.CE.PLE_SETIRQ 0x00000010
                wait

                """
                ).format(
                    ce_id=config.ce_id,
                    isx=isx,
                    isy=isy,
                    isz=isz,
                    osx=osx,
                    osy=osy,
                    osz=osz,
                    scratch=set_ple_scratch_registers(**scratch_args),
                )
                for _ in range(0, isd, num_total_active_ogs):
                    for _ in range(0, ish, bh):
                        for _ in range(0, isw, bw):
                            for og in range(num_active_ogs):
                                test_plan += textwrap.dedent(
                                    """\
                                copy input:{input_offset} in_ram{og}:{inram_offset} length:{total_block_size}
                                """
                                ).format(
                                    input_offset=input_offset,
                                    og=og,
                                    inram_offset=inram_offset,
                                    total_block_size=total_block_size,
                                )
                                input_offset += total_block_size
                            test_plan += textwrap.dedent(
                                """\
                            signal block
                            wait
                            """
                            )
                            inram_offset = (inram_offset + total_block_size) % inram_size

                if size.out_stripe_size[1] == PATCH[1]:
                    if ((isy // stride[1]) % GROUP[1]) == 0:
                        out_stripe_offset += 1
                    else:
                        out_stripe_offset += total_out_stripe_size - 1
                elif size.out_stripe_size[2] == PATCH[2]:
                    if ((isx // stride[0]) % GROUP[2]) == 0:
                        out_stripe_offset += 2
                    else:
                        out_stripe_offset += total_out_stripe_size - 2
                else:
                    out_stripe_offset += total_out_stripe_size

                if (
                    (out_stripe_offset - (out_stripe_offset % 4)) + total_out_stripe_size
                ) > DFC_SRAM_SIZE:
                    out_stripe_offset = 0

                osx += full_osw
            osy += full_osh
        osz += full_osd

    for sram in range(num_srams):
        test_plan += "dump ce_sram{}\n".format(sram)
    test_plan += "dump coderam\n"
    test_plan += "end\n"

    filename = "test.plan"
    with open(filename, "wt") as f:
        f.write(test_plan)


def get_ple_scratch_args(config, size, num_patches_per_sram, num_inputs):
    height, width, depth = size.in_size[1:]

    scratch_args = {
        "ce_id": config.ce_id,
        "addr_in0": 0,
        "addr_in1": num_patches_per_sram,
        "addr_out": num_inputs * num_patches_per_sram,
        "width": width,
        "height": height,
        "depth": depth,
    }
    if config.op_type == "ADDITION":
        scratch_args["zero_point_out"] = config.zero_point
    elif config.op_type == "ADDITION_RESCALE":
        scratch_args["zero_point_in0"] = addition_rescale_zero_point0
        scratch_args["shift_in0"] = addition_rescale_shift0
        scratch_args["multiplier_in0"] = addition_rescale_multiplier0
        scratch_args["zero_point_in1"] = addition_rescale_zero_point1
        scratch_args["shift_in1"] = addition_rescale_shift1
        scratch_args["multiplier_in1"] = addition_rescale_multiplier1
        scratch_args["addr_out"] = num_inputs * num_patches_per_sram
        scratch_args["zero_point_out"] = addition_rescale_output_zero_point
    elif config.op_type == "AVGPOOL_3X3_1_1_UDMA":
        # Avg Pool doesn't need to set additional parameters.
        pass
    else:
        assert False

    return scratch_args


# Generate test plan when only PLE is operating
def gen_test_plan_ple_only(config, size):
    num_srams = config.get_num_sram()
    num_total_srams = num_srams * config.get_ces()

    nhwcb = get_nhwcb_shape(config, size.in_size)

    num_inputs = 2 if config.op_type.startswith("ADDITION") else 1
    size_per_sram = np.prod(nhwcb) // num_total_srams
    num_patches_per_sram = size_per_sram // 16

    scratch_args = get_ple_scratch_args(config, size, num_patches_per_sram, num_inputs)

    test_plan = textwrap.dedent(
        """\
    [plan]
    ; Set mask to enable setirq
    cfg_regwrite CE{ce_id}.CE.PLE_CONTROL_1 0x00410110

    ; Start PLE
    cfg_regwrite CE{ce_id}.CE.PLE_CONTROL_0 0x00000000
    wait

    ; Set runtime parameters in scratch registers
    {scratch}

    ; Copy input data
    """
    ).format(ce_id=config.ce_id, scratch=set_ple_scratch_registers(**scratch_args))

    for sram in range(num_srams):
        total_bytes_per_sram = num_inputs * size_per_sram
        test_plan += "copy input:{} ce_sram{}:0 length:{}\n".format(
            sram * total_bytes_per_sram, sram, total_bytes_per_sram
        )

    test_plan += textwrap.dedent(
        """\

    ; Raise setirq event and wait for completion
    cfg_regwrite CE{ce_id}.CE.PLE_SETIRQ 0x00000010
    wait

    ; Dump srams
    """
    ).format(ce_id=config.ce_id)

    for sram in range(num_srams):
        test_plan += "dump ce_sram{}\n".format(sram)

    test_plan += "end\n"

    filename = "test.plan"
    with open(filename, "wt") as f:
        f.write(test_plan)


# Execute operator
def execute_operator(config, data, size):
    if config.op_type == "ADDITION":
        return addition(config, data)
    elif config.op_type == "ADDITION_RESCALE":
        return addition_rescale(data)
    elif config.op_type == "AVGPOOL_3X3_1_1_UDMA":
        return avgpool_3x3_1_1(data, size)
    elif config.op_type == "DOWNSAMPLE_2X2":
        return downsample_2x2(data)
    elif config.op_type == "INTERLEAVE_2X2_2_2":
        return interleave(config, data)
    elif config.op_type == "LEAKY_RELU":
        return leaky_relu(config, data)
    elif config.op_type == "MAXPOOL_2X2_2_2":
        return maxpool_2x2_2_2(data)
    elif config.op_type == "MAXPOOL_3X3_2_2_EVEN" or config.op_type == "MAXPOOL_3X3_2_2_ODD":
        return maxpool_3x3_2_2(data)
    elif config.op_type in ("MEAN_XY_7X7", "MEAN_XY_8X8"):
        return mean_xy(data)
    elif config.op_type == "PASSTHROUGH":
        return passthrough(data)
    elif config.op_type == "SIGMOID":
        return sigmoid(config, data)
    elif config.op_type == "TRANSPOSE_XY":
        return transpose_xy(data, size)
    return []


# Execute passthrough
def passthrough(data):
    return data


# Execute interleave
def interleave(config, data):
    num_total_srams = config.get_num_sram() * config.get_ces()
    data = right_pad(data, round_up_to_multiple(data.shape, np.array((1, 2, 2, num_total_srams))))
    data = data.reshape(data.shape[:-1] + (-1, 1, num_total_srams))
    out_data = np.concatenate(
        (
            data[:, 0::2, 0::2, ...],
            data[:, 0::2, 1::2, ...],
            data[:, 1::2, 0::2, ...],
            data[:, 1::2, 1::2, ...],
        ),
        axis=-2,
    )
    out_data = out_data.reshape(out_data.shape[:3] + (-1,))
    return out_data


# Execute downsample_2x2
def downsample_2x2(data):
    return data[:, ::2, ::2, :]


# Execute maxpool_2x2_2_2
def maxpool_2x2_2_2(data):
    data = right_pad(data, round_up_to_multiple(data.shape, np.array((1, 2, 2, 1))))
    # pylint: disable=assignment-from-no-return
    out_data = np.maximum(
        np.maximum(data[:, 0::2, 0::2, :], data[:, 0::2, 1::2, :]),
        np.maximum(data[:, 1::2, 0::2, :], data[:, 1::2, 1::2, :]),
    )
    return out_data


# Execute maxpool_3x3_2_2
def maxpool_3x3_2_2(data):
    ones = np.array((0, 1, 1, 0))
    data = right_pad(data, round_up_to_multiple(data.shape - ones, GROUP) + ones)
    # pylint: disable=assignment-from-no-return
    out_data = np.maximum(
        np.maximum(data[:, :, 0:-2:2, :], data[:, :, 1:-1:2, :]), data[:, :, 2::2, :]
    )
    out_data = np.maximum(
        np.maximum(out_data[:, 0:-2:2, :, :], out_data[:, 1:-1:2, :, :]),
        out_data[:, 2::2, :, :],
    )
    return out_data


# Execute mean_xy
def mean_xy(data):
    out_data = np.rint(np.mean(data, axis=(1, 2), keepdims=True)).astype(data.dtype)
    return out_data


# Execute avgpool
def avgpool_3x3_1_1(data, size):
    out_data = np.pad(
        data[:, : size.in_size[1], : size.in_size[2], :],
        [(0, 0), (1, 1), (1, 1), (0, 0)],
        "constant",
    )
    out_data = out_data.astype("i4")

    # pylint: disable=assignment-from-no-return
    out_data = np.add(
        np.add(out_data[:, :, 0:-2, :], out_data[:, :, 1:-1, :]), out_data[:, :, 2:, :]
    )
    out_data = np.add(
        np.add(out_data[:, 0:-2, :, :], out_data[:, 1:-1, :, :]), out_data[:, 2:, :, :]
    )

    out_data[:, 1:-1, 1:-1, :] += 4
    out_data[:, 1:-1, 1:-1, :] //= 9
    out_data[:, 1:-1, (0, -1), :] += 3
    out_data[:, 1:-1, (0, -1), :] //= 6
    out_data[:, (0, -1), 1:-1, :] += 3
    out_data[:, (0, -1), 1:-1, :] //= 6
    out_data[:, (0, 0, -1, -1), (0, -1, 0, -1), :] += 2
    out_data[:, (0, 0, -1, -1), (0, -1, 0, -1), :] //= 4

    return out_data.astype(data.dtype)


# Addition
def addition(config, data):
    tinfo = np.iinfo(data.dtype)
    out_data = np.clip(
        data[0:1].astype("i2") + data[1:2] - config.zero_point, tinfo.min, tinfo.max
    ).astype(data.dtype)
    return out_data


# Addition Rescale
def addition_rescale(data):
    tinfo = np.iinfo(data.dtype)
    input0 = (data[0:1].astype("i4") - addition_rescale_zero_point0) * addition_rescale_multiplier0
    input0 = input0 + (1 << (addition_rescale_shift0 - 1))
    input0 = input0 >> addition_rescale_shift0
    input1 = (data[1:2].astype("i4") - addition_rescale_zero_point1) * addition_rescale_multiplier1
    input1 = input1 + (1 << (addition_rescale_shift1 - 1))
    input1 = input1 >> addition_rescale_shift1
    summation = input0 + input1 + addition_rescale_output_zero_point
    clipped = np.clip(summation, tinfo.min, tinfo.max).astype(data.dtype)
    return clipped


# Execute sigmoid
def sigmoid(config, data):
    # Output zero point is always the minimum representable value in the data type
    out_zero_point = np.iinfo(config.dtype).min

    out_data = ((data.astype("i4") - config.zero_point) * sigmoid_multiplier) >> sigmoid_shift
    out_data = 256.0 / (1.0 + np.exp2(-out_data / 256.0))
    out_data = (np.rint(out_data).clip(0, 255) + out_zero_point).astype(data.dtype)

    return out_data


# Execute leaky relu
def leaky_relu(config, data):
    if config.dtype == "int8":
        leaky_relu_input_multiplier = signed_leaky_relu_input_multiplier
        leaky_relu_input_zero_point = signed_leaky_relu_input_zero_point
        leaky_relu_input_shift = signed_leaky_relu_input_shift
        leaky_relu_alpha_multiplier = signed_leaky_relu_alpha_multiplier
        leaky_relu_alpha_zero_point = signed_leaky_relu_alpha_zero_point
        leaky_relu_alpha_shift = signed_leaky_relu_alpha_shift
        leaky_relu_output_zero_point = signed_leaky_relu_output_zero_point
    else:
        leaky_relu_input_multiplier = unsigned_leaky_relu_input_multiplier
        leaky_relu_input_zero_point = unsigned_leaky_relu_input_zero_point
        leaky_relu_input_shift = unsigned_leaky_relu_input_shift
        leaky_relu_alpha_multiplier = unsigned_leaky_relu_alpha_multiplier
        leaky_relu_alpha_zero_point = unsigned_leaky_relu_alpha_zero_point
        leaky_relu_alpha_shift = unsigned_leaky_relu_alpha_shift
        leaky_relu_output_zero_point = unsigned_leaky_relu_output_zero_point

    mul = (data.astype("i4") - leaky_relu_alpha_zero_point) * leaky_relu_alpha_multiplier
    mul = mul + (1 << (leaky_relu_alpha_shift - 1))
    mul = mul >> leaky_relu_alpha_shift
    mul = mul + leaky_relu_output_zero_point
    input_requant = (data.astype("i4") - leaky_relu_input_zero_point) * leaky_relu_input_multiplier
    input_requant = input_requant + (1 << (leaky_relu_input_shift - 1))
    input_requant = input_requant >> leaky_relu_input_shift
    input_requant = input_requant + leaky_relu_output_zero_point
    out_data = np.maximum(mul, input_requant)  # pylint: disable=assignment-from-no-return
    tinfo = np.iinfo(data.dtype)
    out_data = np.clip(out_data, tinfo.min, tinfo.max).astype(data.dtype)
    return out_data


# Execute transpose with swap of X and Y
def transpose_xy(data, size):
    return data.transpose((0, 2, 1, 3))


# Helper function
def get_in_size(args):
    return (1, args.input_height, args.input_width, args.input_depth)


# Helper function
def get_stripe_size(args):
    return (1, args.stripe_height, args.stripe_width, args.stripe_depth)


# Helper function
def get_mce_block_size(args):
    if args.block_height % GROUP[1]:
        print_error("Block height not multiple of group height")
        return None
    if args.block_width % GROUP[2]:
        print_error("Block width not multiple of group width")
        return None
    return (1, args.block_height, args.block_width, 1)


# Main function
def main():
    parser = argparse.ArgumentParser(description="Generate vectors for ple unit tests.")

    parser.add_argument(
        "-op",
        "--operator_type",
        required=True,
        choices=OP_SUPPORTED,
        type=str.upper,
        help="Operator type",
    )

    parser.add_argument(
        "-bw",
        "--block_width",
        required=False,
        type=int,
        default=16,
        help="MCE block Width",
    )
    parser.add_argument(
        "-bh",
        "--block_height",
        required=False,
        type=int,
        default=16,
        help="MCE block height",
    )

    parser.add_argument(
        "-iw", "--input_width", required=True, type=int, help="Width of the input data"
    )
    parser.add_argument(
        "-ih",
        "--input_height",
        required=True,
        type=int,
        help="Height of the input data",
    )
    parser.add_argument(
        "-ic", "--input_depth", required=True, type=int, help="Depth of the input data"
    )

    parser.add_argument(
        "-sw",
        "--stripe_width",
        required=True,
        type=int,
        help="Width of the input stripe",
    )
    parser.add_argument(
        "-sh",
        "--stripe_height",
        required=True,
        type=int,
        help="Height of the input stripe",
    )
    parser.add_argument(
        "-sc",
        "--stripe_depth",
        required=True,
        type=int,
        help="Depth of the input stripe",
    )

    parser.add_argument(
        "-sd",
        "--signed_data",
        required=False,
        action="store_true",
        dest="is_signed",
        default=False,
        help="When given, signed data are generated.",
    )
    parser.add_argument(
        "-ce",
        "--engine_number",
        required=False,
        type=int,
        default=0,
        help="Engine number",
    )

    parser.add_argument(
        "-mceop",
        "--mce_operation",
        required=False,
        type=int,
        default=0,
        choices=MCE_OPERATION_SUPPORTED,
        help="MCE operation",
    )

    parser.add_argument(
        "-ces",
        "--num_engines",
        required=True,
        type=int,
        default=2,
        help="Number of Engines",
    )

    parser.add_argument(
        "-mceif",
        "--num_mceifs",
        required=True,
        type=int,
        default=4,
        help="Number of ple inrams",
    )

    parser.add_argument(
        "-num_srams",
        "--num_srams",
        required=True,
        type=int,
        default=4,
        help="Number of DFCSRAMS",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose enable: Default: %(default)s",
    )
    args = parser.parse_args()

    verbose = args.verbose

    [config, size] = setup(args)
    if size.mce_block_size is None:
        print_error("Mce block size not correct")
        return 1

    if not config.validate_num_ce():
        print_error("Engine number not supported by this variant")
        return 1

    if verbose:
        print("Input dimensions:", size.in_size)
        print("Output dimensions:", size.out_size)
        print("Operator:", config.op_type)
        print("HW Config:", config.hw_config)
        print("Data Type:", config.dtype)

    return gen_vectors(config, size)


if __name__ == "__main__":
    sys.exit(main())
