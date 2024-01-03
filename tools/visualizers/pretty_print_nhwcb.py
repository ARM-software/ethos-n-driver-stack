#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2018-2019,2021,2024 Arm Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import print_function

import argparse
import os
import sys
import numpy as np


def round_up_to_multiple(value, m):
    return ((np.add(value, m) - 1) // m) * m


def pad_to_multiple(data, m):
    pad_width = round_up_to_multiple(data.shape, m) - data.shape
    return np.pad(data, [(0, p) for p in pad_width], "constant")


def main(width, depth, data_format, hex_file):
    if hex_file == "-":
        hex_file = sys.stdin
        out_file = sys.stdout
    else:
        out_file = os.path.splitext(hex_file)[0] + "-pretty.txt"

    with (open(hex_file) if isinstance(hex_file, str) else hex_file) as f:
        data = [
            [int(e, 16) for e in (line.split(":")[1] if ":" in line else line).split()]
            for line in f
        ]
    data = np.array(data, dtype="<u4")
    if data.shape[-1] == 16:
        data = data.astype("<u1")
    elif data.shape[-1] == 4:
        data = data.view("<u1")
    else:
        raise ValueError("Invalid hex format: found shape {}".format(data.shape))

    if data_format == "nhwc":
        brick_group_shape = (1, 1, 1, 1, 1)
    elif data_format == "nhwcb":
        brick_group_shape = (2, 2, 16, 4, 4)
    elif data_format == "sram":
        brick_group_shape = (2, 2, 1, 4, 4)
    else:
        raise ValueError("Unknown format: {}".format(data_format))

    brick_group_width = brick_group_shape[0] * brick_group_shape[-1]
    brick_group_depth = brick_group_shape[2]

    width = round_up_to_multiple(width, brick_group_width)
    depth = round_up_to_multiple(depth, brick_group_depth)

    nhwcb_shape = (
        -1,
        width // brick_group_width,
        depth // brick_group_depth,
    ) + brick_group_shape

    nhwcb = pad_to_multiple(data.ravel(), np.prod(nhwcb_shape[1:])).reshape(nhwcb_shape)
    nhwc = pad_to_multiple(
        nhwcb.transpose((0, 4, 6, 1, 3, 7, 2, 5)).reshape((-1, width, depth)), (8, 8, 1)
    )

    height, width, depth = nhwc.shape

    num_groups_x = len(range(0, width, 8))

    with (open(out_file, "w") if isinstance(out_file, str) else out_file) as outf:
        for z in range(depth):
            addr = 0x10 * z
            for gy in range(0, height, 8):
                for gx in range(0, width, 8):
                    s = "+- 0x{:04x} -- y={},x={},z={} -".format(addr, gy, gx, z)
                    s += "-" * (30 - len(s))
                    print(s, end="", file=outf)
                    addr += 0x10 * depth
                print("+", file=outf)
                for y in range(gy, gy + 8):
                    if (y % 4) == 0:
                        print((("|" + (" " * 29)) * num_groups_x) + "|", file=outf)
                    for gx in range(0, width, 8):
                        s = "|  {}   {}  ".format(
                            " ".join(["{:02X}".format(v) for v in nhwc[y, gx + 0 : gx + 4, z]]),
                            " ".join(["{:02X}".format(v) for v in nhwc[y, gx + 4 : gx + 8, z]]),
                        )
                        print(s, end="", file=outf)
                    print("|", file=outf)
                print((("|" + (" " * 29)) * num_groups_x) + "|", file=outf)
        print((("+" + ("-" * 29)) * num_groups_x) + "+", file=outf)


# Usage: ./pretty_print_nhwcb.py width depth [[<]HEX_FILE]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce a data-pretty.txt file from a data.hex file."
    )
    parser.add_argument("-w", "--width", type=int, required=True, help="Tensor width")
    parser.add_argument("-d", "--depth", type=int, required=True, help="Tensor depth")
    parser.add_argument(
        "-f",
        "--format",
        choices=("nhwc", "nhwcb", "sram"),
        required=True,
        help="Data format",
    )
    parser.add_argument("hex_file", nargs="?", default="-", help="Input hex file")

    args = parser.parse_args()

    main(args.width, args.depth, args.format, args.hex_file)
