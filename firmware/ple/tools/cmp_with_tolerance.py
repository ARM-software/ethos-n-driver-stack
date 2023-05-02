#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2018-2019,2021 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import sys
import numpy as np


def read_hex_file(hex_file):
    with open(hex_file, "rt") as f:
        return np.array([int(d, 16) for d in f.read().split()])


def cmp_with_tolerance(file1, file2, tolerance):
    return np.max(np.abs(read_hex_file(file1) - read_hex_file(file2))) <= tolerance


def main():
    parser = argparse.ArgumentParser(description="Compare files with the tolerance value provided.")

    parser.add_argument("f1", help="Path to first file")

    parser.add_argument("f2", help="Path to second file")

    parser.add_argument("-t", "--tolerance", help="Accepted tolerance value", type=int, default=0)

    args = parser.parse_args()

    return cmp_with_tolerance(args.f1, args.f2, args.tolerance)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
