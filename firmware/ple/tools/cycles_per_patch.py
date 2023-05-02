#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2018-2019,2021 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import print_function

import fileinput
import os
import re
import sys


class TestAttributes(dict):
    def __missing__(self, key):
        if key in ("sW", "sH", "sC"):
            return self["i{}".format(key[1])]
        if key in ("bW", "bH"):
            return 16
        if key == "mceop":
            return 0
        raise KeyError


def main(in_file, out_file, do_tee):
    test_attrs = ("iW", "iH", "iC", "sW", "sH", "sC", "bW", "bH", "mceop")
    csv_fields = test_attrs + ("cycles", "patches", "cxp", "size")

    col_width = max(len(f) for f in csv_fields) + 1
    fmt = "{{:>{}}}".format(col_width)
    print(",".join(fmt.format(f) for f in csv_fields), file=out_file)

    bin_regex = re.compile(r"ple_test .* --bin (?P<bin>\S+\.bin)")
    cycles_prefix = "Cycles = "
    patches_prefix = "Patches = "

    attrs = TestAttributes()
    for line in in_file:
        if line.startswith("run_test"):
            for a in test_attrs:
                m = re.search(r"{}=(\d+)".format(a), line)
                if m:
                    attrs[a] = int(m.group(1))
        m = bin_regex.search(line)
        if m:
            attrs["size"] = os.path.getsize(m.groupdict()["bin"])
        if line.startswith(cycles_prefix):
            attrs["cycles"] = int(line[len(cycles_prefix) :], 0)
        if line.startswith(patches_prefix):
            attrs["patches"] = int(line[len(patches_prefix) :], 0)
            attrs["cxp"] = attrs["cycles"] / attrs["patches"]
            fmt = ",".join(
                "{{:>{}{}}}".format(col_width, ".2f" if f == "cxp" else "") for f in csv_fields
            )
            print(fmt.format(*(attrs[f] for f in csv_fields)), file=out_file)
            attrs = TestAttributes()
        if do_tee:
            sys.stdout.write(line)


# Usage: ./cycles_per_patch.py [INPUT_FILES...] OUTPUT_FILE
#
# Parses the output of PLE unit tests runs and produces a csv file with performance statistics.
# If no input files are given, input is taken from stdin and forwarded to stdout (as tee would do)
#
if __name__ == "__main__":
    in_files = sys.argv[1:-1]
    with open(sys.argv[-1], "w") as outf:
        main(fileinput.input(in_files), outf, not in_files)
