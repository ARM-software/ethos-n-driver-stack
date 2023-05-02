#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020-2021 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#
# See help text for description of what this script is for.

import argparse
import sys
import re

# Detects stack usage from: SUB sp,sp,#0x50
SUB_PATTERN = re.compile(r"SUB\s+sp,sp,#(.*)")
# Detects stack usage from: PUSH {r1,r2-r4}
PUSH_PATTERN = re.compile(r"PUSH\s+{(.*)}")


def calculate_stack_usage(dump_file, verbose):
    total = 0
    with open(dump_file, "rt") as f:
        for line in f:
            sub_match = SUB_PATTERN.search(line)
            push_match = PUSH_PATTERN.search(line)

            if sub_match:
                # Extract the immediate value subtracted from SP (in hex)
                usage = int(sub_match.group(1), 16)
                total += usage
                if verbose:
                    print("{} -> {}".format(line.strip(), usage))
            elif push_match:
                # Parse the list of ranges to find how many registers are pushed
                num_regs = 0
                for reg_range in push_match.group(1).split(","):
                    if "-" in reg_range:
                        (first, last) = reg_range.split("-")
                        num_regs += int(last[1:]) - int(first[1:])
                    else:
                        num_regs += 1

                usage = num_regs * 4  # Each register takes 4 bytes on the stack
                total += usage
                if verbose:
                    print("{} -> {}".format(line.strip(), usage))
    return total


def main():
    parser = argparse.ArgumentParser(
        description="""Calculates the maximum stack usage from a PLE kernel .dump file.
This can be used to determine an *upper bound* for how much stack space is needed at runtime for a particular kernel.
It works by finding all instructions in the dump file which indicate stack usage (e.g. PUSH).
It assumes there is no recursion and therefore that each PUSH (or equivalent) is never executed multiple times
before the corresponding POP.
Note that this will likely give an over-estimate because there is no guarantee that every PUSH that we find will
actually be executed at runtime, e.g. due to a branch.

Example usage:

    find build/release/ -name LEAKY_RELU*.dump -exec tools/stack_usage.py {} \\; | sort -nr

"""
    )
    parser.add_argument("dump_file", help="Path to the .dump file")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Turn on extra logging to help debugging.",
    )
    args = parser.parse_args()

    usage = calculate_stack_usage(args.dump_file, args.verbose)
    print(usage)
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
