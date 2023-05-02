#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2018-2019,2021 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import print_function

import sys


def get_pc(inst):
    """
    Get the program counter of instruction inst.
    """
    return int(inst[-8:], 16)


def iter_instructions(mcu_trace_lines):
    """
    Generator that filters mcu_trace lines and only outputs instruction lines (tagged with '[DASM]').
    The '[DASM]' tag is removed and the line is stripped of leading and trailing whitespace.
    """
    tag = "[DASM]"
    pc = -1
    for line in mcu_trace_lines:
        if line.startswith(tag):
            inst = line[len(tag) :].strip()
            prev_pc = pc
            pc = get_pc(inst)
            # mcu trace may contain the line after a WFE repeated several times
            if pc != prev_pc:
                yield inst


def is_branch(inst):
    """
    Return whether inst is a branching instruction.
    """
    inst_start = inst[:3].upper()
    return inst_start[0] == "B" and inst_start not in ("BFC", "BFI", "BIC", "BKP")


def iter_code_blocks(mcu_trace_lines):
    """
    Generator that groups the lines in the input sequence to produce tuples of instructions between branches.
    """
    instructions = iter_instructions(mcu_trace_lines)
    # skip until first WFE
    for inst in instructions:
        if inst.upper().startswith("WFE"):
            break
    block = []
    for inst in instructions:
        block.append(inst)
        if is_branch(inst):
            yield tuple(block)
            block = []
    if block:
        yield tuple(block)


def parse_mcu_trace(mcu_trace_lines):
    """
    Return a list of (count, code_block) tuples in descending order by count and ascending order by program counter.

    count is the number of times the code block was run.

    code_block is a sequence of instructions in mcu_trace format.
    """
    counts = {}
    for b in iter_code_blocks(mcu_trace_lines):
        counts[b] = counts.get(b, 0) + 1

    rcounts_dict = {}
    for b, c in counts.items():
        rcounts_dict[c] = rcounts_dict.get(c, ()) + (b,)

    rcounts = [
        (cb[0], b)
        for cb in sorted(rcounts_dict.items(), reverse=True)
        for b in sorted(cb[1], key=lambda x: get_pc(x[0]))
    ]

    return rcounts


def main():
    """
    Parse an mcu_trace read from stdin and print formatted stats on stdout.
    """
    counts = parse_mcu_trace(sys.stdin)
    print(*("{:4}: {}".format(cb[0], "\n      ".join(cb[1])) for cb in counts), sep="\n\n")
    print("\nCycles = {}".format(sum(cb[0] * len(cb[1]) for cb in counts)))


# Usage: ./mcu_trace_profile.py <ple_00.mcu_trace
if __name__ == "__main__":
    main()
