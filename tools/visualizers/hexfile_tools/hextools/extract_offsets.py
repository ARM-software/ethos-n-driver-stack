#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2021,2024 Arm Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import print_function

from collections import OrderedDict

import argparse
import glob
import os
import sys
import click

# Append to import search path to access e2e_utils - with the assumption that devtools is checked out next to driver_stack.
scriptPath = os.path.dirname(sys.argv[0])
importPath = os.path.join(scriptPath, "..", "..", "driver_stack", "tools", "end_to_end")
sys.path.append(importPath)
import e2e_utils


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Extract offsets in binding table for buffers")
    parser.add_argument(
        "-i",
        metavar="build_dir",
        default=os.getcwd(),
        help="The build folder produced by end_to_end. Default: %(default)s",
    )
    parser.add_argument(
        "-o",
        metavar="output",
        default="inout_offsets.txt",
        help="The output text file to write offset. Default: %(default)s",
    )
    args = parser.parse_args(args)
    # We will be changing directory so get the abs paths
    args.i = os.path.abspath(args.i)
    args.o = os.path.abspath(args.o)
    return args


def readWord(line):
    address, data = line.split(":")
    address = int(address, 16)
    return address, data.strip()


def readMemoryMap(filename):
    memoryMap = OrderedDict()
    with open(filename, "rt") as f:
        for line in f:
            address, data = readWord(line.strip())
            memoryMap[address] = data
    return memoryMap


def sliceMemoryMap(memoryMap, bufInfo):
    newMemoryMap = OrderedDict()
    for address, data in memoryMap.items():
        if bufInfo.addr <= address < (bufInfo.addr + bufInfo.size):
            newMemoryMap[address] = data
    return newMemoryMap


def writeMemoryMap(filename, memoryMap):
    with open(filename, "wt") as f:
        for address, data in memoryMap.items():
            print("{:08x}: {}".format(address, data), file=f)


def findUniqueFilename(matchPattern):
    files = glob.glob(matchPattern)
    assert files is not None, 'No match for "{}"'.format(matchPattern)
    assert len(files) < 2, "Multiple matches for {}".format(matchPattern)
    return files[0]


def writeOffsetsFile(
    filename, bufferMap, originalLoadAddress, mailboxAddress, inputIndex, outputIndex
):
    # pylint: disable=too-many-arguments
    with open(filename, "wt") as f:
        print(
            "original load address of inference.hex: 0x{:08x}".format(originalLoadAddress), file=f
        )
        print("original mailbox address: 0x{:08x}".format(mailboxAddress), file=f)
        print("buffer pointers:", file=f)
        for index in bufferMap.keys():
            if index == inputIndex:
                bufType = "input"
            elif index == outputIndex:
                bufType = "output"
            else:
                bufType = "other"
            # address and size (uint32_t)
            sizeofBufferInfo = 4 + 4
            # numBuffers (uint32_t), command stream size (uint32_t),
            # inference address (uint32_t), pending, done, inferenceOk (bool), padding
            offsetofBufferInfoTable = 4 + 4
            offsetFromLoadAddress = offsetofBufferInfoTable + sizeofBufferInfo * index
            print(
                "  0x{:08x}: {} (original value: 0x{:08x})".format(
                    offsetFromLoadAddress, bufType, bufferMap[index].addr
                ),
                file=f,
            )


def _findBindingTableFilename(build_dir):
    return findUniqueFilename(os.path.join(build_dir, "*_BindingTable.xml"))


def _findInputAndOutputIndex(build_dir):
    filename = findUniqueFilename(os.path.join(build_dir, "*_CommandStream.xml"))
    with open(filename, "rt") as f:
        inputBuffers = [int(bufId) for bufId in e2e_utils.iter_input_buffers(f)]
    with open(filename, "rt") as f:
        outputBuffers = [int(bufId) for bufId in e2e_utils.iter_output_buffers(f)]
    return inputBuffers[0], outputBuffers[-1]


def _findMemoryMapFilename(build_dir):
    return os.path.join(build_dir, "inference.hex")


def _findFirmwareFilename(build_dir):
    return os.path.join(build_dir, "firmware.hex")


def main(args):
    bindingTableFilename = _findBindingTableFilename(args.i)
    buff_map = e2e_utils.buffer_map(bindingTableFilename)

    inputIndex, outputIndex = _findInputAndOutputIndex(args.i)

    # Cut out inputData.hex and outputData.hex from CombinedMemoryMap.hex
    inferenceMemoryMapFilename = _findMemoryMapFilename(args.i)
    inferenceMemoryMap = readMemoryMap(inferenceMemoryMapFilename)
    writeMemoryMap(
        os.path.join(args.i, "inputData.hex"),
        sliceMemoryMap(inferenceMemoryMap, buff_map[inputIndex]),
    )
    writeMemoryMap(
        os.path.join(args.i, "outputData.hex"),
        sliceMemoryMap(inferenceMemoryMap, buff_map[outputIndex]),
    )

    # Retrieve address of mailbox from firmware.hex file
    # Default address of mailbox is the last 32-bit value of firmware
    firmwareMemoryMap = readMemoryMap(_findFirmwareFilename(args.i))
    dataWords = list(firmwareMemoryMap.values())
    mailboxAddress = int(dataWords[-1].split()[3], 16)

    # Use first address of inference.hex as original load address
    addresses = list(inferenceMemoryMap.keys())
    originalLoadAddress = addresses[0]

    # Write file containing offsets to buffer table and type of data
    writeOffsetsFile(args.o, buff_map, originalLoadAddress, mailboxAddress, inputIndex, outputIndex)


if __name__ == "__main__":
    parsed_args = parse_args()
    sys.exit(main(parsed_args))
