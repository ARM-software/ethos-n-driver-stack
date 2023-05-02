#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2022-2023 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#


import os
import struct
import re
import argparse
import math
from collections import namedtuple
import subprocess


def getSize(bin):
    with open(bin, "rb") as f_in:
        f_in.seek(0, 2)
        return f_in.tell()


def get_firmware_ver(kernel_module_dir):
    version_re = re.compile(
        r"#define\s+ETHOSN_FIRMWARE_VERSION_(?P<rank>\w+)\s+(?P<value>\d+)",
        re.MULTILINE,
    )
    version = namedtuple("version", ["major", "minor", "patch"])
    header_file_path = os.path.join(kernel_module_dir, "ethosn_firmware.h")
    with open(header_file_path, "r") as fw_ver_in:
        code = fw_ver_in.read()
    return version(**{m[0].lower(): int(m[1]) for m in version_re.findall(code)})


# Returns a dict from .elf section name to tuple of (address, size),
# Note that it's the _address_ we want, not the _offset_, because this is the offset
# in the .bin file.
# Uses the `readelf`` program and parses the table that it outputs.
def get_section_addresses_and_sizes(elf):
    result = {}
    table = subprocess.run(
        ["readelf", "-S", elf], stdout=subprocess.PIPE, check=True, encoding="utf-8"
    ).stdout
    col_name = 0
    col_address = 0
    col_size = 0
    for row in table.splitlines():
        if " Name " in row and " Addr " in row:
            # Header row - extract col addresses
            col_name = row.index("Name ")
            col_address = row.index("Addr ")
            col_size = row.index("Size ")
        elif row.lstrip().startswith("["):
            # Row in table - extract data
            section_name = row[col_name:].split(" ", 2)[0]
            section_address = int(row[col_address:].split(" ", 2)[0], 16)
            section_size = int(row[col_size:].split(" ", 2)[0], 16)
            result[section_name] = (section_address, section_size)
        else:
            # Other line - skip
            pass

    return result


# Big FW header structure (must be in sync with kernel driver)
#
# size in bytes, name, description
# 4, fw_magic,     Firmware magic (FourCC) to identify the binary
# 4, fw_ver_major, Firmware's major version number
# 4, fw_ver_minor, Firmware's minor version number
# 4, fw_ver_patch, Firmware's patch version number
# 4, arch_min,    Minimal supported NPU arch version
# 4, arch_max,    Maximal supported NPU arch version
# 4, offset,      Offset inside BIG image
# 4, size,        Size of FW (all assets). Below fields contain offset and sizes of each individual asset.
# 4, code_offset
# 4, code_size
# 4, ple_offset
# 4, ple_size
# 4, vector_table_offset
# 4, vector_table_size
# 4, unpriv_stack_offset
# 4, unpriv_stack_size
# 4, priv_stack_offset
# 4, priv_stack_size
def write_header(fw_out, bin, elf, arch_max, arch_min, kernel_module_dir, page_size):
    num_headers = 18
    header_size = 4 * num_headers
    # Page aligned offset to the firmware binary in the big fw
    fw_offset = page_size * math.ceil(header_size / page_size)
    # Padding to add after the headers to make the firmware binary page aligned
    header_padding_size = fw_offset - header_size
    fw_size = getSize(bin)
    magic = b"ENFW"
    version = get_firmware_ver(kernel_module_dir)

    print(
        "Packed Firmware magic: {} version: {}.{}.{} arch_min: {:08x} arch_max: {:08x} offset: {} size: {}".format(
            magic.decode(),
            version.major,
            version.minor,
            version.patch,
            arch_min,
            arch_max,
            fw_offset,
            fw_size,
        )
    )

    section_addressses_and_sizes = get_section_addresses_and_sizes(elf)
    fw_out.write(
        struct.pack(
            "<{}I{}x".format(num_headers, header_padding_size),
            int.from_bytes(magic, byteorder="little"),
            version.major,
            version.minor,
            version.patch,
            arch_min,
            arch_max,
            fw_offset,
            fw_size,
            0,
            section_addressses_and_sizes["PLE_DATA"][0],  # Code ends when PLE starts
            *section_addressses_and_sizes["PLE_DATA"],
            *section_addressses_and_sizes["VECTOR_TABLE"],
            *section_addressses_and_sizes["UNPRIV_STACK"],
            *section_addressses_and_sizes["PRIV_STACK"]
        )
    )


def build_big_fw(target, bin, elf, arch_max, arch_min, kernel_module_dir, page_size):
    with open(target, "wb") as fw_out:
        write_header(fw_out, bin, elf, arch_max, arch_min, kernel_module_dir, page_size)

        with open(bin, "rb") as fw_in:
            fw_out.write(fw_in.read())

    return None


def main():
    parser = argparse.ArgumentParser(description="Create packaged firmware binary")
    parser.add_argument("--target", help="Path to output big binary")
    parser.add_argument("--bin", help="Path to firmware binary to be packed")
    parser.add_argument(
        "--elf",
        help="Path to firmware elf file to extract offsets from.",
    )
    parser.add_argument(
        "--arch-max",
        help="Max supported architecture version in packed integer format",
        type=int,
    )
    parser.add_argument(
        "--arch-min",
        help="Min supported architecture version in packed integer format",
        type=int,
    )
    parser.add_argument(
        "--page-size",
        help="Page size the firmware binary should be aligned to (Default: %(default)s)",
        type=int,
        default=65536,
    )
    parser.add_argument("--kernel-module-dir", help="Path to kernel module", type=str)
    args = parser.parse_args()

    build_big_fw(
        args.target,
        args.bin,
        args.elf,
        args.arch_max,
        args.arch_min,
        args.kernel_module_dir,
        args.page_size,
    )


if __name__ == "__main__":
    main()
