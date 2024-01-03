#
# Copyright © 2018-2024 Arm Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import argparse
from collections import defaultdict
import glob
import sys
import numpy as np
import pandas as pd
import os
import re
import click
from tqdm import tqdm
import subprocess

hex_file_cache = {}


def round_up_to_multiple(value, m):
    return ((np.add(value, m) - 1) // m) * m


def pad_to_multiple(data, m):
    pad_width = round_up_to_multiple(data.shape, m) - data.shape
    return np.pad(data, [(0, p) for p in pad_width], "constant")


def dump_pretty(nhwc_data_orig, pretty_filename):
    nhwc_data = nhwc_data_orig.astype("u1")

    nhwc_data = pad_to_multiple(nhwc_data, (8, 8, 1))

    height, width, depth = nhwc_data.shape
    num_groups_x = len(range(0, width, 8))

    with open(pretty_filename, "w") as outf:
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
                            " ".join(
                                ["{:02X}".format(v) for v in nhwc_data[y, gx + 0 : gx + 4, z]]
                            ),
                            " ".join(
                                ["{:02X}".format(v) for v in nhwc_data[y, gx + 4 : gx + 8, z]]
                            ),
                        )
                        print(s, end="", file=outf)
                    print("|", file=outf)
                print((("|" + (" " * 29)) * num_groups_x) + "|", file=outf)
        print((("+" + ("-" * 29)) * num_groups_x) + "+", file=outf)


def nhwcb_to_nhwc(data, height, width, depth):
    brick_group_shape = (2, 2, 16, 4, 4)

    brick_group_width = brick_group_shape[0] * brick_group_shape[-1]
    brick_group_depth = brick_group_shape[2]

    rounded_width = round_up_to_multiple(width, brick_group_width)
    rounded_depth = round_up_to_multiple(depth, brick_group_depth)

    nhwcb_shape = (
        -1,
        rounded_width // brick_group_width,
        rounded_depth // brick_group_depth,
    ) + brick_group_shape

    nhwcb = pad_to_multiple(data.ravel(), np.prod(nhwcb_shape[1:])).reshape(nhwcb_shape)
    nhwc = nhwcb.transpose((0, 4, 6, 1, 3, 7, 2, 5)).reshape((-1, rounded_width, rounded_depth))

    # Crop the valid area (remove padding)
    nhwc = nhwc[:height, :width, :depth]

    return nhwc


def read_hex_file(hex_file, datatype_override=None):
    # Check if we already have a cached binary file for this (converted on a previous run, as this is a lot faster!)
    if not hex_file.endswith(".hex"):
        raise Exception("Must be a hex file")
    npy_filename = hex_file.replace(".hex", ".npy")
    if os.path.isfile(npy_filename):
        return np.load(npy_filename)

    # Decode the filename to figure out the datatype, using the override if specified
    if datatype_override == "QAsymmU8":
        datatype = "<u1"
    elif datatype_override == "QAsymmS8":
        datatype = "<i1"
    elif datatype_override == "Float32":
        datatype = "<f4"
    elif datatype_override:
        raise Exception("Unknown format override: {}".format(datatype_override))
    else:
        if "QAsymmS8" in hex_file:
            datatype = "<i1"
        elif "QAsymmU8" in hex_file:
            datatype = "<u1"
        elif "Float32" in hex_file:
            datatype = "<f4"
        elif "UINT8_QUANTIZED" in hex_file:
            datatype = "<u1"
        elif "INT8_QUANTIZED" in hex_file:
            datatype = "<i1"
        elif "Float32" in hex_file:
            datatype = "<f4"
        else:
            raise Exception(
                "Unable to determine data type for filename: {}. You may specify a datatype override if you wish.".format(
                    hex_file
                )
            )

    # Extract shape from filename
    regex = re.compile(r"_(\d+)(_(\d+))?(_(\d+))?(_(\d+))?\.hex$")
    match = regex.search(hex_file)
    if match:
        # Note if trailing dims are missing we defalt to 1, to match the Ethos-N backend behaviour
        n = int(match.group(1) if match.group(1) else 1)
        h = int(match.group(3) if match.group(3) else 1)
        w = int(match.group(5) if match.group(5) else 1)
        c = int(match.group(7) if match.group(7) else 1)
    else:
        raise Exception("Couldn't extract shape from filename: {}".format(hex_file))

    # Decode the filename to figure out the data layout (NHWC, NHWCB etc)
    if "FCAF_WIDE" in hex_file or "FCAF_DEEP" in hex_file:
        # Run the incredible tool to convert FCAF to NHWCB

        if "FCAF_WIDE" in hex_file:
            deepwide = "wide"
        elif "FCAF_DEEP" in hex_file:
            deepwide = "deep"
        else:
            raise Exception("Invalid FCAF format")

        zeropoint = re.search(r"(-?\d+)_FCAF", hex_file)
        if not zeropoint:
            raise Exception("Couldn't extract zero point from filename: {}".format(hex_file))
        zeropoint = zeropoint.group(1)

        # Workaround for signed zero points before the bug fix in driver stack
        if int(zeropoint) > 2147483648:
            zeropoint = int(zeropoint) - 4294967296

        if "UINT8_QUANTIZED" in hex_file:
            signedness = "unsigned"
        elif "INT8_QUANTIZED" in hex_file:
            signedness = "signed"
        else:
            raise Exception("Invalid signedness")

        new_filename = hex_file.replace("FCAF_WIDE", "NHWCB-decoded").replace(
            "FCAF_DEEP", "NHWCB-decoded"
        )
        if not os.path.isfile(new_filename):
            args = [
                r"E:\Marvin\devtools\FcafDecode\x64\Release\FcafDecode.exe",
                hex_file,
                str(w),
                str(h),
                str(c),
                deepwide,
                str(zeropoint),
                signedness,
                new_filename,
            ]
            if True:
                print(" ".join(args))
            subprocess.check_call(args)
        return read_hex_file(new_filename, datatype_override)

    elif "NHWCB" in hex_file:
        fmt = "nhwcb"
    else:
        fmt = "nhwc"

    with open(hex_file) as f:
        data = [
            [int(e, 16) for e in (line.split(":")[1] if ":" in line else line).split()]
            for line in f
        ]

    data = np.array(data, dtype="<u4")
    if data.shape[-1] != 4:
        raise Exception("Invalid hex format: found shape {}".format(data.shape))

    data = data.view(datatype).flatten()

    # Convert from NHWCB to NHWC if required
    if fmt == "nhwcb":
        data = nhwcb_to_nhwc(data, h, w, c).flatten()
    elif fmt == "nhwc":
        # Remove trailing zero bytes (hex dumps may be rounded up to 16)
        data = data[: (n * h * w * c)]

    # Reshape to shape
    data = np.reshape(data, (h, w, c))

    data = data.astype("f4")

    # Save a pretty-printed version for easier manual diffs
    pretty_filename = hex_file.replace(".hex", ".txt")
    if not os.path.isfile(pretty_filename):
        dump_pretty(data, pretty_filename)

    # Save a cached binary file for this, so that future runs of the script can pick this up to be a lot faster
    np.save(npy_filename, data)

    return data


def load_hex_file_into_cache(hex_file, datatype_override=None):
    key = (hex_file, datatype_override)
    value = read_hex_file(hex_file, datatype_override)
    hex_file_cache[key] = value


def get_hex_file(hex_file, datatype_override=None):
    key = (hex_file, datatype_override)
    return hex_file_cache[key]


def compare_files(first_filename, second_filename, datatype_override):
    first_data = get_hex_file(first_filename, datatype_override)
    second_data = get_hex_file(second_filename, datatype_override)

    if first_data.shape != second_data.shape:
        raise Exception("Different shapes!")

    diff = np.abs(first_data - second_data)
    return diff.max(), diff.std(), diff


def find_closest_candidate(auto_candidates, reference):
    best = None
    for c in auto_candidates:
        try:
            max_diff, sd, diff = compare_files(reference, c, None)

            if not best or max_diff < best[0]:
                best = (max_diff, c)
        except BaseException as error:
            pass

    return best[1] if best else None


def process_csv(csv_filename, armnn_dot_filename):
    # Read CSV. Empty cells -> empty strings rather than NaNs
    df = pd.read_csv(csv_filename, keep_default_na=False)

    # Add new columns to DataFrame that we will fill in with the results, and get the name of the reference and test datasets
    reference_dataset = df.columns[0]
    test_datasets = []
    for idx, column in reversed(list(enumerate(df))):  # Reverse order so that insertion works
        if column != "DataTypeOverride" and column != reference_dataset:
            test_datasets.insert(0, column)  # Reverse order to cancel-out reverse loop
            df.insert(idx + 1, column + " Max Diff", "")
            df.insert(idx + 2, column + " Standard Deviation", "")
    # Compare each test dataset against each other as well
    for test_dataset_a in test_datasets:
        for test_dataset_b in test_datasets:
            if test_dataset_b == test_dataset_a:
                break
            df[test_dataset_a + " vs " + test_dataset_b + " Max Diff"] = ""
            df[test_dataset_a + " vs " + test_dataset_b + " Standard Deviation"] = ""

    # Find all entries in the test columns that don't have a corresponding entry in the other columns, as we will use
    # these to match up <Auto> entries
    # Also load all hex files into the cache
    auto_candidates = defaultdict(list)
    for index, row in tqdm(df.iterrows(), total=len(df.index), desc="Loading hex files"):
        if not row[reference_dataset]:
            found = None
            for col in test_datasets:
                if row[col]:
                    if found:
                        found = None
                        break
                    else:
                        found = col
            if found:
                if row[found].startswith("GLOB:"):
                    expanded = [
                        f for f in glob.glob(row[found][5:]) if not "-decoded" in f
                    ]  # Exclude decoded FCAF files
                    auto_candidates[found].extend(expanded)
                else:
                    auto_candidates[found].append(row[found])

        for col in row:
            if col and col != "<Auto>":
                if col.startswith("GLOB:"):
                    expanded = [
                        f for f in glob.glob(col[5:]) if not "-decoded" in f
                    ]  # Exclude decoded FCAF files
                    for i in expanded:
                        load_hex_file_into_cache(i, row.get("DataTypeOverride", default=None))
                else:
                    load_hex_file_into_cache(col, row.get("DataTypeOverride", default=None))

    # Process each row in turn
    for index, row in tqdm(df.iterrows(), total=len(df.index), desc="Processing comparisons"):
        friendly_index = index + 2  # The data rows in Excel start at 2, not 0

        if row[reference_dataset]:  # Some rows might not be ready for comparison
            for test_dataset in test_datasets:
                if row[test_dataset]:  # Some rows might not be ready for comparison
                    try:
                        if row[test_dataset] == "<Auto>":
                            print(
                                "Row {} - finding best Auto candidate for test dataset {} ...".format(
                                    friendly_index, test_dataset
                                )
                            )
                            closest = find_closest_candidate(
                                auto_candidates[test_dataset], row[reference_dataset]
                            )
                            if not closest:
                                raise Exception("Can't find Auto candidate")
                            row[test_dataset] = closest
                        max_diff, sd, diff = compare_files(
                            row[reference_dataset],
                            row[test_dataset],
                            row.get("DataTypeOverride", default=None),
                        )
                        print(
                            "Row {} vs {} max diff: {}, sd: {}".format(
                                friendly_index, test_dataset, max_diff, sd
                            )
                        )
                    except BaseException as error:
                        print("Row {} vs {}, error: {}".format(friendly_index, test_dataset, error))
                        max_diff = ""
                        sd = ""
                    row[test_dataset + " Max Diff"] = max_diff
                    row[test_dataset + " Standard Deviation"] = sd

            # Compare each test dataset against each other as well. Note we have to do this after resolving auto candidates (in the above loop)
            for test_dataset_a in test_datasets:
                if (
                    row[test_dataset_a] and row[test_dataset_a] != "<Auto>"
                ):  # Some rows might not be ready for comparison
                    for test_dataset_b in test_datasets:
                        if test_dataset_b == test_dataset_a:
                            break
                        if (
                            row[test_dataset_b] and row[test_dataset_b] != "<Auto>"
                        ):  # Some rows might not be ready for comparison
                            max_diff, sd, diff = compare_files(
                                row[test_dataset_a], row[test_dataset_b], None
                            )
                            row[test_dataset_a + " vs " + test_dataset_b + " Max Diff"] = max_diff
                            row[
                                test_dataset_a + " vs " + test_dataset_b + " Standard Deviation"
                            ] = sd

    name, _ = os.path.splitext(csv_filename)
    csv_output_filename = "{}Out.csv".format(name)
    df.to_csv(csv_output_filename)
    print("Wrote CSV output to {}".format(csv_output_filename))

    # If provided, augment the Arm NN dot file
    if armnn_dot_filename:
        with open(armnn_dot_filename) as f:
            dot = ""
            # Remove Constant layers, as there can be a lot of these (weights, biases),
            # and they clutter the graph. This might be unhelpful in some cases (constants used as regular
            # inputs to layers, but this isn't very common), but the helpfulness is most cases outweights this.
            constant_guids = []
            for line in f.readlines():
                parts = line.strip().split(" ", 3)
                # Remember the GUIDs for all the constant layers, so we can remove connections from them too
                if "\lLayerType : Constant\l" in line:
                    constant_guids.append(parts[0])
                elif len(parts) > 2 and parts[1] == "->" and parts[0] in constant_guids:
                    # Connection from a constant layer - ignore
                    pass
                else:
                    # Regular line, retain
                    dot += line

        for index, row in df.iterrows():
            if row[reference_dataset]:  # Some rows might not have been compared
                # Attempt to extract the Arm NN layer GUID from the filename
                # e.g. Ref/Armnn_CpuRef_Tensor_Layer014_Conv2D-0-8_Slot0_QAsymmS8_1_144_256_24.hex
                regex = re.compile(r"_Tensor_Layer(\d+)_")
                match = regex.search(row[reference_dataset])
                if not match:
                    print(
                        'Failed to extract Arm NN guid from "',
                        row[reference_dataset],
                        '", this row will not be added to the dot file.',
                    )
                    continue
                guid = int(match.group(1))

                # Figure out how we are editing this node, based on the test datasets
                label_additions = ""

                # Show the standard deviation and range of the reference data, to give an indication of its distribution
                ref_data = read_hex_file(
                    row[reference_dataset], row.get("DataTypeOverride", default=None)
                )
                label_additions += r"""\l\l{}\l{}\l""".format(
                    reference_dataset
                    + " range = "
                    + str(ref_data.min())
                    + " - "
                    + str(ref_data.max()),
                    reference_dataset + " s.d. = " + str(ref_data.std()),
                )

                max_max_diff = None
                for test_dataset in test_datasets:
                    if (
                        row[test_dataset] and row[test_dataset] != "<Auto>"
                    ):  # Some rows might not be ready for comparison
                        max_diff = row[test_dataset + " Max Diff"]
                        if max_diff != "" and (not max_max_diff or max_diff > max_max_diff):
                            max_max_diff = max_diff

                        label_additions += r"""\l\l{}\l{}\l{}\l""".format(
                            test_dataset + " = " + row[test_dataset],
                            test_dataset + " Max Diff = " + str(max_diff),
                            test_dataset
                            + " Standard Deviation = "
                            + str(row[test_dataset + " Standard Deviation"]),
                        )

                # Compare each test dataset against each other as well
                for test_dataset_a in test_datasets:
                    if (
                        row[test_dataset_a] and row[test_dataset_a] != "<Auto>"
                    ):  # Some rows might not be ready for comparison
                        for test_dataset_b in test_datasets:
                            if test_dataset_b == test_dataset_a:
                                break
                            if (
                                row[test_dataset_b] and row[test_dataset_b] != "<Auto>"
                            ):  # Some rows might not be ready for comparison
                                max_diff = row[
                                    test_dataset_a + " vs " + test_dataset_b + " Max Diff"
                                ]
                                sd = row[
                                    test_dataset_a + " vs " + test_dataset_b + " Standard Deviation"
                                ]
                                label_additions += r"""\l\l{}\l{}\l""".format(
                                    test_dataset_a
                                    + " vs "
                                    + test_dataset_b
                                    + " Max Diff = "
                                    + str(max_diff),
                                    test_dataset_a
                                    + " vs "
                                    + test_dataset_b
                                    + " Standard Deviation = "
                                    + str(sd),
                                )

                if label_additions:  # Some rows might not have been compared
                    # Find the layer with this GUID in the dot file
                    regex = re.compile(r"""(Guid : {}\\l.*)(\\l}}"];)""".format(guid))
                    # Update it with the new information
                    def clamp(x):
                        return min(255, max(0, int(x)))

                    color = "black"
                    if max_max_diff is not None:
                        color = (
                            clamp(max_max_diff * 4),
                            clamp(255 - max_max_diff * 4),
                            0,
                        )  # RGB tuple (0-255)
                        color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

                    (dot, n) = regex.subn(
                        r"""\1{}}}", color="{}"];""".format(re.escape(label_additions), color), dot
                    )
                    if n != 1:
                        print("Failed to augment Arm NN guid: ", guid)

        name, _ = os.path.splitext(armnn_dot_filename)
        output_dot_filename = "{}Out.dot".format(name)
        with open(output_dot_filename, "w") as f:
            f.write(dot)
        print("Wrote augmented dot file to {}".format(output_dot_filename))

    return True


@click.command("compare-dumps")
@click.option(
    "--csv_filename",
    required=True,
    default="",
    help="""Path to the CSV file to process. Each row in the file (after the header)
represents a set of comparisons to perform. The first column defines a 'reference', which is compared against the entry in each
of the other 'test' columns. Therefore the file is expected to have at least two columns, with each column containing a set of dump
files. For example you could have three columns:
'CpuRef': Hex dump files for an Arm NN reference inference.
'Cascading': Hex dump files from a cascading inference
'Legacy': Hex dump files from a legacy inference
Some information (e.g. data type, tensor shape) is extracted from the filename itself.
A special value "<Auto>" can be used to attempt to automatically determine which hex file to use for a test column,
by finding the hex file that gives the minimum diff from any hex files in that column which don't have a corresponding entry
in any other column.
The following columns are optional:
'DataTypeOverride': Overrides the datatype that is normally extracted from the filenames.
""",
)
@click.option(
    "--armnn_dot_filename",
    required=False,
    default="",
    help="""Path to an Arm NN optimised graph dot file.
If provided, an augmented version of this will be written out, showing the difference between the first (reference) column and each other column.
""",
)
def cli(csv_filename, armnn_dot_filename):
    """
    Compare two sets of hex dump files, defined by a CSV file.
    The results are written to a second CSV file adjacent to the input, with a modified name.
    """

    return process_csv(csv_filename, armnn_dot_filename)


if __name__ == "__main__":
    cli()
