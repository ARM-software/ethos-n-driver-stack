#!/usr/bin/python3
#
# Copyright © 2022-2024 Arm Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import sys
import csv
import json
import re

# Result is a dict of operation ID to command range
def extract_operation_id_to_cmds_from_dot(dot_file):
    result = {}
    with open(dot_file) as f:
        current_pass = None
        for line in f:
            # Legacy or Cascading
            m = re.search(r"Pass (\d+)\\n.*Commands (\d+)-(\d+)\\n", line) or re.search(
                r"Pass(\d+)\\n.*Agent IDs: (\d+) - (\d+)\\n", line
            )
            if m:
                current_pass = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            elif line.startswith("}"):
                current_pass = None

            # Legacy or Cascading
            m = re.search(r"CorrespondingOperationIds:(( \d+)*)", line) or re.search(
                r"Operation Ids = \[((\d+, )*\d+)\]", line
            )
            if current_pass and m:
                operation_ids = [int(x) for x in m.group(1).replace(",", "").strip().split(" ")]
                for id in operation_ids:
                    if not id in result:
                        result[id] = (current_pass[1], current_pass[2])
                    else:
                        result[id] = (
                            min(result[id][0], current_pass[1]),
                            max(result[id][1], current_pass[2]),
                        )

    return result


# Result is a dict of operation ID to a tuple of
#  1. string describing the operation (e.g. "Relu, 10x20x30")
#  2. Number of MACs theoretically for this operation
def extract_operation_details_from_dot(dot_file):
    operand_descs = {}
    operation_labels = {}

    with open(dot_file) as f:
        current_pass = None
        for line in f:
            # Operations
            m = re.search(r'''Operation(\d+).*label = "\d+: (.*)"''', line)
            if m:
                id = int(m.group(1))
                label = m.group(2)
                operation_labels[id] = label

            # Operands
            m = re.search(r'''Operand([0-9_]*)\[label = "(.*)"''', line)
            if m:
                id = m.group(1)
                label = m.group(2)
                operand_descs[id] = label

            # Connections
            m = re.search(r"""Operation(\d+) -> Operand([0-9_]*)""", line)
            if m:
                operation_id = int(m.group(1))
                operand_id = m.group(2)
                operation_labels[operation_id] += " Output: " + operand_descs[operand_id]

            m = re.search(r"""Operand([0-9_]*) -> Operation(\d+)""", line)
            if m:
                operand_id = m.group(1)
                operation_id = int(m.group(2))
                operation_labels[operation_id] += " Input: " + operand_descs[operand_id]

    # Extract Num MACs from the labels
    result = {}
    for i, l in operation_labels.items():
        m = re.search(r"""Num MACs: (\d+)""", l)
        num_macs = int(m.group(1)) if m else 0
        result[i] = (l, num_macs)

    return result


# Result is dict of command number to start and end timestamp
def parse_profiling_entries(profiling_file):
    result = {}

    inflight_commands = set()
    with open(profiling_file) as f:
        j = json.load(f)

        # Offset for all values to make the start of the profiling trace at time zero
        offset = j[0]["ts"]

        def extract_cmd_idx(n):
            return int(n.split(" ", 2)[1])

        for entry in j:
            if entry["ph"] == "B" and (
                entry["name"].startswith("Agent ") or entry["name"].startswith("Command ")
            ):
                cmd = extract_cmd_idx(entry["name"])
                if cmd in result:
                    continue  # Ignore duplicate commands. This could happen if we have multiple subgraphs, or if there are commands such as DUMP_DRAM that happen after a cascade
                result[cmd] = (entry["ts"] - offset, None)
                inflight_commands.add(cmd)
            elif (
                entry["ph"] == "E"
                and (entry["name"].startswith("Agent ") or entry["name"].startswith("Command "))
                and extract_cmd_idx(entry["name"]) in inflight_commands
            ):
                cmd = extract_cmd_idx(entry["name"])
                result[cmd] = (result[cmd][0], entry["ts"] - offset)

    return result


def main(args):
    sol_macs_per_cycle = args.tops * (9 * 16 * 32 * 2) / 8

    if len(args.dot_file) != len(args.profiling_entries):
        raise RuntimeError("Must have same number of dot files and profiling entries")

    operation_details = extract_operation_details_from_dot(args.network)

    operation_ids_to_times = {}
    operation_ids_to_cmds = [None] * len(args.dot_file)
    for i in range(len(args.dot_file)):
        operation_ids_to_cmds[i] = extract_operation_id_to_cmds_from_dot(args.dot_file[i])
        # print(operation_ids_to_cmds)

        command_times = parse_profiling_entries(args.profiling_entries[i])
        # print(command_times)

        min_timestamp = None
        for operation_id, cmds in operation_ids_to_cmds[i].items():
            start = None
            end = None
            for cmd in cmds:
                start = min(start, command_times[cmd][0]) if start else command_times[cmd][0]
                end = max(end, command_times[cmd][1]) if end else command_times[cmd][1]

            min_timestamp = min(min_timestamp, start) if min_timestamp is not None else start

            if not operation_id in operation_ids_to_times:
                operation_ids_to_times[operation_id] = [None] * len(args.dot_file)
            operation_ids_to_times[operation_id][i] = (start, end)

        # Offset timestamps so they start at zero
        for operation_id in operation_ids_to_times.keys():
            start_and_end = operation_ids_to_times[operation_id][i]
            if start_and_end:
                operation_ids_to_times[operation_id][i] = (
                    start_and_end[0] - min_timestamp,
                    start_and_end[1] - min_timestamp,
                )

    # print(operation_ids_to_times)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)

        headings1 = [
            "Operation ID",
            "Operation Desc",
            "Operation MACs",
            "Operation speed-of-light cycles (at {} TOPs)".format(args.tops),
        ]
        for p in args.profiling_entries:
            headings1.append(p)
            headings1.extend(["", "", "", ""])
        writer.writerow(headings1)

        headings2 = ["", "", "", ""]
        for p in args.profiling_entries:
            headings2.extend(
                [
                    "Command range",
                    "Start time (ns)",
                    "End time (ns)",
                    "Duration (ns)",
                    "Duration (cycles @ {}Mhz)".format(args.fpga_frequency_mhz),
                ]
            )
        writer.writerow(headings2)

        for operation_id, times in sorted(operation_ids_to_times.items()):
            operation_desc, operation_macs = operation_details[operation_id]
            if operation_desc.startswith("Constant"):
                continue  # Constant layers are boring and clutter the results, so skip them
            operation_sol_cycles = operation_macs / sol_macs_per_cycle
            row = [operation_id, operation_desc, operation_macs, operation_sol_cycles]
            for i in range(len(args.dot_file)):
                dump_times = times[i]
                if dump_times:
                    duration_ns = dump_times[1] - dump_times[0]
                    row.extend(
                        [
                            operation_ids_to_cmds[i][operation_id],
                            dump_times[0],
                            dump_times[1],
                            duration_ns,
                            args.fpga_frequency_mhz * 1000000 * duration_ns / 1000000000,
                        ]
                    )
                else:
                    row.extend(["-", "-", "-", "-", "-"])
            writer.writerow(row)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts the performance of individual operations from one or more profiling dumps."
        "Produces a csv file for further analysis."
    )
    parser.add_argument(
        "-p",
        "--profiling_entries",
        default=[],
        action="append",
        help="File containing profiling entries, output from profiling_converter.py. Default: %(default)s",
    )
    parser.add_argument(
        "-d",
        "--dot_file",
        default=[],
        action="append",
        help="GraphViz dot file containing details of the compiled network. Used to extract what operation IDs correspond to what commands. Default: %(default)s",
    )
    parser.add_argument(
        "-n",
        "--network",
        default="NetworkDetailed.dot",
        help="Path to the NetworkDetailed.dot file, describing what each operation is. Default: %(default)s",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="operation_perf.csv",
        help="Path of the output CSV file to create. Default: %(default)s",
    )
    parser.add_argument(
        "-f",
        "--fpga_frequency_mhz",
        type=int,
        default="5",
        help="FPGA frequency in MHz to use when converting between cycles and seconds. Default: %(default)s",
    )
    parser.add_argument(
        "-t",
        "--tops",
        type=int,
        default="8",
        help="HW config tops number to use when calculating speed-of-light figures. Default: %(default)s",
    )
    cmd_args = parser.parse_args()
    result = main(cmd_args)
    sys.exit(0 if result else 1)
