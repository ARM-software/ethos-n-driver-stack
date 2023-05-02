#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020-2021 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#
from collections import OrderedDict
import csv
import pathlib
import re
import sys


def read_csv(csv_filename):
    path = pathlib.Path(csv_filename).resolve().as_posix()
    m = re.search(r"/(\w+)/perf_(\w+)", str(path))
    op, hw = m.groups() if m else ("", "")
    with open(csv_filename) as csv_file:
        return [
            OrderedDict([("operation", op), ("variant", hw)] + sorted(r.items()))
            for r in csv.DictReader(csv_file)
        ]


def write_csv(rows, csv_file):
    fieldnames = rows[0].keys()
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        assert r.keys() == fieldnames
        writer.writerow(r)


def process_files(csv_filenames, out_file):
    rows = []
    for f in csv_filenames:
        rows += read_csv(f)
    write_csv(rows, out_file)


# Usage: ./process_perf_csvs.py CSV_FILES...
#
# Parses the csv files resulting from PLE perf test runs and prints to stdout an aggregated
# csv file with two extra columns: operation and hw variant.
#
if __name__ == "__main__":
    process_files(sys.argv[1:], sys.stdout)
