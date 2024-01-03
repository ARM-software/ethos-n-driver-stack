#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Copyright © 2021,2024 Arm Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import print_function
import argparse
import os
import re

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import click

from hextools.compare_dumps import read_hex_file

group_cols = ["backend", "group", "layer"]

REGEX_HEX_FILES = re.compile(r"(?P<backend>[a-zA-Z_]+)(?P<layer>\d+)_(?P<desc>.*(?:_\d+){4})\.hex")
REGEX_ARMNN_GROUP = re.compile(r"(?P<group>[A-Za-z]+)")
REGEX_ETHOSN_GROUP = re.compile(r"(?P<group>[\w_\d]+)(?:_\d+){4}")


class Guard:
    pass


guard = Guard()


def cached(func):
    cache = {}

    def wrapper(*args, **kwargs):
        if func not in cache:
            cache[func] = guard
            cache[func] = func(*args, **kwargs)
        elif cache[func] == guard:
            raise Exception("Loading cache in progress.")
        return cache[func]

    return wrapper


class PlotDumps:
    def __init__(
        self,
        folder=None,
        backend=None,
        plots=False,
        max_cols=None,
        verbose=False,
        quiet=False,
        max_files=0,
        std_match=True,
        nbins=16,
        no_progress=False,
    ):
        """
        Create plots from dumps context.

        Args:
            folder (str): The folder to scan for hex files
            backend (str): The name of a backend to filter (Armnn or EthosN)
            plots (list): The plots types and sub-features
            max_cols (int): The maximum number of layers to display in one histogram plot
            verbose (bool): Add verbosity
            quiet (bool): Silence all console outputs
            max_files (int): The maximum number of files to load
            std_match (bool): Match for min standard deviation between backends
            nbins (int): Histogram number of bins
            no_progress (bool): Disable progress bars
        """
        self.folder = folder or os.getcwd()
        self.backend = (backend or "").lower()
        self.plots = plots
        self.max_cols = max_cols or 16
        self.verbose = verbose
        self.quiet = quiet
        self.max_files = max_files
        self.std_match = std_match
        self.nbins = nbins
        self.no_progress = no_progress or quiet

    @property
    @cached
    def data(self):
        """
        Get loaded data.
        """
        return self.load()

    @property
    @cached
    def histograms(self):
        """
        Get loaded histogram data.

        Compute histograms of each buffer output and matches layers from each backend
        """
        df = self.data.copy()

        # Adds hist column to dataframe
        df["hist"], bins = self.get_histogram_data(df, nbins=self.nbins)

        # Matches layers on histogram diff
        # Adds the columns:
        # - diff_id
        # - diff_data
        # - diff_std_data
        # - diff_hist
        # - diff_std_hist
        # - match
        # - layer_ref
        df = self.get_matches(df, "ethosn", "armnn", diff_keys=["data", "hist"], match_key="hist")

        return df, bins

    @property
    @cached
    def diff_histograms(self):
        df, bins = self.histograms

        # Keep only ethosn backend diff data
        df = df[df.backend == "ethosn"]

        # Create histogram of data diff
        df["diff_data_hist"], bins = self.get_histogram_data(
            df, nbins=self.nbins, data_key="diff_data"
        )

        return df, bins

    def output(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)

    def info(self, *args, **kwargs):
        if self.verbose:
            self.output(*args, **kwargs)

    def _load(self):
        files = os.listdir(self.folder)

        matches = list(filter(None, map(REGEX_HEX_FILES.match, files)))
        if self.max_files:
            matches = matches[: self.max_files]

        data = []

        try:
            progress = tqdm(matches, desc="Loading data from files", disable=self.no_progress)
            for m in progress:
                # Identify classes
                name = m[0]
                layer = int(m.group("layer"))
                backend = m.group("backend").split("_")[0].lower().replace("intermediatebuffer", "")
                desc = m.group("desc").strip("_")

                # Backend filter
                if self.backend is not None:
                    if self.backend not in backend:
                        continue

                # Backend group extraction
                if "armnn" in backend:
                    regex_group = REGEX_ARMNN_GROUP
                elif "ethos" in backend:
                    regex_group = REGEX_ETHOSN_GROUP

                desc_match = regex_group.match(desc)
                group = desc_match.group("group")

                try:
                    # Load data from file
                    df = self._load_file_data(name)

                    # Add columns
                    df[group_cols] = [backend, group, layer]

                    data.append(df)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self.output("WARN:", e)

        except KeyboardInterrupt:
            self.output("Interrupted loading.")

        return pd.concat(data).sort_values("layer")  # .set_index("filename")

    def _load_file_data(self, name):
        data = read_hex_file(os.path.join(self.folder, name))

        # Add data basic statistics
        d = {
            "filename": name,
            "data": [data],
            "min": data.min(),
            "max": data.max(),
            "mean": data.mean(),
            "std": data.std(),
        }

        return pd.DataFrame(d).set_index("filename")

    def load(self):
        """
        Loads the data from hex files located in specified folder.
        """
        self.info("Loading data...")

        df = self._load()

        self.info(df)

        return df

    def get_histogram_data(self, df, bins=None, nbins=16, data_key="data"):
        """
        Extract histogram from data.

        Expects a DataFrame with the columns [min, max, backend, layer, data]
        with data as 1D data array.

        Min and max from the whole dataset are used to determine histograms bins for all buffer histograms
        """
        min_all = df["min"].min()
        max_all = df["max"].max()

        # Create a linear bin space for all histograms
        bins = bins or np.linspace(min_all, max_all, nbins)

        def extract_histogram(row):
            return np.histogram(row, bins=bins, density=True)[0]

        # Extract histograms
        tqdm.pandas(desc="Applying histograms", disable=self.no_progress)

        histogram = df[data_key].progress_apply(extract_histogram)

        return histogram, bins

    def get_matches(
        self,
        df,
        backend1,
        backend2,
        diff_keys=["data"],
        match_key="data",
        id_key="filename",
    ):

        # Split histograms dataframes for each backend
        df = df.reset_index().sort_values(["layer"])
        df1 = df[df.backend == backend1]
        df2 = df[df.backend == backend2]

        rows = []

        # Iterate through first backend
        with tqdm(total=len(df1) * len(df2), disable=self.no_progress) as progress:
            for i, df1_row in df1.iterrows():
                # Iterate through second backend
                for j, df2_row in df2.iterrows():
                    lhs = f"{df1_row.backend}-{df1_row.layer}"
                    rhs = f"{df2_row.backend}-{df2_row.layer}"
                    progress.set_description(f"Match layers {lhs}/{rhs}")
                    progress.update()

                    # Match only same size data
                    if (
                        not df2_row[diff_keys]
                        .transform(len)
                        .equals(df1_row[diff_keys].transform(len))
                    ):
                        continue

                    row1 = df1_row.copy()
                    for diff_key in diff_keys:
                        row1["diff_" + diff_key] = diff = np.abs(
                            df2_row[diff_key] - df1_row[diff_key]
                        )
                        row1["diff_std_" + diff_key] = diff.std()
                    row1["diff_id"] = f"{lhs}-{rhs}"
                    row1["match"] = df2_row[id_key]
                    row1["layer_ref"] = df2_row.layer
                    rows.append(row1)

                    cols = np.setdiff1d(row1.index.values, df2_row.index.values)
                    row2 = pd.concat([df2_row, row1[cols]])
                    rows.append(row2)

        df = pd.DataFrame(rows)

        # Extract lowest stddev score for matching backend 1
        match_key = "diff_std_" + match_key

        tqdm.pandas(desc="Get best matches", disable=self.no_progress)
        df = df.groupby("filename", as_index=False).progress_apply(
            lambda g: g[g[match_key] == g[match_key].min()]
        )
        df = df[df.groupby("layer_ref").layer.transform("size") > 1]

        return df.droplevel(1).sort_values(["layer_ref"])

    def plot_histogram(self, specs):

        facet_col = "layer_ref"
        axis_depth = "layer_ref"
        title = "Buffer output histogram evolution accross layers"

        if "diff" in specs:
            df, bins = self.diff_histograms
            df["title"] = "Layer output diff"
            legend = "diff_id"
            axis_histogram = "diff_data_hist"
            facet_row = "title"
            title += " (backends difference)"
        else:
            df, bins = self.histograms
            legend = "filename"
            axis_histogram = "hist"
            facet_row = "backend"

        self.info("Histograms:")
        self.info(df)

        if "3d" in specs:
            self.output("Creating 3D histogram plot")

            self.plot_hist_3d(df, bins, axis_depth, axis_histogram, facet_row, title=title)
        else:
            self.output("Creating 2D histogram plot")

            # Split graphs into pages
            grouper = df.groupby(facet_col)
            pages = len(grouper) // self.max_cols + 1
            df_pages = {}
            for i, (idx, df_page) in enumerate(grouper):
                i = i // self.max_cols
                if len(df_pages) <= i:
                    df_pages[i] = df_page
                else:
                    df_pages[i] = df_pages[i].append(df_page)

            for i, df_page in df_pages.items():
                self.plot_hist_matrix(
                    df_page,
                    bins,
                    y=axis_histogram,
                    legend=legend,
                    facet_row=facet_row,
                    facet_col=facet_col,
                    title=f"{title} (page {i + 1}/{pages})",
                )

    @staticmethod
    def plot_hist_3d(df, bins, y="layer", z="hist", facet_row="backend", title=None):
        """
        Create sub plots rows = <key>
        """
        group = df.groupby(facet_row, as_index=False)
        titles = [k for k, _ in group[facet_row]]
        rows = len(group)

        fig = make_subplots(
            rows=rows,
            cols=1,
            specs=[[{"type": "surface"}]] * rows,
            subplot_titles=titles,
        )

        for row, (name, df) in enumerate(group):

            fig.add_trace(
                go.Surface(x=bins, y=df[y], z=df[z], opacity=0.7, name=name),
                row=row + 1,
                col=1,
            )

        fig.update_layout(
            scene=dict(xaxis_title="Distribution", yaxis_title=y, zaxis_title=z),
            title=title,
        )
        fig.show()

    @staticmethod
    def plot_hist_matrix(
        df,
        bins,
        y="hist",
        legend="filename",
        facet_row="backend",
        facet_col="diff_id",
        title=None,
    ):
        """
        Create sub plots matrix columns = <facet_col>, rows = <facet_row>

                                 0                             1
                    +----------------------------+----------------------------+
                    |                            |                            |
                    |  |    #                    |  |                         |
           Armnn    |  |    # #                  |  |    # #                  |
                    |  |  # # # #                |  |  # # # # #     # #      |
                    |  |  # # # # # #            |  |  # # # # # # # # # #    |
                    |  +----------------------   |  +----------------------   |
                    +----------------------------+----------------------------+
                    |                            |                            |
                    |  |    #                    |  |                         |
           EthosN   |  |    # #                  |  |    #                    |
                    |  |  # # # #                |  |  # # # # #       #      |
                    |  |  # # # # # # #          |  |  # # # # # # # # # #    |
                    |  +----------------------   |  +----------------------   |
                    +----------------------------+----------------------------+
        """
        super_grouper = df.groupby([facet_row, facet_col], as_index=False)
        subplots = len(super_grouper)
        grouper = df.groupby(facet_row, as_index=False)
        rows = len(grouper)
        cols = subplots // rows
        fig = make_subplots(
            rows=rows,
            cols=cols,
            column_titles=list(map(str, df[facet_col].unique())),
            row_titles=list(map(str, df[facet_row].unique())),
        )

        row = 0
        for row_name, g1 in grouper:
            row += 1
            col = 0
            for col_name, g2 in g1.groupby(facet_col, as_index=False):
                col += 1

                for i, grprow in g2.iterrows():
                    fig.add_trace(
                        go.Bar(x=bins, y=grprow[y], name=grprow[legend]),
                        row=row,
                        col=col,
                    )

        fig.update_layout(title=title)
        fig.show()

    def plot_scatter(self, specs):

        axis_x = "layer_ref"
        axis_y = "std"
        series = "group"
        title = "Buffer output standard deviation evolution accross layers"

        if "diff" in specs:
            df, _ = self.diff_histograms
            facet_row = None
            title += " (backends difference)"
            axis_y = "diff_std_data"
            # series = "diff_id"
        else:
            df, _ = self.histograms
            facet_row = "backend"

        if "line" in specs:
            self.output("Creating scatter line plot")

            fig = px.line(df, x=axis_x, y=axis_y, color=series, facet_row=facet_row, title=title)
        else:
            self.output("Creatin scatter plot")

            fig = px.scatter(df, x=axis_x, y=axis_y, color=series, facet_row=facet_row, title=title)

        fig.show()

    def plot(self, specs):
        if "scatter" in specs:
            self.plot_scatter(specs)
        elif "hist" in specs:
            self.plot_histogram(specs)
        else:
            raise Exception(f"Invalid plot option {specs}")

    def show(self):
        """
        Show the plot(s)
        """
        for specs in self.plots:
            try:
                self.plot(specs)
            except Exception as e:
                self.output("WARN:", e)


@click.command("plot-dumps")
@click.option("--folder", required=True, prompt=True, help="Folder to search for hex files in")
@click.option("--backend", "-b", required=False, help="Filter backend.")
@click.option(
    "--plots",
    "-p",
    required=True,
    prompt=True,
    multiple=True,
    help="""\
            Type of plot to create: <plot_type>[<sub_feature>].
            Where plot_type can be 'scatter', 'hist' and sub_feature:
            * `scatter: `line`
            * `hist`: `3d`
            * any: `diff`
            e.g. 'scatter+line' or 'hist+3d' or 'hist-3d-diff'\
            """,
)
@click.option(
    "--max-cols", "-C", required=False, default=16, help="The maximum number of file to load"
)
@click.option(
    "--verbose", "-v", required=False, default=False, help="The maximum number of columns per chart"
)
@click.option("--quiet", "-q", required=False, default=False, help="Don't print any output")
@click.option("--max-files", "-m", required=False, default=0, help="Increase verbosity output")
@click.option("--no-progress", required=False, default=False, help="Hide progress bars")
def cli(folder, backend, plots, max_cols, verbose, quiet, max_files, no_progress):
    """
    Plot the distribution and standard deviation  accross the buffer output hex files.
    """
    plot = PlotDumps(
        folder=os.path.expanduser(folder),
        backend=backend,
        plots=plots,
        max_cols=max_cols,
        verbose=verbose,
        quiet=quiet,
        max_files=max_files,
        std_match=True,
        nbins=16,
        no_progress=False,
    )
    plot.show()


if __name__ == "__main__":
    cli()
