# Hexfile Tools

Copyright © 2020-2021 Arm Limited. All rights reserved.
SPDX-License-Identifier: Apache-2.0

## Prerequisites

Hexfile tools require some python libraries to function. These python libraries are:

* pandas
* numpy
* plotly
* tqdm
* click
* click_completion

Run this to install them:

```
pip install .
```

Once installed, the tools can be run directly from the CLI. Refer below for examples.

## plot_dumps.py

This tool helps in plotting the distribution and standard deviation as it evolves accross the buffer output hex files.

* Extract dump information from hex files names
* Loads data from dumps hex files
* Compute histograms and statistical data for each buffer dump
* Find best matching buffers between EthosN backend and CpuRef based on lowest std dev difference values between
  histograms from one backend to another
* Plot distribution as 2D or 3D plots

**Plot types**

The current plot types are Scatter and Histograms.

Plot options `-p`:

The option can select the type of chart to plot. The option string should be composed of a  of the plot type and of
the sub options. No separator are needed but can be added for clarity (e.g. `hist-3d-diff` or `hist+diff+3d` are
equivalent)

* Plot types:
    - `scatter`: Plot a Scatter type chart
    - `hist`: Plots the evolution of buffer output histograms

* Sub options:
    - `line` (with `scatter`): Plot with lines instead of points
    - `3d` (with `hist`): Plot as 3D Surface histogram chart
    - `diff`: Plot the difference between backends

**Plots layout**

* `scatter`: Plots the evolution of buffer output standard deviation
    - **x axis**: matched layers
    - **y axis**: standard deviation
    - **Subplots rows**: backends

  With `diff`:

    - **y axis**: standard deviation difference between backends
    - **Subplots rows**: None

* `hist`: Plots the evolution of buffer output histograms
    - **x axis**: distribution bins
    - **y axis**: occurrence
    - **Subplots rows**: backends
    - **Subplots columns**: matched layers

  With `diff`:
    - **y axis**: occurence difference between backends

  With `3d`: Plot as 3D Surface histogram chart
    - **x axis**: distribution bins
    - **y axis**: matched layers
    - **z axis**: occurrence
    - **Subplots rows**: backends
    - **Subplots columns**: None

  With `3d` and `diff`:
    - **z axis**: occurence difference between backends


**Examples**

    $ plot-dumps --folder <FOLDER> --plots hist-diff -plots scatter-line

## compare_dumps.py

This tool helps in comparing the standard deviation as it evolves accross the buffer output hex files.

The tool expects path to a CSV file whose each row (after the header) represents a single comparison to perform.

The file is expected to have the following columns:
'First': Path to the first .hex file to compare. Some information (e.g. data type) is extracted from the filename itself.
'Second': Path to the second .hex file to compare. Some information (e.g. data type) is extracted from the filename itself.
The following columns are optional:
'DataTypeOverride': Overrides the datatype that is normally extracted from the filenames.
'''

**Examples**

    $ compare-dumps --csv-filename <CSV-FILE>