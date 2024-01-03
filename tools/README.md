# Arm® Ethos™-N Driver Stack developer tools

Copyright © 2024 Arm Limited. All rights reserved.
SPDX-License-Identifier: Apache-2.0

This folder contains various developer tools that could be helpful, they are not part of the Arm® Ethos™-N Driver Stack and are not supported in any way, they are released AS IS without any warranty.

## System Test

Ethos-N NPU System Tests. This program supports the running of TensorFlow Lite models, networks described in Ggf format, and contains a suite of built-in tests.
It is availible in the folder `system_tests`.

See root folder README.md for some instruction on how to run this.

## Visualizers

Various tools to visualize internal states and performance metrics.

### Command stream viewer

A web-based tool to visualize the generated command stream. You can set the environment variable ETHOSN_DRIVER_LIBRARY_DEBUG to "cmdstream" to generate a command stream file.

In `visualizers/command_stream_viewer`

### Dependency Viewer

A web-based tool to show data dependencies.

In `visualizers/dependency_viewer`

### Dot Viewer

A web-based tool to view the generated dot files and search in them.

In `visualizers/dot_viewer`

### Hexfile tools

Various python tools to plot and compare dumps.

In `visualizers/hexfile_tools`

### Profile converter

A tool to convert the saved profile data so it can be used with common tools. The tool also uses a command stream dump. Set the environment variable ETHOSN_DRIVER_LIBRARY_DEBUG to "cmdstream" to generate a command stream file.

In `visualizers/profiling_converter`

### Extract operation perf

In `visualizers/extract_operation_perf.py`

### Pretty print nhwcb

A python tool to visualize nhwcb arrays.

In `visualizers/pretty_print_nhwcb.py`

