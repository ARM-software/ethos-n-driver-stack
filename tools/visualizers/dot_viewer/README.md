Dot Viewer
===========

Copyright © 2022-2024 Arm Limited. All rights reserved.
SPDX-License-Identifier: Apache-2.0

This is a tool to view GraphViz dot files in an interactive manner. It is designed to be easy-to-use
and have good performance even for large graphs, which none of the existing tools seem to offer.

Usage
=====

```
dot_viewer.py my_dot_file.dot
```

After a few seconds, this will open a web browser window showing the dot file.

There is also a `dot_viewer.bat` batch file for Windows users to associate with .dot files,
so that double-clicking a .dot file will open it in the viewer.

Configuration
=============

The python script needs to run the `dot` executable, which it assumes is on your `PATH`.
You can override the path to the `dot` tool by adding a `Config.json` next to the script (useful if `dot` is not on your `PATH`), e.g.:

```
{
    "dot": "E:\\Utilities\\Graphviz\\bin\\dot.exe"
}
```

Features
========

* Double-click to open .dot files (Windows-only)
* Pan and zoom with keyboard (WASD) and mouse (click and drag, mouse wheel)
* Automatic culling and level-of-detail to keep performance good for large graphs
* Search (Ctrl+F), including case-sensitivity and regex options
* Alt-click on a connection to go to the other end
* Select an element and press 'F' to focus the view on it
* Middle-click or press Home to reset the view

Implementation details
======================

* The python script runs the `dot` executable to convert the given dot file into an SVG
* It then starts an ephemeral web server which serves some static HTML/JS/CSS as well as the SVG
file that was just generated. Once done, the web server shuts down as it is no longer needed.
* A simple web server was chosen rather than simply loading files using file:///, to avoid
cross-origin browser restrictions with local files.
* The javascript on the web page processes the SVG file and implements the rendering and
interactivity features.
* Native browser SVG rendering was not used because it doesn't offer enough flexibility for the
features we want (e.g. LODing)
* The browser is still used to parse the SVG into a DOM, but then we render this
ourselves using a Canvas.
