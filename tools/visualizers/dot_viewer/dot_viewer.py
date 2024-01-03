#!/usr/bin/env python3
#
# Copyright © 2022,2024 Arm Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import http.server
import socketserver
import subprocess
import sys
import os
import json
import webbrowser
import pathlib


class Server(socketserver.TCPServer):
    def __init__(self, dot_filename, svg_filename, is_dev_mode, *args):
        super().__init__(*args)
        self.please_shutdown = False
        self.svg_filename = svg_filename
        self.dot_filename = dot_filename
        self.is_dev_mode = is_dev_mode


# Note that we don't use SimpleHTTPRequestHandler, as this does some caching stuff
# and keeps connections open, which we don't want, so we make a simpler version ourself.
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"

        # The generate SVG file will have different names depending on the dot filename, but we always
        # serve it under the same fixed name.
        if self.path == "/Source.svg":
            self.send_response(200)
            # Send the original dot filename as a header, so that it can be shown as the page title in the browser
            self.send_header("X-Dot-Filename", pathlib.Path(self.server.dot_filename).name)
            self.end_headers()
            with open(self.server.svg_filename, "rb") as f:
                self.wfile.write(f.read())

            # The web server will exit once the content is served,
            # but not in dev mode as this makes iterative development harder.
            if not self.server.is_dev_mode:
                self.server.please_shutdown = True
        elif os.path.isfile(self.path.lstrip("/")):
            self.send_response(200)
            self.end_headers()
            with open(self.path.lstrip("/"), "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def end_headers(self):
        # Disable caching, so the browser will fetch up-to-date stuff every time
        self.send_header("Cache-Control", "max-age=0")
        self.send_header("Expires", "0")
        # Prevent the browser keeping the connection open, so that our timeout works and
        # we can shutdown the server quickly
        self.send_header("Connection", "close")
        super().end_headers()


def main(dot_filename, is_dev_mode):
    dot_program = "dot"
    dot_viewer_dir = pathlib.Path(__file__).parent.resolve()
    config_filename = dot_viewer_dir / "Config.json"

    if os.path.isfile(config_filename):
        with open(config_filename) as f:
            j = json.load(f)
            if "dot" in j:
                dot_program = j["dot"]

    print("Converting dot file to svg using " + dot_program + "...")
    # Place the converted SVG next to the dot file, as it might be useful to inspect later
    svg_filename = pathlib.Path(dot_filename).with_suffix(".svg")
    subprocess.run(
        [dot_program, "-T", "svg", "-o", str(svg_filename), str(dot_filename)], check=True
    )

    # Start a temporary local web server to serve the content - this avoids cross-site origin
    # policy issues when using local files (file://).
    os.chdir(dot_viewer_dir)  # So that the web server serves files from the right folder
    port = 8080
    host = "" if is_dev_mode else "localhost"  # Listen on all interfaces in dev mode
    print("Starting local webserver on port " + str(port) + "...")
    with Server(dot_filename, svg_filename, is_dev_mode, (host, port), Handler) as server:
        print("Opening web browser...")
        webbrowser.open("http://localhost:" + str(port))

        while True:
            server.handle_request()

            if server.please_shutdown:
                break


if __name__ == "__main__":
    main(sys.argv[1], len(sys.argv) >= 3 and sys.argv[2] == "--dev")
