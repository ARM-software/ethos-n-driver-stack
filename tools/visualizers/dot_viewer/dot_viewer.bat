
REM Copyright © 2022 Arm Limited. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0

REM This batch file can be used as an "Open With..." target on Windows, and set as the default if you want

python %~dp0\dot_viewer.py %*

REM If there was an error, don't close the command prompt so that the error can be inspected
IF %ERRORLEVEL% NEQ 0 pause
