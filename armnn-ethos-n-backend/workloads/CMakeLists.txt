#
# Copyright © 2018-2020 Arm Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

list(APPEND armnnEthosNBackendWorkloads_sources
    EthosNPreCompiledWorkload.cpp
    EthosNPreCompiledWorkload.hpp
)

add_library(armnnEthosNBackendWorkloads OBJECT ${armnnEthosNBackendWorkloads_sources})
target_include_directories(armnnEthosNBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnEthosNBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnEthosNBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnEthosNBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnEthosNBackendWorkloads PRIVATE ..)
