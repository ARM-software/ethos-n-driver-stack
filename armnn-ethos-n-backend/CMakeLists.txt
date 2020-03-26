#
# Copyright © 2018-2020 Arm Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

list(APPEND armnnEthosNBackend_sources
    EthosNBackend.cpp
    EthosNBackend.hpp
    EthosNBackendId.hpp
    EthosNConfig.cpp
    EthosNConfig.hpp
    EthosNLayerSupport.cpp
    EthosNLayerSupport.hpp
    EthosNBackendProfilingContext.cpp
    EthosNBackendProfilingContext.hpp
    EthosNMapping.hpp
    EthosNMapping.cpp
    EthosNSubgraphViewConverter.cpp
    EthosNSubgraphViewConverter.hpp
    EthosNTensorHandle.hpp
    EthosNTensorUtils.cpp
    EthosNTensorUtils.hpp
    EthosNWorkloadFactory.cpp
    EthosNWorkloadFactory.hpp
    EthosNWorkloadUtils.hpp
)

add_library(armnnEthosNBackend OBJECT ${armnnEthosNBackend_sources})
target_include_directories(armnnEthosNBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnEthosNBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnEthosNBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnEthosNBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

add_subdirectory(workloads)

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()
