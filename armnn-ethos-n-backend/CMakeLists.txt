#
# Copyright © 2018-2022 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#

list(APPEND armnnEthosNBackend_sources
    EthosNBackend.cpp
    EthosNBackend.hpp
    EthosNBackendId.hpp
    EthosNCaching.cpp
    EthosNCaching.hpp
    EthosNConfig.cpp
    EthosNConfig.hpp
    EthosNTensorHandleFactory.hpp
    EthosNTensorHandleFactory.cpp
    EthosNLayerSupport.cpp
    EthosNLayerSupport.hpp
    EthosNBackendProfilingContext.cpp
    EthosNBackendProfilingContext.hpp
    EthosNSubgraphViewConverter.cpp
    EthosNSubgraphViewConverter.hpp
    EthosNTensorHandle.hpp
    EthosNTensorUtils.cpp
    EthosNTensorUtils.hpp
    EthosNWorkloadFactory.cpp
    EthosNWorkloadFactory.hpp
    EthosNWorkloadUtils.hpp
    EthosNReplaceUnsupported.hpp
    EthosNReplaceUnsupported.cpp
)

add_library(armnnEthosNBackend OBJECT ${armnnEthosNBackend_sources})
target_include_directories(armnnEthosNBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnEthosNBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnEthosNBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnEthosNBackend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
target_include_directories(armnnEthosNBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

# This static library is built into the core Arm NN shared library, so set the appropriate define
target_compile_definitions(armnnEthosNBackend PRIVATE "ARMNN_COMPILING_DLL")

add_subdirectory(workloads)

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()
