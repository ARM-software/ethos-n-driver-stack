#
# Copyright © 2018-2021 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#

# Add the Ethos-N backend unit test to the rest of the test suite
list(APPEND armnnEthosNBackendUnitTests_sources
     EthosNCreateEstimationWorkloadTests.cpp
     EthosNCreateWorkloadTests.cpp
     EthosNLayerTests.cpp
     EthosNMappingTests.cpp
     EthosNMemCopyTests.cpp
     EthosNOptimizeSubgraphViewTests.cpp
     EthosNProfilingTests.cpp
     EthosNSupportTest.cpp
     EthosNTensorUtilsTests.cpp
     EthosNTestUtils.hpp
     EthosNWorkloadFactoryHelper.hpp
     EthosNReplaceUnsupportedTests.cpp
     EthosNBackendTests.cpp
     replacement-tests/SISOCatOneGraphFactory.cpp
)

add_library(armnnEthosNBackendUnitTests OBJECT ${armnnEthosNBackendUnitTests_sources})
target_include_directories(armnnEthosNBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnEthosNBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnEthosNBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnEthosNBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
target_include_directories(armnnEthosNBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
# The "." adds the Ethos-N backend code into the include path. This is the cleanest way because
# Arm NN normally accesses the code of the Ethos-N backend by finding a sub-directory that contains CMake files.
# The name of the sub-directory can be environment specific.
target_include_directories(armnnEthosNBackendUnitTests PRIVATE .)
target_include_directories(armnnEthosNBackendUnitTests PRIVATE ..)
