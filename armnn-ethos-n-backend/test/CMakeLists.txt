#
# Copyright © 2018-2020 Arm Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Add the Ethos-N backend unit test to the rest of the test suite
list(APPEND armnnEthosNBackendUnitTests_sources
     EthosNCreateEstimationWorkloadTests.cpp
     EthosNCreateWorkloadTests.cpp
     EthosNLayerTests.cpp
     EthosNLayerTests.hpp
     EthosNMappingTests.cpp
     EthosNMemCopyTests.cpp
     EthosNOptimizeSubgraphViewTests.cpp
     EthosNProfilingTests.cpp
     EthosNSupportTest.cpp
     EthosNTensorUtilsTests.cpp
     EthosNTestUtils.hpp
     EthosNWorkloadFactoryHelper.hpp
     PreCompiledTestImpl.cpp
     PreCompiledTestImpl.hpp
     EthosNReplacementTests.cpp
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

# Keep the following test separated from the rest of the test suite, as it takes a long time to run (30 or so minutes)
# This test is deployed in the armnn/tests directory along with the other network tests and should be run manually
macro(EthosNPreCompiledVGG16QuantTest testName sources)
    add_executable_ex(${testName} ${sources})

    # By default EthosNPreCompiledVGG16QuantTest would end up to <build dir>/src/backend/ethosn/test,
    # instead add it to the other network tests in <build dir>/armnn/tests
    set_target_properties(${testName} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/armnn/tests)

    target_include_directories(${testName} PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
    target_include_directories(${testName} PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
    target_include_directories(${testName} PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
    target_include_directories(${testName} PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
    target_include_directories(${testName} PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
    target_include_directories(${testName} PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn/test)
    target_include_directories(${testName} PRIVATE ..)

    target_link_libraries(${testName} armnn)
    target_link_libraries(${testName} ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(${testName}
                          ${Boost_SYSTEM_LIBRARY}
                          ${Boost_FILESYSTEM_LIBRARY}
                          ${Boost_PROGRAM_OPTIONS_LIBRARY})
    addDllCopyCommands(${testName})
endmacro()

set(EthosNPreCompiledVGG16Quant-Armnn_sources EthosNPreCompiledVGG16Quant-Armnn.cpp)
EthosNPreCompiledVGG16QuantTest(EthosNPreCompiledVGG16Quant-Armnn "${EthosNPreCompiledVGG16Quant-Armnn_sources}")
