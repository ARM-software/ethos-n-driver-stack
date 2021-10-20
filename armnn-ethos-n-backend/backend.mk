#
# Copyright © 2018-2021 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of Arm NN

# The variable to enable/disable this backend (ARMNN_ETHOSN_ENABLED is declared in android-nn-driver/Android.mk)
ifeq ($(ARMNN_ETHOSN_ENABLED),1)

BACKEND_SOURCES := \
        EthosNBackend.cpp \
        EthosNBackendProfilingContext.cpp \
        EthosNConfig.cpp \
        EthosNLayerSupport.cpp \
        EthosNReplaceUnsupported.cpp \
        EthosNSubgraphViewConverter.cpp \
        EthosNTensorUtils.cpp \
        EthosNWorkloadFactory.cpp \
        workloads/EthosNPreCompiledWorkload.cpp

ETHOSN_DRIVER_STACK := vendor/arm/ethos-n-driver-stack

BACKEND_INCLUDES := $(ETHOSN_DRIVER_STACK)/armnn-ethos-n-backend/ \
                    $(ETHOSN_DRIVER_STACK)/driver/driver_library/include/ \
                    $(ETHOSN_DRIVER_STACK)/driver/support_library/include/ \
                    $(ETHOSN_DRIVER_STACK)/driver/utils/include/

BACKEND_STATIC_LIBRARIES := \
        libEthosNDriver \
        libEthosNSupport

else

# Backend disabled, no source file will be compiled.
BACKEND_SOURCES :=

endif

# BACKEND_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of Arm NN

# The variable to enable/disable this backend (ARMNN_ETHOSN_ENABLED is declared in android-nn-driver/Android.mk)
ifeq ($(ARMNN_ETHOSN_ENABLED),1)


BACKEND_TEST_SOURCES :=

else

# Backend disabled, no test file will be compiled.

BACKEND_TEST_SOURCES :=


endif
