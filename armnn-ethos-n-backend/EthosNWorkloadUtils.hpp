//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNBackendId.hpp"

#include <Profiling.hpp>
#include <armnn/Types.hpp>

#define ARMNN_SCOPED_PROFILING_EVENT_ETHOSN(name) ARMNN_SCOPED_PROFILING_EVENT(armnn::EthosNBackendId(), name)
