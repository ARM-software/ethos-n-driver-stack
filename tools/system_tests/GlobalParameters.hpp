//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include <armnn/ArmNN.hpp>

namespace ethosn
{
namespace system_tests
{

// Globals set from command line arguments in main.cpp
extern std::string g_GgfFilePath;
extern std::string g_TfLiteFilePath;
extern std::string g_TfLiteIfmPath;
extern bool g_SkipReference;
extern bool g_CachedRef;
extern std::string g_CacheFolder;
extern bool g_GgfUseArmnn;
extern bool g_RunProtectedInference;
extern std::vector<armnn::BackendId> g_ArmnnNonEthosNBackends;
extern bool g_UseDmaBuf;
extern std::string g_DmaBufHeap;
extern std::string g_DmaBufProtected;
extern std::string g_Strategies;
extern std::string g_BlockConfigs;
extern std::string g_DefaultConvolutionAlgorithm;
extern size_t g_NumberRuns;
extern size_t g_RunBatchSize;
extern int g_EthosNTimeoutSeconds;
extern std::map<std::string, float> g_ReferenceComparisonTolerances;
extern unsigned int g_DistributionSeed;
extern std::string g_Debug;
extern bool g_StrictPrecision;
extern bool g_BlockInferenceForDebug;
extern std::vector<armnn::BackendOptions> g_ArmnnBackendOptions;
extern bool g_SkipOutputDistributionCheck;

}    // namespace system_tests
}    // namespace ethosn
