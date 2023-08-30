//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "GlobalParameters.hpp"

#include <map>
#include <random>

namespace ethosn
{
namespace system_tests
{

std::string g_GgfFilePath;
std::string g_TfLiteFilePath;
std::string g_TfLiteIfmPath = "";
bool g_SkipReference;
bool g_CachedRef;
std::string g_CacheFolder;
bool g_GgfUseArmnn;
bool g_RunProtectedInference;
std::vector<armnn::BackendId> g_ArmnnNonEthosNBackends = { "CpuRef" };
bool g_UseDmaBuf;
std::string g_DmaBufHeap      = "/dev/dma_heap/system";
std::string g_DmaBufProtected = "/dev/ethosn-tzmp1-test-module";
std::string g_Strategies;
std::string g_BlockConfigs;
size_t g_NumberRuns        = 1;
size_t g_RunBatchSize      = 0;
int g_EthosNTimeoutSeconds = 60;
/// Comparison tolerance to use for each output of the network.
/// The key of the map is the output name, and the value is the tolerance, with -1 meaning "auto".
/// A special key of "*" is used to indicate all outputs should use the specified tolerance.
std::map<std::string, float> g_ReferenceComparisonTolerances = { { "*", -1.0f } };
unsigned int g_DistributionSeed                              = std::mt19937::default_seed;
std::string g_Debug;
bool g_StrictPrecision = false;
std::vector<armnn::BackendOptions> g_ArmnnBackendOptions;
bool g_SkipOutputDistributionCheck = false;

}    // namespace system_tests
}    // namespace ethosn
