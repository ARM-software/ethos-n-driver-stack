//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "LayerData.hpp"
#include "SystemTestsUtils.hpp"

namespace ethosn
{
namespace system_tests
{

void CompareArmnnAndEthosNOutput(const char* ggfFilename,
                                 LayerData& layerData,
                                 bool verifyStatisticalOutput                                      = true,
                                 const std::map<std::string, float>& referenceComparisonTolerances = {
                                     { "*", -1.0f } });

void CompareArmnnAndEthosNOutput(std::istream& ggfFile,
                                 LayerData& layerData,
                                 bool verifyStatisticalOutput                                      = true,
                                 const std::map<std::string, float>& referenceComparisonTolerances = { { "*", -1.0f } },
                                 const std::string& armnnCache                                     = "");

InferenceOutputs RunArmnn(std::istream& ggfFile,
                          LayerData& layerData,
                          const std::string& armnnCacheFilename,
                          std::vector<armnn::BackendId> backends);

}    // namespace system_tests
}    // namespace ethosn
