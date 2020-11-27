//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"

#include <ethosn_utils/Json.hpp>

namespace ethosn
{
namespace support_library
{

std::ostream& PrintPassPerformanceData(std::ostream& os, ethosn::utils::Indent indent, const PassPerformanceData& pass);
std::ostream& PrintFailureReasons(std::ostream& os,
                                  ethosn::utils::Indent indent,
                                  const std::map<uint32_t, std::string>& failureReasons);

}    // namespace support_library
}    // namespace ethosn
