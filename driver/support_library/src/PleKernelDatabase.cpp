//
// Copyright © 2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
//
// Contains the raw list of avilable PLE kernels and their parameters.
// This is in its own cpp file rather than a header to avoid having to recompile
// as much (it can take a good few seconds).

#include <map>
#include <string>
#include <vector>

#include <ethosn_command_stream/PleKernelIds.hpp>

extern const std::vector<std::pair<std::map<std::string, std::string>, ethosn::command_stream::PleKernelId>>
    g_PleKernelParamsToId =
#include "PleKernelDatabaseGenerated.hpp"
    ;
