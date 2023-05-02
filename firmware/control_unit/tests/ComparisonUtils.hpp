//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <model/LoggingHal.hpp>

#include <vector>

/// Performs a Catch test to check that the given lists of LoggingHal::Entries are identical.
/// If they are not, it will print an error and dump the lists to files that can be compared.
void RequireLoggingHalEntriesEqual(const std::vector<ethosn::control_unit::LoggingHal::Entry>& golden,
                                   const std::vector<ethosn::control_unit::LoggingHal::Entry>& actual);

/// Performs a Catch test to check that the given 'actual' list of LoggingHal::Entries are contained
/// within the 'golden' list in the correct order.
/// If they are not, it will print an error and dump the lists to files that can be compared.
void RequireLoggingHalEntriesContainsInOrder(const std::vector<ethosn::control_unit::LoggingHal::Entry>& golden,
                                             const std::vector<ethosn::control_unit::LoggingHal::Entry>& actual);
