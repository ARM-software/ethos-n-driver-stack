//
// Copyright © 2018-2019 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ComparisonUtils.hpp"

#include <catch.hpp>

#include <fstream>

using namespace ethosn::control_unit;

namespace
{

void DumpLoggingHalEntriesToFile(const std::vector<LoggingHal::Entry>& entries, const char* filename)
{
    std::ofstream stream(filename);
    for (const LoggingHal::Entry& e : entries)
    {
        stream << e << std::endl;
    }
}

}    // namespace

void RequireLoggingHalEntriesEqual(const std::vector<LoggingHal::Entry>& golden,
                                   const std::vector<LoggingHal::Entry>& actual)
{
    if (golden != actual)
    {
        std::string goldenFile = Catch::getResultCapture().getCurrentTestName() + "_Golden.txt";
        DumpLoggingHalEntriesToFile(golden, goldenFile.c_str());
        std::string actualFile = Catch::getResultCapture().getCurrentTestName() + "_Actual.txt";
        DumpLoggingHalEntriesToFile(actual, actualFile.c_str());
        FAIL("golden != actual. See files to compare: " << goldenFile << " and " << actualFile);
    }
}

void RequireLoggingHalEntriesContainsInOrder(const std::vector<LoggingHal::Entry>& golden,
                                             const std::vector<LoggingHal::Entry>& actual)
{
    auto actualIt = actual.begin();
    for (const LoggingHal::Entry& g : golden)
    {
        actualIt = std::find(actualIt, actual.end(), g);
        if (actualIt == actual.end())
        {
            std::string goldenFile = Catch::getResultCapture().getCurrentTestName() + "_Golden.txt";
            DumpLoggingHalEntriesToFile(golden, goldenFile.c_str());
            std::string actualFile = Catch::getResultCapture().getCurrentTestName() + "_Actual.txt";
            DumpLoggingHalEntriesToFile(actual, actualFile.c_str());
            FAIL("actual does not contain golden. See files to compare: " << goldenFile << " and " << actualFile);
        }
    }
}
