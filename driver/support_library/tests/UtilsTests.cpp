//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/Utils.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;
using namespace ethosn::support_library::utils;

TEST_CASE("RoundDownToPow2")
{
    CHECK(RoundDownToPow2(0) == 1);
    CHECK(RoundDownToPow2(1) == 1);
    CHECK(RoundDownToPow2(3) == 2);
    CHECK(RoundDownToPow2(4) == 4);
    CHECK(RoundDownToPow2(5) == 4);
    CHECK(RoundDownToPow2(8) == 8);
    CHECK(RoundDownToPow2(9) == 8);
    CHECK(RoundDownToPow2(0x1003) == 0x1000);
    CHECK(RoundDownToPow2(0x80000000) == 0x80000000);
    CHECK(RoundDownToPow2(0x800a5f6e) == 0x80000000);
    CHECK(RoundDownToPow2(0xffffffff) == 0x80000000);
}

TEST_CASE("FindBestConvAlgorithm")
{
    const HardwareCapabilities capabilities = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

    CHECK(FindBestConvAlgorithm(capabilities, 1, 1) == CompilerMceAlgorithm::Direct);
    CHECK(FindBestConvAlgorithm(capabilities, 3, 3) == CompilerMceAlgorithm::Winograd);
#if defined(ETHOSN_SUPPORT_LIBRARY_DISABLE_LARGE_WINOGRAD)
    CHECK(FindBestConvAlgorithm(capabilities, 7, 7) == CompilerMceAlgorithm::Direct);
#else
    CHECK(FindBestConvAlgorithm(capabilities, 7, 7) == CompilerMceAlgorithm::Winograd);
#endif
}

// Note that tests for IsCompressionFormatCompatibleWithStripeShape are covered by the
// tests for IsSramBufferCompatibleWithDramBuffer, which calls IsCompressionFormatCompatibleWithStripeShape.
