//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include <common/Utils.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

#include <cassert>

int main(int argc, char* argv[])
{
    using namespace Catch::clara;

    Catch::Session session;
#if defined(CONTROL_UNIT_ASSERTS)
    ethosn::control_unit::utils::g_AssertCallback = ethosn::control_unit::utils::DefaultAssert;
#endif

    auto cli = session.cli();
    session.cli(cli);

    int ret = session.applyCommandLine(argc, argv);
    if (ret)
    {
        return ret;
    }

    return session.run();
}
