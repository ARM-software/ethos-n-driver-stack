//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "GlobalParameters.hpp"

#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

int main(int argc, char* argv[])
{
    Catch::Session session;

    using namespace Catch::clara;

    auto cli = session.cli() | Opt(g_AllowDotFileGenerationInTests)["--generate-dot-files"](
                                   "Generate GraphViz dot files of network and graphs in tests");

    session.cli(cli);

    int returnCode = session.applyCommandLine(argc, argv);

    if (returnCode != 0)
    {
        // Indicates a command line error
        return returnCode;
    }

    return session.run();
}
