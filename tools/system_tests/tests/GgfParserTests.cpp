//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../EthosNParseRunner.hpp"
#include "../GgfParser.hpp"
#include "../LayerData.hpp"

#include <catch.hpp>

namespace ethosn
{
namespace system_tests
{

/// Checks that GgfParser correctly parses the given file
TEST_CASE("GgfParser Parse Mean valid parameters")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 7, 7, 16
mean layer, name mean1, top mean1, bottom data, keep_dims 1, dimension 3_2
)");
    LayerData layerData;

    EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));

    REQUIRE(parser.GetInputLayerIndex("data") == 0);
}

/// Checks that GgfParser throws an appropriate error message when an incorrect parameter is provided
TEST_CASE("GgfParser Parse Mean invalid dimensions")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 7, 7, 16
mean layer, name mean1, top mean1, bottom data, keep_dims 1, dimension 1_2_3
)");
    LayerData layerData;
    std::string exceptionMessage("only \"dimension 2_3\" is supported");
    try
    {
        EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
    }
    catch (std::exception& e)
    {
        REQUIRE(std::string(e.what()).find(exceptionMessage) != std::string::npos);
    }
}

/// Checks that GgfParser throws an appropriate error message when an incorrect parameter is provided
TEST_CASE("GgfParser Parse Mean no keep_dims")
{
    std::stringstream ggfContents(R"(
input layer, name data, top data, shape 1, 7, 7, 16
mean layer, name mean1, top mean1, bottom data, keep_dims 0, dimension 2_3
)");
    LayerData layerData;
    std::string exceptionMessage("\"keep_dims 0\" is not supported");
    try
    {
        EthosNParseRunner parser(EthosNParseRunner::CreationOptions(ggfContents, layerData));
    }
    catch (std::exception& e)
    {
        REQUIRE(std::string(e.what()).find(exceptionMessage) != std::string::npos);
    }
}

}    // namespace system_tests
}    // namespace ethosn
