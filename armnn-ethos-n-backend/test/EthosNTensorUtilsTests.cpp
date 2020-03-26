//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include <EthosNTensorUtils.hpp>
#include <armnn/ArmNN.hpp>
#include <boost/test/unit_test.hpp>
#include <ethosn_support_library/Support.hpp>

#include <numeric>
#include <vector>

namespace armnn
{

using namespace ethosntensorutils;

BOOST_AUTO_TEST_SUITE(EthosNTensorUtils)

BOOST_AUTO_TEST_CASE(SwizzleConvolutionWeightsDataOHWIToHWIO)
{
    const unsigned int numDimensions             = 4u;
    const unsigned int dimensions[numDimensions] = { 2u, 4u, 4u, 2u };

    TensorShape tensorShape(numDimensions, dimensions);
    const unsigned int numElements = tensorShape.GetNumElements();

    std::vector<uint8_t> inputData(numElements);
    std::iota(inputData.begin(), inputData.end(), 1);

    std::vector<uint8_t> swizzledData(numElements, 0);
    SwizzleOHWIToHWIO<uint8_t>(inputData.data(), swizzledData.data(), tensorShape);

    std::vector<uint8_t> expectedOutputData({ 1,  33, 2,  34, 3,  35, 4,  36, 5,  37, 6,  38, 7,  39, 8,  40,

                                              9,  41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48,

                                              17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56,

                                              25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63, 32, 64 });

    BOOST_CHECK_EQUAL_COLLECTIONS(swizzledData.begin(), swizzledData.end(), expectedOutputData.begin(),
                                  expectedOutputData.end());
}

// cppcheck-suppress syntaxError symbolName=SwizzleConvolutionWeightsDataOIHWToHWIO
BOOST_AUTO_TEST_CASE(SwizzleConvolutionWeightsDataOIHWToHWIO)
{
    const unsigned int numDimensions             = 4u;
    const unsigned int dimensions[numDimensions] = { 2u, 2u, 4u, 4u };

    TensorShape tensorShape(numDimensions, dimensions);
    const unsigned int numElements = tensorShape.GetNumElements();

    std::vector<uint8_t> inputData(numElements);
    std::iota(inputData.begin(), inputData.end(), 1);

    std::vector<uint8_t> swizzledData(numElements, 0);
    SwizzleOIHWToHWIO<uint8_t>(inputData.data(), swizzledData.data(), tensorShape);

    std::vector<uint8_t> expectedOutputData({ 1,  33, 17, 49, 2,  34, 18, 50, 3,  35, 19, 51, 4,  36, 20, 52,

                                              5,  37, 21, 53, 6,  38, 22, 54, 7,  39, 23, 55, 8,  40, 24, 56,

                                              9,  41, 25, 57, 10, 42, 26, 58, 11, 43, 27, 59, 12, 44, 28, 60,

                                              13, 45, 29, 61, 14, 46, 30, 62, 15, 47, 31, 63, 16, 48, 32, 64 });

    BOOST_CHECK_EQUAL_COLLECTIONS(swizzledData.begin(), swizzledData.end(), expectedOutputData.begin(),
                                  expectedOutputData.end());
}

BOOST_AUTO_TEST_SUITE_END()

}    // namespace armnn
