//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include <EthosNTensorUtils.hpp>
#include <armnn/ArmNN.hpp>
#include <doctest/doctest.h>
#include <ethosn_support_library/Support.hpp>

#include <numeric>
#include <vector>

namespace armnn
{

using namespace ethosntensorutils;

TEST_SUITE("EthosNTensorUtils")
{

    TEST_CASE("SwizzleConvolutionWeightsDataOHWIToHWIO")
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

        CHECK(
            std::equal(swizzledData.begin(), swizzledData.end(), expectedOutputData.begin(), expectedOutputData.end()));
    }

    // cppcheck-suppress syntaxError symbolName=SwizzleConvolutionWeightsDataOIHWToHWIO
    TEST_CASE("SwizzleConvolutionWeightsDataOIHWToHWIO")
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

        CHECK(
            std::equal(swizzledData.begin(), swizzledData.end(), expectedOutputData.begin(), expectedOutputData.end()));
    }

    TEST_CASE("SupportedDataTypes")
    {
        // Supported DataTypes
        CHECK(IsDataTypeSupportedOnEthosN(armnn::DataType::QAsymmU8));
        CHECK(IsDataTypeSupportedOnEthosN(armnn::DataType::QAsymmS8));
        CHECK(IsDataTypeSupportedOnEthosN(armnn::DataType::QSymmS8));
        CHECK(IsDataTypeSupportedOnEthosN(armnn::DataType::Signed32));
        // Unsupported DataTypes
        CHECK(!IsDataTypeSupportedOnEthosN(armnn::DataType::Float32));
    }

    TEST_CASE("BuildEthosNTensorShapeTests")
    {
        CHECK((BuildEthosNTensorShape(TensorShape{ 23 }) == ethosn_lib::TensorShape{ 1, 23, 1, 1 }));
        CHECK((BuildEthosNTensorShape(TensorShape{ 23, 45 }) == ethosn_lib::TensorShape{ 1, 23, 45, 1 }));
        CHECK((BuildEthosNTensorShape(TensorShape{ 23, 45, 4 }) == ethosn_lib::TensorShape{ 1, 23, 45, 4 }));
        CHECK((BuildEthosNTensorShape(TensorShape{ 23, 45, 4, 235 }) == ethosn_lib::TensorShape{ 23, 45, 4, 235 }));
        CHECK((BuildEthosNTensorShape(TensorShape{ 1, 23 }) == ethosn_lib::TensorShape{ 1, 23, 1, 1 }));
        CHECK((BuildEthosNTensorShape(TensorShape{ 1, 23, 45 }) == ethosn_lib::TensorShape{ 1, 23, 45, 1 }));
        CHECK((BuildEthosNTensorShape(TensorShape{ 1, 23, 45, 4 }) == ethosn_lib::TensorShape{ 1, 23, 45, 4 }));
    }

    TEST_CASE("BuildEthosNReluInfoTests")
    {
        CHECK(BuildEthosNReluInfo(ActivationDescriptor(ActivationFunction::ReLu, 999.9f, 999.0f), DataType::QAsymmU8,
                                  0.1f, 20) == ethosn_lib::ReluInfo(20, 255));
        CHECK(BuildEthosNReluInfo(ActivationDescriptor(ActivationFunction::ReLu, 999.9f, 999.0f), DataType::QAsymmS8,
                                  0.1f, -20) == ethosn_lib::ReluInfo(-20, 127));
        CHECK(BuildEthosNReluInfo(ActivationDescriptor(ActivationFunction::BoundedReLu, 1.0f, -1.0f),
                                  DataType::QAsymmU8, 0.1f, 20) == ethosn_lib::ReluInfo(10, 30));
        CHECK(BuildEthosNReluInfo(ActivationDescriptor(ActivationFunction::BoundedReLu, 1.0f, -1.0f),
                                  DataType::QAsymmS8, 0.1f, -20) == ethosn_lib::ReluInfo(-30, -10));
    }

    TEST_CASE("BuildEthosNBiasesInfo")
    {
        // Declare a set of parameters that are supported, so we can re-use these for the different subcases
        const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
        const TensorInfo weightInfo({ 1, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0, true);
        TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0, true);

        Convolution2dDescriptor descriptor;
        descriptor.m_BiasEnabled = true;
        descriptor.m_DataLayout  = DataLayout::NHWC;
        descriptor.m_StrideX     = 1;
        descriptor.m_StrideY     = 1;
        auto ExpectFail = [](const TensorInfo& biasInfo, const TensorInfo& inputInfo, const TensorInfo& weightInfo) {
            BuildEthosNBiasesInfo(biasInfo, inputInfo, weightInfo);
        };
        //In this scenario we assume that tolerance is 1%
        SUBCASE("Tolerable difference")
        {
            auto newScales = std::vector<float>{ 0.891999976f };
            biasInfo.SetQuantizationScales(newScales);
            auto res = BuildEthosNBiasesInfo(biasInfo, inputInfo, weightInfo);
            CHECK(res.m_QuantizationInfo.GetScale(0) == 0.899999976f);
        }
        //In this scenario we assume that tolerance is 1%
        SUBCASE("Intolerable difference")
        {
            auto newScales = std::vector<float>{ 0.890999976f };
            biasInfo.SetQuantizationScales(newScales);
            CHECK_THROWS_AS(ExpectFail(biasInfo, inputInfo, weightInfo), InvalidArgumentException);
        }
        SUBCASE("Different amount of biases")
        {
            auto newScales = std::vector<float>{ 0.899999976f, 1.0f };
            biasInfo.SetQuantizationScales(newScales);
            REQUIRE_THROWS_WITH(ExpectFail(biasInfo, inputInfo, weightInfo),
                                "The amount of biases scales(2) is different from weightScales*inputScales(1)");
        }
    }

    TEST_CASE("ExtendPadList")
    {
        // Padding is only allowed in the HW dimensions, but this test uses batch and channel padding
        // to confirm ExtendPadList is extending the padding correctly, as it only inserts {0,0}

        using PadList = std::vector<std::pair<unsigned int, unsigned int>>;

        // H    -> NHWC, (23)               -> (1, 23, 1, 1)
        CHECK(ExtendPadList({ { 1, 1 } }, TensorShape{ 23 }) == PadList({ { 0, 0 }, { 1, 1 }, { 0, 0 }, { 0, 0 } }));
        // HW   -> NHWC, (23, 45)           -> (1, 23, 45, 1)
        CHECK(ExtendPadList({ { 1, 1 }, { 2, 2 } }, TensorShape{ 23, 45 }) ==
              PadList({ { 0, 0 }, { 1, 1 }, { 2, 2 }, { 0, 0 } }));
        // NHWC -> NHWC, (23, 45, 4)        -> (1, 23, 45, 4)
        CHECK(ExtendPadList({ { 1, 1 }, { 2, 2 }, { 3, 3 } }, TensorShape{ 23, 45, 4 }) ==
              PadList({ { 0, 0 }, { 1, 1 }, { 2, 2 }, { 3, 3 } }));
        // NHWC -> NHWC, (23, 45, 4, 235)   -> (23, 45, 4, 235)
        // Invalid as batch > 1 but this function shouldn't change the padding regardless
        CHECK(ExtendPadList({ { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } }, TensorShape{ 23, 45, 4, 235 }) ==
              PadList({ { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } }));
        // NH   -> NHWC, (1, 23)            -> (1, 23, 1, 1)
        CHECK(ExtendPadList({ { 1, 1 }, { 2, 2 } }, TensorShape{ 1, 23 }) ==
              PadList({ { 1, 1 }, { 2, 2 }, { 0, 0 }, { 0, 0 } }));
        // NHW  -> NHWC, (1, 23, 45)        -> (1, 23, 45, 1)
        CHECK(ExtendPadList({ { 1, 1 }, { 2, 2 }, { 3, 3 } }, TensorShape{ 1, 23, 45 }) ==
              PadList({ { 1, 1 }, { 2, 2 }, { 3, 3 }, { 0, 0 } }));
        // NHWC -> NHWC, (1, 23, 45, 4)     -> (1, 23, 45, 4)
        CHECK(ExtendPadList({ { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } }, TensorShape{ 1, 23, 45, 4 }) ==
              PadList({ { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } }));
    }

    TEST_CASE("BuildEthosNPaddingInfo")
    {
        PadDescriptor padding;
        padding.m_PadList = { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } };
        CHECK(BuildEthosNPaddingInfo(padding, TensorShape{ 1, 23, 45, 4 }) == ethosn_lib::Padding(2, 2, 3, 3));
    }
}

}    // namespace armnn
