//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../src/Network.hpp"
#include "../src/Utils.hpp"

#include <catch.hpp>

#include <iostream>
#include <unordered_set>

using namespace ethosn::support_library;

/// Checks that iteration over the network yields operations in topological order.
TEST_CASE("TopologyTest")
{
    TensorInfo inputInfo{
        { { 1, 128, 128, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo{
        { { 1, 1, 1, 16 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo bias2Info{
        { { 1, 1, 1, 16 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.1f },
    };

    TensorInfo bias3Info{
        { { 1, 1, 1, 64 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.0f },
    };

    TensorInfo biasFcInfo{
        { { 1, 1, 1, 4 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.1f },
    };

    TensorInfo weightsInfo{
        { { 3, 3, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 0, 1.f },
    };

    TensorInfo weightsFcInfo{
        { { 1, 1, 122 * 122 * 64, 4 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 0, 1.f },
    };

    TensorInfo weightsHwimInfo{
        { { 3, 3, 64, 1 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIM,
        { 0, 1.f },
    };

    ConvolutionInfo convInfo{
        { 0, 0, 0, 0 },
        { 1, 1 },
        { 0, 1.1f },
    };

    ConvolutionInfo conv2Info{
        { 0, 0, 0, 0 },
        { 1, 1 },
        { 0, 1.2f },
    };

    TensorInfo constInfo{
        { { 1, 1, 1, 1 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    constexpr ReluInfo reluInfo{ 0, 255 };

    constexpr PoolingInfo poolingInfo{ 2, 2, 2, 2, { 0, 0, 0, 0 }, PoolingType::MAX };

    FullyConnectedInfo fullyConnInfo{ { 0, 1.15f } };

    constexpr TensorShape newDimensions{ { 1, 2, 2, 1 } };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> bias2Data(utils::TotalSizeBytes(bias2Info));
    const std::vector<uint8_t> bias3Data(utils::TotalSizeBytes(bias3Info));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));
    const std::vector<uint8_t> constData(utils::TotalSizeBytes(constInfo));
    const std::vector<uint8_t> weightsHwimData(utils::TotalSizeBytes(weightsHwimInfo));
    const std::vector<uint8_t> weightsFcData(utils::TotalSizeBytes(weightsFcInfo));
    const std::vector<uint8_t> biasFcData(utils::TotalSizeBytes(biasFcInfo));

    std::shared_ptr<Network> network = CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Constant -> Output
    std::shared_ptr<Constant> constant     = AddConstant(network, constInfo, constData.data()).tensor;
    std::shared_ptr<Output> constantOutput = AddOutput(network, *GetOperand(constant)).tensor;

    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;

    /*                                                        Convolution
                                                            / Convolution \
    { Input, Constant, Constant } -> Convolution -> Relu ->                 -> Concatenation ->
                                                            \ Convolution /
                                                              Convolution


         DepthwiseConvolution
       /                      \
    ->                          -> Addition -> Reshape -> FullyConnected -> Reshape -> Pooling -> Sigmoid -> Output
       \                      /
         DepthwiseConvolution


    */

    std::shared_ptr<Operand> input = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> conv  = AddConvolution(network, *input, *bias, *weights, convInfo).tensor;
    std::shared_ptr<Operand> relu  = AddRelu(network, *conv, reluInfo).tensor;

    std::shared_ptr<Constant> bias2 = AddConstant(network, bias2Info, bias2Data.data()).tensor;
    uint32_t numLayers              = 4;
    std::vector<Operand*> layers;
    for (uint32_t i = 0; i < numLayers; i++)
    {
        std::shared_ptr<Operand> convA = AddConvolution(network, *relu, *bias2, *weights, conv2Info).tensor;
        layers.push_back(convA.get());
    }
    std::shared_ptr<Operand> concat       = AddConcatenation(network, layers, ConcatenationInfo(3, { 0, 1.0f })).tensor;
    std::shared_ptr<Constant> weightsHwim = AddConstant(network, weightsHwimInfo, weightsHwimData.data()).tensor;
    std::shared_ptr<Constant> bias3       = AddConstant(network, bias3Info, bias3Data.data()).tensor;
    std::shared_ptr<Operand> depthwiseConvA =
        AddDepthwiseConvolution(network, *concat, *bias3, *weightsHwim, convInfo).tensor;
    std::shared_ptr<Operand> depthwiseConvB =
        AddDepthwiseConvolution(network, *concat, *bias3, *weightsHwim, convInfo).tensor;
    std::shared_ptr<Operand> addition =
        AddAddition(network, *depthwiseConvA, *depthwiseConvB, convInfo.m_OutputQuantizationInfo).tensor;
    std::shared_ptr<Operand> reshape1   = AddReshape(network, *addition, TensorShape{ 1, 1, 1, 122 * 122 * 64 }).tensor;
    std::shared_ptr<Constant> weightsFc = AddConstant(network, weightsFcInfo, weightsFcData.data()).tensor;
    std::shared_ptr<Constant> biasFc    = AddConstant(network, biasFcInfo, biasFcData.data()).tensor;
    std::shared_ptr<Operand> fullyConnected =
        AddFullyConnected(network, *reshape1, *biasFc, *weightsFc, fullyConnInfo).tensor;
    std::shared_ptr<Operand> reshape = AddReshape(network, *fullyConnected, newDimensions).tensor;
    std::shared_ptr<Operand> pooling = AddPooling(network, *reshape, poolingInfo).tensor;
    std::shared_ptr<Operand> sigmoid = AddSigmoid(network, *pooling).tensor;
    std::shared_ptr<Output> output   = AddOutput(network, *sigmoid).tensor;

    // { Convolution, Constant, Constant } -> Convolution -> Output
    std::shared_ptr<Constant> weights2 = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv2     = AddConvolution(network, *conv, *bias2, *weights2, conv2Info).tensor;
    std::shared_ptr<Output> output2    = AddOutput(network, *conv2).tensor;

    // Check that operations are visited in topological order
    std::unordered_set<const Operation*> visited;
    for (const auto& operation : *network)
    {
        for (const Operand* input : operation->GetInputs())
        {
            REQUIRE(visited.count(&input->GetProducer()) == 1);
        }
        REQUIRE(visited.emplace(operation.get()).second);
    }
}
