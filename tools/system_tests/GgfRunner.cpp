//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "GgfRunner.hpp"

#include "ArmnnParseRunner.hpp"
#include "EthosNParseRunner.hpp"
#include "GlobalParameters.hpp"

#include <catch.hpp>

namespace ethosn
{
namespace system_tests
{

void CompareArmnnAndEthosNOutput(const char* ggfFilename,
                                 LayerData& layerData,
                                 bool verifyStatisticalOutput,
                                 const std::map<std::string, float>& referenceComparisonTolerances)
{
    CAPTURE(ggfFilename);
    std::string ggfFilenameStr(ggfFilename);

    std::ifstream ggfFile(ggfFilename, std::ios_base::binary | std::ios_base::in);
    if (!ggfFile.is_open())
    {
        std::string error = "Failed to open ggf file: " + ggfFilenameStr + "\n";
        throw std::invalid_argument(error);
    }

    std::string armnnCacheFilename = g_CachedRef ? GetCacheFilename(ggfFilenameStr, g_CacheFolder) : "";

    CompareArmnnAndEthosNOutput(ggfFile, layerData, verifyStatisticalOutput, referenceComparisonTolerances,
                                armnnCacheFilename);
}

InferenceOutputs RunArmnn(std::istream& ggfFile,
                          LayerData& layerData,
                          const std::string& armnnCacheFilename,
                          std::vector<armnn::BackendId> backends)
{
    g_Logger.Debug("Parsing and Executing on Armnn...");

    // Parse Network using Arm NN
    ArmnnParseRunner armnnParseRunner(ggfFile, layerData);

    // Run Network using Arm NN, using the cache if requested
    InferenceOutputs nhwcArmnnOutput =
        RunNetworkCached(armnnCacheFilename, [&]() { return armnnParseRunner.RunNetwork(backends); });

    size_t numOutputs = armnnParseRunner.GetOutputLayerNames().size();

    for (uint32_t i = 0; i < numOutputs; ++i)
    {
        g_Logger.Debug("Output (%s): ", armnnParseRunner.GetOutputLayerNames()[i].c_str());
        DebugTensor("nhwcArmnnOutput", *nhwcArmnnOutput[i], 256);
    }

    return nhwcArmnnOutput;
}

std::tuple<InferenceOutputs, float, std::vector<std::string>> RunEthosN(std::istream& ggfFile, LayerData& layerData)
{
    InferenceOutputs nhwcEthosNOutput;
    g_Logger.Debug("Parsing and Executing on Ethos-N...");

    ggfFile.clear();
    ggfFile.seekg(0);

    if (!g_GgfUseArmnn)
    {
        // Parse Network using Ethos-N Support Library
        EthosNParseRunner::CreationOptions creationOptions =
            EthosNParseRunner::CreationOptions::CreateWithGlobalOptions(ggfFile, layerData);
        EthosNParseRunner ethosnParseRunner(creationOptions);

        ethosnParseRunner.SetStrategies(g_Strategies);
        ethosnParseRunner.SetBlockConfigs(g_BlockConfigs);

        size_t numOutputs = ethosnParseRunner.GetOutputLayerNames().size();
        // Run Network using Ethos-N Driver Library
        InferenceOutputs ethosnOutput = ethosnParseRunner.RunNetwork();

        nhwcEthosNOutput.resize(numOutputs);
        for (uint32_t i = 0; i < numOutputs; ++i)
        {
            g_Logger.Debug("Output (%s)", ethosnParseRunner.GetOutputLayerNames()[i].c_str());
            DebugTensor("ethosnOutput", *ethosnOutput[i], 256);

            // Convert to NHWC if necessary
            if (layerData.GetOutputTensorFormat() == ethosn::support_library::DataFormat::NHWCB)
            {
                ethosn::support_library::TensorShape outputShape =
                    ethosnParseRunner.GetLayerOutputShape(ethosnParseRunner.GetOutputLayerNames()[i]);
                nhwcEthosNOutput[i] =
                    ConvertNhwcbToNhwc(*ethosnOutput[i], outputShape[1], outputShape[2], outputShape[3]);
                DebugTensor("nhwcEthosNOutput[i]", *nhwcEthosNOutput[i], 256);
            }
            else
            {
                nhwcEthosNOutput[i] = std::move(ethosnOutput[i]);
            }
        }

        return std::tuple<InferenceOutputs, float, std::vector<std::string>>{
            std::move(nhwcEthosNOutput), ethosnParseRunner.GetComparisonTolerance(),
            ethosnParseRunner.GetOutputLayerNames()
        };
    }
    else
    {
        ArmnnParseRunner armnnParseRunner(ggfFile, layerData);

        for (size_t k = 0; k < g_NumberRuns; ++k)
        {
            InferenceOutputs nhwcArmnnOutput = armnnParseRunner.RunNetwork({ "EthosNAcc" });

            // Save the first inference output to use as a reference for the other inferences
            if (k == 0)
            {
                nhwcEthosNOutput = std::move(nhwcArmnnOutput);
                continue;
            }

            for (size_t i = 0; i < nhwcEthosNOutput.size(); ++i)
            {
                if (!CompareTensors(*nhwcEthosNOutput[i], *nhwcArmnnOutput[i], 0.f))
                {
                    std::string res = DumpOutputToFiles(*nhwcArmnnOutput[i], *nhwcEthosNOutput[i], "EthosNAcc",
                                                        armnnParseRunner.GetOutputLayerNames()[i], k);
                    throw std::runtime_error(res);
                }
            }
        }

        return std::tuple<InferenceOutputs, float, std::vector<std::string>>{ std::move(nhwcEthosNOutput), 0.f,
                                                                              armnnParseRunner.GetOutputLayerNames() };
    }
}

void CompareArmnnAndEthosNOutput(std::istream& ggfFile,
                                 LayerData& layerData,
                                 bool verifyStatisticalOutput,
                                 const std::map<std::string, float>& referenceComparisonTolerances,
                                 const std::string& armnnCacheFilename)
{
    InferenceOutputs nhwcArmnnOutput;

    if (!g_SkipReference)
    {
        nhwcArmnnOutput = RunArmnn(ggfFile, layerData, armnnCacheFilename, g_ArmnnNonEthosNBackends);

        if (verifyStatisticalOutput && layerData.GetVerifyDistribution())
        {
            if (!IsStatisticalOutputGood(nhwcArmnnOutput))
            {
                FAIL("Distribution of outputs is not good enough (see above histogram).");
            }
        }
    }

    InferenceOutputs nhwcEthosNOutput;
    float ethosnReferenceComparisonTolerance;
    std::vector<std::string> outputNames;
    std::tie(nhwcEthosNOutput, ethosnReferenceComparisonTolerance, outputNames) = RunEthosN(ggfFile, layerData);
    if (g_SkipReference)
    {
        // If we skipped Arm NN then verify the statistics of the Ethos-N output instead.
        if (verifyStatisticalOutput && layerData.GetVerifyDistribution())
        {
            if (!IsStatisticalOutputGood(nhwcEthosNOutput))
            {
                FAIL("Distribution of outputs is not good enough (see above histogram).");
            }
        }
        WARN("Arm NN has been disabled via --skip-ref - no reference comparison is being performed");
    }
    else
    {
        g_Logger.Debug("Number of runs = %zu", g_NumberRuns);

        if (nhwcArmnnOutput.size() != nhwcEthosNOutput.size())
        {
            FAIL("Different number of output tensors");
        }

        for (uint32_t i = 0; i < nhwcEthosNOutput.size(); ++i)
        {
            float tolerance = GetReferenceComparisonTolerance(referenceComparisonTolerances, outputNames[i]);
            // Determine an appropriate comparison tolerance if one has not been specified
            if (tolerance < 0.f)
            {
                tolerance = ethosnReferenceComparisonTolerance;
            }
            g_Logger.Debug("Output %s - comparing to reference with tolerance +/-%f", outputNames[i].c_str(),
                           tolerance);

            bool matchesReference = CompareTensors(*nhwcEthosNOutput[i], *nhwcArmnnOutput[i], tolerance);
            std::string res;
            if (!matchesReference || g_Debug.find("dump-outputs") != std::string::npos)
            {
                res = DumpFiles(*nhwcEthosNOutput[i], *nhwcArmnnOutput[i], outputNames[i], tolerance);
            }
            if (!matchesReference)
            {
                FAIL_CHECK(res);
            }
        }
    }
}

TEST_CASE("strategy3_conv1x1_fixdata", "[!hide]")
{
    LayerData layerData;

    // Dimensions must match the input layer in the ggf file
    const int ifmHeight   = 16;
    const int ifmWidth    = 16;
    const int ifmChannels = 16;
    const int ofmChannels = 16;

    const bool useZeroInput  = false;
    const bool useZeroWeight = false;
    const bool useZeroBias   = false;

    // Generate Input Data
    std::vector<uint8_t> inputData(ifmHeight * ifmWidth * ifmChannels, 0);

    if (useZeroInput == false)
    {
        for (size_t i = 0; i < inputData.size(); ++i)
        {
            // The first channel contain XY pos of elements
            // the other channels contain the channel number + 1 for all elements
            size_t x     = ((i / ifmChannels) % ifmWidth);
            size_t y     = ((i / ifmChannels) / ifmWidth);
            inputData[i] = static_cast<uint8_t>((i % ifmChannels) ? (i % ifmChannels) + 1 : ((y << 4) | x));
        }
    }

    // Generate convolution weights that copy the input layer
    const int conv1InChannels   = ifmChannels;
    const int conv1KernelHeight = 1;
    const int conv1KernelWidth  = 1;
    const int conv1OutChannels  = ofmChannels;

    std::vector<uint8_t> weightsData(conv1InChannels * conv1KernelHeight * conv1KernelWidth * conv1OutChannels, 0);

    if (useZeroWeight == false)
    {
        for (std::vector<int>::size_type i = 0; i < conv1OutChannels; ++i)
        {
            weightsData[conv1InChannels * i + i] = 1;
        }
    }

    // Generate bias data
    std::vector<int32_t> biasData(ofmChannels, 0);

    if (useZeroBias == false)
    {
        for (int32_t i = 0; i < ofmChannels; ++i)
        {
            // Use output channel number + 1 as bias data
            biasData[i] = i + 1;
        }
    }

    // Populate Layer Data
    layerData.SetTensor("layer 0 input - tensor", *MakeTensor(inputData));
    layerData.SetTensor("layer 1 conv - weights", *MakeTensor(weightsData));
    layerData.SetTensor("layer 1 conv - bias", *MakeTensor(biasData));
    layerData.SetQuantInfo("layer 1 conv - weight quantization parameters", { 0, 1.0f });
    layerData.SetQuantInfo("layer 1 conv - bias quantization parameters", { 0, 1.0f });
    layerData.SetQuantInfo("layer 1 conv - output quantization parameters", { 0, 1.01f });
    CompareArmnnAndEthosNOutput("tests/graphs/strategy3_conv1x1relu.ggf", layerData);
}

TEST_CASE("GgfRunner", "[!hide]")
{
    LayerData layerData;
    try
    {
        CompareArmnnAndEthosNOutput(g_GgfFilePath.c_str(), layerData, !g_SkipOutputDistributionCheck,
                                    g_ReferenceComparisonTolerances);
    }
    catch (const std::exception& e)
    {
        FAIL(e.what());
    }
}

}    // namespace system_tests
}    // namespace ethosn
