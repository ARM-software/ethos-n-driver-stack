//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "ArmnnUtils.hpp"
#include "GlobalParameters.hpp"
#include "SystemTestsUtils.hpp"

#include <armnn/ArmNN.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <catch.hpp>
#include <ethosn_utils/Filesystem.hpp>
#include <ethosn_utils/Strings.hpp>

#include <algorithm>
#include <cassert>
#include <inttypes.h>
#include <numeric>
#include <random>

namespace ethosn
{
namespace system_tests
{

template <typename T>
InputTensor GenInputTensor(uint32_t numElements)
{
    std::vector<T> data(numElements);
    std::mt19937 gen;
    std::uniform_int_distribution<> distribution(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
    generate(data.begin(), data.end(), [&]() { return static_cast<T>(distribution(gen)); });

    return MakeTensor(data);
}

template <>
InputTensor GenInputTensor<float>(uint32_t numElements)
{
    std::vector<float> data(numElements);
    std::mt19937 gen;
    std::uniform_real_distribution<> distribution(0.f, 1.f);
    generate(data.begin(), data.end(), [&]() { return static_cast<float>(distribution(gen)); });

    return MakeTensor(data);
}

void CompareArmnnAndEthosNTflite(std::string tfLiteFile,
                                 const std::map<std::string, float>& referenceComparisonTolerances)
{
    using namespace armnn;
    CAPTURE(tfLiteFile);

    armnnTfLiteParser::ITfLiteParser::TfLiteParserOptions options;
    options.m_InferAndValidate                       = true;
    armnnTfLiteParser::ITfLiteParserPtr tfLiteParser = armnnTfLiteParser::ITfLiteParser::Create(options);

    INetworkPtr armnnNetwork(nullptr, nullptr);
    // Special option for reading tflite file from stdin.
    // Note that the standard "-" name doesn't seem to be compatible with the Catch2 command-line parsing,
    // so we use something else instead.
    if (tfLiteFile == "STDIN")
    {
        //  std::ifstream file(fileName, std::ios::binary);
        std::vector<uint8_t> fileContent((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
        g_Logger.Debug("Loaded %zu bytes from stdin", fileContent.size());
        armnnNetwork = tfLiteParser->CreateNetworkFromBinary(fileContent);
        // Attempt to generate a unique name for this stdin data, so that the armnn cache filename is unique (see below)
        // This means we can still benefit from the cache when running different networks through stdin.
        size_t hash = 17;
        for (uint8_t x : fileContent)
        {
            hash = hash * 37 + x;
        }
        tfLiteFile = "STDIN-" + std::to_string(hash);
        g_Logger.Warning("STDIN filename assigned as %s. Beware collisions for Arm NN cache!", tfLiteFile.c_str());
    }
    else
    {
        armnnNetwork = tfLiteParser->CreateNetworkFromBinaryFile(tfLiteFile.c_str());
    }
    std::string armnnCacheFilename = g_CachedRef ? GetCacheFilename(tfLiteFile, g_CacheFolder) : "";

    // We only support single subgraph networks for now.
    assert(tfLiteParser->GetSubgraphCount() == 1);

    std::vector<std::string> inputNames = tfLiteParser->GetSubgraphInputTensorNames(0);
    std::vector<LayerBindingId> inputBindings;
    InferenceInputs inputData;
    for (const std::string& inputName : inputNames)
    {
        BindingPointInfo bindingInfo = tfLiteParser->GetNetworkInputBindingInfo(0, inputName);
        InputTensor data;
        if (g_TfLiteIfmPath == "")
        {
            switch (bindingInfo.second.GetDataType())
            {
                case armnn::DataType::QAsymmU8:
                    data = GenInputTensor<uint8_t>(bindingInfo.second.GetNumElements());
                    break;
                case armnn::DataType::QSymmS8:
                case armnn::DataType::QAsymmS8:
                    data = GenInputTensor<int8_t>(bindingInfo.second.GetNumElements());
                    break;
                case armnn::DataType::Float32:
                    data = GenInputTensor<float>(bindingInfo.second.GetNumElements());
                    break;
                default:
                    throw std::invalid_argument("Unsupported input type");
            }
        }
        else
        {
            std::ifstream reader(const_cast<char*>(g_TfLiteIfmPath.c_str()), std::ios::binary);
            if (!reader.is_open())
            {
                g_Logger.Error("Failed to open Tflite Ifm file: %s", g_TfLiteIfmPath.c_str());
                return;
            }

            bool isFileNhwcb = (strstr(g_TfLiteIfmPath.c_str(), "NHWCB") != nullptr);
            // If the file is in NHWCB format, we might need to load more bytes, due to padding.
            uint32_t numElementsToLoad =
                isFileNhwcb ? GetTotalSizeNhwcb(bindingInfo.second.GetShape()[1], bindingInfo.second.GetShape()[2],
                                                bindingInfo.second.GetShape()[3])
                            : bindingInfo.second.GetNumElements();

            OwnedTensor fileData;
            if (utils::EndsWith(g_TfLiteIfmPath, ".hex"))
            {
                fileData =
                    LoadTensorFromHexStream(reader, GetDataType(bindingInfo.second.GetDataType()), numElementsToLoad);
            }
            else
            {
                fileData = LoadTensorFromBinaryStream(reader, GetDataType(bindingInfo.second.GetDataType()),
                                                      numElementsToLoad);
            }

            if (isFileNhwcb)
            {
                data = ConvertNhwcbToNhwc(*fileData, bindingInfo.second.GetShape()[1], bindingInfo.second.GetShape()[2],
                                          bindingInfo.second.GetShape()[3]);
            }
            else
            {
                data = std::move(fileData);
            }
        }

        DebugTensor(inputName.c_str(), *data, 256);

        inputBindings.push_back(bindingInfo.first);
        inputData.push_back(std::move(data));
    }

    std::vector<std::string> outputNames = tfLiteParser->GetSubgraphOutputTensorNames(0);
    std::vector<LayerBindingId> outputBindings(outputNames.size());
    std::transform(outputNames.begin(), outputNames.end(), outputBindings.begin(), [&](const auto& outputName) {
        BindingPointInfo binding = tfLiteParser->GetNetworkOutputBindingInfo(0, outputName);
        return binding.first;
    });

    tfLiteParser.release();    // Free up memory in the parser (it keeps a copy of the model)

    InferenceInputs cpu;
    if (!g_SkipReference)
    {
        cpu = RunNetworkCached(armnnCacheFilename, [&]() {
            return ArmnnRunNetwork(armnnNetwork.get(), g_ArmnnNonEthosNBackends, inputBindings, outputBindings,
                                   inputData, g_ArmnnBackendOptions, nullptr, false, 1);
        });
    }

    const char* dmaBufHeap = nullptr;

    // g_RunProtectedInference overrides g_DmaBuf
    if (g_RunProtectedInference)
    {
        dmaBufHeap = g_DmaBufProtected.c_str();
    }
    else if (g_UseDmaBuf)
    {
        dmaBufHeap = g_DmaBufHeap.c_str();
    }

    // Prefer the Ethos-N backend, but with fallback to other backends if not supported on Ethos-N
    std::vector<BackendId> backends = { "EthosNAcc" };
    backends.insert(backends.end(), g_ArmnnNonEthosNBackends.begin(), g_ArmnnNonEthosNBackends.end());
    auto ethosn = ArmnnRunNetwork(armnnNetwork.get(), backends, inputBindings, outputBindings, inputData,
                                  g_ArmnnBackendOptions, dmaBufHeap, g_RunProtectedInference, g_NumberRuns);

    if (!g_SkipReference)
    {
        for (uint32_t i = 0; i < cpu.size(); ++i)
        {
            float tolerance = GetReferenceComparisonTolerance(referenceComparisonTolerances, outputNames[i]);
            // -1 is the default value because the GgfRunner uses this to calculate a tolerance
            // We aren't so clever in the TfLiteRunner and use 0 tolerance if none has been provided.
            if (tolerance < 0.f)
            {
                tolerance = 0.f;
            }

            bool matchesReference = CompareTensors(*cpu[i], *ethosn[i], tolerance);
            std::string res;
            if (!matchesReference || g_Debug.find("dump-outputs") != std::string::npos)
            {
                res = DumpFiles(*ethosn[i], *cpu[i], outputNames[i], tolerance);
            }
            if (!matchesReference)
            {
                FAIL_CHECK(res);
            }
        }
    }
}

TEST_CASE("TfLiteRunner", "[!hide]")
{
    CompareArmnnAndEthosNTflite(g_TfLiteFilePath, g_ReferenceComparisonTolerances);
}

}    // namespace system_tests
}    // namespace ethosn
