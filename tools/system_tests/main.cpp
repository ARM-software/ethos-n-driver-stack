//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ArmnnUtils.hpp"
#include "GlobalParameters.hpp"
#include "SystemTestsUtils.hpp"
#include <ethosn_utils/KernelUtils.hpp>
#include <ethosn_utils/Strings.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch.hpp>
#include <iostream>

using namespace ethosn::system_tests;
using namespace ethosn::utils;

constexpr const char SystemTestsName[] = "system_tests";

int main(int argc, char* argv[])
{
    Catch::Session session;

    using namespace Catch::clara;

    auto const setDistributionSeed = [&](std::string const& seed) {
        if (seed != "time")
        {
            return Catch::clara::detail::convertInto(seed, g_DistributionSeed);
        }
        else
        {
            g_DistributionSeed = static_cast<unsigned int>(std::time(nullptr));
            return ParserResult::ok(ParseResultType::Matched);
        }
    };

    bool deprecatedDebugFlag                  = false;
    std::string deprecatedDebugFlagEquivalent = "dump-ram,dump-support-library-debug-files=High,system-tests-logging="
                                                "Debug,armnn-logging=Debug,dump-armnn-tensors,dump-armnn-graph";

    auto cli =
        session.cli() |
        Opt(g_GgfFilePath, "path")["-g"]["--ggf-file"]("Path to ggf-file used by GgfRunner and GgfEstimator") |
        Opt(g_TfLiteFilePath, "path")["--tflite-file"]("Path to tflite file used by TfLiteRunner") |
        Opt(g_TfLiteIfmPath, "path")["--tflite-ifm-file"]("Path to tflite Ifm file used by TfLiteRunner. "
                                                          "Only supports raw binary flattened array of uint8") |
        Opt(g_SkipReference)["--skip-ref"](
            "Do not run the network through the Arm NN reference. No reference comparison"
            " will be performed.") |
        Opt(g_CachedRef)["--cache"]("Use cached reference (Arm NN) output data. See also --cache-folder option.") |
        Opt(g_CacheFolder, "path")["--cache-folder"]("The folder to place cached outputs in (see --cache option).") |
        Opt(g_GgfUseArmnn)["--ggf-use-armnn"]("Use Arm NN with the Ethos-N backend when executing the Ethos-N "
                                              "part of the comparison for GgfRunner. If not specified "
                                              "then the Support Library will be used directly.") |
        Opt(
            [](std::string s) {
                g_ArmnnNonEthosNBackends.clear();
                for (std::string backend : Split(s, ","))
                {
                    g_ArmnnNonEthosNBackends.push_back(armnn::BackendId(backend));
                }

                return ParserResult::ok(ParseResultType::Matched);
            },
            "backends")["--armnn-non-ethosn-backends"](
            "Comma-separated list of Arm NN backends to use when computing the reference "
            "result, and as the fallback backends when running the Ethos-N inference through "
            " Arm NN.") |
        Opt(
            [](std::string s) {
                std::vector<std::string> backends = Split(s, ";");
                for (std::string backend : backends)
                {
                    backend                                      = Trim(backend);
                    std::vector<std::string> backendIdAndOptions = Split(backend, ":");
                    if (backendIdAndOptions.size() != 2)
                    {
                        return ParserResult::runtimeError(
                            "Invalid syntax for backend-options. Expected one colon per backend section.");
                    }
                    armnn::BackendId backendId = Trim(backendIdAndOptions[0]);

                    armnn::BackendOptions res(backendId);

                    std::vector<std::string> options = Split(backendIdAndOptions[1], ",");
                    for (std::string option : options)
                    {
                        std::vector<std::string> keyAndValue = Split(option, "=");
                        if (keyAndValue.size() != 2)
                        {
                            return ParserResult::runtimeError("Invalid syntax for backend-options. Expected one "
                                                              "equals-sign per backend option section.");
                        }
                        std::string key      = Trim(keyAndValue[0]);
                        std::string valueStr = Trim(keyAndValue[1]);
                        if (valueStr == "True")
                        {
                            res.AddOption(armnn::BackendOptions::BackendOption(key, true));
                        }
                        else if (valueStr == "False")
                        {
                            res.AddOption(armnn::BackendOptions::BackendOption(key, false));
                        }
                        else
                        {
                            res.AddOption(armnn::BackendOptions::BackendOption(key, valueStr));
                        }
                    }
                    g_ArmnnBackendOptions.push_back(res);
                }

                return ParserResult::ok(ParseResultType::Matched);
            },
            "backend options")["--backend-options"](
            "Options for Arm NN backends. Options are separated by commas "
            "and backends are separated by semicolons. For example: "
            "CpuRef:Option with spaces=Value1,OptionWithoutSpaces=Value2; EthosNAcc:DisableWinograd=True") |
        Opt(g_RunProtectedInference)["--run-protected-inf"](
            "Run protected inference with buffers from protected memory. Requires ethosn-tzmp1-test-module for "
            "allocating protected buffers. Relevant only when npu security is TZMP1. "
            "This flag overrides --use-dma-buf. ") |
        Opt(g_UseDmaBuf)["--use-dma-buf"]("Use Dma Buffer Heap as a shared memory to run test with zero copying ") |
        Opt(
            [](std::string s) {
                // Custom parser so that we can enable the --use-dma-buf flag automatically if a heap is specified.
                // Otherwise if a user provides --dma-buf-heap but not --use-dma-buf, it would silently be ignored.
                g_DmaBufHeap = std::move(s);
                g_UseDmaBuf  = true;
            },
            "/dev/dma-heaps/XYZ")["--dma-buf-heap"]("Use the specified dev file to allocate DMA bufs. "
                                                    " Relevant only if --use-dma-buf is used.") |
        Opt(g_Strategies,
            "0,1,3,...")["--strategies"]("Comma seperated list of strategy numbers to enable, used by GgfRunner") |
        Opt(g_BlockConfigs,
            "WxH,WxH,...")["--block-configs"]("Comma seperated list of block configs to enable, used by GgfRunner") |
        Opt(
            [](int x) {
                if (x < 1)
                {
                    return ParserResult::runtimeError("numRuns must be at least one");
                }
                g_NumberRuns = static_cast<size_t>(x);
                return ParserResult::ok(ParseResultType::Matched);
            },
            "numRuns")["--num-runs"]("Number of times the same inference has to be executed, used by GgfRunner") |
        Opt(
            [](int x) {
                if (x < 1)
                {
                    return ParserResult::runtimeError("runBatchSize must be at least one");
                }

                g_RunBatchSize = static_cast<size_t>(x);
                return ParserResult::ok(ParseResultType::Matched);
            },
            "runBatchSize")["--run-batch-size"](
            "Max number of inference runs that are allowed to allocate output buffers on the NPU at the same time. "
            "The specified batch size must be less or equal to the number of inference runs. "
            "This option is used by GgfRunner and by default the output buffers for all the inference runs are "
            "allocated at the same time.") |
        Opt(g_DefaultConvolutionAlgorithm, "algorithm")["--default-convolution-algorithm"](
            "Sets the default convolution algorithm to use when not specified in the ggf file. "
            "This overrides the default set in the support library.") |
        Opt(g_EthosNTimeoutSeconds,
            "seconds")["--ethosn-timeout"]("EthosN network timeout override in seconds, used by GgfRunner") |
        Opt(
            [](std::string s) {
                // Clear the default, as it will be replaced by what the user provided
                g_ReferenceComparisonTolerances.clear();
                // If the string is just a single floating point number, then use that for all outputs
                s = Trim(s);
                try
                {
                    float t                         = std::stof(s);
                    g_ReferenceComparisonTolerances = { { "*", t } };
                    return ParserResult::ok(ParseResultType::Matched);
                }
                catch (std::logic_error&)
                {
                    // Do nothing - we will attempt to parse the string differently below.
                }

                // Otherwise, parse it as a map, e.g. "Output1:-1.0,Output2:10"
                std::vector<std::string> outputs = Split(s, ",");
                for (std::string output : outputs)
                {
                    output = Trim(output);
                    // Note that some networks use colons in their output names, so we have to support this by splitting
                    // only on the last colon.
                    size_t colonIdx = output.rfind(':');
                    if (colonIdx == std::string::npos)
                    {
                        return ParserResult::runtimeError("Invalid syntax for --reference-comparison-tolerance. "
                                                          "Expected a colon between output name and tolerance.");
                    }
                    std::string outputName   = Trim(output.substr(0, colonIdx));
                    std::string toleranceStr = Trim(output.substr(colonIdx + 1));
                    float tolerance;
                    try
                    {
                        tolerance = std::stof(toleranceStr);
                    }
                    catch (std::logic_error&)
                    {
                        return ParserResult::runtimeError("Unable to convert to float: " + toleranceStr);
                    }
                    g_ReferenceComparisonTolerances[outputName] = tolerance;
                }

                return ParserResult::ok(ParseResultType::Matched);
            },
            "tolerance")["--reference-comparison-tolerance"](
            "Maximum allowable difference when comparing elements between actual and reference outputs. "
            "Can be specified as either a single floating point number to use for all outputs, or a map "
            "specifying a different value for each output of the network. "
            "For example: Output1:-1.0,Output2:10"
            "A special tolerance value of -1 (which is the default for all outputs if this is omitted) "
            "can be used to indicate a heuristic to automatically determine an appropriate tolerance. ") |
        Opt(setDistributionSeed, "'time'|number")["--distribution-seed"]("Seed for random distribution of weights") |
        Opt(deprecatedDebugFlag)["--debug"](std::string("<DEPRECATED> Enables a set of debugging features. ") +
                                            "This flag is deprecated, please use --debug-options instead. "
                                            "This flag is equivalent to --debug-options " +
                                            deprecatedDebugFlagEquivalent) |
        Opt(g_Debug, "comma separated list")["--debug-options"](
            "Enables debugging features. This is a comma-separated list of options. "
            "The following options are supported: \n"
            "\tdump-ram\n"
            "\tdump-support-library-debug-files=[None|Medium|High]\n"
            "\tdump-inputs\n"
            "\tdump-outputs\n"
            "\tsystem-tests-logging=[Panic|Error|Warning|Info|Debug|Verbose]\n"
            "\tarmnn-logging=[Fatal|Error|Warning|Info|Debug|Trace]\n"
            "\tdump-armnn-tensors\n"
            "\tdump-armnn-graph\n"
            "\armnn-profiling\n") |
        Opt(g_StrictPrecision)["--strict-precision"](
            "Enable this option for more precise but slower compiled network."
            "If not specified then optimization for quantization operations at concat inputs will be applied and less "
            "precision results is expected ") |
        Opt(g_BlockInferenceForDebug)["--block-inferences-debug"](
            "Enable this option for blocking new inferences if the current inference has failed.") |
        Opt(g_SkipOutputDistributionCheck)["--skip-output-distribution-check"](
            "Skips checking that the output of the inference has a good distribution of values.");

    session.cli(cli);

    int returnCode = session.applyCommandLine(argc, argv);

    if (returnCode != 0)
    {
        // Indicates a command line error
        return returnCode;
    }

    if (g_RunProtectedInference)
    {
        try
        {
            // Create a DmaBufferDevice to see if a protected dma buffer can be created (or simulated)
            DmaBufferDevice(g_DmaBufProtected.c_str());
        }
        catch (...)
        {
            std::cerr << g_DmaBufProtected.c_str()
                      << " cannot be used. Check if environment is configured for TZMP1 to run protected inference test"
                      << std::endl;

            return -1;
        }
    }

    if (g_UseDmaBuf)
    {
        try
        {
            // Create a DmaBufferDevice to see if a non-protected dma buffer can be created (or simulated)
            DmaBufferDevice(g_DmaBufHeap.c_str());
        }
        catch (...)
        {
            std::cerr << g_DmaBufHeap.c_str()
                      << " cannot be used. Check if environment is configured run imported buffer test" << std::endl;

            return -1;
        }
    }

    // Convert the deprecated --debug flag into the new debug options string.
    // This is a list of all the debug options that used to be enabled by the --debug flag
    // before we switched to having individually requested debug options.
    if (deprecatedDebugFlag)
    {
        g_Debug += deprecatedDebugFlagEquivalent;
    }

    if (g_NumberRuns < g_RunBatchSize)
    {
        std::cerr << "Run batch size must be less or equal to the number of runs" << std::endl;
        return -1;
    }

    if (g_Debug.find("dump-armnn-tensors") != std::string::npos && g_GgfUseArmnn && g_UseDmaBuf)
    {
        std::cerr << "Error, dump-armnn-tensors is incompatible with --use-dma-buf and --ggf-use-armnn because it "
                     "falls back to CpuRef that is not compatible with importing buffers."
                  << std::endl;
        return -1;
    }

    // Configure system_tests logging
    g_Logger.AddSink(ethosn::utils::log::sinks::StdOut<SystemTestsName>);
    ethosn::utils::log::Severity severity = ethosn::utils::log::Severity::Info;
    if (g_Debug.find("system-tests-logging=Panic") != std::string::npos)
    {
        severity = ethosn::utils::log::Severity::Panic;
    }
    else if (g_Debug.find("system-tests-logging=Error") != std::string::npos)
    {
        severity = ethosn::utils::log::Severity::Error;
    }
    else if (g_Debug.find("system-tests-logging=Warning") != std::string::npos)
    {
        severity = ethosn::utils::log::Severity::Warning;
    }
    else if (g_Debug.find("system-tests-logging=Info") != std::string::npos)
    {
        severity = ethosn::utils::log::Severity::Info;
    }
    else if (g_Debug.find("system-tests-logging=Debug") != std::string::npos)
    {
        severity = ethosn::utils::log::Severity::Debug;
    }
    else if (g_Debug.find("system-tests-logging=Verbose") != std::string::npos)
    {
        severity = ethosn::utils::log::Severity::Verbose;
    }
    g_Logger.SetMaxSeverity(severity);

    // Configure Arm NN logging
    ConfigureArmnnLogging();

    return session.run();
}
