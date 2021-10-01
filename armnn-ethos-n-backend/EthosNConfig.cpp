//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNConfig.hpp"

#include <armnn/Exceptions.hpp>

#include <cstdlib>
#include <fstream>
#include <regex>

namespace
{

bool TryConvertToBool(std::ssub_match submatch, const std::string& line, const size_t lineNo)
{
    if (submatch != "1" && submatch != "0")
    {
        throw armnn::Exception("Unable to convert to boolean in config file on line " + std::to_string(lineNo) + ": " +
                               line);
    }
    return submatch == "1";
}

float TryConvertToFloat(std::ssub_match submatch, const std::string& line, const size_t lineNo)
{
    try
    {
        return std::stof(submatch);
    }
    catch (std::invalid_argument&)
    {
        throw armnn::Exception("Unable to convert to float in config file on line " + std::to_string(lineNo) + ": " +
                               line);
    }
    catch (std::out_of_range&)
    {
        throw armnn::Exception("Floating point out of range in config file on line " + std::to_string(lineNo) + ": " +
                               line);
    }
}

uint32_t TryConvertToUnsigned(std::ssub_match submatch, const std::string& line, const size_t lineNo)
{
    try
    {
        return static_cast<uint32_t>(std::stoul(submatch));
    }
    catch (std::invalid_argument&)
    {
        throw armnn::Exception("Unable to convert to unsigned integer in config file on line " +
                               std::to_string(lineNo) + ": " + line);
    }
    catch (std::out_of_range&)
    {
        throw armnn::Exception("Unsigned integer out of range in config file on line " + std::to_string(lineNo) + ": " +
                               line);
    }
}

}    // namespace

namespace armnn
{

constexpr char EthosNConfig::CONFIG_FILE_ENV[];
constexpr char EthosNConfig::PERF_ONLY_VAR[];
constexpr char EthosNConfig::PERF_VARIANT_VAR[];
constexpr char EthosNConfig::PERF_SRAM_SIZE_BYTES_OVERRIDE_VAR[];
constexpr char EthosNConfig::PERF_OUT_DIR_VAR[];
constexpr char EthosNConfig::DUMP_DEBUG_FILES_VAR[];
constexpr char EthosNConfig::DUMP_RAM_VAR[];
constexpr char EthosNConfig::PERF_WEIGHT_COMPRESSION_SAVING[];
constexpr char EthosNConfig::PERF_ACTIVATION_COMPRESSION_SAVING[];
constexpr char EthosNConfig::PERF_CURRENT[];
constexpr char EthosNConfig::COMPILER_ALGORITHM[];
constexpr char EthosNConfig::INTERMEDIATE_COMPRESSION[];

EthosNConfig ReadEthosNConfig()
{
    EthosNConfig config;

    char* configFilePath = std::getenv(EthosNConfig::CONFIG_FILE_ENV);
    if (configFilePath != nullptr)
    {
        std::ifstream configFile(configFilePath);
        configFile >> config;
    }

    return config;
}

}    // namespace armnn

std::istream& operator>>(std::istream& configFile, armnn::EthosNConfig& config)
{
    if (configFile.good())
    {
        const std::regex varAssignRegex("(?:\\s*([A-Z_][A-Z_0-9]*)\\s*=\\s*(\\S*))\\s*(?:#.*)?");

        std::string line;

        for (size_t lineNo = 1; std::getline(configFile, line); ++lineNo)
        {
            if (line.empty() || line[0] == '#')
            {
                continue;
            }

            std::smatch m;

            if (!std::regex_match(line, m, varAssignRegex))
            {
                throw armnn::Exception("Could not parse config file: line " + std::to_string(lineNo) + ": " + line);
            }
            else if (m[1].matched)
            {
                if (m[1] == armnn::EthosNConfig::PERF_ONLY_VAR)
                {
                    config.m_PerfOnly = TryConvertToBool(m[2], line, lineNo);
                }
                else if (m[1] == armnn::EthosNConfig::PERF_VARIANT_VAR)
                {
                    try
                    {
                        config.m_PerfVariant = ethosn::support_library::EthosNVariantFromString(m[2].str().c_str());
                    }
                    catch (std::invalid_argument&)
                    {
                        throw armnn::Exception("Invalid variant specified on line " + std::to_string(lineNo) + ": " +
                                               line +
                                               "\nMust be one of: "
                                               "Ethos-N78_1TOPS_2PLE_RATIO, Ethos-N78_1TOPS_4PLE_RATIO, "
                                               "Ethos-N78_2TOPS_2PLE_RATIO, Ethos-N78_2TOPS_4PLE_RATIO, "
                                               "Ethos-N78_4TOPS_2PLE_RATIO, Ethos-N78_4TOPS_4PLE_RATIO, "
                                               "Ethos-N78_8TOPS_2PLE_RATIO");
                    }
                }
                else if (m[1] == armnn::EthosNConfig::PERF_SRAM_SIZE_BYTES_OVERRIDE_VAR)
                {
                    config.m_PerfSramSizeBytesOverride = TryConvertToUnsigned(m[2], line, lineNo);
                }
                else if (m[1] == armnn::EthosNConfig::PERF_OUT_DIR_VAR)
                {
                    config.m_PerfOutDir = m[2];
                }
                else if (m[1] == armnn::EthosNConfig::DUMP_DEBUG_FILES_VAR)
                {
                    if (m[2] == "None" || m[2] == "0")
                    {
                        config.m_DumpDebugFiles = ethosn::support_library::CompilationOptions::DebugLevel::None;
                    }
                    else if (m[2] == "Medium")
                    {
                        config.m_DumpDebugFiles = ethosn::support_library::CompilationOptions::DebugLevel::Medium;
                    }
                    else if (m[2] == "High" || m[2] == "1")
                    {
                        config.m_DumpDebugFiles = ethosn::support_library::CompilationOptions::DebugLevel::High;
                    }
                    else
                    {
                        throw armnn::Exception("Unable to convert to DebugLevel in config file on line " +
                                               std::to_string(lineNo) + ": " + line +
                                               ". Supported values are 0/1/None/Medium/High");
                    }
                }
                else if (m[1] == armnn::EthosNConfig::DUMP_RAM_VAR)
                {
                    config.m_DumpRam = TryConvertToBool(m[2], line, lineNo);
                }
                else if (m[1] == armnn::EthosNConfig::PERF_ACTIVATION_COMPRESSION_SAVING)
                {
                    config.m_PerfActivationCompressionSaving = TryConvertToFloat(m[2], line, lineNo);
                }
                else if (m[1] == armnn::EthosNConfig::PERF_WEIGHT_COMPRESSION_SAVING)
                {
                    config.m_PerfUseWeightCompressionOverride = true;
                    config.m_PerfWeightCompressionSaving      = TryConvertToFloat(m[2], line, lineNo);
                }
                else if (m[1] == armnn::EthosNConfig::PERF_CURRENT)
                {
                    config.m_PerfCurrent = TryConvertToBool(m[2], line, lineNo);
                }
                else if (m[1] == armnn::EthosNConfig::COMPILER_ALGORITHM)
                {
                    try
                    {
                        config.m_CompilerAlgorithm =
                            ethosn::support_library::EthosNCompilerAlgorithmFromString(m[2].str().c_str());
                    }
                    catch (std::invalid_argument&)
                    {
                        throw armnn::Exception("Invalid value '" + m[2].str() + "' for option " +
                                               std::string(armnn::EthosNConfig::COMPILER_ALGORITHM) +
                                               ". Must be one of: \n"
#define X(value) +#value + "\n"
                                               COMPILER_ALGORITHM_MODE
#undef X
                        );
                    }
                }
                else if (m[1] == armnn::EthosNConfig::INTERMEDIATE_COMPRESSION)
                {
                    config.m_IntermediateCompression = TryConvertToBool(m[2], line, lineNo);
                }
                else
                {
                    throw armnn::Exception("Unknown var in config file: line " + std::to_string(lineNo) + ": " + line);
                }
            }
            else
            {
                // Empty line or comment. Continue
            }
        }
    }

    return configFile;
}
