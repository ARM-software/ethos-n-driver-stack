//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNConfig.hpp"

#include <armnn/Exceptions.hpp>
#include <boost/lexical_cast.hpp>

#include <cstdlib>
#include <fstream>
#include <regex>

namespace armnn
{

constexpr char EthosNConfig::CONFIG_FILE_ENV[];
constexpr char EthosNConfig::PERF_ONLY_VAR[];
constexpr char EthosNConfig::PERF_VARIANT_VAR[];
constexpr char EthosNConfig::PERF_SRAM_SIZE_BYTES_OVERRIDE_VAR[];
constexpr char EthosNConfig::PERF_MAPPING_FILE_VAR[];
constexpr char EthosNConfig::PERF_OUT_DIR_VAR[];
constexpr char EthosNConfig::DUMP_DEBUG_FILES_VAR[];
constexpr char EthosNConfig::PERF_WEIGHT_COMPRESSION_SAVING[];
constexpr char EthosNConfig::PERF_ACTIVATION_COMPRESSION_SAVING[];
constexpr char EthosNConfig::PERF_CURRENT[];
constexpr char EthosNConfig::CASCADING[];

EthosNConfig GetEthosNConfig()
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

std::ostream& operator<<(std::ostream& configFile, const armnn::EthosNConfig& config)
{
    configFile << armnn::EthosNConfig::PERF_ONLY_VAR << " = " << config.m_PerfOnly << std::endl;
    configFile << armnn::EthosNConfig::PERF_VARIANT_VAR << " = "
               << ethosn::support_library::EthosNVariantAsString(config.m_PerfVariant) << std::endl;
    configFile << armnn::EthosNConfig::PERF_SRAM_SIZE_BYTES_OVERRIDE_VAR << " = " << config.m_PerfSramSizeBytesOverride
               << std::endl;
    configFile << armnn::EthosNConfig::PERF_OUT_DIR_VAR << " = " << config.m_PerfOutDir << std::endl;
    configFile << armnn::EthosNConfig::PERF_MAPPING_FILE_VAR << " = " << config.m_PerfMappingFile << std::endl;
    configFile << armnn::EthosNConfig::DUMP_DEBUG_FILES_VAR << " = " << config.m_DumpDebugFiles << std::endl;
    configFile << armnn::EthosNConfig::PERF_WEIGHT_COMPRESSION_SAVING << " = " << config.m_PerfWeightCompressionSaving
               << std::endl;
    configFile << armnn::EthosNConfig::PERF_ACTIVATION_COMPRESSION_SAVING << " = "
               << config.m_PerfActivationCompressionSaving << std::endl;
    configFile << armnn::EthosNConfig::PERF_CURRENT << " = " << config.m_PerfCurrent << std::endl;
    configFile << armnn::EthosNConfig::CASCADING << " = " << config.m_EnableCascading << std::endl;
    configFile.flush();

    return configFile;
}

std::istream& operator>>(std::istream& configFile, armnn::EthosNConfig& config)
{
    if (configFile.good())
    {
        const std::regex varAssignRegex("(?:\\s*([A-Z_][A-Z_0-9]*)\\s*=\\s*(\\S*))\\s*(?:#.*)?");

        std::string line;

        for (size_t lineNo = 1; std::getline(configFile, line); ++lineNo)
        {
            std::smatch m;

            if (!std::regex_match(line, m, varAssignRegex))
            {
                throw armnn::Exception("Could not parse config file: line " + std::to_string(lineNo) + ": " + line);
            }
            else if (m[1].matched)
            {
                if (m[1] == armnn::EthosNConfig::PERF_ONLY_VAR)
                {
                    config.m_PerfOnly = boost::lexical_cast<bool>(m[2]);
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
                                               "\nMust be one of: Ethos-N37, Ethos-N57, Ethos-N77, "
                                               "Ethos-N78_1TOPS_2PLE_RATIO, Ethos-N78_1TOPS_4PLE_RATIO, "
                                               "Ethos-N78_2TOPS_2PLE_RATIO, Ethos-N78_2TOPS_4PLE_RATIO, "
                                               "Ethos-N78_4TOPS_2PLE_RATIO, Ethos-N78_4TOPS_4PLE_RATIO, "
                                               "Ethos-N78_8TOPS_2PLE_RATIO");
                    }
                }
                else if (m[1] == armnn::EthosNConfig::PERF_SRAM_SIZE_BYTES_OVERRIDE_VAR)
                {
                    config.m_PerfSramSizeBytesOverride = boost::lexical_cast<uint32_t>(m[2]);
                }
                else if (m[1] == armnn::EthosNConfig::PERF_OUT_DIR_VAR)
                {
                    config.m_PerfOutDir = m[2];
                }
                else if (m[1] == armnn::EthosNConfig::PERF_MAPPING_FILE_VAR)
                {
                    config.m_PerfMappingFile = m[2];
                }
                else if (m[1] == armnn::EthosNConfig::DUMP_DEBUG_FILES_VAR)
                {
                    config.m_DumpDebugFiles = boost::lexical_cast<bool>(m[2]);
                }
                else if (m[1] == armnn::EthosNConfig::PERF_ACTIVATION_COMPRESSION_SAVING)
                {
                    config.m_PerfActivationCompressionSaving = boost::lexical_cast<float>(m[2]);
                }
                else if (m[1] == armnn::EthosNConfig::PERF_WEIGHT_COMPRESSION_SAVING)
                {
                    config.m_PerfUseWeightCompressionOverride = true;
                    config.m_PerfWeightCompressionSaving      = boost::lexical_cast<float>(m[2]);
                }
                else if (m[1] == armnn::EthosNConfig::PERF_CURRENT)
                {
                    config.m_PerfCurrent = boost::lexical_cast<bool>(m[2]);
                }
                else if (m[1] == armnn::EthosNConfig::CASCADING)
                {
                    config.m_EnableCascading = boost::lexical_cast<bool>(m[2]);
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
