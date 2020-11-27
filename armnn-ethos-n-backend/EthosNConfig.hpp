//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <ethosn_driver_library/Network.hpp>
#include <ethosn_support_library/Support.hpp>

#include <string>

namespace armnn
{

/// Ethos-N backend configuration. It should be obtained via GetEthosNConfig()
struct EthosNConfig
{
    // Environment variable that points to the config file
    static constexpr char CONFIG_FILE_ENV[] = "ARMNN_ETHOSN_BACKEND_CONFIG_FILE";

    // Variables that may be configured inside the config file
    // clang-format off
    static constexpr char PERF_ONLY_VAR[]                       = "PERFORMANCE_ONLY";                             // boolean
    static constexpr char PERF_VARIANT_VAR[]                    = "PERFORMANCE_VARIANT";                          // enum
    static constexpr char PERF_SRAM_SIZE_BYTES_OVERRIDE_VAR[]   = "PERFORMANCE_SRAM_SIZE_BYTES_OVERRIDE";         // string
    static constexpr char PERF_OUT_DIR_VAR[]                    = "PERFORMANCE_OUTPUT_DIR";                       // string
    static constexpr char PERF_MAPPING_FILE_VAR[]               = "PERFORMANCE_MAPPING_FILE";                     // string
    static constexpr char DUMP_DEBUG_FILES_VAR[]                = "DUMP_DEBUG_FILES";                             // boolean
    static constexpr char DUMP_RAM_VAR[]                        = "DUMP_RAM";                                     // boolean
    static constexpr char PERF_WEIGHT_COMPRESSION_SAVING[]      = "PERFORMANCE_WEIGHT_COMPRESSION_SAVING";        // float
    static constexpr char PERF_ACTIVATION_COMPRESSION_SAVING[]  = "PERFORMANCE_ACTIVATION_COMPRESSION_SAVING";    // float
    static constexpr char PERF_CURRENT[]                        = "PERFORMANCE_CURRENT";                          // boolean
    static constexpr char COMPILER_ALGORITHM[]                  = "COMPILER_ALGORITHM";                           // enum
    static constexpr char INTERMEDIATE_COMPRESSION[]            = "INTERMEDIATE_COMPRESSION";                     // boolean
    // clang-format on

    bool m_PerfOnly                                      = false;
    ethosn::support_library::EthosNVariant m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N77;
    uint32_t m_PerfSramSizeBytesOverride                 = 0;
    std::string m_PerfOutDir                             = "ethosn_perf";
    std::string m_PerfMappingFile                        = "";
    bool m_DumpDebugFiles                                = false;
    bool m_DumpRam                                       = false;
    float m_PerfActivationCompressionSaving              = 0.0f;
    bool m_PerfUseWeightCompressionOverride              = false;
    float m_PerfWeightCompressionSaving                  = 0.0f;
    bool m_PerfCurrent                                   = false;
    ethosn::support_library::CompilerAlgorithm m_CompilerAlgorithm =
        ethosn::support_library::CompilerAlgorithm::NonCascadingOnly;
    bool m_IntermediateCompression = true;

    std::vector<char> GetCapabilities()
    {
        return m_PerfOnly ? ethosn::support_library::GetFwAndHwCapabilities(m_PerfVariant, m_PerfSramSizeBytesOverride)
                          : ethosn::driver_library::GetFirmwareAndHardwareCapabilities();
    }
};

/// Reads the configuration for the Ethos-N backend from the file pointed by the environment
/// variable with name EthosNConfig::CONFIG_FILE_ENV
EthosNConfig GetEthosNConfig();

}    // namespace armnn

std::ostream& operator<<(std::ostream& configFile, const armnn::EthosNConfig& config);
std::istream& operator>>(std::istream& configFile, armnn::EthosNConfig& config);
