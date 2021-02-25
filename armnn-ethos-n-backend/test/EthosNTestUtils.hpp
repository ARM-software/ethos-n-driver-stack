//
// Copyright © 2018-2021 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "EthosNConfig.hpp"

#include <Filesystem.hpp>
#include <SubgraphView.hpp>

#include <cstdlib>
#include <fstream>
#include <string>

#define ARRAY_SIZE(X) (sizeof(X) / sizeof(X[0]))

namespace testing_utils
{

class TempDir
{
public:
    TempDir()
    {
        static std::atomic<int> g_Counter;
        int uniqueId = g_Counter++;
        m_Dir        = "TempDir-" + std::to_string(uniqueId);
        fs::create_directories(m_Dir);
    }

    ~TempDir()
    {
        fs::remove_all(m_Dir);
    }

    std::string Str() const
    {
        return m_Dir.string();
    }

private:
    fs::path m_Dir;
};

inline void SetEnv(const char* const name, const char* const value)
{
#if defined(__unix__)
    setenv(name, value, true);
#else
    // Speculative windows support (not tested)
    _putenv_s(name, value);
#endif
}

inline void CreateConfigFile(const std::string& configFile, const armnn::EthosNConfig& config)
{
    std::ofstream os(configFile);
    os << armnn::EthosNConfig::PERF_VARIANT_VAR << " = "
       << ethosn::support_library::EthosNVariantAsString(config.m_PerfVariant) << "\n";
    os << armnn::EthosNConfig::PERF_ONLY_VAR << " = " << config.m_PerfOnly << "\n";
    os << armnn::EthosNConfig::PERF_OUT_DIR_VAR << " = " << config.m_PerfOutDir << "\n";
    os << armnn::EthosNConfig::PERF_MAPPING_FILE_VAR << " = " << config.m_PerfMappingFile << "\n";
    os << armnn::EthosNConfig::PERF_ACTIVATION_COMPRESSION_SAVING << " = " << config.m_PerfActivationCompressionSaving
       << "\n";
    if (config.m_PerfUseWeightCompressionOverride)
    {
        os << armnn::EthosNConfig::PERF_WEIGHT_COMPRESSION_SAVING << " = " << config.m_PerfWeightCompressionSaving
           << "\n";
    }
    os << armnn::EthosNConfig::PERF_CURRENT << " = " << config.m_PerfCurrent << "\n";
}

inline std::string ReadFile(const std::string& file)
{
    std::ifstream is(file);
    std::ostringstream contents;
    contents << is.rdbuf();
    return contents.str();
}

inline bool operator==(const armnn::SubgraphView& lhs, const armnn::SubgraphView& rhs)
{
    if (lhs.GetInputSlots() != rhs.GetInputSlots())
    {
        return false;
    }

    if (lhs.GetOutputSlots() != rhs.GetOutputSlots())
    {
        return false;
    }

    auto lhsLayerI = lhs.cbegin();
    auto rhsLayerI = rhs.cbegin();

    if (std::distance(lhsLayerI, lhs.cend()) != std::distance(rhsLayerI, rhs.cend()))
    {
        return false;
    }

    while (lhsLayerI != lhs.cend() && rhsLayerI != rhs.cend())
    {
        if (*lhsLayerI != *rhsLayerI)
        {
            return false;
        }
        ++lhsLayerI;
        ++rhsLayerI;
    }

    return (lhsLayerI == lhs.cend() && rhsLayerI == rhs.cend());
}

}    // namespace testing_utils
