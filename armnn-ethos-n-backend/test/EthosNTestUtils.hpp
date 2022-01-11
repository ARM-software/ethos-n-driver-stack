//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "EthosNBackend.hpp"
#include "EthosNConfig.hpp"

#include <SubgraphView.hpp>
#include <armnnUtils/Filesystem.hpp>

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

inline std::string ReadFile(const std::string& file)
{
    std::ifstream is(file);
    std::ostringstream contents;
    contents << is.rdbuf();
    return contents.str();
}

inline bool operator==(const armnn::SubgraphView& lhs, const armnn::SubgraphView& rhs)
{
    ARMNN_NO_DEPRECATE_WARN_BEGIN
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
    ARMNN_NO_DEPRECATE_WARN_END
}

/// Sets the globally cached backend config data, so that different tests can run with different configs.
/// Without this, the first test which instantiates an EthosNBackend object would load and set the config for all future
/// tests using EthosNBackend and there would be no way to change this. Tests can use this function to override
/// the cached values.
inline void SetBackendGlobalConfig(const armnn::EthosNConfig& config, const std::vector<char>& capabilities)
{
    class EthosNBackendEx : public armnn::EthosNBackend
    {
    public:
        void SetBackendGlobalConfigForTesting(const armnn::EthosNConfig& config, const std::vector<char>& capabilities)
        {
            ms_Config       = config;
            ms_Capabilities = capabilities;
        }
    };

    EthosNBackendEx().SetBackendGlobalConfigForTesting(config, capabilities);
}

/// Scope-controlled version of SetBackendGlobalConfig, which automatically restores
/// default settings after being destroyed. This can be used to avoid messing up the config for tests
/// that run afterwards.
class BackendGlobalConfigSetter
{
public:
    BackendGlobalConfigSetter(const armnn::EthosNConfig& config, const std::vector<char>& capabilities)
    {
        SetBackendGlobalConfig(config, capabilities);
    }
    ~BackendGlobalConfigSetter()
    {
        // Setting an empty capabilities vector will trigger a reload on next EthosNBackend instantiation
        SetBackendGlobalConfig(armnn::EthosNConfig(), {});
    }
};

}    // namespace testing_utils
