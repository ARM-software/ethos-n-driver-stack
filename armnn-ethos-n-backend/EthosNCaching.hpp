//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <armnn/BackendOptions.hpp>
#include <armnn/Optional.hpp>
#include <armnn/utility/Assert.hpp>

#include <algorithm>
#include <map>
#include <string>

namespace armnn
{

/// User options used to determine wheter to whether to save or load a cached network.
struct EthosNCachingOptions
{
    // Enables caching of the compiled network.
    // Used in conjunction with m_CachedNetworkFilePath to write compiled networks to a file.
    bool m_SaveCachedNetwork = false;

    // If non-empty, the given file will be used to load/save compiled networks.
    std::string m_CachedNetworkFilePath = "";
};

/// Storage object which contains all functionality required to save and load a network.
class EthosNCaching
{
public:
    struct CachedNetwork
    {
        std::vector<char> m_CompiledNetwork;
        uint32_t m_IntermediateDataSize;
    };

    EthosNCaching();
    ~EthosNCaching() = default;

    EthosNCachingOptions GetEthosNCachingOptions()
    {
        return m_EthosNCachingOptions;
    };

    void SetEthosNCachingOptions(const armnn::ModelOptions& modelOptions);

    uint32_t GetNumCachedNetworked() const
    {
        return static_cast<uint32_t>(m_CachedNetworks.size());
    }

    void AddCachedNetwork(uint32_t subgraphIdx, CachedNetwork cachedNetwork);
    armnn::Optional<const CachedNetwork&> GetCachedNetwork(uint32_t subgraphIdx) const;

    bool GetIsLoaded()
    {
        return m_IsLoaded;
    };

    void SetIsLoaded(const bool isLoaded)
    {
        m_IsLoaded = isLoaded;
    };

    // Helper methods
    bool IsLoading();

    bool IsSaving();

    bool Load();

    bool Save();

private:
    bool LoadCachedSubgraphs();

    bool SaveCachedSubgraphs();

    void Reset();

private:
    /// Caching options used to save or load compiled networks from all subgraphs.
    EthosNCachingOptions m_EthosNCachingOptions;

    /// Holds serialized compiled networks temporarily from all subgraphs.
    /// This is used to load or save the compiled networks.
    std::map<uint32_t, CachedNetwork> m_CachedNetworks;

    // Used to determine if the m_EthosNCachingOptions or m_CompiledNetworks have been loaded or not.
    bool m_IsLoaded;
};

/// Acts as a Singlton and stores an instance of EthosNCaching,
class EthosNCachingService
{
public:
    // Getter for the singleton instance
    static EthosNCachingService& GetInstance();

    EthosNCaching* GetEthosNCachingPtr()
    {
        return m_SharedEthosNCaching.get();
    }

    void SetEthosNCachingPtr(std::shared_ptr<EthosNCaching> shared)
    {
        m_SharedEthosNCaching = shared;
    }

private:
    std::shared_ptr<EthosNCaching> m_SharedEthosNCaching;
};

// Returns a populated EthosNCachingOptions based on the given ModelOptions.
EthosNCachingOptions GetEthosNCachingOptionsFromModelOptions(const armnn::ModelOptions& modelOptions);

}    // namespace armnn
