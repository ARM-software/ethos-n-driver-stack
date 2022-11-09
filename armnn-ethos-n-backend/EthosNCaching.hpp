//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <armnn/BackendOptions.hpp>
#include <armnn/utility/Assert.hpp>

#include <algorithm>
#include <string>
#include <vector>

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
    EthosNCaching();
    ~EthosNCaching() = default;

    // Getters and Setters
    uint32_t GetSubgraphCount()
    {
        return m_SubgraphCount;
    };

    void IncrementSubgraphCount()
    {
        m_SubgraphCount++;
    };

    EthosNCachingOptions GetEthosNCachingOptions()
    {
        return m_EthosNCachingOptions;
    };

    void SetEthosNCachingOptions(const armnn::ModelOptions& modelOptions);

    uint32_t GetNumCachedNetworked() const
    {
        return static_cast<uint32_t>(m_CompiledNetworks.size());
    }

    std::pair<std::vector<char>, uint32_t> GetCompiledNetworkAndIntermediateSize(uint32_t id) const
    {
        std::vector<char> compiledNetwork = m_CompiledNetworks.at(id);
        ARMNN_ASSERT(compiledNetwork.size() > sizeof(uint32_t));
        uint32_t intermediateSize = 0;
        std::copy_n(compiledNetwork.end() - sizeof(uint32_t), sizeof(uint32_t),
                    reinterpret_cast<char*>(&intermediateSize));
        compiledNetwork.resize(compiledNetwork.size() - sizeof(uint32_t));
        return { compiledNetwork, intermediateSize };
    };

    void AddCompiledNetwork(std::vector<char> compiledSubgraph, uint32_t bufferSize)
    {
        std::copy_n(reinterpret_cast<const char*>(&bufferSize), sizeof(bufferSize),
                    std::back_inserter(compiledSubgraph));
        m_CompiledNetworks.push_back(compiledSubgraph);
    }
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

    void Load();

    void Save();

private:
    void LoadCachedSubgraphs();

    void SaveCachedSubgraphs();

    void Reset();

private:
    /// Number of subgraphs.
    uint32_t m_SubgraphCount;

    /// Caching options used to save or load compiled networks from all subgraphs.
    EthosNCachingOptions m_EthosNCachingOptions;

    /// Holds serialized compiled networks temporarily from all subgraphs.
    /// This is used to load or save the compiled networks.
    std::vector<std::vector<char>> m_CompiledNetworks;

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
