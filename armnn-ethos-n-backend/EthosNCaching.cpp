//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNCaching.hpp"

#include "EthosNBackend.hpp"

#include <armnn/Exceptions.hpp>
#include <armnnUtils/Filesystem.hpp>

#include <fstream>

namespace armnn
{

// Gets the caching options to use based on the given ModelOptions.
EthosNCachingOptions GetEthosNCachingOptionsFromModelOptions(const armnn::ModelOptions& modelOptions)
{
    EthosNCachingOptions result;

    for (const auto& optionsGroup : modelOptions)
    {
        if (optionsGroup.GetBackendId() == EthosNBackend::GetIdStatic())
        {
            for (size_t i = 0; i < optionsGroup.GetOptionCount(); i++)
            {
                const BackendOptions::BackendOption& option = optionsGroup.GetOption(i);

                if (option.GetName() == "SaveCachedNetwork")
                {
                    if (option.GetValue().IsBool())
                    {
                        result.m_SaveCachedNetwork = option.GetValue().AsBool();
                    }
                    else
                    {
                        throw armnn::InvalidArgumentException(
                            "Invalid option type for SaveCachedNetwork - must be bool.");
                    }
                }
                else if (option.GetName() == "CachedNetworkFilePath")
                {
                    if (option.GetValue().IsString() && option.GetValue().AsString() != "")
                    {
                        result.m_CachedNetworkFilePath = option.GetValue().AsString();
                    }
                    else
                    {
                        throw armnn::InvalidArgumentException(
                            "Invalid option type for CachedNetworkFilePath - must be a non-empty string.");
                    }
                }
            }
        }
    }

    return result;
}

EthosNCaching::EthosNCaching()
    : m_EthosNCachingOptions({ false, "" })
    , m_IsLoaded(false)
{}

void EthosNCaching::SetEthosNCachingOptions(const armnn::ModelOptions& modelOptions)
{
    m_EthosNCachingOptions = GetEthosNCachingOptionsFromModelOptions(modelOptions);
}

armnn::Optional<const EthosNCaching::CachedNetwork&> EthosNCaching::GetCachedNetwork(uint32_t subgraphIdx) const
{
    auto foundNetwork = m_CachedNetworks.find(subgraphIdx);
    if (foundNetwork == m_CachedNetworks.end())
    {
        return {};
    }
    return foundNetwork->second;
}

void EthosNCaching::AddCachedNetwork(uint32_t subgraphIdx, CachedNetwork cachedNetwork)
{
    m_CachedNetworks[subgraphIdx] = std::move(cachedNetwork);
}

bool EthosNCaching::IsSaving()
{
    bool isSaving = false;
    if (m_EthosNCachingOptions.m_SaveCachedNetwork == true && m_EthosNCachingOptions.m_CachedNetworkFilePath != "")
    {
        isSaving = true;
    }

    return isSaving;
}

bool EthosNCaching::IsLoading()
{
    bool isLoading = false;
    if (m_EthosNCachingOptions.m_SaveCachedNetwork == false && m_EthosNCachingOptions.m_CachedNetworkFilePath != "")
    {
        isLoading = true;
    }
    return isLoading;
}

bool EthosNCaching::Load()
{
    bool loaded = LoadCachedSubgraphs();
    if (loaded)
    {
        SetIsLoaded(true);
    }
    return loaded;
}

bool EthosNCaching::Save()
{
    bool saved = SaveCachedSubgraphs();
    Reset();
    return saved;
}

// Private function to save compiled subgraphs to file.
// The file is laid out as follows.
// <number of subgraphs><compiled network sizes><compiled networks binary>
bool EthosNCaching::SaveCachedSubgraphs()
{
    // If user options aren't set then do nothing.
    if (!IsSaving())
    {
        return true;
    }

    // Save map to a filepath provided in ModelOptions. This will have been validated by now.
    std::string filePath = m_EthosNCachingOptions.m_CachedNetworkFilePath;
    ARMNN_LOG(info) << "Saving cached network " << filePath;
    std::ofstream out(filePath, std::ios::binary);

    // Write the number of subgraphs, used for the loop limit.
    uint32_t numOfSubgraphs = static_cast<uint32_t>(m_CachedNetworks.size());
    out.write(reinterpret_cast<const char*>(&numOfSubgraphs), sizeof(numOfSubgraphs));

    // Write the sizes of each of the cached networks in order, this is used when reading in.
    for (auto&& it : m_CachedNetworks)
    {
        size_t compiledNetworkSize = it.second.m_CompiledNetwork.size() + sizeof(it.second.m_IntermediateDataSize);
        out.write(reinterpret_cast<const char*>(&compiledNetworkSize), sizeof(compiledNetworkSize));
    }

    // Write the subgraph index associated with each compiled network
    for (auto&& it : m_CachedNetworks)
    {
        uint32_t subgraphIdx = it.first;
        out.write(reinterpret_cast<const char*>(&subgraphIdx), sizeof(subgraphIdx));
    }

    // Write the compiled network's binary and intermediate data size
    for (auto&& it : m_CachedNetworks)
    {
        auto bytesToWrite = static_cast<std::streamsize>(sizeof(char) * it.second.m_CompiledNetwork.size());
        out.write(reinterpret_cast<const char*>(&it.second.m_CompiledNetwork[0]), bytesToWrite);

        size_t intermediateDataSize = it.second.m_IntermediateDataSize;
        out.write(reinterpret_cast<const char*>(&intermediateDataSize), sizeof(intermediateDataSize));
    }

    if (!out)
    {
        ARMNN_LOG(error) << "Error trying to write to " << filePath << " cached network cannot be saved";
        return false;
    }
    return true;
}

// Private function to load compiled subgraphs from a file.
// The file is laid out as follows.
// <number of subgraphs><compiled network sizes><compiled networks binary>
bool EthosNCaching::LoadCachedSubgraphs()
{
    // If user options aren't set then do nothing.
    if (!IsLoading())
    {
        return true;
    }

    // Save map to a filepath provided in ModelOptions. This will be validated by now.
    std::string filePath = m_EthosNCachingOptions.m_CachedNetworkFilePath;
    ARMNN_LOG(info) << "Loading cached network " << filePath;
    std::ifstream in(filePath, std::ios::binary);
    if (!in)
    {
        ARMNN_LOG(error) << "Error reading cached network file";
        return false;
    }

    // Read in the number of subgraphs, used for the loop limit.
    uint32_t numOfNetworks;
    in.read(reinterpret_cast<char*>(&numOfNetworks), sizeof(numOfNetworks));
    if (!in)
    {
        ARMNN_LOG(error) << "Error trying to read the number of cached subgraphs";
        return false;
    }

    // Read in sizes of each of the cached networks
    std::vector<size_t> cachedNetworkSizes;
    for (uint32_t i = 0; i < numOfNetworks; ++i)
    {
        size_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (!in)
        {
            ARMNN_LOG(error) << "Error trying to read the size of subgraph " << i;
            return false;
        }

        cachedNetworkSizes.emplace_back(size);
    }

    // Read the subgraph index for each compiled network
    std::vector<uint32_t> compiledNetworkSubgraphIdxs;
    for (uint32_t i = 0; i < numOfNetworks; ++i)
    {
        uint32_t subgraphIdx;
        in.read(reinterpret_cast<char*>(&subgraphIdx), sizeof(subgraphIdx));
        if (!in)
        {
            ARMNN_LOG(error) << "Error trying to read the subgraph index " << i;
            return false;
        }
        compiledNetworkSubgraphIdxs.emplace_back(subgraphIdx);
    }

    // Read in the compiled networks binary using the sizes.
    for (uint32_t i = 0; i < numOfNetworks; ++i)
    {
        size_t compiledNetworkSize = cachedNetworkSizes[i] - sizeof(CachedNetwork::m_IntermediateDataSize);
        CachedNetwork cachedNetwork;
        cachedNetwork.m_CompiledNetwork.resize(compiledNetworkSize);

        auto bytesToRead = static_cast<std::streamsize>(sizeof(char) * compiledNetworkSize);
        in.read(reinterpret_cast<char*>(&cachedNetwork.m_CompiledNetwork[0]), bytesToRead);
        if (!in)
        {
            ARMNN_LOG(error) << "Error trying to read subgraph " << i;
            return false;
        }

        size_t intermediateDataSize;
        in.read(reinterpret_cast<char*>(&intermediateDataSize), sizeof(intermediateDataSize));

        if (!in)
        {
            ARMNN_LOG(error) << "Error trying to read intermediate data size " << i;
            return false;
        }

        cachedNetwork.m_IntermediateDataSize = static_cast<uint32_t>(intermediateDataSize);

        m_CachedNetworks.emplace(compiledNetworkSubgraphIdxs[i], std::move(cachedNetwork));
    }

    if (in.peek() != EOF)
    {
        ARMNN_LOG(error) << "Leftover data in cached network file";
        return false;
    }

    return true;
}

void EthosNCaching::Reset()
{
    if (IsSaving() || IsLoading())
    {
        m_EthosNCachingOptions = { false, "" };
        m_CachedNetworks.clear();
        m_IsLoaded = false;
    }
}

EthosNCachingService& EthosNCachingService::GetInstance()
{
    static EthosNCachingService cachingInstance;
    return cachingInstance;
}

}    // namespace armnn
