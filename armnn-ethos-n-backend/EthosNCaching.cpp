//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNBackend.hpp"
#include "EthosNCaching.hpp"

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
                    if (option.GetValue().IsString())
                    {
                        std::string filePath = option.GetValue().AsString();
                        if (filePath != "" && fs::exists(filePath) && fs::is_regular_file(filePath))
                        {
                            result.m_CachedNetworkFilePath = filePath;
                        }
                        else
                        {
                            throw armnn::InvalidArgumentException(
                                "The file used to write cached networks to is invalid or doesn't exist.");
                        }
                    }
                    else
                    {
                        throw armnn::InvalidArgumentException(
                            "Invalid option type for CachedNetworkFilePath - must be string.");
                    }
                }
            }
        }
    }

    return result;
}

EthosNCaching::EthosNCaching()
    : m_EthosNCachingOptions({ false, "" })
    , m_CompiledNetworks()
    , m_IsLoaded(false)
{}

void EthosNCaching::SetEthosNCachingOptions(const armnn::ModelOptions& modelOptions)
{
    m_EthosNCachingOptions = GetEthosNCachingOptionsFromModelOptions(modelOptions);
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

    // Save map to a filepath provided in ModelOptions. This will be validated by now.
    std::string filePath = m_EthosNCachingOptions.m_CachedNetworkFilePath;
    ARMNN_LOG(info) << "Saving cached network " << filePath;
    std::ofstream out(filePath, std::ios::binary);

    // Write the number of subgraphs, used for the loop limit.
    uint32_t numOfSubgraphs = static_cast<uint32_t>(m_CompiledNetworks.size());
    out.write(reinterpret_cast<const char*>(&numOfSubgraphs), sizeof(numOfSubgraphs));

    // Write the sizes of each of the compiled networks in order, this is used when reading in.
    for (auto&& it : m_CompiledNetworks)
    {
        size_t compiledNetworkSize = it.second.size();
        out.write(reinterpret_cast<const char*>(&compiledNetworkSize), sizeof(compiledNetworkSize));
    }

    // Write the subgraph index associated with each compiled network
    for (auto&& it : m_CompiledNetworks)
    {
        uint32_t subgraphIdx = it.first;
        out.write(reinterpret_cast<const char*>(&subgraphIdx), sizeof(subgraphIdx));
    }

    // Write the compiled networks binary, this is used when reading in using the sizes.
    for (auto&& it : m_CompiledNetworks)
    {
        auto bytesToWrite = static_cast<std::streamsize>(sizeof(char) * it.second.size());
        out.write(reinterpret_cast<const char*>(&it.second[0]), bytesToWrite);
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

    // Read in the number of subgraphs, used for the loop limit.
    uint32_t numOfNetworks;
    in.read(reinterpret_cast<char*>(&numOfNetworks), sizeof(numOfNetworks));
    if (!in)
    {
        ARMNN_LOG(error) << "Error trying to read the number of cached subgraphs";
        return false;
    }

    // Read in sizes of each of the compiled networks
    std::vector<size_t> compiledNetworkSizes;
    for (uint32_t i = 0; i < numOfNetworks; ++i)
    {
        size_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (!in)
        {
            ARMNN_LOG(error) << "Error trying to read the size of subgraph " << i;
            return false;
        }

        compiledNetworkSizes.emplace_back(size);
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
    for (uint32_t i = 0; i < compiledNetworkSizes.size(); ++i)
    {
        size_t size        = compiledNetworkSizes[i];
        size_t subgraphIdx = compiledNetworkSubgraphIdxs[i];
        std::vector<char> binaryContent(size);

        auto bytesToRead = static_cast<std::streamsize>(sizeof(char) * size);
        in.read(reinterpret_cast<char*>(&binaryContent[0]), bytesToRead);
        if (!in)
        {
            ARMNN_LOG(error) << "Error trying to read subgraph " << i;
            return false;
        }

        m_CompiledNetworks.emplace(subgraphIdx, binaryContent);
    }

    return true;
}

void EthosNCaching::Reset()
{
    if (IsSaving() || IsLoading())
    {
        m_EthosNCachingOptions = { false, "" };
        m_CompiledNetworks.clear();
        m_IsLoaded = false;
    }
}

EthosNCachingService& EthosNCachingService::GetInstance()
{
    static EthosNCachingService cachingInstance;
    return cachingInstance;
}

}    // namespace armnn
