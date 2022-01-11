//
// Copyright © 2022 Arm Limited.
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
{
    m_SubgraphCount        = 0;
    m_EthosNCachingOptions = { false, "" };
    m_CompiledNetworks     = {};
    m_IsLoaded             = false;
}

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

void EthosNCaching::Load()
{
    LoadCachedSubgraphs();
    SetIsLoaded(true);
}

void EthosNCaching::Save()
{
    SaveCachedSubgraphs();
    Reset();
}

// Private function to save compiled subgraphs to file.
// The file is laid out as follows.
// <number of subgraphs><compiled network sizes><compiled networks binary>
void EthosNCaching::SaveCachedSubgraphs()
{
    // If user options aren't set then do nothing.
    if (!IsSaving())
    {
        return;
    }

    // Save map to a filepath provided in ModelOptions. This will be validated by now.
    std::string filePath = m_EthosNCachingOptions.m_CachedNetworkFilePath;
    std::ofstream out(filePath, std::ios::binary);

    // Write the number of subgraphs, used for the loop limit.
    uint32_t numOfSubgraphs = m_SubgraphCount;
    out.write(reinterpret_cast<const char*>(&numOfSubgraphs), sizeof(numOfSubgraphs));

    // Write the sizes of each of the compiled networks in order, this is used when reading in.
    for (uint32_t i = 0; i < numOfSubgraphs; ++i)
    {
        size_t compiledNetworkSize = m_CompiledNetworks[i].size();
        out.write(reinterpret_cast<const char*>(&compiledNetworkSize), sizeof(compiledNetworkSize));
    }

    // Write the compiled networks binary, this is used when reading in using the sizes.
    for (uint32_t i = 0; i < numOfSubgraphs; ++i)
    {
        auto bytesToWrite = static_cast<std::streamsize>(sizeof(char) * m_CompiledNetworks[i].size());
        out.write(reinterpret_cast<const char*>(&m_CompiledNetworks[i][0]), bytesToWrite);
    }

    out.close();
}

// Private function to load compiled subgraphs from a file.
// The file is laid out as follows.
// <number of subgraphs><compiled network sizes><compiled networks binary>
void EthosNCaching::LoadCachedSubgraphs()
{
    // If user options aren't set then do nothing.
    if (!IsLoading())
    {
        return;
    }

    // Save map to a filepath provided in ModelOptions. This will be validated by now.
    std::string filePath = m_EthosNCachingOptions.m_CachedNetworkFilePath;
    std::ifstream in(filePath, std::ios::binary);

    // Read in the number of subgraphs, used for the loop limit.
    uint32_t numOfSubgraphs;
    in.read(reinterpret_cast<char*>(&numOfSubgraphs), sizeof(numOfSubgraphs));

    // Read in sizes of each of the compiled networks in order.
    std::vector<size_t> compiledNetworkSizes;
    for (uint32_t i = 0; i < numOfSubgraphs; ++i)
    {
        size_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));

        compiledNetworkSizes.emplace_back(size);
    }

    // Read in the compiled networks binary using the sizes.
    for (uint32_t i = 0; i < compiledNetworkSizes.size(); ++i)
    {
        size_t size = compiledNetworkSizes[i];
        std::vector<char> binaryContent(size);

        auto bytesToRead = static_cast<std::streamsize>(sizeof(char) * size);
        in.read(reinterpret_cast<char*>(&binaryContent[0]), bytesToRead);

        m_CompiledNetworks.push_back(binaryContent);
    }

    in.close();
}

void EthosNCaching::Reset()
{
    if (IsSaving() || IsLoading())
    {
        m_SubgraphCount        = 0;
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
