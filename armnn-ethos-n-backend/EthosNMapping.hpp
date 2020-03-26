//
// Copyright © 2019-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <map>
#include <string>
#include <vector>

namespace armnn
{

struct SimpleInputOutput
{
    SimpleInputOutput(std::string name, std::vector<uint32_t> shape)
        : m_Name(name)
        , m_Shape(shape)
    {}

    bool operator==(const SimpleInputOutput& other) const
    {
        return m_Name == other.m_Name && m_Shape == other.m_Shape;
    }

    std::string m_Name;
    std::vector<uint32_t> m_Shape;
};

struct SimpleLayer
{
    SimpleLayer(std::string name,
                std::vector<SimpleInputOutput> inputs,
                std::vector<std::string> outputs,
                std::map<std::string, std::string> extraArgs)
        : m_Name(name)
        , m_Inputs(inputs)
        , m_Outputs(outputs)
        , m_ExtraArgs(extraArgs)
    {}

    bool operator==(const SimpleLayer& rhs) const
    {
        return (m_Name == rhs.m_Name && m_Inputs == rhs.m_Inputs && m_Outputs == rhs.m_Outputs &&
                m_ExtraArgs == rhs.m_ExtraArgs);
    }

    std::string m_Name;
    std::vector<SimpleInputOutput> m_Inputs;
    std::vector<std::string> m_Outputs;
    std::map<std::string, std::string> m_ExtraArgs;
};

std::vector<std::string> Split(std::string s, char delim);
std::map<std::string, std::string> Split(std::string s, char delim, char secondDelim);
std::string Trim(const std::string& s);
void Prune(std::string& s);

// single mapping inside the mapping file
struct Mapping
{
    Mapping(std::map<std::string, SimpleInputOutput>& inOut,
            std::vector<SimpleLayer>& pattern,
            std::vector<SimpleLayer>& replacement)
        : m_InputsOutputs(inOut)
        , m_PatternLayers(pattern)
        , m_ReplacementLayers(replacement)
    {}

    std::map<std::string, SimpleInputOutput> m_InputsOutputs;
    std::vector<SimpleLayer> m_PatternLayers;
    std::vector<SimpleLayer> m_ReplacementLayers;
};

using EthosNMappings = std::vector<Mapping>;

void ProcessPattern(const std::vector<std::string>& buf,
                    std::map<std::string, SimpleInputOutput>& tensors,
                    std::vector<SimpleLayer>& layers);
EthosNMappings GetMappings(std::string mappingFileFromConfig);

}    // namespace armnn
