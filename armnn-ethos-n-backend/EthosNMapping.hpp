//
// Copyright © 2019-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <map>
#include <regex>
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

typedef std::map<std::string, std::string> AdditionalLayerParams;

struct SimpleLayer
{
    SimpleLayer(std::string typeName, std::vector<SimpleInputOutput> inputs, std::vector<std::string> outputs)
        : m_LayerTypeName(typeName)
        , m_Inputs(inputs)
        , m_Outputs(outputs){};

    SimpleLayer(std::string typeName,
                std::vector<SimpleInputOutput> inputs,
                std::vector<std::string> outputs,
                AdditionalLayerParams layerParams)
        : m_LayerTypeName(typeName)
        , m_Inputs(inputs)
        , m_Outputs(outputs)
        , m_LayerParams(layerParams)
    {}

    bool operator==(const SimpleLayer& rhs) const
    {
        return (m_LayerTypeName == rhs.m_LayerTypeName && m_Inputs == rhs.m_Inputs && m_Outputs == rhs.m_Outputs &&
                m_LayerParams == rhs.m_LayerParams);
    }

    // m_LayerTypeName should be mapped to a valid LayerType by calling GetMapStringToLayerType()
    std::string m_LayerTypeName;
    std::vector<SimpleInputOutput> m_Inputs;
    std::vector<std::string> m_Outputs;
    AdditionalLayerParams m_LayerParams;
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

    bool operator==(const Mapping& other) const
    {
        bool inOutEqual       = this->m_InputsOutputs == other.m_InputsOutputs;
        bool patternEqual     = this->m_PatternLayers == other.m_PatternLayers;
        bool replacementEqual = this->m_ReplacementLayers == other.m_ReplacementLayers;

        return inOutEqual && patternEqual && replacementEqual;
    }
};

using EthosNMappings = std::vector<Mapping>;

void ProcessPattern(const std::vector<std::string>& buf,
                    std::map<std::string, SimpleInputOutput>& tensors,
                    std::vector<SimpleLayer>& layers);

EthosNMappings GetMappings(std::string mappingFileFromConfig);

std::vector<uint32_t> ParseNumbers(std::string& buf);

std::map<std::string, std::string> ParseAdditionalParameters(std::string& buf, std::string& errors);

std::vector<uint32_t> GetLayerParameterValue(std::map<std::string, std::string> paramList, std::string param);

std::pair<std::string, SimpleInputOutput> GetInputOutput(std::smatch match);

std::string GetLayerName(std::string buf, std::string& errors);

std::vector<armnn::SimpleInputOutput>
    GetLayerInputs(std::map<std::string, armnn::SimpleInputOutput>& tensors, std::string buf, std::string& errors);

std::vector<std::string> GetLayerOutputs(std::string buf, std::string& errors);

}    // namespace armnn
