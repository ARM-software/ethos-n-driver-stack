//
// Copyright © 2020-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNLayerSupport.hpp"
#include "EthosNMapping.hpp"

#include <armnn/Exceptions.hpp>
#include <armnn/Logging.hpp>

#include <fstream>
#include <regex>

namespace
{

enum class State
{
    Comments,
    Pattern,
    GraphReplacement,
};
}

std::vector<std::string> armnn::Split(std::string s, char delim)
{
    std::stringstream ss(s);
    std::string token;
    std::vector<std::string> results;
    while (std::getline(ss, token, delim))
    {
        results.push_back(token);
    }
    return results;
}

std::map<std::string, std::string> armnn::Split(std::string s, char delim, char secondDelim)
{
    std::stringstream ss(s);
    std::string token;
    std::map<std::string, std::string> strStrMap;
    while (std::getline(ss, token, delim))
    {
        size_t eqIdx       = token.find(secondDelim);
        std::string first  = token.substr(0, eqIdx);
        std::string second = token.substr(eqIdx + 1, token.size());
        strStrMap.insert(std::pair<std::string, std::string>(first, second));
    }

    return strStrMap;
}

std::string armnn::Trim(const std::string& s)
{
    //
    // Remove blank characters from the front and back of the string.
    //

    std::string::size_type start = s.find_first_not_of(" \n\r\t");    // could be an empty line
    std::string::size_type end   = s.find_last_not_of(" \n\r\t");
    return (start == std::string::npos ? std::string() : s.substr(start, end + 1 - start));
}

void armnn::Prune(std::string& s)
{
    //
    // Remove all blank characters from the string.
    //

    s.erase(std::remove_if(s.begin(), s.end(),
                           [](std::string::const_reference c) -> bool {
                               return (c == ' ' || c == '\t' || c == '\n' || c == '\r');
                           }),
            s.end());
}

// Parsing numbers from "1x_x_x_" into {1, 0, 0, 0}
std::vector<uint32_t> armnn::ParseNumbers(std::string& buf)
{
    std::vector<uint32_t> numbers;

    std::vector<std::string> tokens = armnn::Split(buf, 'x');

    for (auto tok : tokens)
    {
        if (tok != "_")
        {
            numbers.push_back(static_cast<uint32_t>(std::stoul(tok)));
        }
        else
        {
            numbers.push_back(0);
        }
    }

    ARMNN_LOG(trace) << "The numbers are { ";

    for (uint32_t i = 0; i < numbers.size(); ++i)
    {
        ARMNN_LOG(trace) << numbers.at(i) << " ";
    }

    ARMNN_LOG(trace) << " } \n";
    return numbers;
}

std::vector<uint32_t> armnn::GetLayerParameterValue(std::map<std::string, std::string> paramList, std::string param)
{
    bool useDefault = false;
    std::string errors;
    std::vector<uint32_t> value;

    if ((paramList.empty()) || (paramList.find(param) == paramList.end()))
    {
        useDefault = true;
    }

    if (!((param.compare("stride")) && (param.compare("kernel")) && (param.compare("dilation"))))
    {
        if (useDefault)
        {
            value = std::vector<uint32_t>{ 1, 1 };
        }
        else
        {
            value = armnn::ParseNumbers(paramList.find(param)->second);
            if (value.size() != 2)
            {
                errors += "Invalid Value: The expected format is ((";
                errors += param;
                errors += "=_x_))";
            }
        }
    }
    else if (!(param.compare("padding")))
    {
        if (useDefault)
        {
            value = std::vector<uint32_t>{ 1, 1, 1, 1 };
        }
        else
        {
            value = armnn::ParseNumbers(paramList.find(param)->second);
            if (value.size() != 4)
            {
                errors += "Invalid Value: The expected format is ((";
                errors += param;
                errors += "=_x_x_x_))";
            }
        }
    }

    if (!errors.empty())
    {
        throw armnn::InvalidArgumentException(errors);
    }
    else
    {
        return value;
    }
}

// The buf is like "(arg1=value1),(arg2=value2),(arg3=value3)"
std::map<std::string, std::string> armnn::ParseAdditionalParameters(std::string& buf, std::string& errors)
{
    std::map<std::string, std::string> paramsList;

    auto regexSettings          = std::regex_constants::ECMAScript | std::regex_constants::icase;
    std::string regexParamMatch = R"(\s*\((.*?)\)\s*)";
    static const std::regex regexParamPair(regexParamMatch, regexSettings);
    constexpr unsigned int cntAdditionalParamsSubGroups = 2;
    constexpr unsigned int cntKeyValuePair              = 2;
    constexpr unsigned int keyValuePairIndex            = 1;

    // Extracting "(arg1=value1)", "(arg2=value2)", ... pairs
    armnn::Prune(buf);
    auto args = armnn::Split(buf, ',');

    // Iterating through each "(arg1=value1)", "(arg2=value2)", ... pair
    for (auto arg : args)
    {
        std::smatch match;

        // Extracting "arg1=value1"
        if (std::regex_match(arg, match, regexParamPair) && (match.size() == cntAdditionalParamsSubGroups))
        {
            // Extracting "arg1", "value1"
            std::string parameter = match.str(keyValuePairIndex);

            auto paramNameValue = armnn::Split(parameter, '=');

            if (paramNameValue.size() != cntKeyValuePair)
            {
                errors += "Syntax error: Additional parameters should be in (name1=value1),(name2=value2) format\n";
                errors += buf;
                errors += "\n";
            }
            else
            {
                paramsList.insert(std::make_pair(paramNameValue[0], paramNameValue[1]));
            }
        }
        else
        {
            errors += "Syntax error: Additional parameters should be specified as (name1=value1) (name2=value2)";
            errors += buf;
            errors += "\n";
        }
    }

    return paramsList;
}

std::pair<std::string, armnn::SimpleInputOutput> armnn::GetInputOutput(std::smatch match)
{
    std::vector<uint32_t> shape;
    std::map<std::string, SimpleInputOutput> inputOutput;

    // Get name of the input/output tensor
    const std::string name = match.str(2);

    // Get the dimensions ie for eg "1x_x_x_"
    std::string buffer = match.str(3);

    shape = ParseNumbers(buffer);

    return std::pair<std::string, SimpleInputOutput>(name, SimpleInputOutput(name, shape));
}

std::vector<armnn::SimpleInputOutput> armnn::GetLayerInputs(std::map<std::string, armnn::SimpleInputOutput>& tensors,
                                                            std::string buf,
                                                            std::string& errors)
{
    std::vector<armnn::SimpleInputOutput> layerInputs;
    armnn::Prune(buf);

    for (auto matchedInput : armnn::Split(buf, ','))
    {
        try
        {
            layerInputs.push_back(tensors.at(matchedInput));
        }
        catch (const std::out_of_range&)
        {
            errors += "Undefined input: '";
            errors += matchedInput;
            errors += "'\n";
        }
    }

    return layerInputs;
}

std::vector<std::string> armnn::GetLayerOutputs(std::string buf, std::string& errors)
{
    armnn::Prune(buf);
    std::vector<std::string> layerOutputs;

    for (auto matchedOutput : armnn::Split(buf, ','))
    {
        layerOutputs.push_back(matchedOutput);
    }

    if (layerOutputs.size() == 0)
    {
        errors += "No outputs specified for the layer\n";
    }
    return layerOutputs;
}

void armnn::ProcessPattern(const std::vector<std::string>& buf,
                           std::map<std::string, SimpleInputOutput>& tensors,
                           std::vector<SimpleLayer>& layers)
{
    auto regexSettings = std::regex_constants::ECMAScript | std::regex_constants::icase;
    // Match string on either 'input' or 'output' followed by two words: input/output name and matrix size.
    // Any number of spaces, tabs or even a single comma could separate words.
    static const std::regex inOrOut(R"(\s*(input|output)(?:\s+|,\s*)(\w+)\s*,?\s*(\w+).*)", regexSettings);

    // Match string on any layer type string followed by three words in brackets: input name, output name and mapping function.
    std::string regexLayerMatch = R"(s*()";
#define X(name) regexLayerMatch += #name "|";
    LIST_OF_LAYER_TYPE
#undef X
    // 'Excluded' means that the layer is not considered for estimation.
    // Excluded is a word defined by us, it is not a standard layer type.
    regexLayerMatch += "Excluded)";
    regexLayerMatch += R"((?:\s+|,\s*)\((.*?)\)\s*,?\s*\((.*?)\)(?:\s*,?\s*\({2}(.*?)\){2})?(.*?))";

    static const std::regex layerType(regexLayerMatch, regexSettings);
    std::smatch match;
    std::string errors;
    constexpr unsigned int minSubGroups                     = 4;
    constexpr unsigned int minSubGroupsWithAdditionalParams = 5;
    constexpr unsigned int minSubGroupsWithExtraneousParams = 6;
    constexpr unsigned int layerTypeNameIndex               = 1;
    constexpr unsigned int inputsIndex                      = 2;
    constexpr unsigned int outputsIndex                     = 3;
    constexpr unsigned int additionalParamsIndex            = 4;
    constexpr unsigned int extraneousParamsIndex            = 5;

    for (auto line : buf)
    {
        if (std::regex_match(line, match, inOrOut) && match.size() >= minSubGroups)
        {
            tensors.emplace(GetInputOutput(match));
        }
        else if (std::regex_match(line, match, layerType) && match.size() >= minSubGroups)
        {
            armnn::AdditionalLayerParams layerParams;
            std::map<std::string, std::string> funcName, args;

            // Finding the name of LayerType
            std::string typeName = match.str(layerTypeNameIndex);
            armnn::Prune(typeName);

            // Finding the inputs
            auto layerInputs = GetLayerInputs(tensors, match.str(inputsIndex), errors);

            // Finding the outputs
            auto layerOutputs = GetLayerOutputs(match.str(outputsIndex), errors);

            uint32_t i = additionalParamsIndex;
            // Let's figure out the additional parameters
            if ((match.size() >= minSubGroupsWithAdditionalParams) && (!match.str(i).empty()))
            {
                std::string additionalParam = match.str(i);
                armnn::Prune(additionalParam);

                // The braces get consumed during the parsing of the
                // Layer. We need to put the braces back for additional
                // parameters to be parsed correctly.
                additionalParam = "(" + additionalParam + ")";

                layerParams = ParseAdditionalParameters(additionalParam, errors);
            }

            // Check if the user has provided any extraneous parameters
            if (match.size() >= minSubGroupsWithExtraneousParams)
            {
                std::string extraneousParams = match.str(extraneousParamsIndex);
                armnn::Prune(extraneousParams);

                if (!extraneousParams.empty())
                {
                    // We assume that the user wanted to specify the additional parameters
                    // enclosed within ((...)).
                    if (match.str(additionalParamsIndex).empty())
                    {
                        errors += "Syntax error:\n";
                        errors += extraneousParams;
                        errors += "\n Additional parameters are to be enclosed in (( ))\n";
                    }
                    // The user has specified too many parameters
                    else
                    {
                        errors += "Syntax error: Too many parameters specified\n";
                    }
                }
            }
            layers.push_back(SimpleLayer(typeName, layerInputs, layerOutputs, layerParams));
        }
        else
        {
            //
            // Line not processed.
            // If blank, ignore. Else, signal error on exit.
            //
            if (!line.empty())
            {
                errors += "Syntax error:\n";
                errors += line;
                errors += "\n";
            }
        }
    }

    if (!errors.empty())
    {
        throw armnn::ParseException(errors);
    }
}

armnn::EthosNMappings armnn::ParseMappings(const char* mappingContents)
{
    std::stringstream ss(mappingContents);
    return ParseMappings(ss);
}

std::vector<armnn::Mapping> armnn::ReadMappingsFromFile(const char* mappingFilename)
{
    if (mappingFilename == nullptr || strlen(mappingFilename) == 0)
    {
        return {};
    }

    std::ifstream mappingFile(mappingFilename, std::ios_base::binary | std::ios_base::in);
    if (!mappingFile.is_open())
    {
        std::string error = "Failed to open mapping file: " + std::string(mappingFilename) + "\n";
        throw std::invalid_argument(error);
    }

    return ParseMappings(mappingFile);
}

armnn::EthosNMappings armnn::ParseMappings(std::istream& stream)
{
    std::vector<armnn::Mapping> mappingsFromFile;
    bool isEmpty       = true;
    bool isCommentOnly = true;
    std::string line;

    State state = State::Comments;

    std::vector<std::string> buf;
    std::map<std::string, SimpleInputOutput> tensors;
    std::vector<SimpleLayer> patternLayers;
    std::vector<SimpleLayer> replacementLayers;

    while (getline(stream, line))
    {
        line = Trim(line);
        if (line.size() == 0)
        {
            continue;
        }

        isEmpty = false;

        if (line[0] == '#')
        {
            continue;
        }

        isCommentOnly = false;

        switch (state)
        {
            case State::Comments:
                if (line == "pattern:")
                {
                    state = State::Pattern;
                }
                break;
            case State::Pattern:
                if (line == "graph-replacement:")
                {
                    ProcessPattern(buf, tensors, patternLayers);
                    buf.clear();
                    state = State::GraphReplacement;
                }
                else
                {
                    buf.push_back(line);
                }
                break;
            case State::GraphReplacement:
                // end of a complete Mapping structure
                if (line == "pattern:")
                {
                    ProcessPattern(buf, tensors, replacementLayers);
                    buf.clear();
                    state = State::Pattern;
                    mappingsFromFile.emplace_back(tensors, patternLayers, replacementLayers);
                    patternLayers.clear();
                    replacementLayers.clear();
                }
                else
                {
                    buf.push_back(line);
                }
                break;
        }
    }

    if (!(buf.empty()) && (state == State::GraphReplacement))
    {
        // Process the last line since there is no more "pattern:" coming up next to trigger
        //      a ProcessPattern() call (it's the end of the file)
        ProcessPattern(buf, tensors, replacementLayers);
        mappingsFromFile.emplace_back(tensors, patternLayers, replacementLayers);
    }
    else if (isEmpty)
    {
        ARMNN_LOG(warning) << "WARNING: Empty mapping file provided";
    }
    else if (isCommentOnly)
    {
        ARMNN_LOG(warning) << "WARNING: Mapping file contains only comments";
    }
    else
    {
        throw armnn::ParseException("Syntax error in mapping file");
    }

    return mappingsFromFile;
}
