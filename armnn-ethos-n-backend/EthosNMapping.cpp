//
// Copyright © 2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNMapping.hpp"
#include "EthosNLayerSupport.hpp"

#include <armnn/Exceptions.hpp>

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

void armnn::ProcessPattern(const std::vector<std::string>& buf,
                           std::map<std::string, SimpleInputOutput>& tensors,
                           std::vector<SimpleLayer>& layers)
{
    auto regexSettings = std::regex_constants::ECMAScript | std::regex_constants::icase;
    // Match string on either 'input' or 'output' followed by two words: input/output name and matrix size.
    // Any number of spaces, tabs or even a single comma could separate words.
    static const std::regex inOrOut(R"(\s*(input|output)(?:\s+|,\s*)(\w+)\s*,?\s*(\w+).*)", regexSettings);
    // Match string on either 'Activation', 'Convolution2d' or 'StandIn' followed by three words in brackets: input name, output name and mapping function.
    // Any number of spaces, tabs or even a single comma could separate words.
    std::string regexLayerMatch = R"(s*()";
    regexLayerMatch += "Activation";
    regexLayerMatch += "|Convolution2d";
    regexLayerMatch += "|StandIn";
    regexLayerMatch += "|Excluded)";
    regexLayerMatch += R"((?:\s+|,\s*)\((.*?)\)\s*,?\s*\((.*?)\)(?:\s*,?\s*\((.*?)\))?.*)";

    static const std::regex layerType(regexLayerMatch, regexSettings);
    std::smatch match;
    std::string errors;

    for (auto line : buf)
    {
        if (std::regex_match(line, match, inOrOut) && match.size() >= 4)
        {
            // split the shape by x and check if any are not _ and record those in the shape and ) for _
            std::vector<std::string> tokens = armnn::Split(match.str(3), 'x');
            std::vector<uint32_t> shape;

            for (auto tok : tokens)
            {
                if (tok != "_")
                {
                    shape.push_back(static_cast<uint32_t>(std::stoul(tok)));
                }
                else
                {
                    shape.push_back(0);
                }
            }

            const std::string name = match.str(2);
            tensors.emplace(name, SimpleInputOutput(name, shape));
        }
        else if (std::regex_match(line, match, layerType) && match.size() >= 4)
        {
            std::vector<SimpleInputOutput> layerInputs;
            std::vector<std::string> layerOutputs;

            std::string name = match.str(1);

            std::string matchedInputs = match.str(2);
            armnn::Prune(matchedInputs);
            for (auto matchedInput : armnn::Split(matchedInputs, ','))
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

            std::string matchedOutputs = match.str(3);
            armnn::Prune(matchedOutputs);
            for (auto matchedOutput : armnn::Split(matchedOutputs, ','))
            {
                layerOutputs.push_back(matchedOutput);
            }
            std::string extra;
            if (match.size() >= 5)
            {
                extra = match.str(4);
                armnn::Prune(extra);
            }
            std::map<std::string, std::string> extraArgs = armnn::Split(extra, ',', '=');

            layers.push_back(SimpleLayer(name, layerInputs, layerOutputs, extraArgs));
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

std::vector<armnn::Mapping> armnn::GetMappings(std::string mappingFileFromConfig)
{
    std::vector<armnn::Mapping> mappingsFromFile;
    if (mappingFileFromConfig.empty())
    {
        return mappingsFromFile;
    }

    std::ifstream mappingFile(mappingFileFromConfig, std::ios_base::binary | std::ios_base::in);
    if (!mappingFile.is_open())
    {
        std::string error = "Failed to open mapping file: " + mappingFileFromConfig + "\n";
        throw std::invalid_argument(error);
    }

    std::string line;

    State state = State::Comments;

    std::vector<std::string> buf;
    std::map<std::string, SimpleInputOutput> tensors;
    std::vector<SimpleLayer> patternLayers;
    std::vector<SimpleLayer> replacementLayers;

    while (getline(mappingFile, line))
    {
        line = Trim(line);
        if (line.size() == 0)
        {
            continue;
        }
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

    // Process the last line since there is no more "pattern:" coming up next to trigger
    //      a ProcessPattern() call (it's the end of the file)
    armnn::ProcessPattern(buf, tensors, replacementLayers);
    mappingsFromFile.emplace_back(tensors, patternLayers, replacementLayers);

    return mappingsFromFile;
}
