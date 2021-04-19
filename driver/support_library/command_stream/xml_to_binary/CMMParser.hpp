//
// Copyright © 2019-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Common.hpp"

#include <cstdio>
#include <vector>

class CMMParser
{
public:
    CMMParser(std::istream& input);

    void ExtractCSFromCMM(std::ostream& output, bool doXmlToBinary);

    void ExtractBTFromCMM(std::ostream& output);

private:
    std::istream& m_Input;

    uint32_t GetInferenceAddress(std::istream& input);
};
