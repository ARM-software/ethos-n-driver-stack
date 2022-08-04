//
// Copyright © 2018-2019,2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "Common.hpp"

#include <cstdio>
#include <sstream>
#include <vector>

class BinaryParser
{
public:
    BinaryParser(std::istream& input);
    BinaryParser(const std::vector<uint32_t>& data);

    void WriteXml(std::ostream& output);

private:
    std::stringstream out;
};
