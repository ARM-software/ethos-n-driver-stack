//
// Copyright © 2018-2019,2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "Common.hpp"

#include <cstdio>

class BinaryParser
{
public:
    BinaryParser(std::istream& input);

    void WriteXml(std::ostream& output, int wrapMargin = 120);

private:
    XmlHandle m_XmlDoc;
};
