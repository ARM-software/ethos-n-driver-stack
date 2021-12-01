//
// Copyright © 2018-2019,2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mxml.h>

#include <memory>
#include <stdexcept>
#include <string>

const auto g_XmlRootName = "STREAM";

class Exception : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class ParseException : public Exception
{
    using Exception::Exception;
};

class IOException : public Exception
{
    using Exception::Exception;
};

struct XmlHandleDeleter
{
    void operator()(mxml_node_t* raw) const
    {
        mxmlDelete(raw);
    }
};

using XmlHandle = std::unique_ptr<mxml_node_t, XmlHandleDeleter>;
