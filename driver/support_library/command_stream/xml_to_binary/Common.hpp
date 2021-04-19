//
// Copyright © 2018-2019,2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mxml.h>

#include <exception>
#include <memory>
#include <string>

const auto g_XmlRootName = "STREAM";

class Exception : public std::exception
{
public:
    explicit Exception(const std::string msg)
        : m_Msg(msg)
    {}
    const char* what() const noexcept override
    {
        return m_Msg.c_str();
    }

private:
    std::string m_Msg;
};

class ParseException : public Exception
{
public:
    using Exception::Exception;
};

class IOException : public Exception
{
public:
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
