//
// Copyright © 2020-2021 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <armnn/INetwork.hpp>

#include <memory>
#include <string>
#include <vector>

class IReplacementTestGraphFactory
{
public:
    virtual const std::string& GetName() const                                   = 0;
    virtual std::unique_ptr<armnn::NetworkImpl> GetInitialGraph() const          = 0;
    virtual std::string GetMappingFileName() const                               = 0;
    virtual std::unique_ptr<armnn::NetworkImpl> GetExpectedModifiedGraph() const = 0;
    virtual ~IReplacementTestGraphFactory()
    {}
};
