//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "IReplacementTestGraphFactory.hpp"

class SISOCatOneGraphFactory : public IReplacementTestGraphFactory
{
public:
    const std::string& GetName() const override;
    armnn::INetworkPtr GetInitialGraph() const override;
    std::string GetMappingFileName() const override;
    armnn::INetworkPtr GetExpectedModifiedGraph() const override;
};
