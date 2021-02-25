//
// Copyright © 2020-2021 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "IReplacementTestGraphFactory.hpp"

class SISOCatOneGraphFactory : public IReplacementTestGraphFactory
{
public:
    const std::string& GetName() const override;
    std::unique_ptr<armnn::NetworkImpl> GetInitialGraph() const override;
    std::string GetMappingFileName() const override;
    std::unique_ptr<armnn::NetworkImpl> GetExpectedModifiedGraph() const override;
};
