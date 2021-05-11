//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <Graph.hpp>

namespace armnn
{
namespace ethosnbackend
{

void ReplaceUnsupportedLayers(Graph& graph);

}    // namespace ethosnbackend
}    // namespace armnn
