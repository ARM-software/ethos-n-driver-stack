//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <backendsCommon/CpuTensorHandle.hpp>

template <typename ConvolutionLayer>
const std::unique_ptr<armnn::ScopedCpuTensorHandle>& GetWeight(ConvolutionLayer* layer)
{
    return layer->m_Weight;
}

template <typename ConvolutionLayer>
const std::unique_ptr<armnn::ScopedCpuTensorHandle>& GetBias(ConvolutionLayer* layer)
{
    return layer->m_Bias;
}