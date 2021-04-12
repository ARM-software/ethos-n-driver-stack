//
// Copyright © 2020-2021 Arm Ltd.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <backendsCommon/CpuTensorHandle.hpp>

template <typename ConvolutionLayer>
const std::shared_ptr<armnn::ConstCpuTensorHandle> GetWeight(ConvolutionLayer* layer)
{
    return layer->m_Weight;
}

template <typename ConvolutionLayer>
const std::shared_ptr<armnn::ConstCpuTensorHandle> GetBias(ConvolutionLayer* layer)
{
    return layer->m_Bias;
}
