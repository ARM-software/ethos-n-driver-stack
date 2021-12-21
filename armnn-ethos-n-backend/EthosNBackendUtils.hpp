//
// Copyright © 2020-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <armnn/backends/TensorHandle.hpp>

template <typename ConvolutionLayer>
const std::shared_ptr<armnn::ConstTensorHandle> GetWeight(ConvolutionLayer* layer)
{
    return layer->m_Weight;
}

template <typename ConvolutionLayer>
const std::shared_ptr<armnn::ConstTensorHandle> GetBias(ConvolutionLayer* layer)
{
    return layer->m_Bias;
}
