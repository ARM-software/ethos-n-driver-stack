//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <backendsCommon/test/LayerTests.hpp>

LayerTestResult<uint8_t, 4>
    PreCompiledConvolution2dTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                     const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4>
    PreCompiledConvolution2dStride2x2TestImpl(armnn::IWorkloadFactory& workloadFactory,
                                              const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4>
    PreCompiledDepthwiseConvolution2dTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                              const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> PreCompiledDepthwiseConvolution2dStride2x2TestImpl(
    armnn::IWorkloadFactory& workloadFactory, const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> PreCompiledTransposeConvolution2dStride2x2TestImpl(
    armnn::IWorkloadFactory& workloadFactory, const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4>
    PreCompiledMaxPooling2dTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4>
    PreCompiledActivationRelu6TestImpl(armnn::IWorkloadFactory& workloadFactory,
                                       const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4>
    PreCompiledActivationReluTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                      const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4>
    PreCompiledActivationRelu1TestImpl(armnn::IWorkloadFactory& workloadFactory,
                                       const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2>
    PreCompiledFullyConnectedTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                      const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                      const armnn::TensorShape& inputShape);

std::vector<LayerTestResult<uint8_t, 4>>
    PreCompiledSplitterTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4>
    PreCompiledDepthToSpaceTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4>
    PreCompiledAdditionTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<boost::uint8_t, 4>
    PreCompiledMultiInputTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                  const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

std::vector<LayerTestResult<boost::uint8_t, 4>>
    PreCompiledMultiOutputTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                   const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<boost::uint8_t, 1>
    PreCompiled1dTensorTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<boost::uint8_t, 2>
    PreCompiled2dTensorTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<boost::uint8_t, 3>
    PreCompiled3dTensorTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);
