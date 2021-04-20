//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "EthosNWorkloadFactoryHelper.hpp"
#include "PreCompiledTestImpl.hpp"

#include <EthosNWorkloadFactory.hpp>

#include <vector>

LayerTestResult<uint8_t, 4>
    PreCompiledConvolution2dTest(armnn::IWorkloadFactory& workloadFactory,
                                 const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledConvolution2dTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4>
    PreCompiledConvolution2dStride2x2Test(armnn::IWorkloadFactory& workloadFactory,
                                          const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledConvolution2dStride2x2TestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4>
    PreCompiledDepthwiseConvolution2dTest(armnn::IWorkloadFactory& workloadFactory,
                                          const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledDepthwiseConvolution2dTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> PreCompiledDepthwiseConvolution2dStride2x2Test(
    armnn::IWorkloadFactory& workloadFactory, const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledDepthwiseConvolution2dStride2x2TestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> PreCompiledTransposeConvolution2dStride2x2Test(
    armnn::IWorkloadFactory& workloadFactory, const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledTransposeConvolution2dStride2x2TestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> PreCompiledConvolution2dWithAssymetricSignedWeightsTest(
    armnn::IWorkloadFactory& workloadFactory, const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledConvolution2dWithAssymetricSignedWeightsTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> PreCompiledConvolution2dWithSymetricSignedWeightsTest(
    armnn::IWorkloadFactory& workloadFactory, const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledConvolution2dWithSymetricSignedWeightsTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> PreCompiledMeanXyTest(armnn::IWorkloadFactory& workloadFactory,
                                                  const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledMeanXyTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4>
    PreCompiledMaxPooling2dTest(armnn::IWorkloadFactory& workloadFactory,
                                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledMaxPooling2dTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4>
    PreCompiledActivationRelu6Test(armnn::IWorkloadFactory& workloadFactory,
                                   const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledActivationRelu6TestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4>
    PreCompiledActivationReluTest(armnn::IWorkloadFactory& workloadFactory,
                                  const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledActivationReluTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4>
    PreCompiledActivationRelu1Test(armnn::IWorkloadFactory& workloadFactory,
                                   const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledActivationRelu1TestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2>
    PreCompiledFullyConnectedTest(armnn::IWorkloadFactory& workloadFactory,
                                  const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledFullyConnectedTestImpl(workloadFactory, memoryManager, { 1, 8 });
}

LayerTestResult<uint8_t, 2>
    PreCompiledFullyConnected4dTest(armnn::IWorkloadFactory& workloadFactory,
                                    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledFullyConnectedTestImpl(workloadFactory, memoryManager, { 1, 2, 2, 3 });
}

std::vector<LayerTestResult<uint8_t, 4>>
    PreCompiledSplitterTest(armnn::IWorkloadFactory& workloadFactory,
                            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledSplitterTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4>
    PreCompiledDepthToSpaceTest(armnn::IWorkloadFactory& workloadFactory,
                                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledDepthToSpaceTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4>
    PreCompiledLeakyReluTest(armnn::IWorkloadFactory& workloadFactory,
                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledLeakyReluTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4>
    PreCompiledAdditionTest(armnn::IWorkloadFactory& workloadFactory,
                            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledAdditionTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4>
    PreCompiledMultiInputTest(armnn::IWorkloadFactory& workloadFactory,
                              const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledMultiInputTestImpl(workloadFactory, memoryManager);
}

std::vector<LayerTestResult<uint8_t, 4>>
    PreCompiledMultiOutputTest(armnn::IWorkloadFactory& workloadFactory,
                               const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledMultiOutputTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 1>
    PreCompiled1dTensorTest(armnn::IWorkloadFactory& workloadFactory,
                            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiled1dTensorTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2>
    PreCompiled2dTensorTest(armnn::IWorkloadFactory& workloadFactory,
                            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiled2dTensorTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 3>
    PreCompiled3dTensorTest(armnn::IWorkloadFactory& workloadFactory,
                            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiled3dTensorTestImpl(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4>
    PreCompiledConstMulToDepthwiseTest(armnn::IWorkloadFactory& workloadFactory,
                                       const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return PreCompiledConstMulToDepthwiseTestImpl(workloadFactory, memoryManager);
}
