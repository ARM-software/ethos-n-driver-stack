//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include <Graph.hpp>
#include <Network.hpp>
#include <QuantizeHelper.hpp>
#include <Runtime.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnnUtils/Permute.hpp>
#include <backendsCommon/WorkloadFactory.hpp>
#include <backendsCommon/test/LayerTests.hpp>
#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <test/TensorHelpers.hpp>
#include <test/UnitTests.hpp>

#include "EthosNWorkloadFactoryHelper.hpp"

#include <boost/test/unit_test.hpp>

#include <algorithm>

using namespace armnn;

namespace
{

IConnectableLayer* AddConvolutionLayerToNetwork(INetwork& network,
                                                const Convolution2dDescriptor& descriptor,
                                                const ConstTensor& weights,
                                                const ConstTensor& biases)
{
    return network.AddConvolution2dLayer(descriptor, weights, Optional<ConstTensor>(biases), "convolution");
}

IConnectableLayer* AddConvolutionLayerToNetwork(INetwork& network,
                                                const DepthwiseConvolution2dDescriptor& descriptor,
                                                const ConstTensor& weights,
                                                const ConstTensor& biases)
{
    return network.AddDepthwiseConvolution2dLayer(descriptor, weights, Optional<ConstTensor>(biases),
                                                  "depthwiseConvolution");
}

IConnectableLayer* AddConvolutionLayerToNetwork(INetwork& network,
                                                const TransposeConvolution2dDescriptor& descriptor,
                                                const ConstTensor& weights,
                                                const ConstTensor& biases)
{
    return network.AddTransposeConvolution2dLayer(descriptor, weights, Optional<ConstTensor>(biases),
                                                  "transposeConvolution");
}

template <typename ConvolutionDescriptor>
ConvolutionDescriptor CreateConvolutionDescriptor(unsigned int stride, const unsigned int (&padding)[2])
{
    ConvolutionDescriptor descriptor;

    descriptor.m_StrideX     = stride;
    descriptor.m_StrideY     = stride;
    descriptor.m_PadLeft     = padding[0];
    descriptor.m_PadRight    = padding[1];
    descriptor.m_PadTop      = padding[0];
    descriptor.m_PadBottom   = padding[1];
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = DataLayout::NHWC;

    return descriptor;
}

template <typename ConvolutionDescriptor>
ConvolutionDescriptor CreateConvolutionDescriptor(unsigned int stride, unsigned int padding)
{
    return CreateConvolutionDescriptor<ConvolutionDescriptor>(stride, { padding, padding });
}

std::vector<uint8_t> CreateIdentityConvolutionKernel(unsigned int kernelSize, unsigned int channels)
{
    ARMNN_ASSERT(kernelSize % 2 == 1);    // kernelSize need to be an odd number

    const unsigned int numElements = channels * (kernelSize * kernelSize);
    std::vector<uint8_t> kernel(numElements, 0u);

    unsigned int centerIndex = kernelSize / 2;
    for (unsigned int y = 0u; y < kernelSize; y++)
    {
        for (unsigned int x = 0u; x < kernelSize; x++)
        {
            for (unsigned int channel = 0u; channel < channels; channel++)
            {
                if (x == centerIndex && y == centerIndex)
                {
                    const unsigned int flatIndex = (y * kernelSize * channels) + (x * channels) + channel;

                    kernel[flatIndex] = 1u;
                }
            }
        }
    }

    return kernel;
}

template <typename ConvolutionDescriptor>
std::vector<uint8_t> GetIdentityConvolutionExpectedOutputData(const TensorInfo& inputInfo,
                                                              const TensorInfo& outputInfo,
                                                              const ConvolutionDescriptor& descriptor,
                                                              const std::vector<uint8_t>& inputData)
{
    const unsigned int outputDataSize = outputInfo.GetNumElements();
    std::vector<uint8_t> expectedOutputData(outputDataSize);

    const unsigned int channels = outputInfo.GetShape()[3];
    ARMNN_ASSERT(channels == inputInfo.GetShape()[3]);

    const unsigned int inputW = inputInfo.GetShape()[2];

    const unsigned int outputH = outputInfo.GetShape()[1];
    const unsigned int outputW = outputInfo.GetShape()[2];

    // Pick values from the input buffer, but after each iteration skip a number of
    // rows and columns equal to the stride in the respective dimension.
    // For transpose convolution the stride applies to the output rather than the input:
    const bool isTranspose           = std::is_same<ConvolutionDescriptor, TransposeConvolution2dDescriptor>::value;
    const unsigned int inputStrideY  = isTranspose ? 1 : descriptor.m_StrideY;
    const unsigned int inputStrideX  = isTranspose ? 1 : descriptor.m_StrideX;
    const unsigned int outputStrideY = isTranspose ? descriptor.m_StrideY : 1;
    const unsigned int outputStrideX = isTranspose ? descriptor.m_StrideX : 1;

    for (unsigned int inputY = 0, outputY = 0; outputY < outputH; inputY += inputStrideY, outputY += outputStrideY)
    {
        for (unsigned int inputX = 0, outputX = 0; outputX < outputW; inputX += inputStrideX, outputX += outputStrideX)
        {
            for (unsigned int channel = 0u; channel < channels; channel++)
            {
                const unsigned int inputIndex  = (inputY * inputW * channels) + (inputX * channels) + channel;
                const unsigned int outputIndex = (outputY * outputW * channels) + (outputX * channels) + channel;

                expectedOutputData[outputIndex] = inputData[inputIndex];
            }
        }
    }

    return expectedOutputData;
}

template <typename ActivationDescriptor>
std::vector<uint8_t> GetActivationExpectedOutputData(const TensorInfo& inputInfo,
                                                     const TensorInfo& outputInfo,
                                                     const ActivationDescriptor& descriptor,
                                                     const std::vector<uint8_t>& inputData)
{
    const unsigned int inputDataSize  = inputInfo.GetNumElements();
    const unsigned int outputDataSize = outputInfo.GetNumElements();

    ARMNN_ASSERT(outputDataSize == inputDataSize);
    std::vector<uint8_t> expectedOutputData(outputDataSize);

    switch (descriptor.m_Function)
    {
        case ActivationFunction::BoundedReLu:
        {
            const uint8_t lowerBound = Quantize<uint8_t>(descriptor.m_B, outputInfo.GetQuantizationScale(),
                                                         outputInfo.GetQuantizationOffset());
            const uint8_t upperBound = Quantize<uint8_t>(descriptor.m_A, outputInfo.GetQuantizationScale(),
                                                         outputInfo.GetQuantizationOffset());
            for (unsigned int i = 0u; i < inputDataSize; ++i)
            {
                expectedOutputData[i] = std::max(lowerBound, std::min(inputData[i], upperBound));
            }
            break;
        }
        case ActivationFunction::ReLu:
        {
            constexpr uint8_t lowerBound = 0u;
            for (unsigned int i = 0u; i < inputDataSize; ++i)
            {
                expectedOutputData[i] = std::max(lowerBound, inputData[i]);
            }
            break;
        }
        case ActivationFunction::LeakyReLu:
        {
            for (unsigned int i = 0u; i < inputDataSize; ++i)
            {
                const float dequantizedInput =
                    Dequantize(inputData[i], inputInfo.GetQuantizationScale(), inputInfo.GetQuantizationOffset());
                expectedOutputData[i] =
                    Quantize<uint8_t>(std::max(descriptor.m_A * dequantizedInput, dequantizedInput),
                                      outputInfo.GetQuantizationScale(), outputInfo.GetQuantizationOffset());
            }
            break;
        }
        default:
        {
            ARMNN_ASSERT_MSG(false, "Unsupported Activation function");
            break;
        }
    }

    return expectedOutputData;
}

armnn::PreCompiledLayer* FindPreCompiledLayer(armnn::Graph& optimisedGraph)
{
    for (auto& layer : optimisedGraph)
    {
        if (layer->GetType() == armnn::LayerType::PreCompiled)
        {
            return PolymorphicPointerDowncast<armnn::PreCompiledLayer>(layer);
        }
    }

    // No pre-compiled layer found
    return nullptr;
}

armnn::IConnectableLayer* AddFusedActivationLayer(armnn::IConnectableLayer* prevLayer,
                                                  unsigned int outputSlotIndex,
                                                  const ActivationDescriptor& descriptor,
                                                  INetwork& network)
{
    std::string layerName = "activation" + std::string(GetActivationFunctionAsCString(descriptor.m_Function));
    IConnectableLayer* activationLayer = network.AddActivationLayer(descriptor, layerName.c_str());

    auto& prevOutputSlot = prevLayer->GetOutputSlot(outputSlotIndex);
    prevOutputSlot.Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).SetTensorInfo(prevOutputSlot.GetTensorInfo());
    return activationLayer;
}

template <uint32_t NumDims = 4>
std::vector<LayerTestResult<uint8_t, NumDims>>
    OptimiseAndRunNetworkMultiple(armnn::IWorkloadFactory& workloadFactory,
                                  INetwork& network,
                                  std::map<LayerBindingId, TensorInfo> inputInfos,
                                  std::map<LayerBindingId, std::vector<uint8_t>> inputData,
                                  std::map<LayerBindingId, TensorInfo> outputInfos,
                                  std::map<LayerBindingId, std::vector<uint8_t>> expectedOutputData)
{
    ARMNN_ASSERT(inputInfos.size() == inputData.size());
    ARMNN_ASSERT(outputInfos.size() == expectedOutputData.size());

    // Optimize the network for the backend supported by the factory
    std::vector<BackendId> backends = { workloadFactory.GetBackendId() };
    IRuntimePtr runtime(IRuntime::Create(IRuntime::CreationOptions()));
    std::vector<std::string> messages;
    IOptimizedNetworkPtr optimizedNet(nullptr, nullptr);
    try
    {
        optimizedNet = Optimize(network, backends, runtime->GetDeviceSpec(), OptimizerOptions(), messages);
    }
    catch (const armnn::Exception&)
    {
        throw RuntimeException(std::string("Failed to optimize network for ") + std::string(backends[0]),
                               CHECK_LOCATION());
    }
    ARMNN_ASSERT(GetGraphForTesting(optimizedNet.get()).GetNumInputs() == inputInfos.size());
    ARMNN_ASSERT(GetGraphForTesting(optimizedNet.get()).GetNumOutputs() == outputInfos.size());
    // Find the pre-compiled layer in the optimised graph
    Graph& optimisedGraph              = GetGraphForTesting(optimizedNet.get());
    PreCompiledLayer* preCompiledLayer = FindPreCompiledLayer(optimisedGraph);
    if (!preCompiledLayer)
    {
        throw RuntimeException("Could not find pre-compiled layer in optimised graph", CHECK_LOCATION());
    }

    // Lookup the mapping of input and output binding IDs to the input/output indices of the precompiled layer.
    // We assume that the network consists entirely of the precompiled layer (i.e. no other layers present).
    std::map<uint32_t, LayerBindingId> inputIdxsToBindingId;
    for (uint32_t i = 0; i < preCompiledLayer->GetNumInputSlots(); ++i)
    {
        Layer& inputLayer =
            PolymorphicPointerDowncast<OutputSlot>(preCompiledLayer->GetInputSlot(i).GetConnection())->GetOwningLayer();
        ARMNN_ASSERT(inputLayer.GetType() == LayerType::Input);
        LayerBindingId bindingId = PolymorphicPointerDowncast<InputLayer>(&inputLayer)->GetBindingId();
        inputIdxsToBindingId[i]  = bindingId;
    }

    std::map<uint32_t, LayerBindingId> outputIdxsToBindingId;
    for (uint32_t i = 0; i < preCompiledLayer->GetNumOutputSlots(); ++i)
    {
        ARMNN_ASSERT(preCompiledLayer->GetOutputSlot(i).GetNumConnections() == 1);
        Layer& outputLayer = PolymorphicPointerDowncast<InputSlot>(preCompiledLayer->GetOutputSlot(i).GetConnection(0))
                                 ->GetOwningLayer();
        ARMNN_ASSERT(outputLayer.GetType() == LayerType::Output);
        LayerBindingId bindingId = PolymorphicPointerDowncast<OutputLayer>(&outputLayer)->GetBindingId();
        outputIdxsToBindingId[i] = bindingId;
    }

    // Create the tensor handles
    TensorHandleFactoryRegistry tmpRegistry;
    for (auto&& layer : optimisedGraph.TopologicalSort())
    {
        layer->CreateTensorHandles(tmpRegistry, workloadFactory);
    }

    // Create the pre-compiled workload
    auto workload = preCompiledLayer->CreateWorkload(workloadFactory);

    // Set the input data
    const QueueDescriptor& workloadData =
        static_cast<BaseWorkload<PreCompiledQueueDescriptor>*>(workload.get())->GetData();
    ARMNN_ASSERT(inputInfos.size() == workloadData.m_Inputs.size());
    for (uint32_t i = 0; i < inputInfos.size(); ++i)
    {
        LayerBindingId bindingId = inputIdxsToBindingId.at(i);
        CopyDataToITensorHandle(workloadData.m_Inputs[i], inputData.at(bindingId).data());
    }

    // Execute the workload
    workload->Execute();

    // Set the expected and actual outputs
    std::vector<LayerTestResult<uint8_t, NumDims>> results;
    ARMNN_ASSERT(outputInfos.size() == workloadData.m_Outputs.size());
    for (uint32_t i = 0; i < outputInfos.size(); ++i)
    {
        LayerBindingId bindingId = outputIdxsToBindingId.at(i);
        LayerTestResult<uint8_t, NumDims> result(outputInfos.at(bindingId));
        result.outputExpected =
            MakeTensor<uint8_t, NumDims>(outputInfos.at(bindingId), expectedOutputData.at(bindingId));
        CopyDataFromITensorHandle(result.output.data(), workloadData.m_Outputs[i]);
        results.push_back(result);
    }
    return results;
}

/// Simpler version of the above function for single input and single output networks.
template <uint32_t NumDims = 4>
LayerTestResult<uint8_t, NumDims> OptimiseAndRunNetwork(armnn::IWorkloadFactory& workloadFactory,
                                                        INetwork& network,
                                                        LayerBindingId inputBindingId,
                                                        TensorInfo inputInfo,
                                                        std::vector<uint8_t> inputData,
                                                        LayerBindingId outputBindingId,
                                                        TensorInfo outputInfo,
                                                        std::vector<uint8_t> expectedOutputData)
{
    return OptimiseAndRunNetworkMultiple<NumDims>(
        workloadFactory, network, { { inputBindingId, inputInfo } }, { { inputBindingId, inputData } },
        { { outputBindingId, outputInfo } }, { { outputBindingId, expectedOutputData } })[0];
}

template <typename ConvolutionDescriptor>
LayerTestResult<uint8_t, 4> PreCompiledConvolution2dTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
                                                             unsigned int inputSize,
                                                             unsigned int outputSize,
                                                             unsigned int channels,
                                                             unsigned int kernelSize,
                                                             const ConvolutionDescriptor& descriptor)
{
    constexpr bool isDepthwise = std::is_same<ConvolutionDescriptor, DepthwiseConvolution2dDescriptor>::value;

    ARMNN_ASSERT(descriptor.m_BiasEnabled == true);
    ARMNN_ASSERT(descriptor.m_DataLayout == DataLayout::NHWC);

    // Set up tensor shapes and infos
    const TensorShape inputShape({ 1, inputSize, inputSize, channels });
    const TensorShape outputShape({ 1, outputSize, outputSize, channels });
    const TensorShape kernelShape = isDepthwise ? TensorShape({ 1, channels, kernelSize, kernelSize })
                                                : TensorShape({ 1, kernelSize, kernelSize, channels });
    const TensorShape biasesShape({ 1, 1, 1, channels });

    // NOTE: inputScale * weightsScale / outputScale must be >= 0.0 and < 1.0
    TensorInfo inputInfo(inputShape, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo outputInfo(outputShape, DataType::QAsymmU8, 2.0f, 0);
    TensorInfo weightsInfo(kernelShape, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo biasesInfo(biasesShape, DataType::Signed32, 1.0f, 0);

    // Populate weight and bias data
    std::vector<uint8_t> weightsData = CreateIdentityConvolutionKernel(kernelSize, channels);

    if (isDepthwise)
    {
        // Swizzle weight data
        std::vector<uint8_t> swizzledWeightsData(weightsData.size(), 0);
        const PermutationVector permutationVector = { 0, 2, 3, 1 };
        armnnUtils::Permute(kernelShape, permutationVector, reinterpret_cast<const uint8_t*>(weightsData.data()),
                            swizzledWeightsData.data(), sizeof(uint8_t));

        weightsData = std::move(swizzledWeightsData);
    }

    // NOTE: We need to multiply the elements of the identity kernel by 2
    // to compensate for the scaling factor
    std::transform(weightsData.begin(), weightsData.end(), weightsData.begin(),
                   [](uint8_t w) -> uint8_t { return static_cast<uint8_t>(w * 2); });

    const unsigned int biasDataSize = biasesInfo.GetNumElements();
    std::vector<int32_t> biasesData(biasDataSize, 0);

    // Construct network
    INetworkPtr network = armnn::INetwork::Create();
    ConstTensor weights(weightsInfo, weightsData);
    ConstTensor biases(biasesInfo, biasesData);

    IConnectableLayer* const inputLayer       = network->AddInputLayer(0, "input");
    IConnectableLayer* const convolutionLayer = AddConvolutionLayerToNetwork(*network, descriptor, weights, biases);
    IConnectableLayer* const outputLayer      = network->AddOutputLayer(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convolutionLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    convolutionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    convolutionLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // Generate input data: sequence [0, 1 .. 255]
    const unsigned int inputDataSize = inputInfo.GetNumElements();
    std::vector<uint8_t> inputData(inputDataSize);
    std::iota(inputData.begin(), inputData.end(), 0);

    // Set expected output
    std::vector<uint8_t> expectedOutputData =
        GetIdentityConvolutionExpectedOutputData(inputInfo, outputInfo, descriptor, inputData);

    return OptimiseAndRunNetwork(workloadFactory, *network, 0, inputInfo, inputData, 0, outputInfo, expectedOutputData);
}

// Test a [1, 1, 1, 1] tensor with signed weights
template <typename ConvolutionDescriptor>
LayerTestResult<uint8_t, 4>
    PreCompiledConvolution2dWithSignedWeightsTest(armnn::IWorkloadFactory& workloadFactory,
                                                  const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
                                                  const ConvolutionDescriptor& descriptor,
                                                  const armnn::DataType weightDataType)
{
    constexpr int inputSize  = 1;
    constexpr int outputSize = 1;
    constexpr int channels   = 1;
    constexpr int kernelSize = 1;
    // We must set a zero point bigger than the absolute value of our final
    // results, else the ouput is clamped into the range [0, 255] because
    // output values are uint8_t
    constexpr int32_t outputZeroPoint = 100;
    constexpr float weightScale       = 0.5f;

    ARMNN_ASSERT(descriptor.m_BiasEnabled == true);
    ARMNN_ASSERT(descriptor.m_DataLayout == DataLayout::NHWC);

    // Set up tensor shapes and infos
    const TensorShape inputShape({ 1, inputSize, inputSize, channels });
    const TensorShape outputShape({ 1, outputSize, outputSize, channels });
    const TensorShape kernelShape({ 1, kernelSize, kernelSize, channels });
    const TensorShape biasesShape({ 1, 1, 1, channels });

    // NOTE: inputScale * weightsScale / outputScale must be >= 0.0 and < 1.0
    TensorInfo inputInfo(inputShape, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo outputInfo(outputShape, DataType::QAsymmU8, 1.0f, outputZeroPoint);
    // We set on purpose a non zero zero point when the data type is symmetric
    // to check that the backend reset it to zero.
    const bool isWeightDataTypeSymmetric = weightDataType == armnn::DataType::QSymmS8;
    const int32_t weightZeroPoint        = isWeightDataTypeSymmetric ? 42 : 0;
    TensorInfo weightsInfo(kernelShape, weightDataType, weightScale, weightZeroPoint);
    TensorInfo biasesInfo(biasesShape, DataType::Signed32, 1.0f, 0);

    // input weight is -42
    // the weight data are quantized
    // -84 comes from armnn::Quantize<int8_t>(-42, weightScale, 0)
    std::vector<int8_t> weightsData = { -84 };

    const unsigned int biasDataSize = biasesInfo.GetNumElements();
    std::vector<int32_t> biasesData(biasDataSize, 0);

    // Construct network
    INetworkPtr network = armnn::INetwork::Create();
    ConstTensor weights(weightsInfo, weightsData);
    ConstTensor biases(biasesInfo, biasesData);

    IConnectableLayer* const inputLayer       = network->AddInputLayer(0, "input");
    IConnectableLayer* const convolutionLayer = AddConvolutionLayerToNetwork(*network, descriptor, weights, biases);
    IConnectableLayer* const outputLayer      = network->AddOutputLayer(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convolutionLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    convolutionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    convolutionLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    std::vector<uint8_t> inputData = { 2 };

    // Set expected output
    // 16 comes from armnn::Dequantize(weightsData[0], weightScale, 0) * inputData[0] + outputZeroPoint
    uint8_t signedExpectedOutput            = 16;
    std::vector<uint8_t> expectedOutputData = { signedExpectedOutput };

    return OptimiseAndRunNetwork(workloadFactory, *network, 0, inputInfo, inputData, 0, outputInfo, expectedOutputData);
}

}    // anonymous namespace

LayerTestResult<uint8_t, 4>
    PreCompiledConvolution2dTest(armnn::IWorkloadFactory& workloadFactory,
                                 const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputSize  = 16;
    const unsigned int outputSize = 16;
    const unsigned int channels   = 1;
    const unsigned int kernelSize = 3;
    const unsigned int stride     = 1;
    const unsigned int padding    = 1;

    Convolution2dDescriptor descriptor = CreateConvolutionDescriptor<Convolution2dDescriptor>(stride, padding);

    return PreCompiledConvolution2dTestImpl(workloadFactory, memoryManager, inputSize, outputSize, channels, kernelSize,
                                            descriptor);
}

LayerTestResult<uint8_t, 4>
    PreCompiledConvolution2dStride2x2Test(armnn::IWorkloadFactory& workloadFactory,
                                          const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputSize  = 16;
    const unsigned int outputSize = 8;
    const unsigned int channels   = 1;
    const unsigned int kernelSize = 3;
    const unsigned int stride     = 2;
    const unsigned int padding[2] = { 1, 0 };

    Convolution2dDescriptor descriptor = CreateConvolutionDescriptor<Convolution2dDescriptor>(stride, padding);

    return PreCompiledConvolution2dTestImpl(workloadFactory, memoryManager, inputSize, outputSize, channels, kernelSize,
                                            descriptor);
}

LayerTestResult<uint8_t, 4>
    PreCompiledDepthwiseConvolution2dTest(armnn::IWorkloadFactory& workloadFactory,
                                          const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputSize  = 16;
    const unsigned int outputSize = 16;
    const unsigned int channels   = 3;
    const unsigned int kernelSize = 1;
    const unsigned int stride     = 1;
    const unsigned int padding    = 0;

    DepthwiseConvolution2dDescriptor descriptor =
        CreateConvolutionDescriptor<DepthwiseConvolution2dDescriptor>(stride, padding);

    return PreCompiledConvolution2dTestImpl(workloadFactory, memoryManager, inputSize, outputSize, channels, kernelSize,
                                            descriptor);
}

LayerTestResult<uint8_t, 4> PreCompiledDepthwiseConvolution2dStride2x2Test(
    armnn::IWorkloadFactory& workloadFactory, const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputSize  = 16;
    const unsigned int outputSize = 8;
    const unsigned int channels   = 3;
    const unsigned int kernelSize = 3;
    const unsigned int stride     = 2;
    const unsigned int padding[2] = { 1, 0 };

    DepthwiseConvolution2dDescriptor descriptor =
        CreateConvolutionDescriptor<DepthwiseConvolution2dDescriptor>(stride, padding);

    return PreCompiledConvolution2dTestImpl(workloadFactory, memoryManager, inputSize, outputSize, channels, kernelSize,
                                            descriptor);
}

LayerTestResult<uint8_t, 4> PreCompiledTransposeConvolution2dStride2x2Test(
    armnn::IWorkloadFactory& workloadFactory, const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputSize  = 8;
    const unsigned int outputSize = 16;
    const unsigned int channels   = 1;
    const unsigned int kernelSize = 3;
    const unsigned int stride     = 2;
    const unsigned int padding[2] = { 1, 0 };

    TransposeConvolution2dDescriptor descriptor =
        CreateConvolutionDescriptor<TransposeConvolution2dDescriptor>(stride, padding);

    return PreCompiledConvolution2dTestImpl(workloadFactory, memoryManager, inputSize, outputSize, channels, kernelSize,
                                            descriptor);
}

LayerTestResult<uint8_t, 4> PreCompiledConvolution2dWithAssymetricSignedWeightsTest(
    armnn::IWorkloadFactory& workloadFactory, const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int stride  = 1;
    const unsigned int padding = 0;

    Convolution2dDescriptor descriptor = CreateConvolutionDescriptor<Convolution2dDescriptor>(stride, padding);

    return PreCompiledConvolution2dWithSignedWeightsTest(workloadFactory, memoryManager, descriptor,
                                                         armnn::DataType::QAsymmS8);
}

LayerTestResult<uint8_t, 4> PreCompiledConvolution2dWithSymetricSignedWeightsTest(
    armnn::IWorkloadFactory& workloadFactory, const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int stride  = 1;
    const unsigned int padding = 0;

    Convolution2dDescriptor descriptor = CreateConvolutionDescriptor<Convolution2dDescriptor>(stride, padding);

    return PreCompiledConvolution2dWithSignedWeightsTest(workloadFactory, memoryManager, descriptor,
                                                         armnn::DataType::QSymmS8);
}

LayerTestResult<uint8_t, 4> PreCompiledMeanXyTest(armnn::IWorkloadFactory& workloadFactory,
                                                  const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
{
    // Set up the input/output tensor info
    TensorInfo inputInfo({ 1, 7, 7, 1 }, DataType::QAsymmU8, 2.0f, 0);
    TensorInfo outputInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 2.0f, 0);
    unsigned int numElements = inputInfo.GetNumElements();
    std::vector<uint8_t> inputData(numElements);

    for (uint8_t i = 0; i < numElements; i++)
    {
        inputData[i] = i;
    }

    std::vector<uint8_t> expectedOutputData = { 24 };

    // Set up the Mean descriptor to calculate the mean along height and width
    armnn::MeanDescriptor desc;
    desc.m_KeepDims = true;
    desc.m_Axis     = { 1, 2 };

    // Construct the network
    armnn::INetworkPtr net               = armnn::INetwork::Create();
    IConnectableLayer* const inputLayer  = net->AddInputLayer(0, "input");
    IConnectableLayer* const meanLayer   = net->AddMeanLayer(desc, "mean");
    IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output");

    // Connect the layers
    inputLayer->GetOutputSlot(0).Connect(meanLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    meanLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    meanLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return OptimiseAndRunNetwork(workloadFactory, *net, 0, inputInfo, inputData, 0, outputInfo, expectedOutputData);
}

LayerTestResult<uint8_t, 4> PreCompiledMaxPooling2dTest(armnn::IWorkloadFactory& workloadFactory,
                                                        const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
{
    // Pooling cannot be run in isolation, it must be fused with the previous layer, e.g. Convolution2d.

    // Set up the Convolution descriptor
    Convolution2dDescriptor convDescriptor;
    convDescriptor.m_StrideX     = 1;
    convDescriptor.m_StrideY     = 1;
    convDescriptor.m_BiasEnabled = true;
    convDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Set up the Convolution weights
    TensorInfo weightsInfo(TensorShape({ 16, 1, 1, 16 }), DataType::QAsymmU8, 2.0f, 0);
    const unsigned int weightsDataSize = weightsInfo.GetNumElements();
    std::vector<uint8_t> weightsData(weightsDataSize);
    for (unsigned int i = 0; i < 16; ++i)
    {
        for (unsigned int j = 0; j < 16; ++j)
        {
            weightsData[(i * 16) + j] = i == j ? 1 : 0;
        }
    }
    ConstTensor weights(weightsInfo, weightsData);

    // Set up the Convolution biases
    TensorInfo biasInfo(TensorShape({ 1, 1, 1, 16 }), DataType::Signed32, 1.0f * 2.0f, 0);
    const unsigned int biasDataSize = biasInfo.GetNumElements();
    std::vector<int32_t> biasData(biasDataSize, 0);
    ConstTensor biases(biasInfo, biasData);

    // Set up the Convolution input
    TensorInfo inputInfo(TensorShape({ 1, 16, 16, 16 }), DataType::QAsymmU8, 1.0f, 0);
    const unsigned int inputDataSize = inputInfo.GetNumElements();
    std::vector<uint8_t> inputData(inputDataSize);
    for (unsigned int i = 0; i < inputDataSize; ++i)
    {
        inputData[i] = numeric_cast<uint8_t>((i * 4) % 250);
    }

    // Set up the Convolution output / Pooling input info
    TensorInfo convOutputInfo(TensorShape({ 1, 16, 16, 16 }), DataType::QAsymmU8, 4.0f, 0);

    // Set up the Pooling descriptor
    Pooling2dDescriptor poolDescriptor;
    poolDescriptor.m_PoolType      = PoolingAlgorithm::Max;
    poolDescriptor.m_PoolWidth     = 2;
    poolDescriptor.m_PoolHeight    = 2;
    poolDescriptor.m_StrideX       = 2;
    poolDescriptor.m_StrideY       = 2;
    poolDescriptor.m_PaddingMethod = PaddingMethod::Exclude;
    poolDescriptor.m_DataLayout    = DataLayout::NHWC;

    // Set the expected output from the Pooling layer
    TensorInfo outputInfo(TensorShape({ 1, 8, 8, 16 }), DataType::QAsymmU8, 4.0f, 0);
    const unsigned int outputDataSize = outputInfo.GetNumElements();
    std::vector<uint8_t> expectedOutputData(outputDataSize);
    // The Maxpooling inputs are the Convolution outputs, i.e. (Convolution inputs / 2) after scale adjustments
    // Maxpooling selects the max value in each pool from its inputs and our pool size is 2x2
    for (unsigned int channel = 0; channel < 16; ++channel)
    {
        for (unsigned int row = 0; row < 8; ++row)
        {
            for (unsigned int column = 0; column < 8; ++column)
            {
                // The input and output data indexes are calculated for NHWC data layout.
                // Output index: (row * columns * channels) + (column * channels) + channel
                auto outIndex = (row * 8 * 16) + (column * 16) + channel;
                // Input index: (row * strideY * columns * channels) + (column * strideX * channels) + channel
                //      and we take 4 entries for the 2x2 pool
                auto in0Index = ((row * 2) * 16 * 16) + ((column * 2) * 16) + channel;
                auto in1Index = ((row * 2) * 16 * 16) + (((column * 2) + 1) * 16) + channel;
                auto in2Index = (((row * 2) + 1) * 16 * 16) + ((column * 2) * 16) + channel;
                auto in3Index = (((row * 2) + 1) * 16 * 16) + (((column * 2) + 1) * 16) + channel;
                // output value is the maximum of the input pool values, adjusted for the quantization scale change
                auto maxIn = std::max<uint8_t>(
                    { inputData[in0Index], inputData[in1Index], inputData[in2Index], inputData[in3Index] });
                expectedOutputData[outIndex] = maxIn / 2;
            }
        }
    }

    // Construct the network
    armnn::INetworkPtr net              = armnn::INetwork::Create();
    IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input");
    IConnectableLayer* const convLayer =
        net->AddConvolution2dLayer(convDescriptor, weights, Optional<ConstTensor>(biases), "conv");
    IConnectableLayer* const poolingLayer = net->AddPooling2dLayer(poolDescriptor, "pooling2d");
    IConnectableLayer* const outputLayer  = net->AddOutputLayer(0, "output");

    // Connect the layers
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    convLayer->GetOutputSlot(0).Connect(poolingLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).SetTensorInfo(convOutputInfo);
    poolingLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    poolingLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return OptimiseAndRunNetwork(workloadFactory, *net, 0, inputInfo, inputData, 0, outputInfo, expectedOutputData);
}

template <typename ConvolutionDescriptor>
LayerTestResult<uint8_t, 4> PreCompiledFusedActivationTest(armnn::IWorkloadFactory& workloadFactory,
                                                           const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
                                                           unsigned int inputSize,
                                                           unsigned int outputSize,
                                                           unsigned int channels,
                                                           unsigned int kernelSize,
                                                           const ConvolutionDescriptor& convDescriptor,
                                                           const ActivationDescriptor& activationDescriptor)
{
    ARMNN_ASSERT(convDescriptor.m_BiasEnabled == true);
    ARMNN_ASSERT(convDescriptor.m_DataLayout == DataLayout::NHWC);

    // Set up tensor shapes and infos
    const TensorShape inputShape({ 1, inputSize, inputSize, channels });
    const TensorShape outputShape({ 1, outputSize, outputSize, channels });
    const TensorShape kernelShape({ 1, kernelSize, kernelSize, channels });
    const TensorShape biasesShape({ 1, 1, 1, channels });

    // NOTE: inputScale * weightsScale / outputScale must be >= 0.0 and < 1.0
    TensorInfo inputInfo(inputShape, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo convOutputInfo(outputShape, DataType::QAsymmU8, 2.0f, 0);
    TensorInfo weightsInfo(kernelShape, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo biasesInfo(biasesShape, DataType::Signed32, 1.0f, 0);

    // Populate weight and bias data
    std::vector<uint8_t> weightsData = CreateIdentityConvolutionKernel(kernelSize, channels);

    // NOTE: We need to multiply the elements of the identity kernel by 2
    // to compensate for the scaling factor
    std::transform(weightsData.begin(), weightsData.end(), weightsData.begin(),
                   [](uint8_t w) -> uint8_t { return static_cast<uint8_t>(w * 2); });

    const unsigned int biasDataSize = biasesInfo.GetNumElements();
    std::vector<int32_t> biasesData(biasDataSize, 0);

    // Generate input data: sequence [0, 1 .. 255]
    const unsigned int inputDataSize = inputInfo.GetNumElements();
    std::vector<uint8_t> inputData(inputDataSize);
    std::iota(inputData.begin(), inputData.end(), 0);

    // Set expected convolution output
    std::vector<uint8_t> expectedConvOutputData =
        GetIdentityConvolutionExpectedOutputData(inputInfo, convOutputInfo, convDescriptor, inputData);

    // Set the expected output shape from the activation layer
    TensorInfo outputInfo(TensorShape({ 1, inputSize, inputSize, channels }), DataType::QAsymmU8, 2.0f, 0);

    // Set expected output for ReLu
    std::vector<uint8_t> expectedOutputData =
        GetActivationExpectedOutputData(convOutputInfo, outputInfo, activationDescriptor, expectedConvOutputData);

    // Construct network
    armnn::INetworkPtr net = armnn::INetwork::Create();
    ConstTensor weights(weightsInfo, weightsData);
    ConstTensor biases(biasesInfo, biasesData);

    IConnectableLayer* const inputLayer  = net->AddInputLayer(0, "input");
    IConnectableLayer* const convLayer   = AddConvolutionLayerToNetwork(*net, convDescriptor, weights, biases);
    IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output");

    // Connect the layers
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    auto activationLayer = AddFusedActivationLayer(convLayer, 0, activationDescriptor, *net);

    activationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    return OptimiseAndRunNetwork(workloadFactory, *net, 0, inputInfo, inputData, 0, outputInfo, expectedOutputData);
}

LayerTestResult<uint8_t, 4>
    PreCompiledActivationRelu6Test(armnn::IWorkloadFactory& workloadFactory,
                                   const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputSize  = 16;
    const unsigned int outputSize = 16;
    const unsigned int channels   = 1;
    const unsigned int kernelSize = 3;
    const unsigned int stride     = 1;
    const unsigned int padding    = 1;

    Convolution2dDescriptor convolutionDescriptor =
        CreateConvolutionDescriptor<Convolution2dDescriptor>(stride, padding);

    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A        = 6.0f;
    activationDescriptor.m_B        = 0.0f;

    return PreCompiledFusedActivationTest(workloadFactory, memoryManager, inputSize, outputSize, channels, kernelSize,
                                          convolutionDescriptor, activationDescriptor);
}

LayerTestResult<uint8_t, 4>
    PreCompiledActivationReluTest(armnn::IWorkloadFactory& workloadFactory,
                                  const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputSize  = 16;
    const unsigned int outputSize = 16;
    const unsigned int channels   = 1;
    const unsigned int kernelSize = 3;
    const unsigned int stride     = 1;
    const unsigned int padding    = 1;

    Convolution2dDescriptor convolutionDescriptor =
        CreateConvolutionDescriptor<Convolution2dDescriptor>(stride, padding);

    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::ReLu;

    return PreCompiledFusedActivationTest(workloadFactory, memoryManager, inputSize, outputSize, channels, kernelSize,
                                          convolutionDescriptor, activationDescriptor);
}

LayerTestResult<uint8_t, 4>
    PreCompiledActivationRelu1Test(armnn::IWorkloadFactory& workloadFactory,
                                   const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputSize  = 16;
    const unsigned int outputSize = 16;
    const unsigned int channels   = 1;
    const unsigned int kernelSize = 3;
    const unsigned int stride     = 1;
    const unsigned int padding    = 1;

    Convolution2dDescriptor convolutionDescriptor =
        CreateConvolutionDescriptor<Convolution2dDescriptor>(stride, padding);

    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::BoundedReLu;
    activationDescriptor.m_A        = 1.0f;
    activationDescriptor.m_B        = 0.0f;

    return PreCompiledFusedActivationTest(workloadFactory, memoryManager, inputSize, outputSize, channels, kernelSize,
                                          convolutionDescriptor, activationDescriptor);
}

LayerTestResult<uint8_t, 2> PreCompiledFullyConnectedTest(armnn::IWorkloadFactory& workloadFactory,
                                                          const armnn::IBackendInternal::IMemoryManagerSharedPtr&,
                                                          const armnn::TensorShape& inputShape)
{
    const unsigned int numInputs  = inputShape.GetNumElements();
    const unsigned int numOutputs = 2;

    // Set up tensor shapes and infos
    const TensorShape outputShape({ 1, numOutputs });
    const TensorShape weightShape({ numInputs, numOutputs });
    const TensorShape biasesShape({ numOutputs });

    TensorInfo inputInfo(inputShape, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo outputInfo(outputShape, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo weightsInfo(weightShape, DataType::QAsymmU8, 0.5f, 0);
    TensorInfo biasesInfo(biasesShape, DataType::Signed32, 0.5f, 0);

    // Populate weight data such that output channel n is 1 * input channel n (i.e. an identity transformation).
    const uint8_t quantizedWeight =
        Quantize<uint8_t>(1.0f, weightsInfo.GetQuantizationScale(), weightsInfo.GetQuantizationOffset());
    const unsigned int weightsDataSize = weightsInfo.GetNumElements();
    std::vector<uint8_t> weightsData(weightsDataSize, 0);

    weightsData[0 * numOutputs + 0] = quantizedWeight;
    weightsData[1 * numOutputs + 1] = quantizedWeight;

    // Populate bias data (all ones)
    const unsigned int biasDataSize = biasesInfo.GetNumElements();
    std::vector<int32_t> biasesData(biasDataSize, 1);

    // Generate input data: sequence [1, 2, 3, ..., n-1, n]
    std::vector<uint8_t> inputData(numInputs);
    std::iota(inputData.begin(), inputData.end(), 1u);

    FullyConnectedDescriptor descriptor;
    descriptor.m_BiasEnabled           = true;
    descriptor.m_TransposeWeightMatrix = false;

    // Set expected output
    std::vector<uint8_t> expectedOutputData{ 1, 2 };

    // Construct network
    armnn::INetworkPtr net = armnn::INetwork::Create();
    ConstTensor weights(weightsInfo, weightsData);
    ConstTensor biases(biasesInfo, biasesData);

    IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input");
    IConnectableLayer* const fullyConnectedLayer =
        net->AddFullyConnectedLayer(descriptor, weights, Optional<ConstTensor>(biases), "fullyConnected");
    IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output");

    // Connect the layers
    inputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    fullyConnectedLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    return OptimiseAndRunNetwork<2>(workloadFactory, *net, 0, inputInfo, inputData, 0, outputInfo, expectedOutputData);
}

/// A simple split of a 1x1x2x1 tensor into two 1x1x1x1 tensors.
std::vector<LayerTestResult<uint8_t, 4>>
    PreCompiledSplitterTest(armnn::IWorkloadFactory& workloadFactory,
                            const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
{
    // Construct network
    armnn::INetworkPtr net = armnn::INetwork::Create();

    TensorInfo inputInfo({ 1, 1, 2, 1 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo outputInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 1.0f, 0);

    ViewsDescriptor descriptor(2, 4);
    // First view takes the left element
    descriptor.SetViewOriginCoord(0, 0, 0);
    descriptor.SetViewOriginCoord(0, 1, 0);
    descriptor.SetViewOriginCoord(0, 2, 0);
    descriptor.SetViewOriginCoord(0, 3, 0);
    descriptor.SetViewSize(0, 0, 1);
    descriptor.SetViewSize(0, 1, 1);
    descriptor.SetViewSize(0, 2, 1);
    descriptor.SetViewSize(0, 3, 1);
    // Second view takes the right element
    descriptor.SetViewOriginCoord(1, 0, 0);
    descriptor.SetViewOriginCoord(1, 1, 0);
    descriptor.SetViewOriginCoord(1, 2, 1);
    descriptor.SetViewOriginCoord(1, 3, 0);
    descriptor.SetViewSize(1, 0, 1);
    descriptor.SetViewSize(1, 1, 1);
    descriptor.SetViewSize(1, 2, 1);
    descriptor.SetViewSize(1, 3, 1);

    IConnectableLayer* const inputLayer    = net->AddInputLayer(0, "input");
    IConnectableLayer* const splitterLayer = net->AddSplitterLayer(descriptor, "splitter");
    IConnectableLayer* const outputLayer0  = net->AddOutputLayer(0, "output0");
    IConnectableLayer* const outputLayer1  = net->AddOutputLayer(1, "output1");

    // Connect the layers
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    inputLayer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));

    splitterLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    splitterLayer->GetOutputSlot(0).Connect(outputLayer0->GetInputSlot(0));
    splitterLayer->GetOutputSlot(1).SetTensorInfo(outputInfo);
    splitterLayer->GetOutputSlot(1).Connect(outputLayer1->GetInputSlot(0));

    // Set input data and expected output
    std::vector<uint8_t> inputData{ 1, 2 };
    std::vector<uint8_t> expectedOutputData0{ 1 };
    std::vector<uint8_t> expectedOutputData1{ 2 };

    return OptimiseAndRunNetworkMultiple(workloadFactory, *net, { { 0, inputInfo } }, { { 0, inputData } },
                                         { { 0, outputInfo }, { 1, outputInfo } },
                                         { { 0, expectedOutputData0 }, { 1, expectedOutputData1 } });
}

LayerTestResult<uint8_t, 4> PreCompiledDepthToSpaceTest(armnn::IWorkloadFactory& workloadFactory,
                                                        const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
{
    // Construct network
    armnn::INetworkPtr net = armnn::INetwork::Create();

    TensorInfo inputInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo outputInfo({ 1, 4, 4, 1 }, DataType::QAsymmU8, 1.0f, 0);

    IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    DepthToSpaceDescriptor desc(2, DataLayout::NHWC);
    IConnectableLayer* const spaceToDepthLayer = net->AddDepthToSpaceLayer(desc, "depthToSpace");
    spaceToDepthLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    inputLayer->GetOutputSlot(0).Connect(spaceToDepthLayer->GetInputSlot(0));

    IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output");
    spaceToDepthLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    std::vector<uint8_t> inputData{
        // clang-format off
        1, 2, 3, 4,           10, 20, 30, 40,
        5, 6, 7, 8,           11, 21, 31, 41,
        // clang-format on
    };
    std::vector<uint8_t> expectedOutputData{
        // clang-format off
        1, 2,                 10, 20,
        3, 4,                 30, 40,

        5, 6,                 11, 21,
        7, 8,                 31, 41
        // clang-format on
    };

    return OptimiseAndRunNetwork(workloadFactory, *net, 0, inputInfo, inputData, 0, outputInfo, expectedOutputData);
}

LayerTestResult<uint8_t, 4>
    PreCompiledLeakyReluTest(armnn::IWorkloadFactory& workloadFactory,
                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const unsigned int inputSize  = 16;
    const unsigned int outputSize = 16;
    const unsigned int channels   = 1;
    const unsigned int kernelSize = 3;
    const unsigned int stride     = 1;
    const unsigned int padding    = 1;

    Convolution2dDescriptor convolutionDescriptor =
        CreateConvolutionDescriptor<Convolution2dDescriptor>(stride, padding);

    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::LeakyReLu;
    activationDescriptor.m_A        = 0.1f;
    activationDescriptor.m_B        = 0.0f;

    return PreCompiledFusedActivationTest(workloadFactory, memoryManager, inputSize, outputSize, channels, kernelSize,
                                          convolutionDescriptor, activationDescriptor);
}

LayerTestResult<uint8_t, 4> PreCompiledAdditionTest(armnn::IWorkloadFactory& workloadFactory,
                                                    const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
{
    armnn::INetworkPtr net = armnn::INetwork::Create();

    // Note the use of non-trivial quantization parameters to make sure that these are correctly passed to Ethos-N.
    TensorInfo inputInfo0({ 1, 2, 2, 1 }, DataType::QAsymmU8, 2.0f, 1);
    TensorInfo inputInfo1({ 1, 2, 2, 1 }, DataType::QAsymmU8, 4.0f, 1);
    TensorInfo outputInfo({ 1, 2, 2, 1 }, DataType::QAsymmU8, 0.2f, 2);

    IConnectableLayer* const inputLayer0 = net->AddInputLayer(0, "input0");
    inputLayer0->GetOutputSlot(0).SetTensorInfo(inputInfo0);
    IConnectableLayer* const inputLayer1 = net->AddInputLayer(1, "input1");
    inputLayer1->GetOutputSlot(0).SetTensorInfo(inputInfo1);

    IConnectableLayer* const additionLayer = net->AddAdditionLayer("addition");
    additionLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    inputLayer0->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(1));

    IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output");
    additionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    std::vector<uint8_t> inputData0{ 1, 2, 3, 4 };               // Dequantised: 0.0, 2.0, 4.0, 6.0
    std::vector<uint8_t> inputData1{ 1, 2, 3, 4 };               // Dequantised: 0.0, 4.0, 8.0, 12.0
    std::vector<uint8_t> expectedOutputData{ 2, 32, 62, 92 };    // Dequantised: 0.0, 6.0, 12.0, 18.0

    return OptimiseAndRunNetworkMultiple(workloadFactory, *net, { { 0, inputInfo0 }, { 1, inputInfo1 } },
                                         { { 0, inputData0 }, { 1, inputData1 } }, { { 0, outputInfo } },
                                         { { 0, expectedOutputData } })[0];
}

/// Checks the results from a 2-input network are correct.
/// The network topology is:
///
///   input0 -> relu0
///                    \'
///                      concat -> output
///                    /
///   input1 -> relu1
///
/// The two inputs are provided with different values, so the output
/// relies on the order of the inputs being correct.
LayerTestResult<uint8_t, 4> PreCompiledMultiInputTest(armnn::IWorkloadFactory& workloadFactory,
                                                      const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
{
    // Set up tensor infos
    TensorInfo inputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo intermediateInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo outputInfo({ 1, 8, 8, 32 }, DataType::QAsymmU8, 1.0f, 0);

    ActivationDescriptor reluDesc;
    reluDesc.m_Function = ActivationFunction::BoundedReLu;
    reluDesc.m_A        = 255.0f;
    reluDesc.m_B        = 0.0f;

    // Construct network
    armnn::INetworkPtr net               = armnn::INetwork::Create();
    IConnectableLayer* const input0Layer = net->AddInputLayer(0, "input0");
    input0Layer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    IConnectableLayer* const relu0Layer = net->AddActivationLayer(reluDesc, "relu0");
    relu0Layer->GetOutputSlot(0).SetTensorInfo(intermediateInfo);
    input0Layer->GetOutputSlot(0).Connect(relu0Layer->GetInputSlot(0));

    IConnectableLayer* const input1Layer = net->AddInputLayer(1, "input1");
    input1Layer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    IConnectableLayer* const relu1Layer = net->AddActivationLayer(reluDesc, "relu1");
    relu1Layer->GetOutputSlot(0).SetTensorInfo(intermediateInfo);
    input1Layer->GetOutputSlot(0).Connect(relu1Layer->GetInputSlot(0));

    std::array<TensorShape, 2> concatInputShapes = { intermediateInfo.GetShape(), intermediateInfo.GetShape() };
    IConnectableLayer* const concatLayer         = net->AddConcatLayer(
        CreateDescriptorForConcatenation(concatInputShapes.begin(), concatInputShapes.end(), 3), "concat");
    concatLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    relu0Layer->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
    relu1Layer->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(1));

    IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output");
    concatLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Use different input data for each input
    std::vector<uint8_t> inputData0(inputInfo.GetNumElements(), 64);
    std::vector<uint8_t> inputData1(inputInfo.GetNumElements(), 192);

    // Output data should be the inputs concatenated along the channels dimension
    std::vector<uint8_t> expectedOutputData;
    for (uint32_t i = 0; i < outputInfo.GetNumElements(); ++i)
    {
        expectedOutputData.push_back(i % 32 < 16 ? 64 : 192);
    }

    return OptimiseAndRunNetworkMultiple(workloadFactory, *net, { { 0, inputInfo }, { 1, inputInfo } },
                                         { { 0, inputData0 }, { 1, inputData1 } }, { { 0, outputInfo } },
                                         { { 0, expectedOutputData } })[0];
}

/// Checks the results from a 2-output network are correct.
/// The network topology is:
///
///   input0 -> relu0 -> relu1 -> output1
///                    \'
///                      -> output0
///
/// The two relus force their output to different specific values, so each output
/// should produce a tensor with the a value filled to all elements, but different for each output.
std::vector<LayerTestResult<uint8_t, 4>>
    PreCompiledMultiOutputTest(armnn::IWorkloadFactory& workloadFactory,
                               const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
{
    // Set up tensor infos
    TensorInfo inputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo intermediateInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo outputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);

    ActivationDescriptor reluDesc0;
    reluDesc0.m_Function = ActivationFunction::BoundedReLu;
    reluDesc0.m_A        = 64.0f;
    reluDesc0.m_B        = 64.0f;

    ActivationDescriptor reluDesc1;
    reluDesc1.m_Function = ActivationFunction::BoundedReLu;
    reluDesc1.m_A        = 192.0f;
    reluDesc1.m_B        = 192.0f;

    // Construct network
    armnn::INetworkPtr net               = armnn::INetwork::Create();
    IConnectableLayer* const input0Layer = net->AddInputLayer(0, "input0");
    input0Layer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    IConnectableLayer* const relu0Layer = net->AddActivationLayer(reluDesc0, "relu0");
    relu0Layer->GetOutputSlot(0).SetTensorInfo(intermediateInfo);
    input0Layer->GetOutputSlot(0).Connect(relu0Layer->GetInputSlot(0));
    IConnectableLayer* const relu1Layer = net->AddActivationLayer(reluDesc1, "relu1");
    relu1Layer->GetOutputSlot(0).SetTensorInfo(intermediateInfo);
    relu0Layer->GetOutputSlot(0).Connect(relu1Layer->GetInputSlot(0));

    IConnectableLayer* const output1Layer = net->AddOutputLayer(1, "output1");
    relu1Layer->GetOutputSlot(0).Connect(output1Layer->GetInputSlot(0));
    IConnectableLayer* const output0Layer = net->AddOutputLayer(0, "output0");
    relu0Layer->GetOutputSlot(0).Connect(output0Layer->GetInputSlot(0));

    // Input data is unimportant (as the relus will effectively overwrite the values)
    std::vector<uint8_t> inputData(inputInfo.GetNumElements(), 0);

    // Output data should be different for each output
    std::vector<uint8_t> expectedOutputData0(outputInfo.GetNumElements(), 64);
    std::vector<uint8_t> expectedOutputData1(outputInfo.GetNumElements(), 192);

    return OptimiseAndRunNetworkMultiple(workloadFactory, *net, { { 0, inputInfo } }, { { 0, inputData } },
                                         { { 0, outputInfo }, { 1, outputInfo } },
                                         { { 0, expectedOutputData0 }, { 1, expectedOutputData1 } });
}

/// Checks that a reshape to a 1D tensor is supported and ran by the Ethos-N.
LayerTestResult<uint8_t, 1> PreCompiled1dTensorTest(armnn::IWorkloadFactory& workloadFactory,
                                                    const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
{
    // Set up tensor infos
    TensorInfo inputInfo({ 1, 2, 2, 60 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo reluInfo({ 1, 2, 2, 60 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo reshapeInfo({ 240 }, DataType::QAsymmU8, 1.0f, 0);

    // Construct network
    armnn::INetworkPtr net              = armnn::INetwork::Create();
    IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    ActivationDescriptor reluDesc;
    reluDesc.m_Function                = ActivationFunction::BoundedReLu;
    reluDesc.m_A                       = 255.0f;
    reluDesc.m_B                       = 0.0f;
    IConnectableLayer* const reluLayer = net->AddActivationLayer(reluDesc, "relu");
    reluLayer->GetOutputSlot(0).SetTensorInfo(reluInfo);
    inputLayer->GetOutputSlot(0).Connect(reluLayer->GetInputSlot(0));

    IConnectableLayer* const reshapeLayer = net->AddReshapeLayer(ReshapeDescriptor(reshapeInfo.GetShape()), "reshape");
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapeInfo);
    reluLayer->GetOutputSlot(0).Connect(reshapeLayer->GetInputSlot(0));

    IConnectableLayer* const outputLayer = net->AddOutputLayer(1, "output");
    reshapeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Generate arbitrary input data
    std::vector<uint8_t> inputData(inputInfo.GetNumElements(), 0);
    std::iota(inputData.begin(), inputData.end(), 0);

    // Output data should be the the same as the input when expressed linearly as NHWC
    std::vector<uint8_t> expectedOutputData = inputData;

    return OptimiseAndRunNetwork<1>(workloadFactory, *net, 0, inputInfo, inputData, 1, reshapeInfo, expectedOutputData);
}

/// Checks that a reshape to a 2D tensor is supported and ran by the Ethos-N.
LayerTestResult<uint8_t, 2> PreCompiled2dTensorTest(armnn::IWorkloadFactory& workloadFactory,
                                                    const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
{
    // Set up tensor infos
    TensorInfo inputInfo({ 1, 2, 2, 60 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo reluInfo({ 1, 2, 2, 60 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo reshapeInfo({ 24, 10 }, DataType::QAsymmU8, 1.0f, 0);

    // Construct network
    armnn::INetworkPtr net              = armnn::INetwork::Create();
    IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    ActivationDescriptor reluDesc;
    reluDesc.m_Function                = ActivationFunction::BoundedReLu;
    reluDesc.m_A                       = 255.0f;
    reluDesc.m_B                       = 0.0f;
    IConnectableLayer* const reluLayer = net->AddActivationLayer(reluDesc, "relu");
    reluLayer->GetOutputSlot(0).SetTensorInfo(reluInfo);
    inputLayer->GetOutputSlot(0).Connect(reluLayer->GetInputSlot(0));

    IConnectableLayer* const reshapeLayer = net->AddReshapeLayer(ReshapeDescriptor(reshapeInfo.GetShape()), "reshape");
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapeInfo);
    reluLayer->GetOutputSlot(0).Connect(reshapeLayer->GetInputSlot(0));

    IConnectableLayer* const outputLayer = net->AddOutputLayer(1, "output");
    reshapeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Generate arbitrary input data
    std::vector<uint8_t> inputData(inputInfo.GetNumElements(), 0);
    std::iota(inputData.begin(), inputData.end(), 0);

    // Output data should be the the same as the input when expressed linearly as NHWC
    std::vector<uint8_t> expectedOutputData = inputData;

    return OptimiseAndRunNetwork<2>(workloadFactory, *net, 0, inputInfo, inputData, 1, reshapeInfo, expectedOutputData);
}

/// Checks that a reshape to a 3D tensor is supported and ran by the Ethos-N.
LayerTestResult<uint8_t, 3> PreCompiled3dTensorTest(armnn::IWorkloadFactory& workloadFactory,
                                                    const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
{
    // Set up tensor infos
    TensorInfo inputInfo({ 1, 2, 2, 60 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo reluInfo({ 1, 2, 2, 60 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo reshapeInfo({ 1, 24, 10 }, DataType::QAsymmU8, 1.0f, 0);

    // Construct network
    armnn::INetworkPtr net              = armnn::INetwork::Create();
    IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    ActivationDescriptor reluDesc;
    reluDesc.m_Function                = ActivationFunction::BoundedReLu;
    reluDesc.m_A                       = 255.0f;
    reluDesc.m_B                       = 0.0f;
    IConnectableLayer* const reluLayer = net->AddActivationLayer(reluDesc, "relu");
    reluLayer->GetOutputSlot(0).SetTensorInfo(reluInfo);
    inputLayer->GetOutputSlot(0).Connect(reluLayer->GetInputSlot(0));

    IConnectableLayer* const reshapeLayer = net->AddReshapeLayer(ReshapeDescriptor(reshapeInfo.GetShape()), "reshape");
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapeInfo);
    reluLayer->GetOutputSlot(0).Connect(reshapeLayer->GetInputSlot(0));

    IConnectableLayer* const outputLayer = net->AddOutputLayer(1, "output");
    reshapeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Generate arbitrary input data
    std::vector<uint8_t> inputData(inputInfo.GetNumElements(), 0);
    std::iota(inputData.begin(), inputData.end(), 0);

    // Output data should be the the same as the input when expressed linearly as NHWC
    std::vector<uint8_t> expectedOutputData = inputData;

    return OptimiseAndRunNetwork<3>(workloadFactory, *net, 0, inputInfo, inputData, 1, reshapeInfo, expectedOutputData);
}

/// Checks that the backend optimization substituting the Constant-Multiplication layer
/// pattern with a DepthwiseConvolution2d will produce correct results when ran by the Ethos-N.
LayerTestResult<uint8_t, 4> PreCompiledConstMulToDepthwiseTest(armnn::IWorkloadFactory& workloadFactory,
                                                               const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
{
    // Set up tensor infos
    TensorInfo inputInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo constInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 0.5f, 0);
    TensorInfo outputInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0);

    std::vector<uint8_t> inputData{
        // clang-format off
        1, 2, 3, 4,           10, 20, 15, 30,
        8, 6, 5, 4,           11, 21, 31, 41,
        // clang-format on
    };

    std::vector<uint8_t> constData{ 5, 8, 2, 6 };

    ConstTensor constantTensor{ constInfo, constData };

    // Construct a network with the Constant-Multiplication pattern
    armnn::INetworkPtr net = armnn::INetwork::Create();

    IConnectableLayer* const inputLayer  = net->AddInputLayer(0, "input");
    IConnectableLayer* const constLayer  = net->AddConstantLayer(constantTensor);
    IConnectableLayer* const mulLayer    = net->AddMultiplicationLayer("multiplication");
    IConnectableLayer* const outputLayer = net->AddOutputLayer(1, "output");

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    constLayer->GetOutputSlot(0).SetTensorInfo(constInfo);
    mulLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    inputLayer->GetOutputSlot(0).Connect(mulLayer->GetInputSlot(0));
    constLayer->GetOutputSlot(0).Connect(mulLayer->GetInputSlot(1));
    mulLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    std::vector<uint8_t> expectedOutputData{
        // clang-format off
        3, 8, 3, 12,          25, 80, 15, 90,
        20, 24, 5, 12,        28, 84, 31, 123,
        // clang-format on
    };

    return OptimiseAndRunNetwork(workloadFactory, *net, 0, inputInfo, inputData, 1, outputInfo, expectedOutputData);
}

BOOST_AUTO_TEST_SUITE(Compute_EthosN)

using FactoryType = armnn::EthosNWorkloadFactory;

ARMNN_AUTO_TEST_CASE(PreCompiledActivationRelu, PreCompiledActivationReluTest)
ARMNN_AUTO_TEST_CASE(PreCompiledActivationRelu1, PreCompiledActivationRelu1Test)
ARMNN_AUTO_TEST_CASE(PreCompiledActivationRelu6, PreCompiledActivationRelu6Test)

ARMNN_AUTO_TEST_CASE(PreCompiledConvolution2d, PreCompiledConvolution2dTest)
ARMNN_AUTO_TEST_CASE(PreCompiledConvolution2dStride2x2, PreCompiledConvolution2dStride2x2Test)

ARMNN_AUTO_TEST_CASE(PreCompiledDepthwiseConvolution2d, PreCompiledDepthwiseConvolution2dTest)
ARMNN_AUTO_TEST_CASE(PreCompiledDepthwiseConvolution2dStride2x2, PreCompiledDepthwiseConvolution2dStride2x2Test)

ARMNN_AUTO_TEST_CASE(PreCompiledTransposeConvolution2dStride2x2, PreCompiledTransposeConvolution2dStride2x2Test)

ARMNN_AUTO_TEST_CASE(PreCompiledConvolution2dWithAssymetricSignedWeights,
                     PreCompiledConvolution2dWithAssymetricSignedWeightsTest)

ARMNN_AUTO_TEST_CASE(PreCompiledConvolution2dWithSymetricSignedWeights,
                     PreCompiledConvolution2dWithSymetricSignedWeightsTest)

ARMNN_AUTO_TEST_CASE(PreCompiledFullyConnected, PreCompiledFullyConnectedTest, TensorShape{ 1, 8 })
ARMNN_AUTO_TEST_CASE(PreCompiledFullyConnected4d, PreCompiledFullyConnectedTest, TensorShape{ 1, 2, 2, 3 })

ARMNN_AUTO_TEST_CASE(PreCompiledMaxPooling2d, PreCompiledMaxPooling2dTest)

ARMNN_AUTO_TEST_CASE(PreCompiledMeanXy, PreCompiledMeanXyTest)

ARMNN_AUTO_TEST_CASE(PreCompiledSplitter, PreCompiledSplitterTest)

ARMNN_AUTO_TEST_CASE(PreCompiledDepthToSpace, PreCompiledDepthToSpaceTest)

ARMNN_AUTO_TEST_CASE(PreCompiledLeakyRelu, PreCompiledLeakyReluTest)

ARMNN_AUTO_TEST_CASE(PreCompiledAddition, PreCompiledAdditionTest)

ARMNN_AUTO_TEST_CASE(PreCompiledMultiInput, PreCompiledMultiInputTest)
ARMNN_AUTO_TEST_CASE(PreCompiledMultiOutput, PreCompiledMultiOutputTest)

ARMNN_AUTO_TEST_CASE(PreCompiled1dTensor, PreCompiled1dTensorTest)
ARMNN_AUTO_TEST_CASE(PreCompiled2dTensor, PreCompiled2dTensorTest)
ARMNN_AUTO_TEST_CASE(PreCompiled3dTensor, PreCompiled3dTensorTest)

ARMNN_AUTO_TEST_CASE(PreCompiledConstMulToDepthwise, PreCompiledConstMulToDepthwiseTest)

BOOST_AUTO_TEST_CASE(TestInvalidLayerName)
{
    BOOST_CHECK_THROW(armnn::ethosnbackend::GetLayerType("Excluded"), armnn::InvalidArgumentException);

    try
    {
        armnn::ethosnbackend::GetLayerType("Excluded");
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        std::string err = "layername \"Excluded\" is not valid";
        BOOST_CHECK_EQUAL(err, e.what());
    }
}

BOOST_AUTO_TEST_SUITE_END()
