//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ArmnnUtils.hpp"

#include "GlobalParameters.hpp"
#include "SystemTestsUtils.hpp"

#include "../profiling/common/include/ProfilingGuid.hpp"
#include "ProtectedAllocator.hpp"
#include <armnn/ArmNN.hpp>
#include <armnn/IProfiler.hpp>
#include <armnn/backends/ITensorHandle.hpp>
#include <ethosn_utils/Strings.hpp>

namespace ethosn
{
namespace system_tests
{

using namespace arm::pipe;

namespace
{

/// Visitor for armnn::INetwork which gathers a map of layer GUID -> IConnectableLayer.
/// We use this to gather information for layers in the debug callback, as some information is not directly available
/// in the callback, and there is no Arm NN API to lookup a layer by its guid.
class ArmnnLayerVisitor : public armnn::IStrategy
{
public:
    const std::map<LayerGuid, const armnn::IConnectableLayer*>& GetLayerMap()
    {
        return m_Layers;
    }

    // Interface implementation
public:
    virtual void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                                 const armnn::BaseDescriptor&,
                                 const std::vector<armnn::ConstTensor>&,
                                 const char*,
                                 const armnn::LayerBindingId)
    {
        m_Layers[layer->GetGuid()] = layer;
    }

    std::map<LayerGuid, const armnn::IConnectableLayer*> m_Layers;
};

class ArmnnLogSink : public armnn::LogSink
{
public:
    void Consume(const std::string& s) override
    {
        std::cout << "[Arm NN] " << s << std::endl;
    }
};

}    // namespace

void ConfigureArmnnLogging()
{
    using namespace armnn;

    // Configure Arm NN's logging. Pass this through a custom LogSink so that
    // we can prepend Arm NN's log messages to indicate that they are from Arm NN
    // (and not other other parts of the driver stack)
    LogSeverity logSeverity = LogSeverity::Warning;
    if (g_Debug.find("armnn-logging=Fatal") != std::string::npos)
    {
        logSeverity = LogSeverity::Fatal;
    }
    else if (g_Debug.find("armnn-logging=Error") != std::string::npos)
    {
        logSeverity = LogSeverity::Error;
    }
    else if (g_Debug.find("armnn-logging=Warning") != std::string::npos)
    {
        logSeverity = LogSeverity::Warning;
    }
    else if (g_Debug.find("armnn-logging=Info") != std::string::npos)
    {
        logSeverity = LogSeverity::Info;
    }
    else if (g_Debug.find("armnn-logging=Debug") != std::string::npos)
    {
        logSeverity = LogSeverity::Debug;
    }
    else if (g_Debug.find("armnn-logging=Trace") != std::string::npos)
    {
        logSeverity = LogSeverity::Trace;
    }
    armnn::SetLogFilter(logSeverity);
    std::shared_ptr<ArmnnLogSink> sink = std::make_shared<ArmnnLogSink>();
    armnn::SimpleLogger<LogSeverity::Fatal>::Get().RemoveAllSinks();
    armnn::SimpleLogger<LogSeverity::Fatal>::Get().AddSink(sink);
    armnn::SimpleLogger<LogSeverity::Error>::Get().RemoveAllSinks();
    armnn::SimpleLogger<LogSeverity::Error>::Get().AddSink(sink);
    armnn::SimpleLogger<LogSeverity::Warning>::Get().RemoveAllSinks();
    armnn::SimpleLogger<LogSeverity::Warning>::Get().AddSink(sink);
    armnn::SimpleLogger<LogSeverity::Info>::Get().RemoveAllSinks();
    armnn::SimpleLogger<LogSeverity::Info>::Get().AddSink(sink);
    armnn::SimpleLogger<LogSeverity::Debug>::Get().RemoveAllSinks();
    armnn::SimpleLogger<LogSeverity::Debug>::Get().AddSink(sink);
    armnn::SimpleLogger<LogSeverity::Trace>::Get().RemoveAllSinks();
    armnn::SimpleLogger<LogSeverity::Trace>::Get().AddSink(sink);
}

InferenceOutputs ArmnnRunNetwork(armnn::INetwork* network,
                                 const std::vector<armnn::BackendId>& devices,
                                 const std::vector<armnn::LayerBindingId>& inputBindings,
                                 const std::vector<armnn::LayerBindingId>& outputBindings,
                                 const InferenceInputs& inputData,
                                 const std::vector<armnn::BackendOptions>& backendOptions,
                                 const char* dmaBufHeapDevFilename,
                                 bool runProtected,
                                 size_t numInferences)
{
    using namespace armnn;

    NetworkId networkIdentifier;

    // Create runtime
    IRuntime::CreationOptions options;
    options.m_BackendOptions                    = backendOptions;
    std::string id                              = "EthosNAcc";
    armnn::ICustomAllocator* customAllocatorRef = nullptr;
    options.m_ProtectedMode                     = runProtected;
    if (dmaBufHeapDevFilename != nullptr)
    {
        if (options.m_ProtectedMode)
        {
            auto customAllocator         = std::make_shared<ProtectedAllocator>();
            options.m_CustomAllocatorMap = { { id, std::move(customAllocator) } };
            customAllocatorRef           = static_cast<ProtectedAllocator*>(options.m_CustomAllocatorMap[id].get());
        }
        else
        {
            auto customAllocator         = std::make_shared<CustomAllocator>();
            options.m_CustomAllocatorMap = { { id, std::move(customAllocator) } };
            customAllocatorRef           = static_cast<CustomAllocator*>(options.m_CustomAllocatorMap[id].get());
        }
    }

    IRuntimePtr run = IRuntime::Create(options);

    // Include the backend(s) in the dump name, as we may be running armnn twice in the same test - once
    // for reference and once for ethosn.
    const std::string backends = ethosn::utils::Join("+", devices, [](auto x) { return x; });
    // Enabling this will dump Arm NN's output after each layer - useful for debugging.
    const bool dumpArmnnOutput = (g_Debug.find("dump-armnn-tensors") != std::string::npos);

    OptimizerOptionsOpaque optOpts(false, dumpArmnnOutput);
    optOpts.SetImportEnabled(dmaBufHeapDevFilename != nullptr);
    optOpts.SetExportEnabled(dmaBufHeapDevFilename != nullptr);
    optOpts.SetShapeInferenceMethod(armnn::ShapeInferenceMethod::InferAndValidate);
    for (auto& option : backendOptions)
    {
        optOpts.AddModelOption(option);
    }
    std::vector<std::string> errorsAndWarnings;
    IOptimizedNetworkPtr optNet(nullptr, nullptr);
    try
    {
        optNet = Optimize(*network, devices, run->GetDeviceSpec(), optOpts, errorsAndWarnings);
    }
    catch (const armnn::Exception& e)
    {
        std::cout << "Arm NN exception: " << e.what() << std::endl;
    }
    for (const std::string& msg : errorsAndWarnings)
    {
        std::cout << "Arm NN warning/error: " << msg << std::endl;
    }
    if (!optNet)
    {
        throw std::runtime_error("Arm NN failed to optimize network");
    }

    if (g_Debug.find("dump-armnn-graph") != std::string::npos)
    {
        std::ofstream f(std::string("Armnn_") + backends + "_OptimisedGraph.dot");
        optNet->SerializeToDot(f);
    }

    // Load graph into runtime
    INetworkProperties networkProperties(
        false, customAllocatorRef != nullptr ? customAllocatorRef->GetMemorySourceType() : MemorySource::Undefined,
        customAllocatorRef != nullptr ? customAllocatorRef->GetMemorySourceType() : MemorySource::Undefined);
    std::string errMsgs;
    Status status = run->LoadNetwork(networkIdentifier, std::move(optNet), errMsgs, networkProperties);
    if (status != Status::Success)
    {
        g_Logger.Error("%s", errMsgs.c_str());
        throw std::runtime_error("Arm NN failed to load network");
    }

    // Enable profiling, if requested
    std::shared_ptr<IProfiler> profiler;
    if (g_Debug.find("armnn-profiling") != std::string::npos)
    {
        profiler = run->GetProfiler(networkIdentifier);
        profiler->EnableProfiling(true);
    }

    // Register callback to save the output of each Arm NN layer - see 'dumpArmnnOutput' flag above
    ArmnnLayerVisitor layerVisitor;    // Note this must live until the end of the inference (due to the callback)
    const std::map<LayerGuid, const armnn::IConnectableLayer*>& layers = layerVisitor.GetLayerMap();
    if (dumpArmnnOutput)
    {
        // Gather the map of layer GUID -> IConnectableLayer, to look up information in the callback
        network->ExecuteStrategy(layerVisitor);

        const DebugCallbackFunction callback = [&](LayerGuid guid, unsigned int slotIndex,
                                                   const ITensorHandle* tensorHandle) {
            std::string filename;
            {
                std::string layerNameSafe = "NONAME";
                const char* dataTypeName  = "UNKNOWN";
                {
                    const auto layerIt = layers.find(guid);

                    // Layer may not be found if this was added to the graph as part of optimisation (e.g. mem copy)
                    if (layerIt != layers.end())
                    {
                        const armnn::IConnectableLayer* const layer = layerIt->second;
                        assert(layer != nullptr);

                        // Make the layer name into something that is safe as a filename
                        layerNameSafe = utils::ReplaceAll(layer->GetName(), ":", "-");
                        layerNameSafe = utils::ReplaceAll(layerNameSafe, "/", "-");
                        dataTypeName  = GetDataTypeName(layer->GetOutputSlot(slotIndex).GetTensorInfo().GetDataType());
                    }
                }

                std::stringstream ss;

                ss << "Armnn_" << backends;
                // Pad the buffer ID for easy sorting of dumped file names
                ss << "_Tensor_Layer" << std::setfill('0') << std::setw(3) << guid << std::setw(0);
                ss << "_" << layerNameSafe << "_Slot" << slotIndex;
                ss << "_" << dataTypeName;
                for (unsigned int i = 0; i < tensorHandle->GetShape().GetNumDimensions(); ++i)
                {
                    ss << "_" << tensorHandle->GetShape()[i];
                }
                ss << ".hex";

                filename = ss.str();
            }

            const auto data     = static_cast<const uint8_t*>(tensorHandle->Map());
            const uint32_t size = tensorHandle->GetStrides()[0];

            {
                std::ofstream fs(filename.c_str());
                WriteHex(fs, 0, data, size);
            }

            tensorHandle->Unmap();

            g_Logger.Debug("Dumped Arm NN intermediate tensor to %s", filename.c_str());
        };

        run->RegisterDebugCallback(networkIdentifier, callback);
    }

    std::unique_ptr<DmaBufferDevice> dmaBufHeap;
    if (dmaBufHeapDevFilename != nullptr)
    {
        dmaBufHeap = std::make_unique<DmaBufferDevice>(dmaBufHeapDevFilename);
    }

    InputTensors inputTensors;
    std::vector<DmaBuffer> inputDmaBuffers(inputData.size());
    std::vector<int> inputDmaBufFds(inputData.size());
    {
        auto bindingIt = inputBindings.begin();
        for (size_t i = 0; i < inputData.size(); ++i)
        {
            TensorInfo tensorInfo = run->GetInputTensorInfo(networkIdentifier, *bindingIt);
            tensorInfo.SetConstant();
            void* memOrFd;
            if (dmaBufHeap)
            {
                DmaBuffer dmaBuf(dmaBufHeap, inputData[i]->GetNumBytes());
                dmaBuf.PopulateData(const_cast<uint8_t*>(inputData[i]->GetByteData()), inputData[i]->GetNumBytes());
                inputDmaBufFds[i]  = dmaBuf.GetFd();
                inputDmaBuffers[i] = std::move(dmaBuf);

                memOrFd = &inputDmaBufFds[i];
            }
            else
            {
                memOrFd = inputData[i]->GetByteData();
            }
            ConstTensor constTensor = ConstTensor(tensorInfo, memOrFd);
            inputTensors.emplace_back(*bindingIt, constTensor);
            ++bindingIt;
        }
    }

    InferenceOutputs outputData(outputBindings.size());
    std::vector<DmaBuffer> outputDmaBuffers(outputData.size());
    std::vector<int> outputDmaBufFds(outputData.size());
    OutputTensors outputTensors;
    {
        for (size_t i = 0; i < outputData.size(); ++i)
        {
            TensorInfo tensorInfo = run->GetOutputTensorInfo(networkIdentifier, outputBindings[i]);
            void* memOrFd;
            outputData[i] = MakeTensor(tensorInfo);
            if (dmaBufHeap)
            {
                DmaBuffer dmaBuf(dmaBufHeap, tensorInfo.GetNumBytes());
                outputDmaBufFds[i]  = dmaBuf.GetFd();
                outputDmaBuffers[i] = std::move(dmaBuf);

                memOrFd = &outputDmaBufFds[i];
            }
            else
            {
                memOrFd = outputData[i]->GetByteData();
            }
            armnn::Tensor tensor =
                armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, outputBindings[i]), memOrFd);
            outputTensors.emplace_back(outputBindings[i], tensor);
        }
    }

    // Execute network, potentially multiple times if requested
    for (size_t k = 0; k < numInferences; ++k)
    {
        status = run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
        if (status != Status::Success)
        {
            throw std::runtime_error("Arm NN failed to enqueue workload");
        }
    }
    run->UnloadNetwork(networkIdentifier);

    if (dmaBufHeap)
    {
        for (size_t i = 0; i < outputData.size(); ++i)
        {
            outputDmaBuffers[i].RetrieveData(outputData[i]->GetByteData(), outputData[i]->GetNumBytes());
        }
    }

    // Dump profiling JSON file, if enabled
    if (profiler)
    {
        std::ofstream f(std::string("Armnn_") + backends + "_Profiling.json");
        profiler->Print(f);
    }

    return outputData;
}

ethosn::system_tests::OwnedTensor MakeTensor(const armnn::TensorInfo& t)
{
    return MakeTensor(GetDataType(t.GetDataType()), t.GetNumElements());
}

void* CustomAllocator::allocate(size_t size, size_t alignment)
{
    // This function implementation does not support alignment
    armnn::IgnoreUnused(alignment);
    std::unique_ptr<DmaBuffer> dataDmaBuf = std::make_unique<DmaBuffer>(m_DmaBufHeap, size);
    int fd                                = dataDmaBuf->GetFd();
    if (fd < 0)
        throw std::runtime_error("Arm NN failed to allocate intermediate buffer");
    m_Map[fd].m_DataDmaBuf = std::move(dataDmaBuf);
    m_Map[fd].m_Fd         = fd;
    return static_cast<void*>(&m_Map[fd].m_Fd);
}

void CustomAllocator::free(void* ptr)
{
    int index = *static_cast<int*>(ptr);
    m_Map[index].m_DataDmaBuf.reset();
    m_Map.erase(index);
}

armnn::MemorySource CustomAllocator::GetMemorySourceType()
{
    return armnn::MemorySource::DmaBuf;
}

}    // namespace system_tests
}    // namespace ethosn
