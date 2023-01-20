//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNBackend.hpp"

#include "EthosNTensorUtils.hpp"
#include "EthosNWorkloadUtils.hpp"

#include "armnn/Exceptions.hpp"
#include "armnn/TypesUtils.hpp"
#include "armnn/backends/ITensorHandle.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnnUtils/CompatibleTypes.hpp>
#include <ethosn_driver_library/Buffer.hpp>
#include <ethosn_driver_library/ProcMemAllocator.hpp>

namespace armnn
{

// Tensor handle wrapping a Ethos-N readable region of memory, interpreting it as tensor data.
class EthosNTensorHandle : public ITensorHandle
{
public:
    EthosNTensorHandle(const TensorInfo& tensorInfo, const std::string& deviceId)
        : m_TensorInfo(tensorInfo)
        , m_DeviceId(deviceId)
        , m_Buffer(CreateBuffer(tensorInfo, deviceId))
    {
        using namespace ethosntensorutils;
        // NOTE: The Ethos-N API is unclear on whether the size specified for a Buffer is the number of elements, or
        //       the number of bytes; this can be ignored for now, as the only supported data types are QAsymmU8,
        //       QAsymmS8 and QSymmS8.
        // NOTE: The only supported DataFormat is NHWC.
        // NOTE: The DataFormat parameter is unused and may be removed in a future Ethos-N version.
        if (!IsDataTypeSupportedOnEthosN(tensorInfo.GetDataType()))
        {
            throw InvalidArgumentException(std::string("Unsupported data type ") +
                                               std::string(GetDataTypeName(tensorInfo.GetDataType())),
                                           CHECK_LOCATION());
        }
    }

    virtual ~EthosNTensorHandle()
    {}

    virtual void Manage() override
    {}

    virtual void Allocate() override
    {}

    virtual ITensorHandle* GetParent() const override
    {
        return nullptr;
    }

    TensorShape GetStrides() const override
    {
        TensorShape shape(m_TensorInfo.GetShape());
        auto numDims = shape.GetNumDimensions();
        std::vector<unsigned int> strides(numDims);
        strides[numDims - 1] = GetDataTypeSize(m_TensorInfo.GetDataType());
        for (unsigned int i = numDims - 1; i > 0; --i)
        {
            strides[i - 1] = strides[i] * shape[i];
        }
        return TensorShape(numDims, strides.data());
    }

    TensorShape GetShape() const override
    {
        return m_TensorInfo.GetShape();
    }
    const TensorInfo& GetTensorInfo() const
    {
        return m_TensorInfo;
    }

    const void* Map(bool) const override
    {
        return static_cast<const void*>(m_Buffer->Map());
    }

    void Unmap() const override
    {
        m_Buffer->Unmap();
    }

    ethosn::driver_library::Buffer& GetBuffer()
    {
        return *m_Buffer;
    }

    ethosn::driver_library::Buffer const& GetBuffer() const
    {
        return *m_Buffer;
    }

    MemorySourceFlags GetImportFlags() const override
    {
        return static_cast<MemorySourceFlags>(armnn::MemorySource::DmaBuf);
    }

    bool Import(void* memory, MemorySource source) override
    {
        auto& procMemAllocator = armnn::EthosNBackendAllocatorService::GetInstance().GetProcMemAllocator(m_DeviceId);

        // The driver library only currently works with dma buf sources.
        if (source != MemorySource::DmaBuf)
        {
            return false;
        }
        // The driver library expects this to be a file descriptor.
        // Assume that this is a pointer to a file descriptor which is just an int.
        int fd = *reinterpret_cast<int*>(memory);

        // The input buffer size of fully connected is rounded up to the next 1024
        // byte boundary by the support library. The backend needs to do
        // the same to avoid buffer size mismatch.
        uint32_t bufferSize = armnn::ethosnbackend::RoundUpToNearestMultiple(GetTensorInfo().GetNumElements(),
                                                                             static_cast<uint32_t>(1024));

        m_Buffer = std::make_unique<ethosn::driver_library::Buffer>(procMemAllocator.ImportBuffer(fd, bufferSize));
        return true;
    };

    bool CanBeImported(void* memory, MemorySource source) override
    {
        IgnoreUnused(memory);
        return source == MemorySource::DmaBuf;
    };

    /// Unimport externally allocated memory
    void Unimport() override
    {
        m_Buffer.reset();
    };

    void CopyOutTo(void* memory) const override
    {
        const void* data = Map(true);
        memcpy(memory, data, GetTensorInfo().GetNumBytes());
        Unmap();
    }

    void CopyInFrom(const void* memory) override
    {
        void* data = ITensorHandle::Map(true);
        memcpy(data, memory, GetTensorInfo().GetNumBytes());
        Unmap();
    }

private:
    EthosNTensorHandle(const EthosNTensorHandle& other) = delete;
    EthosNTensorHandle& operator=(const EthosNTensorHandle& other) = delete;

    std::unique_ptr<ethosn::driver_library::Buffer> CreateBuffer(const TensorInfo& tensorInfo,
                                                                 const std::string& deviceId)
    {
        using ethosn::driver_library::Buffer;

        auto& procMemAllocator = armnn::EthosNBackendAllocatorService::GetInstance().GetProcMemAllocator(deviceId);

        // The input buffer size of fully connected is rounded up to the next 1024
        // byte boundary by the support library. The backend needs to do
        // the same to avoid buffer size mismatch.
        uint32_t bufferSize =
            armnn::ethosnbackend::RoundUpToNearestMultiple(tensorInfo.GetNumElements(), static_cast<uint32_t>(1024));

        return std::make_unique<Buffer>(
            procMemAllocator.CreateBuffer(bufferSize, ethosn::driver_library::DataFormat::NHWC));
    }

    TensorInfo m_TensorInfo;
    std::string m_DeviceId;
    std::unique_ptr<ethosn::driver_library::Buffer> m_Buffer;
};

}    // namespace armnn
