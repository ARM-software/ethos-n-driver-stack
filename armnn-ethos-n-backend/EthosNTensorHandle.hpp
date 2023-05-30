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

#include <fmt/format.h>
namespace armnn
{

// Abstract tensor handles wrapping a Ethos-N readable region of memory, interpreting it as tensor data.
class EthosNBaseTensorHandle : public ITensorHandle
{
public:
    EthosNBaseTensorHandle(const TensorInfo& tensorInfo, const std::string& deviceId)
        : m_TensorInfo(tensorInfo)
        , m_DeviceId(deviceId)
        , m_Buffer(nullptr)
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
    virtual ~EthosNBaseTensorHandle()
    {}

    virtual void Manage() override
    {}

    virtual void Allocate() override
    {}

    const void* Map(bool) const override
    {
        if (!m_Buffer)
        {
            throw NullPointerException(fmt::format("{}: Buffer pointer is null", __func__));
        }
        return static_cast<const void*>(m_Buffer->Map());
    }

    void Unmap() const override
    {
        if (!m_Buffer)
        {
            throw NullPointerException(fmt::format("{}: Buffer pointer is null", __func__));
        }
        m_Buffer->Unmap();
    }

    ethosn::driver_library::Buffer& GetBuffer()
    {
        if (!m_Buffer)
        {
            throw NullPointerException(fmt::format("{}: Buffer pointer is null", __func__));
        }
        return *m_Buffer;
    }

    ethosn::driver_library::Buffer const& GetBuffer() const
    {
        if (!m_Buffer)
        {
            throw NullPointerException(fmt::format("{}: Buffer pointer is null", __func__));
        }
        return *m_Buffer;
    }

    bool CanBeImported(void* memory, MemorySource source) override
    {
        return (memory && CheckFlag(GetImportFlags(), source));
    };

    /// Unimport externally allocated memory
    void Unimport() override
    {
        // According to Arm NN Unimport is considered a no-op for non-existing buffers
        if (!m_Buffer)
        {
            return;
        }
        m_Buffer.reset();
    };

    bool Import(void* memory, MemorySource source) override
    {
        auto& procMemAllocator = armnn::EthosNBackendAllocatorService::GetInstance().GetProcMemAllocator(m_DeviceId);

        if (!memory)
        {
            throw NullPointerException("Import from invalid memory");
        }
        // The driver library only currently works with dma buf sources.
        if (!CanBeImported(memory, source))
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
    EthosNBaseTensorHandle(const EthosNBaseTensorHandle& other) = delete;
    EthosNBaseTensorHandle& operator=(const EthosNBaseTensorHandle& other) = delete;

    TensorInfo m_TensorInfo;

protected:
    std::string m_DeviceId;

    std::unique_ptr<ethosn::driver_library::Buffer> m_Buffer;
};

// Tensor handle wrapping a Ethos-N readable region of memory, interpreting it as tensor data.
class EthosNTensorHandle : public EthosNBaseTensorHandle
{
public:
    EthosNTensorHandle(const TensorInfo& tensorInfo, const std::string& deviceId)
        : EthosNBaseTensorHandle(tensorInfo, deviceId)
    {
        m_Buffer = CreateBuffer(tensorInfo, deviceId);
    }

    MemorySourceFlags GetImportFlags() const override
    {
        return static_cast<MemorySourceFlags>(armnn::MemorySource::DmaBuf);
    }

private:
    EthosNTensorHandle(const EthosNTensorHandle& other) = delete;
    EthosNTensorHandle& operator=(const EthosNTensorHandle& other) = delete;

    std::unique_ptr<ethosn::driver_library::Buffer> CreateBuffer(const TensorInfo& tensorInfo,
                                                                 const std::string& deviceId)
    {
        using ethosn::driver_library::Buffer;

        auto& procMemAllocator = armnn::EthosNBackendAllocatorService::GetInstance().GetProcMemAllocator(deviceId);
        if (procMemAllocator.GetProtected())
        {
            throw RuntimeException("Backend does not support CreateBuffer in protected mode");
        }

        // The input buffer size of fully connected is rounded up to the next 1024
        // byte boundary by the support library. The backend needs to do
        // the same to avoid buffer size mismatch.
        uint32_t bufferSize =
            armnn::ethosnbackend::RoundUpToNearestMultiple(tensorInfo.GetNumElements(), static_cast<uint32_t>(1024));

        return std::make_unique<Buffer>(procMemAllocator.CreateBuffer(bufferSize));
    }
};

// Tensor handle wrapping a Ethos-N readable region of memory, interpreting it as tensor data.
class EthosNProtectedTensorHandle : public EthosNBaseTensorHandle
{
public:
    EthosNProtectedTensorHandle(const TensorInfo& tensorInfo, const std::string& deviceId)
        : EthosNBaseTensorHandle(tensorInfo, deviceId)
    {}

    MemorySourceFlags GetImportFlags() const override
    {
        return static_cast<MemorySourceFlags>(armnn::MemorySource::DmaBufProtected);
    }

    void CopyOutTo(void* memory) const override
    {
        IgnoreUnused(memory);
        throw RuntimeException(fmt::format("{} not allowed in protected mode", __func__));
    }

    void CopyInFrom(const void* memory) override
    {
        IgnoreUnused(memory);
        throw RuntimeException(fmt::format("{} not allowed in protected mode", __func__));
    }

    const void* Map(bool) const override
    {
        ARMNN_LOG(info) << __func__ << " not allowed in protected mode";
        return nullptr;
    }

    void Unmap() const override
    {
        ARMNN_LOG(info) << __func__ << " not allowed in protected mode";
    }

private:
    EthosNProtectedTensorHandle(const EthosNProtectedTensorHandle& other) = delete;
    EthosNProtectedTensorHandle& operator=(const EthosNProtectedTensorHandle& other) = delete;
};

}    // namespace armnn
