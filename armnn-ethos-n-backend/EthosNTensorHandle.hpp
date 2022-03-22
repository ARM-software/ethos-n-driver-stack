//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNTensorUtils.hpp"
#include "EthosNWorkloadUtils.hpp"

#include "armnn/Exceptions.hpp"
#include "armnn/TypesUtils.hpp"
#include "armnn/backends/ITensorHandle.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnnUtils/CompatibleTypes.hpp>
#include <ethosn_driver_library/Buffer.hpp>

namespace armnn
{

// Abstract tensor handles wrapping a Ethos-N readable region of memory, interpreting it as tensor data.
class EthosNTensorHandle : public ITensorHandle
{
public:
    explicit EthosNTensorHandle(const TensorInfo& tensorInfo)
        : EthosNTensorHandle(tensorInfo, {})
    {}

    explicit EthosNTensorHandle(const TensorInfo& tensorInfo, const std::string& deviceId)
        : m_TensorInfo(tensorInfo)
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

    ethosn::driver_library::Buffer CreateBuffer(const TensorInfo& tensorInfo, const std::string& deviceId)
    {
        // The input buffer size of fully connected is rounded up to the next 1024
        // byte boundary by the support library. The backend needs to do
        // the same to avoid buffer size mismatch.
        uint32_t bufferSize =
            armnn::ethosnbackend::RoundUpToNearestMultiple(tensorInfo.GetNumElements(), static_cast<uint32_t>(1024));

        if (deviceId.empty())
        {
            return ethosn::driver_library::Buffer(bufferSize, ethosn::driver_library::DataFormat::NHWC);
        }
        else
        {
            return ethosn::driver_library::Buffer(bufferSize, ethosn::driver_library::DataFormat::NHWC, deviceId);
        }
    }

    virtual void Manage() override
    {}
    virtual void Allocate() override
    {}

    virtual ITensorHandle* GetParent() const override
    {
        return nullptr;
    }

    virtual const void* Map(bool /* blocking = true */) const override
    {
        return static_cast<const void*>(m_Buffer.Map());
    }

    virtual void Unmap() const override
    {
        m_Buffer.Unmap();
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

    ethosn::driver_library::Buffer& GetBuffer()
    {
        return m_Buffer;
    }
    ethosn::driver_library::Buffer const& GetBuffer() const
    {
        return m_Buffer;
    }

    void CopyOutTo(void* memory) const override
    {
        const uint8_t* data = m_Buffer.Map();
        memcpy(memory, data, GetTensorInfo().GetNumBytes());
        m_Buffer.Unmap();
    }

    void CopyInFrom(const void* memory) override
    {
        uint8_t* data = m_Buffer.Map();
        memcpy(data, memory, GetTensorInfo().GetNumBytes());
        m_Buffer.Unmap();
    }

private:
    EthosNTensorHandle(const EthosNTensorHandle& other) = delete;
    EthosNTensorHandle& operator=(const EthosNTensorHandle& other) = delete;

    TensorInfo m_TensorInfo;
    mutable ethosn::driver_library::Buffer m_Buffer;
};

// Abstract tensor handles wrapping a Ethos-N readable region of memory, interpreting it as tensor data.
class EthosNImportTensorHandle : public ITensorHandle
{
public:
    explicit EthosNImportTensorHandle(const TensorInfo& tensorInfo, MemorySourceFlags importFlags)
        : EthosNImportTensorHandle(tensorInfo, {}, importFlags)
    {}

    explicit EthosNImportTensorHandle(const TensorInfo& tensorInfo,
                                      const std::string& deviceId,
                                      MemorySourceFlags importFlags)
        : m_TensorInfo(tensorInfo)
        , m_ImportFlags(importFlags)
        , m_DeviceId(deviceId)
        , m_Buffer(nullptr)
    {
        using namespace ethosntensorutils;
        // NOTE: The Support Library API is unclear on whether the size specified for a Buffer is the number of elements, or
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

    void Manage() override
    {}
    void Allocate() override
    {}

    ITensorHandle* GetParent() const override
    {
        return nullptr;
    }

    const void* Map(bool /* blocking = true */) const override
    {
        return static_cast<const void*>(m_Buffer->Map());
    }

    void Unmap() const override
    {
        m_Buffer->Unmap();
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

    ethosn::driver_library::Buffer& GetBuffer()
    {
        return *m_Buffer;
    }
    ethosn::driver_library::Buffer const& GetBuffer() const
    {
        return *m_Buffer;
    }

    void CopyOutTo(void* memory) const override
    {
        const uint8_t* data = m_Buffer->Map();
        memcpy(memory, data, GetTensorInfo().GetNumBytes());
        m_Buffer->Unmap();
    }

    void CopyInFrom(const void* memory) override
    {
        uint8_t* data = m_Buffer->Map();
        memcpy(data, memory, GetTensorInfo().GetNumBytes());
        m_Buffer->Unmap();
    }

    unsigned int GetImportFlags() const override
    {
        return static_cast<unsigned int>(m_ImportFlags);
    }

    bool Import(void* memory, MemorySource source) override
    {
        if (m_Buffer)
        {
            m_Buffer.reset();
        }
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
        uint32_t bufferSize =
            armnn::ethosnbackend::RoundUpToNearestMultiple(m_TensorInfo.GetNumElements(), static_cast<uint32_t>(1024));

        if (m_DeviceId.empty())
        {
            m_Buffer = std::make_unique<ethosn::driver_library::Buffer>(fd, bufferSize);
            return true;
        }
        else
        {
            m_Buffer = std::make_unique<ethosn::driver_library::Buffer>(fd, bufferSize, m_DeviceId);
            return true;
        }
    };

    bool CanBeImported(void* memory, MemorySource source) override
    {
        IgnoreUnused(memory, source);
        return true;
    };

    /// Unimport externally allocated memory
    void Unimport() override
    {

        m_Buffer.reset();
    };

private:
    EthosNImportTensorHandle(const EthosNImportTensorHandle& other) = delete;
    EthosNImportTensorHandle& operator=(const EthosNImportTensorHandle& other) = delete;

    TensorInfo m_TensorInfo;
    MemorySourceFlags m_ImportFlags;
    std::string m_DeviceId;
    std::unique_ptr<ethosn::driver_library::Buffer> m_Buffer;
};

}    // namespace armnn
