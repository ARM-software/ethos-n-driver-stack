//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNTensorUtils.hpp"
#include "EthosNWorkloadUtils.hpp"

#include "armnn/Exceptions.hpp"
#include "armnn/TypesUtils.hpp"
#include "armnn/backends/ITensorHandle.hpp"

#include <CompatibleTypes.hpp>
#include <armnn/utility/Assert.hpp>
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
        return static_cast<const void*>(m_Buffer.GetMappedBuffer());
    }

    virtual void Unmap() const override
    {}

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

    template <typename T>
    T* GetTensor() const
    {
        ARMNN_ASSERT(CompatibleTypes<T>(GetTensorInfo().GetDataType()));
        return reinterpret_cast<T*>(m_Buffer.GetMappedBuffer());
    }

    void CopyOutTo(void* memory) const override
    {
        memcpy(memory, GetTensor<uint8_t>(), GetTensorInfo().GetNumBytes());
    }

    void CopyInFrom(const void* memory) override
    {
        memcpy(GetTensor<uint8_t>(), memory, GetTensorInfo().GetNumBytes());
    }

private:
    EthosNTensorHandle(const EthosNTensorHandle& other) = delete;
    EthosNTensorHandle& operator=(const EthosNTensorHandle& other) = delete;

    TensorInfo m_TensorInfo;
    mutable ethosn::driver_library::Buffer m_Buffer;
};

}    // namespace armnn
