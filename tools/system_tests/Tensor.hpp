//
// Copyright © 2020,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <armnn/ArmNN.hpp>

#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

namespace ethosn
{
namespace system_tests
{

/// Data types that system_tests works with. This is distinct from both Support Library's and Arm NN's DataType enum,
/// as we need to operate with the *union* of the types declared in those libraries.
/// It also provides a neutral choice for code which is specific to neither Support Library nor Arm NN, e.g. the GgfParser.
enum class DataType
{
    U8,
    S8,
    S32,
    F32,
};

/// Gets the number of bytes required to store a single element of the given data type.
uint32_t GetNumBytes(DataType dt);

/// Conversion from system_tests::DataType to any of [system_tests/armnn/support_library]::DataType.
template <typename T>
T GetDataType(DataType dataType);

/// Conversion from armnn::DataType to system_tests::DataType.
DataType GetDataType(armnn::DataType dt);

/// Conversion from compile-time datatype (e.g. uint8_t) to system_tests::DataType (e.g. U8).
template <typename T>
DataType GetDataType();

template <typename T>
class TypedTensor;

/// Polymorphic base tensor type that can contain a vector of any datatype.
/// References or pointers to these can be passed around without needing to know the underlying datatype.
/// The contents of the tensor can only be accessed by converting to a TypedTensor, which requires knowing the
/// compile-time data type, for example using a switch-statement on the run-time data type.
class BaseTensor
{
protected:
    BaseTensor(DataType dataType)
        : m_DataType(dataType)
    {}

public:
    virtual ~BaseTensor() = default;

    DataType GetDataType() const
    {
        return m_DataType;
    }

    /// Gets the number of elements stored in this tensor.
    virtual uint32_t GetNumElements() const = 0;

    /// Gets the number of bytes stored by this tensor.
    /// This may be different to the number of elements if each element is not a single byte.
    uint32_t GetNumBytes() const
    {
        return GetNumElements() * system_tests::GetNumBytes(m_DataType);
    }

    /// Downcasts to a TypedTensor given the compile-time data type.
    /// The given compile-time data type must be correct, otherwise it asserts.
    /// @{
    template <typename T>
    TypedTensor<T>& AsTyped()
    {
        assert(m_DataType == system_tests::GetDataType<T>());
        return static_cast<TypedTensor<T>&>(*this);
    }

    template <typename T>
    const TypedTensor<T>& AsTyped() const
    {
        assert(m_DataType == system_tests::GetDataType<T>());
        return static_cast<const TypedTensor<T>&>(*this);
    }
    /// @}

    /// Gets the std::vector backing store given the compile-time data type.
    /// The given compile-time data type must be correct, otherwise it asserts.
    /// @{
    template <typename T>
    std::vector<T>& GetData()
    {
        return AsTyped<T>().GetData();
    }

    template <typename T>
    const std::vector<T>& GetData() const
    {
        return AsTyped<T>().GetData();
    }
    /// @}

    /// Gets a typed raw pointer to the backing store given the compile-time  data type.
    /// The given compile-time data type must be correct, otherwise it asserts.
    /// @{
    template <typename T>
    T* GetDataPtr()
    {
        return AsTyped<T>().GetData().data();
    }

    template <typename T>
    const T* GetDataPtr() const
    {
        return AsTyped<T>().GetData().data();
    }
    /// @}

    /// Gets a raw byte pointer to the backing store, independent of the actual data type.
    /// This possible reinterpreting of the data is OK because the spec allows examining any type as a byte array.
    /// @{
    virtual const uint8_t* GetByteData() const = 0;
    virtual uint8_t* GetByteData()             = 0;
    /// @}

private:
    const DataType m_DataType;
};

/// Concrete tensor type with storage for a known datatype.
/// By storing the data in the correct type (rather than a generic byte array),
/// it avoids having to reinterpret data and thus breaking aliasing rules.
/// It also avoids alignment issues.
template <typename T>
class TypedTensor : public BaseTensor
{
public:
    TypedTensor()
        : BaseTensor(ethosn::system_tests::GetDataType<T>())
    {}

    TypedTensor(const TypedTensor& rhs)
        : BaseTensor(ethosn::system_tests::GetDataType<T>())
        , m_Data(rhs.GetData())
    {}

    /// Convenience constructor to take the contents of the given std::vector.
    TypedTensor(std::vector<T> data)
        : BaseTensor(ethosn::system_tests::GetDataType<T>())
        , m_Data(std::move(data))
    {}

    const std::vector<T>& GetData() const
    {
        return m_Data;
    }

    std::vector<T>& GetData()
    {
        return m_Data;
    }

    uint32_t GetNumElements() const override
    {
        return static_cast<uint32_t>(m_Data.size());
    }

    const uint8_t* GetByteData() const override
    {
        // This reinterpret is OK because the spec allows examining any type as a byte array
        return reinterpret_cast<const uint8_t*>(m_Data.data());
    }

    uint8_t* GetByteData() override
    {
        // This reinterpret is OK because the spec allows examining any type as a byte array
        return reinterpret_cast<uint8_t*>(m_Data.data());
    }

private:
    std::vector<T> m_Data;
};

/// Because BaseTensor is polymorphic, you normally need to create one on the heap
/// (or more accurately, create a TypedTensor on the heap).
/// This alias is therefore useful for storing a newly created tensor.
using OwnedTensor = std::unique_ptr<BaseTensor>;

/// Convenience factory function which makes a TypedTensor of the given compile-time type, on the heap,
/// forwarding the given arguments to its constructor.
template <typename T, typename... Args>
OwnedTensor MakeTensor(Args&&... args)
{
    return std::make_unique<TypedTensor<T>>(std::forward<Args>(args)...);
}

/// Convenience factory function which makes a TypedTensor of the given compile-time type, on the heap.
/// This overload can be used to infer the data type from the given std::vector type.
/// This prevents having to specify the type twice, e.g.:  MakeTensor<uint8_t>(std::vector<uint8_t>(...));
template <typename T>
OwnedTensor MakeTensor(std::vector<T> x)
{
    return std::make_unique<TypedTensor<T>>(std::move(x));
}

/// Creates a new heap-allocated tensor of the given data type and size, with all elements set to zero.
OwnedTensor MakeTensor(DataType dataType, uint64_t initialSize);

/// Copies the contents of the given tensor into a new heap-allocated tensor of the appropriate type.
OwnedTensor MakeTensor(const BaseTensor& t);

/// Maps every element of the given tensor using the given function-like object, overwriting each input value
/// with the result of the function applied to it.
template <typename T, typename TFunc>
void MapTensor(TypedTensor<T>& t, TFunc func)
{
    std::transform(t.GetData().begin(), t.GetData().end(), t.GetData().begin(), func);
}

/// Maps every element of the given tensor using the given function-like object, overwriting each input value
/// with the result of the function applied to it.
template <typename TFunc>
void MapTensor(BaseTensor& t, TFunc func)
{
    switch (t.GetDataType())
    {
        case DataType::U8:
            MapTensor(t.AsTyped<uint8_t>(), func);
            break;
        case DataType::S8:
            MapTensor(t.AsTyped<int8_t>(), func);
            break;
        case DataType::S32:
            MapTensor(t.AsTyped<int32_t>(), func);
            break;
        default:
            throw std::invalid_argument("Unknown data type");
    }
}

using InputTensor              = OwnedTensor;
using OutputTensor             = OwnedTensor;
using WeightTensor             = OwnedTensor;
using InferenceInputs          = std::vector<InputTensor>;
using InferenceOutputs         = std::vector<OutputTensor>;
using MultipleInferenceOutputs = std::vector<InferenceOutputs>;

}    // namespace system_tests
}    // namespace ethosn
