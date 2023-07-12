//
// Copyright © 2018-2020,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Tensor.hpp"

#include <armnn/ArmNN.hpp>
#include <ethosn_support_library/Support.hpp>

namespace ethosn
{
namespace system_tests
{

uint32_t GetNumBytes(DataType dt)
{
    switch (dt)
    {
        case DataType::U8:
            return 1;
        case DataType::S8:
            return 1;
        case DataType::S32:
            return 4;
        case DataType::F32:
            return 4;
        default:
            throw std::invalid_argument("Unknown data type");
    }
}

DataType GetDataType(armnn::DataType dt)
{
    switch (dt)
    {
        case armnn::DataType::QAsymmS8:
            return DataType::S8;
        case armnn::DataType::QSymmS8:
            return DataType::S8;
        case armnn::DataType::QAsymmU8:
            return DataType::U8;
        case armnn::DataType::Float32:
            return DataType::F32;
        default:
            throw std::invalid_argument("Unknown data type");
    }
}

template <>
DataType GetDataType(ethosn::system_tests::DataType dataType)
{
    return dataType;
}

template <>
ethosn::support_library::DataType GetDataType(ethosn::system_tests::DataType dataType)
{
    switch (dataType)
    {
        case ethosn::system_tests::DataType::S8:
            return ethosn::support_library::DataType::INT8_QUANTIZED;
        case ethosn::system_tests::DataType::U8:
            return ethosn::support_library::DataType::UINT8_QUANTIZED;
        default:
            throw std::invalid_argument("Unknown data type");
    }
}

template <>
armnn::DataType GetDataType(ethosn::system_tests::DataType dataType)
{
    switch (dataType)
    {
        case ethosn::system_tests::DataType::S8:
            return armnn::DataType::QAsymmS8;
        case ethosn::system_tests::DataType::U8:
            return armnn::DataType::QAsymmU8;
        default:
            throw std::invalid_argument("Unknown data type");
    }
}

template <>
DataType GetDataType<uint8_t>()
{
    return DataType::U8;
}

template <>
DataType GetDataType<int8_t>()
{
    return DataType::S8;
}

template <>
DataType GetDataType<int32_t>()
{
    return DataType::S32;
}

template <>
DataType GetDataType<float>()
{
    return DataType::F32;
}

OwnedTensor MakeTensor(DataType dataType, uint64_t initialSize)
{
    switch (dataType)
    {
        case DataType::U8:
            return MakeTensor(std::vector<uint8_t>(initialSize));
        case DataType::S8:
            return MakeTensor(std::vector<int8_t>(initialSize));
        case DataType::S32:
            return MakeTensor(std::vector<int32_t>(initialSize));
        case DataType::F32:
            return MakeTensor(std::vector<float>(initialSize));
        default:
            throw std::invalid_argument("Unknown data type");
    }
}

ethosn::system_tests::OwnedTensor MakeTensor(const BaseTensor& t)
{
    switch (t.GetDataType())
    {
        case DataType::U8:
            return MakeTensor<uint8_t>(t.AsTyped<uint8_t>());
        case DataType::S8:
            return MakeTensor<int8_t>(t.AsTyped<int8_t>());
        case DataType::S32:
            return MakeTensor<int32_t>(t.AsTyped<int32_t>());
        default:
            throw std::invalid_argument("Unknown data type");
    }
}

}    // namespace system_tests
}    // namespace ethosn
