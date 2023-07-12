//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "LayerData.hpp"

#include "GlobalParameters.hpp"
#include "SystemTestsUtils.hpp"

#include <ethosn_utils/Quantization.hpp>

#include <algorithm>
#include <cmath>

using namespace ethosn::support_library;

namespace ethosn
{
namespace system_tests
{

static float GetQuantizedMinValue(bool signedData)
{
    return static_cast<float>(signedData ? g_SignedQuantizedMinValue : g_UnsignedQuantizedMinValue);
}

static float GetQuantizedMaxValue(bool signedData)
{
    return static_cast<float>(signedData ? g_SignedQuantizedMaxValue : g_UnsignedQuantizedMaxValue);
}

ConvolutionAlgorithm ParseConvolutionAlgorithm(const char* str)
{
    if (strcmp(str, "Direct") == 0)
    {
        return ConvolutionAlgorithm::Direct;
    }
    else if (strcmp(str, "BestEffort") == 0)
    {
        return ConvolutionAlgorithm::BestEffort;
    }
    else
    {
        throw std::invalid_argument("Invalid convolution algorithm. Must be 'Direct' or 'BestEffort'.");
    }
}

struct ApplyClustering
{
    explicit ApplyClustering(float scale, float fillerMin, float fillerMax, const QuantizationInfo& qInfo)
        : m_Scale(scale)
        , m_fillerMin(fillerMin)
        , m_fillerMax(fillerMax)
        , m_QInfo(qInfo)
    {}
    const float m_Scale;
    const float m_fillerMin;
    const float m_fillerMax;
    const QuantizationInfo m_QInfo;

    template <typename T>
    T operator()(T data) const
    {
        float dequantizedData = utils::Dequantize<T>(data, m_QInfo.GetScale(), m_QInfo.GetZeroPoint());
        float clusteredData   = std::max(
            m_fillerMin,
            std::min(((std::round((dequantizedData - m_fillerMin) / m_Scale)) * m_Scale + m_fillerMin), m_fillerMax));
        return utils::Quantize<T>(clusteredData, m_QInfo.GetScale(), m_QInfo.GetZeroPoint());
    }
};

LayerData::LayerData()
    : m_LocalReluInfo()
    , m_LocalQuantInfo()
    , m_LocalTensors()
    , m_LocalLayerDataKeyMap()
    , m_InputTensorFormat(g_DefaultInputDataFormat)
    , m_OutputTensorFormat(g_DefaultOutputDataFormat)
    , m_InputDataType(g_DefaultInputDataType)
    , m_WeightDataType(g_DefaultWeightDataType)
    , m_ConvolutionAlgorithm(ConvolutionAlgorithm::SupportLibraryDefault)
    , m_MaxKernelSize(0)
    , m_MinInput(g_DefaultInputMin)
    , m_MaxInput(g_DefaultInputMax)
    , m_ZeroPercentageInput(g_DefaultInputZeroPercentage)
    , m_NoEntriesInput(g_DefaultInputNoEntries)
    , m_StdGaussianInput(g_DefaultInputGaussianStd)
    , m_MeanGaussianInput(g_DefaultInputGaussianMean)
    , m_MinOutputGlobal(g_DefaultGlobalOutputMin)
    , m_MaxOutputGlobal(g_DefaultGlobalOutputMax)
    , m_BlockConfigs()
    , m_UseGlobalOutputMinMax(false)
    , m_IntermediateCompression(true)
    , m_VerifyDistribution(true)
    , m_PerChannelQuantization(false)
    , m_InputQuantZeroPoint(127)
    , m_InputQuantScale(1.0f)
    , m_UserInputQuantZeroPoint(false)
    , m_UserInputQuantScale(false)
    , m_WeightQuantZeroPoint(127)
    , m_WeightQuantScale(1.0f)
    , m_UserWeightQuantZeroPoint(false)
    , m_UserWeightQuantScale(false)
    , m_OutputQuantZeroPoint(127)
    , m_OutputQuantScale(1.0f)
    , m_UserOutputQuantZeroPoint(false)
    , m_UserOutputQuantScale(false)
    , m_RandomGenerator()
{
    if (!g_DefaultConvolutionAlgorithm.empty())
    {
        SetConvolutionAlgorithm(ParseConvolutionAlgorithm(g_DefaultConvolutionAlgorithm.c_str()));
    }

    SetSeed(g_DistributionSeed);
}

void LayerData::SetTensor(std::string key, const BaseTensor& data)
{
    g_Logger.Debug("key=%s", key.c_str());
    m_LocalTensors[key.c_str()] = MakeTensor(data);
}

namespace
{

const char* GetDataFormatStr(DataFormat f)
{
    switch (f)
    {
        case DataFormat::NCHW:
            return "NCHW";
        case DataFormat::NHWC:
            return "NHWC";
        case DataFormat::NHWCB:
            return "NHWCB";
        default:
            assert(false);
            return "?";
    }
}

}    // namespace

void LayerData::SetInputTensorFormat(DataFormat dataFormat)
{
    g_Logger.Debug("InputTensorFormat=%s", GetDataFormatStr(dataFormat));
    m_InputTensorFormat = dataFormat;
}

void LayerData::SetOutputTensorFormat(DataFormat dataFormat)
{
    g_Logger.Debug("OutputTensorFormat=%s", GetDataFormatStr(dataFormat));
    m_OutputTensorFormat = dataFormat;
}

void LayerData::SetInputMin(float inputMin)
{
    m_MinInput = inputMin;
}

void LayerData::SetInputMax(float inputMax)
{
    m_MaxInput = inputMax;
}

void LayerData::SetInputZeroPercentage(float inputZeroPercentage)
{
    m_ZeroPercentageInput = inputZeroPercentage;
}

void LayerData::SetInputNoEntries(int32_t inputNoEntries)
{
    m_NoEntriesInput = inputNoEntries;
}

void LayerData::SetGaussianInputStd(float inputStd)
{
    m_StdGaussianInput = inputStd;
}

void LayerData::SetGaussianInputMean(float inputMean)
{
    m_MeanGaussianInput = inputMean;
}

void LayerData::SetGlobalOutputMin(float globalOutputMin)
{
    m_MinOutputGlobal = globalOutputMin;
}

void LayerData::SetGlobalOutputMax(float globalOutputMax)
{
    m_MaxOutputGlobal = globalOutputMax;
}

void LayerData::SetUseGlobalOutputMinMax(bool enable)
{
    m_UseGlobalOutputMinMax = enable;
}

void LayerData::SetSeed(unsigned seed)
{
    g_Logger.Debug("LayerData::SetSeed(%d)", seed);
    m_RandomGenerator.seed(seed);
}

void LayerData::SetQuantInfo(std::string key, QuantizationInfo quantInfo)
{
    g_Logger.Debug("key=%s zeroPoint=%d scale=%0.17f", key.c_str(), quantInfo.GetZeroPoint(), quantInfo.GetScale());
    m_LocalQuantInfo[key] = quantInfo;
};

void LayerData::SetReluInfo(std::string key, ReluInfo reluInfo)
{
    g_Logger.Debug("key=%s lowerBound=%u upperBound=%u", key.c_str(), reluInfo.m_LowerBound, reluInfo.m_UpperBound);
    m_LocalReluInfo[key] = reluInfo;
};

void LayerData::SetConvolutionAlgorithm(const ConvolutionAlgorithm algo)
{
    auto ToString = [](ConvolutionAlgorithm algo) {
        if (algo == ConvolutionAlgorithm::Direct)
        {
            return "Direct";
        }
        else if (algo == ConvolutionAlgorithm::BestEffort)
        {
            return "BestEffort";
        }
        else
        {
            throw std::invalid_argument("Cannot convert Convolution Algorithm to string");
        }
    };
    g_Logger.Debug("LayerData::ConvolutionAlgorithm=%s", ToString(algo));
    m_ConvolutionAlgorithm = algo;
}

void LayerData::SetMaxKernelSize(const uint32_t val)
{
    if (val > m_MaxKernelSize)
    {
        m_MaxKernelSize = val;
    }
    g_Logger.Debug("LayerData::MaxKernelSize=%u", m_MaxKernelSize);
}

void LayerData::SetIntermediateCompression(bool b)
{
    g_Logger.Debug("LayerData::IntermediateCompression=%i", static_cast<uint32_t>(b));
    m_IntermediateCompression = b;
}

void LayerData::SetVerifyDistribution(bool b)
{
    g_Logger.Debug("LayerData::VerifyDistribution=%i", static_cast<uint32_t>(b));
    m_VerifyDistribution = b;
}

void LayerData::SetPerChannelQuantization(bool b)
{
    g_Logger.Debug("LayerData::PerChannelQuatization=%i", static_cast<uint32_t>(b));
    m_PerChannelQuantization = b;
}

void LayerData::SetInputDataType(DataType dataType)
{
    g_Logger.Debug("LayerData::SetInputDataType=%i", static_cast<uint32_t>(dataType));
    m_InputDataType = dataType;
}

void LayerData::SetWeightDataType(DataType dataType)
{
    g_Logger.Debug("LayerData::SetWeightDataType=%i", static_cast<uint32_t>(dataType));
    m_WeightDataType = dataType;
}

void LayerData::SetInputQuantZeroPoint(int32_t zeroPoint)
{
    g_Logger.Debug("LayerData::SetInputQuantZeroPoint=%d", zeroPoint);
    m_InputQuantZeroPoint = zeroPoint;
}

void LayerData::SetInputQuantScale(float scale)
{
    g_Logger.Debug("LayerData::SetInputQuantScale=%f", scale);
    m_InputQuantScale = scale;
}

void LayerData::SetUserInputQuantScale(bool value)
{
    g_Logger.Debug("LayerData::SetUserInputQuantScale%d", value);
    m_UserInputQuantScale = value;
}

void LayerData::SetUserInputQuantZeroPoint(bool value)
{
    g_Logger.Debug("LayerData::SetUserInputQuantZeroPoint%d", value);
    m_UserInputQuantZeroPoint = value;
}

void LayerData::SetWeightQuantZeroPoint(int32_t zeroPoint)
{
    g_Logger.Debug("LayerData::SetWeightQuantZeroPoint=%d", zeroPoint);
    m_WeightQuantZeroPoint = zeroPoint;
}

void LayerData::SetWeightQuantScale(float scale)
{
    g_Logger.Debug("LayerData::SetWeightQuantScale=%f", scale);
    m_WeightQuantScale = scale;
}

void LayerData::SetUserWeightQuantScale(bool value)
{
    g_Logger.Debug("LayerData::SetUserWeightQuantScale%d", value);
    m_UserWeightQuantScale = value;
}

void LayerData::SetUserWeightQuantZeroPoint(bool value)
{
    g_Logger.Debug("LayerData::SetUserWeightQuantZeroPoint%d", value);
    m_UserWeightQuantZeroPoint = value;
}

void LayerData::SetOutputQuantZeroPoint(int32_t zeroPoint)
{
    g_Logger.Debug("LayerData::SetOutputQuantZeroPoint=%d", zeroPoint);
    m_OutputQuantZeroPoint = zeroPoint;
}

void LayerData::SetOutputQuantScale(float scale)
{
    g_Logger.Debug("LayerData::SetOutputQuantScale=%f", scale);
    m_OutputQuantScale = scale;
}

void LayerData::SetUserOutputQuantScale(bool value)
{
    g_Logger.Debug("LayerData::SetUserOutputQuantScale%d", value);
    m_UserOutputQuantScale = value;
}

void LayerData::SetUserOutputQuantZeroPoint(bool value)
{
    g_Logger.Debug("LayerData::SetUserOutputQuantZeroPoint%d", value);
    m_UserOutputQuantZeroPoint = value;
}

template <typename T>
struct QuantizeIfZero
{
    explicit QuantizeIfZero(int32_t zeroPoint)
        : m_ZeroPoint(zeroPoint)
    {}
    int32_t m_ZeroPoint;
    T operator()(uint8_t a, T b)
    {
        if (a == 0)
        {
            return static_cast<T>(m_ZeroPoint);
        }
        return b;
    }
};

void LayerData::ApplyZeroPercentage(
    BaseTensor& t, uint32_t numElements, const std::string& name, float zeroPercentage, int32_t zeroPoint)
{
    // Apply Zero Percentage
    // Note this may deliver a higher percentage than requested.
    // For example say we want 40%. We are choosing a 40% slice from the tensor
    // which is random. The remaining 60% could also have zeros.
    std::bernoulli_distribution bernoulliDistribution(1 - zeroPercentage);
    // Note zeroData is always u8, as it is used simply as a flag
    std::vector<uint8_t> zeroData =
        GetTensor<uint8_t>(name, "zeroingVector", numElements,
                           [&]() -> uint8_t { return static_cast<uint8_t>(bernoulliDistribution(m_RandomGenerator)); })
            .GetData<uint8_t>();

    switch (t.GetDataType())
    {
        case DataType::S8:
            std::transform(zeroData.begin(), zeroData.end(), t.GetData<int8_t>().begin(), t.GetData<int8_t>().begin(),
                           QuantizeIfZero<int8_t>(zeroPoint));
            break;
        case DataType::U8:
            std::transform(zeroData.begin(), zeroData.end(), t.GetData<uint8_t>().begin(), t.GetData<uint8_t>().begin(),
                           QuantizeIfZero<uint8_t>(zeroPoint));
            break;
        default:
            throw std::invalid_argument("Error in " + std::string(__func__) + ": Input dataType is not supported");
    }
}

bool LayerData::AreWeightsSigned()
{
    return IsDataTypeSigned(m_WeightDataType);
}

bool LayerData::AreInputsSigned()
{
    return IsDataTypeSigned(m_InputDataType);
}

void LayerData::SetBlockConfigs(std::string blockConfigs)
{
    g_Logger.Debug("LayerData::BlockConfigs=%s", blockConfigs.c_str());
    m_BlockConfigs = blockConfigs;
}

template <typename T, typename TDist>
T LayerData::SampleClampAndQuantize(TDist& distribution, float min, float max, const QuantizationInfo& qInfo)
{
    float value = static_cast<float>(distribution(m_RandomGenerator));
    value       = (value < min) ? min : value;
    value       = (value > max) ? max : value;
    return utils::Quantize<T>(value, qInfo.GetScale(), qInfo.GetZeroPoint());
}

template <typename TDist>
const BaseTensor& LayerData::GetRandomTensor(std::string name,
                                             std::string keyQuirk,
                                             uint32_t numElements,
                                             DataType dataType,
                                             TDist& distribution,
                                             float min,
                                             float max,
                                             const QuantizationInfo& qInfo)
{
    switch (dataType)
    {
        case DataType::S8:
            return GetTensor<int8_t>(name, keyQuirk, numElements,
                                     [&]() { return SampleClampAndQuantize<int8_t>(distribution, min, max, qInfo); });
        case DataType::U8:
            return GetTensor<uint8_t>(name, keyQuirk, numElements,
                                      [&]() { return SampleClampAndQuantize<uint8_t>(distribution, min, max, qInfo); });
        case DataType::S32:
            return GetTensor<int32_t>(name, keyQuirk, numElements,
                                      [&]() { return SampleClampAndQuantize<int32_t>(distribution, min, max, qInfo); });
        default:
            throw std::exception();
    }
}

void LayerData::SetPerChannelScales(ethosn::support_library::QuantizationInfo& qInfo,
                                    const uint32_t noOfScales,
                                    const float baseScale)
{
    std::uniform_real_distribution<float> distribution((0.5f * baseScale), (2.0f * baseScale));
    std::vector<float> scalesVec(noOfScales);
    std::generate(scalesVec.begin(), scalesVec.end(), [&]() { return distribution(m_RandomGenerator); });
    g_Logger.Debug("SetPerChannelScales:");
    for (const auto i : scalesVec)
    {
        g_Logger.Debug("%f  ", i);
    }
    qInfo.SetScales(std::move(scalesVec));
}

InputTensor LayerData::GetInputData(std::string name, const TensorShape& shape)
{
    g_Logger.Debug("LayerData::GetInputData name=%s", name.c_str());
    InputTensor inputData;
    QuantizationInfo qInfo = GetInputQuantInfo(name);

    if (m_StdGaussianInput != 0)
    {
        g_Logger.Debug("Drawing input from gaussian distribution {%f, %f} clamped to range [%f, %f]",
                       m_MeanGaussianInput, m_StdGaussianInput, m_MinInput, m_MaxInput);
        // Generate gaussian distributed results based on default or user defined values.
        std::normal_distribution<> distribution(m_MeanGaussianInput, m_StdGaussianInput);

        inputData = MakeTensor(GetRandomTensor(name, "tensor", TensorShapeGetNumBytes(shape), m_InputDataType,
                                               distribution, m_MinInput, m_MaxInput, qInfo));
    }
    else
    {
        g_Logger.Debug("Drawing input from uniform distribution {%f, %f}, scale:%0.17f, zeroPoint:%d ", m_MinInput,
                       m_MaxInput, qInfo.GetScale(), qInfo.GetZeroPoint());
        // Generate uniformally distributed results based on default or user defined range.
        std::uniform_real_distribution<float> distribution(m_MinInput, m_MaxInput);

        inputData =
            MakeTensor(GetRandomTensor(name, "tensor", TensorShapeGetNumBytes(shape), m_InputDataType, distribution,
                                       std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), qInfo));
    }

    if (m_NoEntriesInput > 1)
    {
        g_Logger.Debug("Applying Input Clustering: %d", m_NoEntriesInput);
        // Choose (N Linear Points) between min, max
        float scale = (m_MaxInput - m_MinInput) / static_cast<float>(m_NoEntriesInput - 1);
        // we'll add in zeros too. let's check if zero is representable
        if (m_ZeroPercentageInput > 0.0 && fabsf((0 - m_MinInput) / scale) > 1e-5)
        {
            // m_NoEntriesInput should not be 2 if zeros are requested
            // and not included in possible input values range.
            if (m_NoEntriesInput == 2)
            {
                throw std::invalid_argument("#Input_No_Entries must be bigger than 2 if #Input_Zero_Percentage > 0.0"
                                            " and input values range does not allow 0");
            }
            // nope. need one extra slot for zero
            scale = (m_MaxInput - m_MinInput) / static_cast<float>(m_NoEntriesInput - 2);
        }

        MapTensor(*inputData, ApplyClustering(scale, m_MinInput, m_MaxInput, qInfo));
        DebugTensor("ClusteredInputData", *inputData, 256);
    }

    if (m_ZeroPercentageInput > 0.0)
    {
        ApplyZeroPercentage(*inputData, TensorShapeGetNumBytes(shape), name, m_ZeroPercentageInput,
                            qInfo.GetZeroPoint());
        DebugTensor("ZeroFilledInputData", *inputData, 256);
    }

    if (g_Debug.find("dump-inputs") != std::string::npos)
    {
        DumpData((std::string("input-") + name + ".hex").c_str(), *inputData);
    }

    if (GetInputTensorFormat() == DataFormat::NHWCB)
    {
        return ConvertNhwcToNhwcb(*inputData, shape[1], shape[2], shape[3]);
    }
    else
    {
        return inputData;
    }
}

// Given the min and max values of a float array, return
// reasonable quantization parameters to use for this array.
QuantizationInfo LayerData::ChooseQuantizationParams(float min,
                                                     float max,
                                                     bool signedData,
                                                     uint32_t numScales,
                                                     ethosn::support_library::utils::Optional<uint32_t> quantDim)
{
    // the min and max quantized values, as floating-point values
    float qmin = GetQuantizedMinValue(signedData);
    float qmax = GetQuantizedMaxValue(signedData);

    // First determine the scale.
    const double scale = (max - min) / (qmax - qmin);

    // Zero-point computation.
    // First the initial floating-point computation. The zero-point can be
    // determined from solving an affine equation for any known pair
    // (real value, corresponding quantized value).
    // We know two such pairs: (rmin, qmin) and (rmax, qmax).
    // Let's use the first one here.
    const double initialZeroPoint = qmin - (min / scale);

    // Now we need to nudge the zero point to be an integer
    // (our zero points are integer, and this is motivated by the requirement
    // to be able to represent the real value "0" exactly as a quantized value.
    int32_t nudgedZeroPoint = 0;
    if (initialZeroPoint < qmin)
    {
        nudgedZeroPoint = static_cast<int32_t>(qmin);
    }
    else if (initialZeroPoint > qmax)
    {
        nudgedZeroPoint = static_cast<int32_t>(qmax);
    }
    else
    {
        double roundedInitialZeroPoint = std::round(initialZeroPoint);
        assert(roundedInitialZeroPoint >= std::numeric_limits<int32_t>::lowest() &&
               roundedInitialZeroPoint <= std::numeric_limits<int32_t>::max());
        nudgedZeroPoint = static_cast<int32_t>(roundedInitialZeroPoint);
    }

    QuantizationInfo result;

    if (GetPerChannelQuantization() && numScales > 0)
    {
        SetPerChannelScales(result, numScales, static_cast<float>(scale));
        result.SetQuantizationDim(quantDim.value());
        result.SetZeroPoint(nudgedZeroPoint);
    }
    else
    {
        result.SetScale(static_cast<float>(scale));
        result.SetZeroPoint(nudgedZeroPoint);
    }

    return result;
}

WeightTensor LayerData::GetGenericWeightData(std::string name,
                                             std::string key,
                                             const ethosn::support_library::TensorShape& shape,
                                             const ethosn::support_library::QuantizationInfo& qInfo,
                                             const WeightParams& params)
{
    uint32_t tensorSize = TensorShapeGetNumBytes(shape);
    WeightTensor weightData;

    if (params.weightFillerStd != 0)
    {
        g_Logger.Debug("Drawing weight from gaussian distribution {%f, %f} clamped to range [%f, %f]",
                       params.weightFillerMean, params.weightFillerStd, params.weightFillerMin, params.weightFillerMax);
        // Generate gaussian distributed results based on default or user defined values.
        std::normal_distribution<> distribution(params.weightFillerMean, params.weightFillerStd);

        weightData = MakeTensor(GetRandomTensor(name, key + " weights", tensorSize, m_WeightDataType, distribution,
                                                params.weightFillerMin, params.weightFillerMax, qInfo));
    }
    else
    {
        switch (m_WeightDataType)
        {
            case DataType::S8:
                weightData = MakeTensor(GenerateWeightData<int8_t>(tensorSize, key, name));
                break;
            case DataType::U8:
                weightData = MakeTensor(GenerateWeightData<uint8_t>(tensorSize, key, name));
                break;
            default:
            {
                std::string errorMessage = "Error in " + std::string(__func__) + ": Weight dataType is not supported";
                throw std::invalid_argument(errorMessage);
            }
        }
    }

    if (params.weightFillerNoEntries > 1)
    {
        g_Logger.Debug("Applying Weight Clustering: %d", params.weightFillerNoEntries);
        // Choose (N Linear Points) between min, max
        float scale =
            (params.weightFillerMax - params.weightFillerMin) / static_cast<float>(params.weightFillerNoEntries - 1);
        // we'll add in zeros too. let's check if zero is representable
        if (params.weightFillerZeroPercentage > 0.0 && fabsf((0 - params.weightFillerMin) / scale) > 1e-5)
        {
            // params.weightFillerNoEntries should not be 2 if zeros are requested
            // and not included in possible weights values range.
            if (params.weightFillerNoEntries == 2)
            {
                throw std::invalid_argument("'weight filler no_entries' must be bigger than 2 if "
                                            "'weight filler zero_percentage' > 0.0 and weight values "
                                            "range doesn't allow 0");
            }
            // nope. need one extra slot for zero
            scale = (params.weightFillerMax - params.weightFillerMin) /
                    static_cast<float>(params.weightFillerNoEntries - 2);
        }

        MapTensor(*weightData, ApplyClustering(scale, params.weightFillerMin, params.weightFillerMax, qInfo));
        DebugTensor("ClusteredWeightData", *weightData, 256);
    }

    if (params.weightFillerZeroPercentage > 0.0)
    {
        ApplyZeroPercentage(*weightData, tensorSize, name, params.weightFillerZeroPercentage, qInfo.GetZeroPoint());
        DebugTensor("ZeroFilledWeightData", *weightData, 256);
    }

    return weightData;
}

WeightTensor LayerData::GetConvWeightData(std::string name,
                                          const TensorShape& shape,
                                          const QuantizationInfo& qInfo,
                                          const WeightParams& params)
{
    g_Logger.Debug("LayerData::GetConvWeightData name=%s", name.c_str());
    return GetGenericWeightData(name, "conv", shape, qInfo, params);
}

template <typename T>
const BaseTensor& LayerData::GetTensor(std::string name,
                                       const std::string& keyQuirk,
                                       uint32_t numElements,
                                       std::function<T(void)> generator)
{
    std::string key              = name + " - " + keyQuirk;
    m_LocalLayerDataKeyMap[name] = key;

    g_Logger.Debug("LayerData::GetTensor name=%s key='%s' local=%d", name.c_str(), key.c_str(),
                   m_LocalTensors.find(key.c_str()) != m_LocalTensors.end());

    if (m_LocalTensors.find(key.c_str()) != m_LocalTensors.end())
    {
        // Use existing local data
    }
    else
    {
        // Generate new data and add to local storage
        m_LocalTensors[key.c_str()] = MakeTensor(std::vector<T>(numElements));
        std::vector<T>& data        = m_LocalTensors[key.c_str()]->GetData<T>();
        generate(data.begin(), data.end(), generator);
    }

    DebugTensor("GetTensor", *m_LocalTensors[key.c_str()], 256);

    return *m_LocalTensors[key.c_str()];
}

QuantizationInfo LayerData::GetQuantInfo(std::string name,
                                         const std::string& keyQuirk,
                                         std::function<QuantizationInfo(void)> generator)
{
    std::string key = name + " - " + keyQuirk + "quantization parameters";

    g_Logger.Debug("LayerData::GetQuantInfo name=%s key='%s' local=%d", name.c_str(), key.c_str(),
                   m_LocalQuantInfo.find(key) != m_LocalQuantInfo.end());

    if (m_LocalQuantInfo.find(key) != m_LocalQuantInfo.end())
    {
        // Use existing local data
    }
    else
    {
        // Generate new data and add to local storage
        QuantizationInfo quantInfo = generator();
        m_LocalQuantInfo[key]      = quantInfo;
    }

    g_Logger.Debug("GetQuantInfo scales[0]=%0.17f zeroPoint=%d", m_LocalQuantInfo[key].GetScale(0),
                   m_LocalQuantInfo[key].GetZeroPoint());

    return m_LocalQuantInfo[key];
}

const QuantizationInfo LayerData::GetInputQuantInfo(std::string name)
{
    if (GetUserInputQuantScale() && GetUserInputQuantZeroPoint())
    {
        g_Logger.Debug("LayerData::GetInputQuantInfo user defined value name=%s zeroPoint=%d scale=%f ", name.c_str(),
                       GetInputQuantZeroPoint(), GetInputQuantScale());

        return { GetInputQuantZeroPoint(), GetInputQuantScale() };
    }

    g_Logger.Debug("LayerData::GetInputQuantInfo name=%s m_MinInput=%f m_MaxInput=%f ", name.c_str(), m_MinInput,
                   m_MaxInput);
    return GetQuantInfo(name, "", [this]() -> QuantizationInfo {
        QuantizationInfo qInfo = ChooseQuantizationParams(m_MinInput, m_MaxInput, AreInputsSigned());
        return { qInfo.GetZeroPoint(), qInfo.GetScale() };
    });
}

const QuantizationInfo
    LayerData::GetAdditionQuantInfo(std::string name,
                                    const std::vector<ethosn::support_library::QuantizationInfo>& inputQuantInfos)
{
    auto maxScalePredicate = [](const ethosn::support_library::QuantizationInfo& lhs,
                                const ethosn::support_library::QuantizationInfo& rhs) {
        return lhs.GetScale() < rhs.GetScale();
    };
    ethosn::support_library::QuantizationInfo quantInfo =
        *std::max_element(inputQuantInfos.begin(), inputQuantInfos.end(), maxScalePredicate);

    g_Logger.Debug("LayerData::GetAdditionQuantInfo name=%s", name.c_str());
    return GetQuantInfo(name, "", [quantInfo]() -> QuantizationInfo { return quantInfo; });
}

const QuantizationInfo
    LayerData::GetMultiplicationQuantInfo(std::string name,
                                          const std::vector<ethosn::support_library::QuantizationInfo>& inputQuantInfos)
{
    auto maxScalePredicate = [](const ethosn::support_library::QuantizationInfo& lhs,
                                const ethosn::support_library::QuantizationInfo& rhs) {
        return lhs.GetScale() < rhs.GetScale();
    };
    ethosn::support_library::QuantizationInfo quantInfo =
        *std::max_element(inputQuantInfos.begin(), inputQuantInfos.end(), maxScalePredicate);

    g_Logger.Debug("LayerData::GetMultiplicationQuantInfo name=%s", name.c_str());
    return GetQuantInfo(name, "", [quantInfo]() -> QuantizationInfo { return quantInfo; });
}

static float GetMinRepresentableValue(const QuantizationInfo& qInfo, const float qmin)
{
    const QuantizationScales& scales = qInfo.GetScales();

    return std::min((qmin - static_cast<float>(qInfo.GetZeroPoint())) * scales.min(),
                    (qmin - static_cast<float>(qInfo.GetZeroPoint())) * scales.max());
}

static float GetMaxRepresentableValue(const QuantizationInfo& qInfo, const float qmax)
{
    const QuantizationScales& scales = qInfo.GetScales();
    return std::max((qmax - static_cast<float>(qInfo.GetZeroPoint())) * scales.max(),
                    (qmax - static_cast<float>(qInfo.GetZeroPoint())) * scales.min());
}

// Get standard deviation of an uniform distribution
static float GetUniformDistributionSd(float min, float max)
{
    return (sqrtf(((max - min) * (max - min)) / 12));
}

// Get mean of an uniform distribution
static float GetUniformDistributionMean(float min, float max)
{
    return ((max + min) / 2);
}

const ethosn::support_library::QuantizationInfo LayerData::CalculateWeightQuantInfoForDotProductOperations(
    uint32_t numSummedTerms, uint32_t numScales, ethosn::support_library::utils::Optional<uint32_t> quantDim)
{
    // Choose a range such that the range of the output of the dot-product operation is similar to its input.
    // This prevents the scale of tensors increasing throughout the network and eventually overflowing.
    // To achieve this we want the weight variance to be 1/numSummedTerms, as this is the factor that the variance
    // of the output will be increased by due to the dot-product operation.
    // We will be generating a uniform distribution and so can reverse the formula in GetUniformDistributionSd()
    // to choose a range such that the standard deviation is 1/root(n):
    float range = sqrtf(12.0f / static_cast<float>(numSummedTerms));
    QuantizationInfo result =
        ChooseQuantizationParams(-0.5f * range, 0.5f * range, AreWeightsSigned(), numScales, quantDim);
    if (GetPerChannelQuantization())
    {
        // Depending on rounding etc., ChooseQuantizationParams can sometimes return -1 as the zero point, which causes problems with
        // per-channel quant. Fix it to zero here.
        // cppcheck-suppress assertWithSideEffect
        assert(AreWeightsSigned());    // Per-channel quantisation requires signed weights - already validated elsewhere
        result.SetZeroPoint(0);
    }
    return result;
}

/// Calculates appropriate quantisation parameters for output of layers which perform a dot product of inputs
/// and weights (i.e. convolutions and fully connected).
/// 'numSummedTerms' should be the number of terms in the dot product.
const QuantizationInfo LayerData::CalculateOutputQuantInfoForDotProductOperations(QuantizationInfo inputQuantInfo,
                                                                                  QuantizationInfo weightQuantInfo,
                                                                                  uint32_t numSummedTerms,
                                                                                  const OutputParams& outputParams)
{
    float outputMin;
    float outputMax;

    if (m_UseGlobalOutputMinMax == true)
    {
        outputMin = m_MinOutputGlobal;
        outputMax = m_MaxOutputGlobal;
    }
    else if (!std::isnan(outputParams.outputMin) && !std::isnan(outputParams.outputMax))
    {
        outputMin = outputParams.outputMin;
        outputMax = outputParams.outputMax;
    }
    else
    {
        // Estimate the output range in order to set up an acceptable scale so that most of the expected values
        // can be represented in the quantised space.
        // We cannot assume much about the weight distribution because it might be user defined and we have to deal with this.

        // Assume the input distribution is uniform for simplicity.
        const float inputMax = GetMaxRepresentableValue(inputQuantInfo, GetQuantizedMaxValue(AreInputsSigned()));
        const float inputMin = GetMinRepresentableValue(inputQuantInfo, GetQuantizedMinValue(AreInputsSigned()));

        const float inputSd   = GetUniformDistributionSd(inputMin, inputMax);
        const float inputVar  = inputSd * inputSd;
        const float inputMean = GetUniformDistributionMean(inputMin, inputMax);

        // Assume the weights distribution is uniform for simplicity.
        const float weightMax = GetMaxRepresentableValue(weightQuantInfo, GetQuantizedMaxValue(AreWeightsSigned()));
        const float weightMin = GetMinRepresentableValue(weightQuantInfo, GetQuantizedMinValue(AreWeightsSigned()));

        const float weightSd   = GetUniformDistributionSd(weightMin, weightMax);
        const float weightVar  = weightSd * weightSd;
        const float weightMean = GetUniformDistributionMean(weightMin, weightMax);

        // Calculate the output mean and variance based on a sum-of-products of independent random variables
        const float outputMean = inputMean * weightMean * static_cast<float>(numSummedTerms);
        const float outputVar =
            (inputVar * weightVar + inputVar * weightMean * weightMean + weightVar * inputMean * inputMean) *
            static_cast<float>(numSummedTerms);
        const float outputSd = sqrtf(outputVar);

        // Choose an output scale that fits a reasonable amount of the distribution.
        // This number is basically a fudge-factor tuned based on some GGF test cases.
        // If it is too small then we'll chop off too much data and get lots of clamping.
        // If it is too large then we will be under-utilising the quantised space and eventually all values
        // will converge to be the same.
        const float scale = 1.2f;
        outputMin         = outputMean - outputSd * scale;
        outputMax         = outputMean + outputSd * scale;
    }

    return ChooseQuantizationParams(outputMin, outputMax, AreInputsSigned());
}

const QuantizationInfo LayerData::GetConvWeightQuantInfo(
    std::string name, const WeightParams& params, uint32_t numSummedTerms, uint32_t numOutputChannels, bool isDepthwise)
{
    if (GetUserWeightQuantScale() && GetUserWeightQuantZeroPoint())
    {
        g_Logger.Debug("LayerData::GetWeightQuantInfo user defined value name=%s zeroPoint=%d scale=%f ", name.c_str(),
                       GetWeightQuantZeroPoint(), GetWeightQuantScale());

        return { GetWeightQuantZeroPoint(), GetWeightQuantScale() };
    }

    g_Logger.Debug("LayerData::GetConvWeightQuantInfo name=%s", name.c_str());
    return GetQuantInfo(name, "weight ", [&]() -> QuantizationInfo {
        if (params.isUserDefined)
        {
            return ChooseQuantizationParams(params.weightFillerMin, params.weightFillerMax, AreWeightsSigned(),
                                            numOutputChannels, isDepthwise ? 2 : 3);
        }
        else
        {
            // We cannot have the weights uniform in [-1, 1] as each layer will increase the range of the outputs so
            // much that our quantization scale will overflow and become infinity. Therefore in the case that the
            // user hasn't provided any overrides we choose weights with a smaller range to avoid this overflow.
            return CalculateWeightQuantInfoForDotProductOperations(numSummedTerms, numOutputChannels,
                                                                   isDepthwise ? 2 : 3);
        }
    });
}

const QuantizationInfo LayerData::GetConvBiasQuantInfo(std::string name,
                                                       float inputScale,
                                                       const ethosn::support_library::QuantizationScales& weightScales)
{
    g_Logger.Debug("LayerData::GetConvBiasQuantInfo name=%s", name.c_str());
    // Bias quantisation info is always fixed based on the weight and input scales.
    QuantizationInfo defaultQuantInfo(
        0, QuantizationScales(inputScale * static_cast<const std::valarray<float>&>(weightScales)));
    return GetQuantInfo(name, "bias ", [defaultQuantInfo]() -> QuantizationInfo { return defaultQuantInfo; });
}

const QuantizationInfo LayerData::GetConstantQuantInfo(std::string name, float constMin, float constMax)
{
    // Use a very small range for constant to avoid saturation
    g_Logger.Debug("LayerData::GetConstantQuantInfo name=%s min=%f max=%f ", name.c_str(), constMin, constMax);
    return GetQuantInfo(name, "", [&]() -> QuantizationInfo {
        QuantizationInfo qInfo = ChooseQuantizationParams(constMin, constMax, AreInputsSigned());
        return { qInfo.GetZeroPoint(), qInfo.GetScale() };
    });
}

const QuantizationInfo LayerData::GetConvOutputQuantInfo(std::string name,
                                                         QuantizationInfo inputQuantInfo,
                                                         QuantizationInfo weightQuantInfo,
                                                         uint32_t numSummedTerms,
                                                         const OutputParams& outputParams)
{
    if (GetUserOutputQuantScale() && GetUserOutputQuantZeroPoint())
    {
        g_Logger.Debug("LayerData::GetOutputQuantInfo user defined value name=%s zeroPoint=%d scale=%f ", name.c_str(),
                       GetOutputQuantZeroPoint(), GetOutputQuantScale());

        return { GetOutputQuantZeroPoint(), GetOutputQuantScale() };
    }

    g_Logger.Debug("LayerData::GetConvOutputQuantInfo name=%s", name.c_str());
    return GetQuantInfo(name, "output ", [&]() -> QuantizationInfo {
        return CalculateOutputQuantInfoForDotProductOperations(inputQuantInfo, weightQuantInfo, numSummedTerms,
                                                               outputParams);
    });
}

ReluInfo LayerData::GetReluInfo(std::string name, std::function<ReluInfo(void)> generator)
{
    std::string key = name + " - parameters";

    g_Logger.Debug("LayerData::GetReluInfo name=%s type=%s local=%d", name.c_str(), key.c_str(),
                   m_LocalReluInfo.find(key) != m_LocalReluInfo.end());

    if (m_LocalReluInfo.find(key) != m_LocalReluInfo.end())
    {
        // Use existing local data
    }
    else
    {
        // Generate new data and add to local storage
        ReluInfo reluInfo    = generator();
        m_LocalReluInfo[key] = reluInfo;
    }

    g_Logger.Debug("GetReluInfo lowerBound=%d upperBound=%d", m_LocalReluInfo[key].m_LowerBound,
                   m_LocalReluInfo[key].m_UpperBound);

    return m_LocalReluInfo[key];
}

const ethosn::support_library::ReluInfo LayerData::GetReluInfo(const std::string& name)
{
    // Choose default Relu parameters such that they perform some clamping, but not so much as to disturb the
    // distribution of outputs away from being Normal. We also choose numbers that fit nicely into
    // the histogram buckets of the Stats class :)
    auto GetReluBounds = [&]() -> ReluInfo {
        const int16_t min = m_InputDataType == DataType::S8 ? -128 : 0;
        const int16_t max = m_InputDataType == DataType::S8 ? 127 : 255;

        return ReluInfo(static_cast<int16_t>(min + 32), static_cast<int16_t>(max - 32));
    };
    return GetReluInfo(name, GetReluBounds);
}

const QuantizationInfo LayerData::GetFCWeightQuantInfo(std::string name,
                                                       const WeightParams& params,
                                                       uint32_t numSummedTerms,
                                                       uint32_t numOutputChannels)
{
    g_Logger.Debug("LayerData::GetFCWeightQuantInfo name=%s", name.c_str());
    return GetQuantInfo(name, "weight ", [&]() -> QuantizationInfo {
        if (params.isUserDefined)
        {
            return ChooseQuantizationParams(params.weightFillerMin, params.weightFillerMax, AreWeightsSigned(),
                                            numOutputChannels, 3);
        }
        else
        {
            // We cannot have the weights uniform in [-1, 1] as each layer will increase the range of the outputs so
            // much that our quantization scale will overflow and become infinity. Therefore in the case that the
            // user hasn't provided any overrides we choose weights with a smaller range to avoid this overflow.
            return CalculateWeightQuantInfoForDotProductOperations(numSummedTerms, numOutputChannels, 3);
        }
    });
}

const QuantizationInfo LayerData::GetFCBiasQuantInfo(std::string name, float inputScale, float weightsScale)
{
    g_Logger.Debug("LayerData::GetFCBiasQuantInfo name=%s", name.c_str());
    // Bias quantisation info is always fixed based on the weight and input scales.
    QuantizationInfo defaultQuantInfo(0, inputScale * weightsScale);
    return GetQuantInfo(name, "bias ", [defaultQuantInfo]() -> QuantizationInfo { return defaultQuantInfo; });
}

const QuantizationInfo LayerData::GetFCOutputQuantInfo(std::string name,
                                                       QuantizationInfo inputQuantInfo,
                                                       QuantizationInfo weightQuantInfo,
                                                       uint32_t numSummedTerms,
                                                       const OutputParams& outputParams)
{
    g_Logger.Debug("LayerData::GetFCOutputQuantInfo name=%s", name.c_str());
    return GetQuantInfo(name, "output ", [&]() -> QuantizationInfo {
        return CalculateOutputQuantInfoForDotProductOperations(inputQuantInfo, weightQuantInfo, numSummedTerms,
                                                               outputParams);
    });
}

const QuantizationInfo LayerData::GetConcatOutputQuantInfo(std::string name,
                                                           std::vector<QuantizationInfo> inputQuantInfos)
{
    g_Logger.Debug("LayerData::GetConcatOutputQuantInfo name=%s", name.c_str());
    auto CalculateConcatQuantInfo = [&]() {
        float max = std::numeric_limits<float>::lowest();
        float min = std::numeric_limits<float>::max();
        for (QuantizationInfo it : inputQuantInfos)
        {
            max = std::max(max, GetMaxRepresentableValue(it, 255.f));
            max = std::max(max, GetMinRepresentableValue(it, 0.f));
            min = std::min(min, GetMaxRepresentableValue(it, 255.f));
            min = std::min(min, GetMinRepresentableValue(it, 0.f));
        }
        return ChooseQuantizationParams(min, max, AreInputsSigned());
    };
    return GetQuantInfo(name, "output ", [&]() -> QuantizationInfo { return CalculateConcatQuantInfo(); });
}

const ethosn::support_library::QuantizationInfo LayerData::GetLeakyReluOutputQuantInfo(
    std::string name, ethosn::support_library::QuantizationInfo inputQuantInfo, float alpha)
{
    g_Logger.Debug("LayerData::GetLeakyReluOutputQuantInfo name=%s", name.c_str());

    auto CalculateLeakyReluQuantInfo = [&]() {
        const float quantMin = m_InputDataType == DataType::S8 ? -128.f : 0.f;
        const float quantMax = m_InputDataType == DataType::S8 ? 127.f : 255.f;

        float beginRange = GetMinRepresentableValue(inputQuantInfo, quantMin);
        float endRange   = GetMaxRepresentableValue(inputQuantInfo, quantMax);

        float min = std::min(beginRange, endRange);
        float max = std::max(beginRange, endRange);

        float outputMin = min < 0 ? alpha * min : min;
        float outputMax = max < 0 ? alpha * max : max;

        return ChooseQuantizationParams(outputMin, outputMax, m_InputDataType == DataType::S8);
    };
    return GetQuantInfo(name, "output ", [&]() -> QuantizationInfo { return CalculateLeakyReluQuantInfo(); });
}

const BaseTensor& LayerData::GetConstantData(std::string name, const TensorShape& shape, float constMin, float constMax)
{
    uint32_t tensorSize    = TensorShapeGetNumBytes(shape);
    QuantizationInfo qInfo = GetConstantQuantInfo(name, constMin, constMax);

    g_Logger.Debug("LayerData::GetConstantData name=%s numElement=%d", name.c_str(), tensorSize);

    // Generate uniformly distributed constant data, use a very small range for constant
    // to avoid saturation.
    g_Logger.Debug("Drawing constant from uniform distribution {%f, %f} ", constMin, constMax);
    std::uniform_real_distribution<float> distribution(constMin, constMax);
    const BaseTensor& constData =
        GetRandomTensor(name, "const", tensorSize, m_InputDataType, distribution, std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::max(), qInfo);

    DebugTensor("Storage", constData, 64);

    return constData;
}

const BaseTensor& LayerData::GetConvBiasData(std::string name, uint32_t numOutput)
{
    g_Logger.Debug("LayerData::GetConvBiasData name=%s numOutput=%d", name.c_str(), numOutput);

    // Generate new data and add to local storage
    // Generate bias as very flat normal distribution.
    std::normal_distribution<> normalDistribution(
        0, sqrt(g_DefaultBiasDataStandardDeviation * g_DefaultBiasDataStandardDeviation + 256 * 256 / 12));
    return GetRandomTensor(name, " - bias", numOutput, DataType::S32, normalDistribution,
                           std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), QuantizationInfo());
}

WeightTensor LayerData::GetFullyConnectedWeightData(std::string name,
                                                    const TensorShape& shape,
                                                    QuantizationInfo& qInfo,
                                                    const WeightParams& params)
{
    g_Logger.Debug("LayerData::GetFullyConnectedWeightData name=%s", name.c_str());
    return GetGenericWeightData(name, "fc", shape, qInfo, params);
}

const BaseTensor& LayerData::GetFullyConnectedBiasData(std::string name, uint32_t numOutput)
{
    std::string key = name + " - bias";

    g_Logger.Debug("LayerData::GetFullyConnectedBiasData name=%s numOutput=%d", name.c_str(), numOutput);

    // Generate new data and add to local storage
    // Generate bias as very flat normal distribution.
    std::normal_distribution<> normalDistribution(
        0, sqrt(g_DefaultBiasDataStandardDeviation * g_DefaultBiasDataStandardDeviation + 256 * 256 / 12));
    return GetRandomTensor(name, " - bias", numOutput, DataType::S32, normalDistribution,
                           std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), QuantizationInfo());
}

uint32_t LayerData::TensorShapeGetNumBytes(const TensorShape& shape)
{
    // Arm NN uses tensors that don't always have all 4 dimensions specified.
    uint32_t result = 0;
    for (uint32_t i = 0; i < 4; ++i)
    {
        if (shape[i] > 0)
        {
            if (result == 0)
            {
                result = shape[i];
            }
            else
            {
                result = result * shape[i];
            }
        }
    }
    return result;
};

}    // namespace system_tests

}    // namespace ethosn
