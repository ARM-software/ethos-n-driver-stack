//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "SystemTestsUtils.hpp"

#include <ethosn_support_library/Support.hpp>

#include <functional>
#include <list>
#include <map>
#include <random>

namespace ethosn
{
namespace system_tests
{

constexpr uint32_t g_DefaultBiasDataStandardDeviation = 40;
constexpr uint8_t g_UnsignedQuantizedMaxValue         = std::numeric_limits<uint8_t>::max();
constexpr uint8_t g_UnsignedQuantizedMinValue         = std::numeric_limits<uint8_t>::lowest();
constexpr int8_t g_SignedQuantizedMaxValue            = std::numeric_limits<int8_t>::max();
constexpr int8_t g_SignedQuantizedMinValue            = std::numeric_limits<int8_t>::lowest();

constexpr float g_DefaultInputMin             = 0.0f;
constexpr float g_DefaultInputMax             = 0.5f;
constexpr float g_DefaultInputGaussianStd     = 0.0f;
constexpr float g_DefaultInputGaussianMean    = 0.0f;
constexpr float g_DefaultInputZeroPercentage  = 0.0f;
constexpr int32_t g_DefaultInputNoEntries     = -1;
constexpr float g_DefaultWeightMin            = -1.0f;
constexpr float g_DefaultWeightMax            = 1.0f;
constexpr float g_DefaultWeightGaussianMean   = 0.0f;
constexpr float g_DefaultWeightGaussianStd    = 0.0f;
constexpr float g_DefaultWeightZeroPercentage = 0.0f;
constexpr int32_t g_DefaultWeightNoEntries    = -1;
constexpr float g_DefaultGlobalOutputMin      = -1.0f;
constexpr float g_DefaultGlobalOutputMax      = 1.0f;
// Make the constant range very small in order to avoid saturation when
// the constant is input of a layer.
constexpr float g_DefaultConstantMin       = 0.0f;
constexpr float g_DefaultConstantMax       = 0.01f;
constexpr float g_UnsetFloat               = std::numeric_limits<float>::quiet_NaN();
constexpr DataType g_DefaultInputDataType  = DataType::U8;
constexpr DataType g_DefaultWeightDataType = DataType::U8;

constexpr ethosn::support_library::DataFormat g_DefaultInputDataFormat  = ethosn::support_library::DataFormat::NHWCB;
constexpr ethosn::support_library::DataFormat g_DefaultOutputDataFormat = ethosn::support_library::DataFormat::NHWCB;

struct WeightParams
{
    float weightFillerMin            = g_DefaultWeightMin;
    float weightFillerMax            = g_DefaultWeightMax;
    float weightFillerMean           = g_DefaultWeightGaussianMean;
    float weightFillerStd            = g_DefaultWeightGaussianStd;
    float weightFillerZeroPercentage = g_DefaultWeightZeroPercentage;
    int32_t weightFillerNoEntries    = g_DefaultWeightNoEntries;
    bool isUserDefined               = false;
    bool isSignedWeight              = false;
};

struct OutputParams
{
    float outputMin = g_UnsetFloat;
    float outputMax = g_UnsetFloat;
};

enum class ConvolutionAlgorithm
{
    SupportLibraryDefault,
    Direct,
    BestEffort
};

ConvolutionAlgorithm ParseConvolutionAlgorithm(const char* str);

/// The LayerData class provides data to the layers during parsing and running
/// of a network. Data will be sourced from from internal storage if available.
/// Data will be (randomly) generated if not found in internal storage.
/// The generated data will be added to internal storage.
/// The internal storage can be pre-populated to run with fixed data.
///
/// The Arm NN and Ethos-N runners will share a LayerData instance in order to ensure
/// the same data is used for the two runs, as data generated in the first run
/// will be available in internal storage for the second run.
///
/// The layers are identified by the unique "name" in the class.
class LayerData
{
public:
    LayerData();

    const ethosn::support_library::QuantizationInfo GetInputQuantInfo(std::string name);

    const ethosn::support_library::QuantizationInfo
        GetAdditionQuantInfo(std::string name,
                             const std::vector<ethosn::support_library::QuantizationInfo>& inputQuantInfos);

    const ethosn::support_library::QuantizationInfo
        GetMultiplicationQuantInfo(std::string name,
                                   const std::vector<ethosn::support_library::QuantizationInfo>& inputQuantInfos);

    const ethosn::support_library::QuantizationInfo GetConvWeightQuantInfo(std::string name,
                                                                           const WeightParams& params,
                                                                           uint32_t numSummedTerms,
                                                                           uint32_t numOutputChannels,
                                                                           bool isDepthwise);

    const ethosn::support_library::QuantizationInfo GetConvBiasQuantInfo(
        std::string name, float inputScale, const ethosn::support_library::QuantizationScales& weightScales);

    const ethosn::support_library::QuantizationInfo
        GetConstantQuantInfo(std::string name, float constMin, float constMax);

    const ethosn::support_library::QuantizationInfo
        GetConvOutputQuantInfo(std::string name,
                               ethosn::support_library::QuantizationInfo inputQuantInfo,
                               ethosn::support_library::QuantizationInfo weightQuantInfo,
                               uint32_t numSummedTerms,
                               const OutputParams& outputParams);

    const ethosn::support_library::ReluInfo GetReluInfo(const std::string& name);

    const ethosn::support_library::QuantizationInfo GetFCWeightQuantInfo(std::string name,
                                                                         const WeightParams& params,
                                                                         uint32_t numSummedTerms,
                                                                         uint32_t numOutputChannels = 0U);

    const ethosn::support_library::QuantizationInfo
        GetFCBiasQuantInfo(std::string name, float inputScale, float weightsScale);

    const ethosn::support_library::QuantizationInfo
        GetFCOutputQuantInfo(std::string name,
                             ethosn::support_library::QuantizationInfo inputQuantInfo,
                             ethosn::support_library::QuantizationInfo weightQuantInfo,
                             uint32_t numSummedTerms,
                             const OutputParams& outputParams);

    const ethosn::support_library::QuantizationInfo
        GetConcatOutputQuantInfo(std::string name,
                                 std::vector<ethosn::support_library::QuantizationInfo> inputQuantInfos);

    const ethosn::support_library::QuantizationInfo GetLeakyReluOutputQuantInfo(
        std::string name, ethosn::support_library::QuantizationInfo inputQuantInfo, float alpha);

    InputTensor GetInputData(std::string name, const ethosn::support_library::TensorShape& shape);

    WeightTensor GetConvWeightData(std::string name,
                                   const ethosn::support_library::TensorShape& shape,
                                   const ethosn::support_library::QuantizationInfo& qInfo,
                                   const WeightParams& params);

    const BaseTensor& GetConvBiasData(std::string name, uint32_t numOutput);

    const BaseTensor& GetConstantData(std::string name,
                                      const ethosn::support_library::TensorShape& shape,
                                      float constMin,
                                      float constMax);

    WeightTensor GetFullyConnectedWeightData(std::string name,
                                             const ethosn::support_library::TensorShape& shape,
                                             ethosn::support_library::QuantizationInfo& qInfo,
                                             const WeightParams& params);

    const BaseTensor& GetFullyConnectedBiasData(std::string name, uint32_t numOutput);

    // Possible types are:
    // -ethosn::support_library::DataType
    // -armnn::DataType
    template <typename T>
    T GetWeightDataType()
    {
        return GetDataType<T>(m_WeightDataType);
    }

    std::string GetLayerDataKey(std::string name)
    {
        return m_LocalLayerDataKeyMap[name];
    }

    ethosn::support_library::DataFormat GetInputTensorFormat()
    {
        return m_InputTensorFormat;
    }

    ethosn::support_library::DataFormat GetOutputTensorFormat()
    {
        return m_OutputTensorFormat;
    }

    ConvolutionAlgorithm GetConvolutionAlgorithm()
    {
        return m_ConvolutionAlgorithm;
    }

    uint32_t GetMaxKernelSize()
    {
        return m_MaxKernelSize;
    }

    std::string GetBlockConfigs()
    {
        return m_BlockConfigs;
    }

    bool GetIntermediateCompression()
    {
        return m_IntermediateCompression;
    }

    bool GetVerifyDistribution()
    {
        return m_VerifyDistribution;
    }

    int32_t GetInputQuantZeroPoint()
    {
        return m_InputQuantZeroPoint;
    }

    float GetInputQuantScale()
    {
        return m_InputQuantScale;
    }

    bool GetUserInputQuantZeroPoint()
    {
        return m_UserInputQuantZeroPoint;
    }

    bool GetUserInputQuantScale()
    {
        return m_UserInputQuantScale;
    }

    int32_t GetWeightQuantZeroPoint()
    {
        return m_WeightQuantZeroPoint;
    }

    float GetWeightQuantScale()
    {
        return m_WeightQuantScale;
    }

    bool GetUserWeightQuantZeroPoint()
    {
        return m_UserWeightQuantZeroPoint;
    }

    bool GetUserWeightQuantScale()
    {
        return m_UserWeightQuantScale;
    }

    int32_t GetOutputQuantZeroPoint()
    {
        return m_OutputQuantZeroPoint;
    }

    float GetOutputQuantScale()
    {
        return m_OutputQuantScale;
    }

    bool GetUserOutputQuantZeroPoint()
    {
        return m_UserOutputQuantZeroPoint;
    }

    bool GetUserOutputQuantScale()
    {
        return m_UserOutputQuantScale;
    }

    void SetTensor(std::string key, const BaseTensor& data);

    bool GetPerChannelQuantization()
    {
        return m_PerChannelQuantization;
    }

    void SetPerChannelScales(ethosn::support_library::QuantizationInfo& qInfo,
                             const uint32_t noOfScales,
                             const float baseScale);

    // Possible types are:
    // -ethosn::support_library::DataType
    // -armnn::DataType
    template <typename T>
    T GetInputsDataType()
    {
        return GetDataType<T>(m_InputDataType);
    }

    void SetQuantInfo(std::string key, ethosn::support_library::QuantizationInfo quantInfo);
    void SetReluInfo(std::string key, ethosn::support_library::ReluInfo reluInfo);

    void SetInputTensorFormat(ethosn::support_library::DataFormat dataFormat);
    void SetInputTensorShape(ethosn::support_library::TensorShape shape);
    void SetOutputTensorFormat(ethosn::support_library::DataFormat dataFormat);

    void SetInputMin(float inputMin);
    void SetInputMax(float inputMax);
    void SetInputZeroPercentage(float inputZeroPercentage);
    void SetInputNoEntries(int32_t inputNoEntriesPercentage);

    void SetGaussianInputStd(float inputStd);
    void SetGaussianInputMean(float inputMean);

    void SetGlobalOutputMin(float globalOutputMin);
    void SetGlobalOutputMax(float globalOutputMax);
    void SetUseGlobalOutputMinMax(bool enable);

    void SetBlockConfigs(std::string blockConfigs);
    void SetSeed(unsigned seed);
    void SetConvolutionAlgorithm(const ConvolutionAlgorithm algo);
    void SetMaxKernelSize(const uint32_t val);

    void SetIntermediateCompression(bool b);
    void SetPerChannelQuantization(bool b);

    void SetVerifyDistribution(bool b);

    void SetInputDataType(DataType dataType);
    void SetWeightDataType(DataType dataType);

    void SetInputQuantZeroPoint(int32_t zeroPoint);
    void SetInputQuantScale(float scale);
    void SetUserInputQuantZeroPoint(bool value);
    void SetUserInputQuantScale(bool value);

    void SetWeightQuantZeroPoint(int32_t zeroPoint);
    void SetWeightQuantScale(float scale);
    void SetUserWeightQuantZeroPoint(bool value);
    void SetUserWeightQuantScale(bool value);

    void SetOutputQuantZeroPoint(int32_t zeroPoint);
    void SetOutputQuantScale(float scale);
    void SetUserOutputQuantZeroPoint(bool value);
    void SetUserOutputQuantScale(bool value);

private:
    /// Retrieves the tensor with the given name and keyQuirk, or if it doesn't exist, generate one using the given
    /// per-element generator function.
    template <typename T>
    const BaseTensor& GetTensor(std::string name,
                                const std::string& keyQuirk,
                                uint32_t numElements,
                                std::function<T(void)> generator);

    ethosn::support_library::QuantizationInfo
        GetQuantInfo(std::string name,
                     const std::string& keyQuirk,
                     std::function<ethosn::support_library::QuantizationInfo(void)> generator);

    ethosn::support_library::ReluInfo GetReluInfo(std::string name,
                                                  std::function<ethosn::support_library::ReluInfo(void)> generator);

    WeightTensor GetGenericWeightData(std::string name,
                                      std::string key,
                                      const ethosn::support_library::TensorShape& shape,
                                      const ethosn::support_library::QuantizationInfo& qInfo,
                                      const WeightParams& params);

    const ethosn::support_library::QuantizationInfo CalculateWeightQuantInfoForDotProductOperations(
        uint32_t numSummedTerms,
        uint32_t numScales                                          = 0,
        ethosn::support_library::utils::Optional<uint32_t> quantDim = ethosn::support_library::utils::EmptyOptional());

    const ethosn::support_library::QuantizationInfo
        CalculateOutputQuantInfoForDotProductOperations(ethosn::support_library::QuantizationInfo inputQuantInfo,
                                                        ethosn::support_library::QuantizationInfo weightQuantInfo,
                                                        uint32_t numSummedTerms,
                                                        const OutputParams& outputParams);

    uint32_t TensorShapeGetNumBytes(const ethosn::support_library::TensorShape& shape);

    ethosn::support_library::QuantizationInfo ChooseQuantizationParams(
        float min,
        float max,
        bool signedData,
        uint32_t numScales                                          = 0,
        ethosn::support_library::utils::Optional<uint32_t> quantDim = ethosn::support_library::utils::EmptyOptional());

    bool AreWeightsSigned();
    bool AreInputsSigned();

    template <typename T>
    const BaseTensor& GenerateWeightData(const uint32_t tensorSize, const std::string& key, const std::string& name)
    {
        // Generate uniformly distributed results filling the quantised space.
        const T min = std::numeric_limits<T>::lowest();
        const T max = std::numeric_limits<T>::max();
        // Mind the unary opeator on the min and max variable that allows
        // char and unsigned char type to be promoted to int
        std::string debugMsg = "Drawing weight from uniform distribution {" + std::to_string(+min) + ", " +
                               std::to_string(+max) + "} (in quantized space)";
        g_Logger.Debug("%s", debugMsg.c_str());
        std::uniform_real_distribution<float> distribution(min, max);
        return GetTensor<T>(name, key + " weights", tensorSize,
                            [&]() -> T { return static_cast<T>(distribution(m_RandomGenerator)); });
    }

    /// Takes a sample from the given distribution (e.g. std::normal_distribution), clamps the resulting
    /// floating point values to the given min/max and then quantises using the given quantisation info.
    template <typename T, typename TDist>
    T SampleClampAndQuantize(TDist& distribution,
                             float min,
                             float max,
                             const ethosn::support_library::QuantizationInfo& qInfo);

    /// Retrieves the tensor with the given name and keyQuirk, or if it doesn't exist, generate one randomly using the
    /// remaining parameters.
    template <typename TDist>
    const BaseTensor& GetRandomTensor(std::string name,
                                      std::string keyQuirk,
                                      uint32_t numElements,
                                      DataType dataType,
                                      TDist& distribution,
                                      float min,
                                      float max,
                                      const ethosn::support_library::QuantizationInfo& qInfo);

    /// Randomly sets elements of the given tensor to zero, with the given chance of each element being zeroed.
    /// If a zeroing has already been performed with the same name, the same elements will be zeroed as before.
    void ApplyZeroPercentage(
        BaseTensor& t, uint32_t numElements, const std::string& name, float zeroPercentage, int32_t zeroPoint);

    std::map<std::string, ethosn::support_library::ReluInfo> m_LocalReluInfo;
    std::map<std::string, ethosn::support_library::QuantizationInfo> m_LocalQuantInfo;
    std::map<std::string, OwnedTensor> m_LocalTensors;
    std::map<std::string, std::string> m_LocalLayerDataKeyMap;
    ethosn::support_library::DataFormat m_InputTensorFormat;
    ethosn::support_library::DataFormat m_OutputTensorFormat;
    DataType m_InputDataType;
    DataType m_WeightDataType;

    ConvolutionAlgorithm m_ConvolutionAlgorithm;
    uint32_t m_MaxKernelSize;

    float m_MinInput;                  // #InputMin
    float m_MaxInput;                  // #InputMax
    float m_ZeroPercentageInput;       // #Input_Zero_Percentage
    int32_t m_NoEntriesInput;          // #Input_No_Entries
    float m_StdGaussianInput;          // #InputStd
    float m_MeanGaussianInput;         // #InputMean
    float m_MinOutputGlobal;           // #Global_OutputMin
    float m_MaxOutputGlobal;           // #Global_OutputMax
    std::string m_BlockConfigs;        // #Block_Configs
    bool m_UseGlobalOutputMinMax;      // Enable #Global_OutputMin #Global_OutputMax
    bool m_IntermediateCompression;    // #EnableIntermediateCompression

    // This is used to determine whether to check if the distribution of random weights is good enough
    bool m_VerifyDistribution;

    // To Enable perchannnel quantization for the network.
    bool m_PerChannelQuantization;    // #EnablePerChannelQuantization

    int32_t m_InputQuantZeroPoint;    // #InputQuantizationZeroPoint
    float m_InputQuantScale;          // #InputQuantizationScale
    bool m_UserInputQuantZeroPoint;
    bool m_UserInputQuantScale;

    int32_t m_WeightQuantZeroPoint;    // #WeightQuantizationZeroPoint
    float m_WeightQuantScale;          // #WeightQuantizationScale
    bool m_UserWeightQuantZeroPoint;
    bool m_UserWeightQuantScale;

    int32_t m_OutputQuantZeroPoint;    // #OutputQuantizationZeroPoint
    float m_OutputQuantScale;          // #OutputQuantizationScale
    bool m_UserOutputQuantZeroPoint;
    bool m_UserOutputQuantScale;

    std::mt19937 m_RandomGenerator;
};

}    // namespace system_tests
}    // namespace ethosn
