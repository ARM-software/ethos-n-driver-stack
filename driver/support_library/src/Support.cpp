//
// Copyright © 2018-2024 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"

#include "CapabilitiesInternal.hpp"
#include "Compiler.hpp"
#include "ConcreteOperations.hpp"
#include "Network.hpp"
#include "PerformanceData.hpp"

#include <ethosn_utils/Json.hpp>

#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace ethosn::utils;

namespace ethosn
{
namespace support_library
{

namespace
{

TensorAndId<Operand> GetSingleOutputResult(const std::shared_ptr<Network>& network, Operation& op)
{
    // cppcheck-suppress assertWithSideEffect
    assert(op.GetOutputs().size() == 1);
    return { std::shared_ptr<Operand>(network, &op.GetOutput(0)), op.GetId() };
}

TensorsAndId GetMultipleOutputResult(const std::shared_ptr<Network>& network, Operation& op)
{
    std::vector<std::shared_ptr<Operand>> tensors;
    for (Operand& operand : op.GetOutputs())
    {
        tensors.push_back(std::shared_ptr<Operand>(network, &operand));
    }
    return { tensors, op.GetId() };
}

}    // namespace

Version::Version()
    : Major(0)
    , Minor(0)
    , Patch(0)
{}

Version::Version(const uint32_t Major, const uint32_t Minor, const uint32_t Patch)
    : Major(Major)
    , Minor(Minor)
    , Patch(Patch)
{}

Version::Version(const char* version)
    : Major(0)
    , Minor(0)
    , Patch(0)
{
    std::stringstream ss(version);
    char d1, d2;

    ss >> Major >> d1 >> Minor >> d2 >> Patch;
    if (ss.fail() || d1 != '.' || d2 != '.')
    {
        throw std::invalid_argument(std::string("Invalid version string") + version);
    }
}

const std::string Version::ToString() const
{
    std::stringstream ss;
    ss << Major << "." << Minor << "." << Patch;
    return ss.str();
}

const Version GetLibraryVersion()
{
    return Version(ETHOSN_SUPPORT_LIBRARY_VERSION_MAJOR, ETHOSN_SUPPORT_LIBRARY_VERSION_MINOR,
                   ETHOSN_SUPPORT_LIBRARY_VERSION_PATCH);
}

std::vector<char> GetFwAndHwCapabilities(EthosNVariant variant, uint32_t sramSizeBytes)
{
    FirmwareAndHardwareCapabilities capabilities = GetEthosN78FwHwCapabilities(variant, sramSizeBytes);

    std::vector<char> ret;
    ret.resize(sizeof(capabilities));
    std::copy_n(reinterpret_cast<const char*>(&capabilities), sizeof(capabilities), &ret[0]);

    return ret;
}

std::shared_ptr<Network> CreateNetwork(const std::vector<char>& caps)
{
    return std::make_shared<Network>(caps);
}

std::shared_ptr<Network> CreateEstimationNetwork(const std::vector<char>& caps)
{
    return std::make_shared<Network>(caps, true);
}

TensorAndId<Operand> AddInput(const std::shared_ptr<Network>& network, const TensorInfo& info)
{
    Input& input = network->AddInput(info);
    return { std::shared_ptr<Operand>(network, &input.GetOutput(0)), input.GetId() };
}

TensorAndId<Output> AddOutput(const std::shared_ptr<Network>& network, Operand& operand, const DataFormat outputFormat)
{
    Output& output = network->AddOutput(operand, outputFormat);
    return { std::shared_ptr<Output>(network, &output), output.GetId() };
}

TensorAndId<Constant> AddConstant(const std::shared_ptr<Network>& network, const TensorInfo& info, const void* data)
{
    Constant& constant = network->AddConstant(info, data);
    return { std::shared_ptr<Constant>(network, &constant), constant.GetId() };
}

std::shared_ptr<Operand> GetOperand(const std::shared_ptr<Constant>& constant)
{
    return std::shared_ptr<Operand>(constant, &constant->GetOutput(0));
}

TensorAndId<Operand> AddConvolution(const std::shared_ptr<Network>& network,
                                    Operand& input,
                                    Constant& bias,
                                    Constant& weights,
                                    const ConvolutionInfo& convInfo)
{
    return GetSingleOutputResult(network, network->AddConvolution(input, bias, weights, convInfo));
}

TensorAndId<Operand> AddDepthwiseConvolution(const std::shared_ptr<Network>& network,
                                             Operand& input,
                                             Constant& bias,
                                             Constant& weights,
                                             const ConvolutionInfo& convInfo)
{
    return GetSingleOutputResult(network, network->AddDepthwiseConvolution(input, bias, weights, convInfo));
}

TensorAndId<Operand>
    AddStandalonePadding(const std::shared_ptr<Network>& network, Operand& input, const Padding& padding)
{
    return GetSingleOutputResult(network, network->AddStandalonePadding(input, padding));
}

TensorAndId<Operand> AddTransposeConvolution(const std::shared_ptr<Network>& network,
                                             Operand& input,
                                             Constant& bias,
                                             Constant& weights,
                                             const ConvolutionInfo& convInfo)
{
    return GetSingleOutputResult(network, network->AddTransposeConvolution(input, bias, weights, convInfo));
}

TensorAndId<Operand> AddConcatenation(const std::shared_ptr<Network>& network,
                                      const std::vector<Operand*>& layers,
                                      const ConcatenationInfo& concatInfo)
{
    return GetSingleOutputResult(network, network->AddConcatenation(layers, concatInfo));
}

TensorsAndId AddSplit(const std::shared_ptr<Network>& network, Operand& input, const SplitInfo& splitInfo)
{
    return GetMultipleOutputResult(network, network->AddSplit(input, splitInfo));
}

TensorAndId<Operand> AddAddition(const std::shared_ptr<Network>& network,
                                 Operand& layer1,
                                 Operand& layer2,
                                 const QuantizationInfo& outputQuantizationInfo)
{
    return GetSingleOutputResult(network, network->AddAddition(layer1, layer2, outputQuantizationInfo));
}

TensorAndId<Operand> AddMultiplication(const std::shared_ptr<Network>& network,
                                       Operand& layer1,
                                       Operand& layer2,
                                       const QuantizationInfo& outputQuantizationInfo)
{
    return GetSingleOutputResult(network, network->AddMultiplication(layer1, layer2, outputQuantizationInfo));
}

TensorAndId<Operand> AddFullyConnected(const std::shared_ptr<Network>& network,
                                       Operand& input,
                                       Constant& bias,
                                       Constant& weights,
                                       const FullyConnectedInfo fullyConnectedInfo)
{
    return GetSingleOutputResult(network, network->AddFullyConnected(input, bias, weights, fullyConnectedInfo));
}

TensorAndId<Operand> AddReinterpretQuantization(const std::shared_ptr<Network>& network,
                                                Operand& input,
                                                const ReinterpretQuantizationInfo& reinterpretQuantizationInfo)
{
    return GetSingleOutputResult(network, network->AddReinterpretQuantization(input, reinterpretQuantizationInfo));
}

TensorAndId<Operand> AddRelu(const std::shared_ptr<Network>& network, Operand& input, const ReluInfo& reluInfo)
{
    return GetSingleOutputResult(network, network->AddRelu(input, reluInfo));
}

TensorAndId<Operand>
    AddLeakyRelu(const std::shared_ptr<Network>& network, Operand& input, const LeakyReluInfo& leakyReluInfo)
{
    return GetSingleOutputResult(network, network->AddLeakyRelu(input, leakyReluInfo));
}

TensorAndId<Operand>
    AddRequantize(const std::shared_ptr<Network>& network, Operand& input, const RequantizeInfo& requantizeInfo)
{
    return GetSingleOutputResult(network, network->AddRequantize(input, requantizeInfo));
}

TensorAndId<Operand> AddSigmoid(const std::shared_ptr<Network>& network, Operand& input)
{
    return GetSingleOutputResult(network, network->AddSigmoid(input));
}

TensorAndId<Operand> AddTanh(const std::shared_ptr<Network>& network, Operand& input)
{
    return GetSingleOutputResult(network, network->AddTanh(input));
}

TensorAndId<Operand> AddMeanXy(const std::shared_ptr<Network>& network, Operand& input)
{
    return GetSingleOutputResult(network, network->AddMeanXy(input));
}

TensorAndId<Operand> AddPooling(const std::shared_ptr<Network>& network, Operand& input, const PoolingInfo& poolingInfo)
{
    return GetSingleOutputResult(network, network->AddPooling(input, poolingInfo));
}

TensorAndId<Operand>
    AddReshape(const std::shared_ptr<Network>& network, Operand& input, const TensorShape& newDimensions)
{
    return GetSingleOutputResult(network, network->AddReshape(input, newDimensions));
}

TensorAndId<Operand>
    AddDepthToSpace(const std::shared_ptr<Network>& network, Operand& input, const DepthToSpaceInfo& depthToSpaceInfo)
{
    return GetSingleOutputResult(network, network->AddDepthToSpace(input, depthToSpaceInfo));
}

TensorAndId<Operand>
    AddSpaceToDepth(const std::shared_ptr<Network>& network, Operand& input, const SpaceToDepthInfo& spaceToDepthInfo)
{
    return GetSingleOutputResult(network, network->AddSpaceToDepth(input, spaceToDepthInfo));
}

TensorAndId<Operand>
    AddTranspose(const std::shared_ptr<Network>& network, Operand& input, const TransposeInfo& transposeInfo)
{
    return GetSingleOutputResult(network, network->AddTranspose(input, transposeInfo));
}

TensorAndId<Operand> AddResize(const std::shared_ptr<Network>& network, Operand& input, const ResizeInfo& resizeInfo)
{
    if (resizeInfo.m_Algo == ResizeAlgorithm::BILINEAR && input.GetTensorInfo().m_DataType == DataType::INT8_QUANTIZED)
    {
        /* Workaround for an MCE issue where, for bilinear resizing, regardless of input type the input is treated as unsigned.
         * This results in errors when dealing with negative numbers. To fix this the input is shifted into unsigned range
         * before resizing, and shifted back to int8 range after resizing. */
        QuantizationInfo inputInfo = input.GetTensorInfo().m_QuantizationInfo;
        int32_t newZeroPoint       = inputInfo.GetZeroPoint() + 128;

        RequantizeInfo uintQuantInfo     = RequantizeInfo(QuantizationInfo(newZeroPoint, inputInfo.GetScale()));
        uintQuantInfo.m_OutputDataType   = DataType::UINT8_QUANTIZED;
        TensorAndId<Operand> resizeInput = GetSingleOutputResult(network, network->AddRequantize(input, uintQuantInfo));

        ResizeInfo newInfo(resizeInfo);
        newInfo.m_OutputQuantizationInfo.SetZeroPoint(newZeroPoint);
        TensorAndId<Operand> intQuantInput =
            GetSingleOutputResult(network, network->AddResize(*resizeInput.tensor, newInfo));

        RequantizeInfo intQuantInfo = RequantizeInfo(QuantizationInfo(inputInfo.GetZeroPoint(), inputInfo.GetScale()));
        intQuantInfo.m_OutputDataType = DataType::INT8_QUANTIZED;
        TensorAndId<Operand> operationOutput =
            GetSingleOutputResult(network, network->AddRequantize(*intQuantInput.tensor, intQuantInfo));

        return operationOutput;
    }
    else
    {
        return GetSingleOutputResult(network, network->AddResize(input, resizeInfo));
    }
}

TensorsAndId AddEstimateOnly(const std::shared_ptr<Network>& network,
                             const std::vector<Operand*>& inputs,
                             const EstimateOnlyInfo& estimateOnly)
{
    return GetMultipleOutputResult(network, network->AddEstimateOnly(inputs, estimateOnly));
}

TensorInfo GetTensorInfo(const std::shared_ptr<Operand>& operand)
{
    return operand->GetTensorInfo();
}

std::vector<std::unique_ptr<CompiledNetwork>> Compile(const Network& network, const CompilationOptions& options)
{
    FirmwareAndHardwareCapabilities caps = GetValidCapabilities(network.GetCapabilities());

    if (!AreCapabilitiesSupported(caps))
    {
        throw NotSupportedException("Support library does not support compilation for the given target capabilities");
    }

    Compiler compiler(network, caps, options, utils::EmptyOptional());

    std::vector<std::unique_ptr<CompiledNetwork>> result;
    try
    {
        result.push_back(compiler.Compile().compiledOpGraph.m_CompiledNetwork);
    }
    catch (const std::runtime_error& e)
    {
        // Need better approach for error reporting from support library.
        g_Logger.Error("Error: %s", e.what());
    }
    return result;
}

NetworkPerformanceData EstimatePerformance(const Network& network,
                                           const CompilationOptions& compilationOptions,
                                           const EstimationOptions& estimationOptions)
{
    FirmwareAndHardwareCapabilities caps = GetValidCapabilities(network.GetCapabilities());

    if (!AreCapabilitiesSupported(caps))
    {
        throw NotSupportedException("Support library does not support compilation for the given target capabilities");
    }

    Compiler compiler(network, caps, compilationOptions, estimationOptions);

    try
    {
        return compiler.Compile().GetLegacyNetworkPerformanceData();
    }
    catch (const std::exception& e)
    {
        g_Logger.Warning("Estimation failed with: %s", e.what());
        throw NotSupportedException("Estimation didn't find any valid performance data to return");
    }
}

void PrintNetworkPerformanceDataJson(std::ostream& os, uint32_t indentNumTabs, const NetworkPerformanceData& perfData)
{
    Indent indent(indentNumTabs);

    os << indent << "{\n";
    ++indent;

    const auto printPass = [indent](std::ostream& os, const PassPerformanceData& pass) {
        PrintPassPerformanceData(os, Indent(indent + 1), pass);
    };

    os << indent << JsonField("Stream") << '\n';
    Print(os, indent, JsonArray(perfData.m_Stream), printPass, true) << ",\n";

    os << indent << JsonField("Issues") << '\n';
    PrintFailureReasons(os, indent, perfData.m_OperationIdFailureReasons) << '\n';

    --indent;
    os << indent << "}\n";
}

const char* EthosNVariantAsString(EthosNVariant npuType)
{
    switch (npuType)
    {
        case EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO:
            return "Ethos-N78_1TOPS_2PLE_RATIO";
        case EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO:
            return "Ethos-N78_1TOPS_4PLE_RATIO";
        case EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO:
            return "Ethos-N78_2TOPS_2PLE_RATIO";
        case EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO:
            return "Ethos-N78_2TOPS_4PLE_RATIO";
        case EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO:
            return "Ethos-N78_4TOPS_2PLE_RATIO";
        case EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO:
            return "Ethos-N78_4TOPS_4PLE_RATIO";
        case EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO:
            return "Ethos-N78_8TOPS_2PLE_RATIO";
        default:
            return "Unknown NPU type";
    }
}

EthosNVariant EthosNVariantFromString(const char* npuType)
{
    if (!std::strcmp(npuType, EthosNVariantAsString(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO)))
    {
        return EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO;
    }
    else if (!std::strcmp(npuType, EthosNVariantAsString(EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO)))
    {
        return EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO;
    }
    else if (!std::strcmp(npuType, EthosNVariantAsString(EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO)))
    {
        return EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO;
    }
    else if (!std::strcmp(npuType, EthosNVariantAsString(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO)))
    {
        return EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO;
    }
    else if (!std::strcmp(npuType, EthosNVariantAsString(EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO)))
    {
        return EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO;
    }
    else if (!std::strcmp(npuType, EthosNVariantAsString(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO)))
    {
        return EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
    }
    else if (!std::strcmp(npuType, EthosNVariantAsString(EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO)))
    {
        return EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO;
    }
    else
    {
        throw std::invalid_argument("Unknown NPU type");
    }
}

namespace
{
template <typename Op>
QuantizationScales ApplyWithBroadcast(const QuantizationScales& lhs, const QuantizationScales& rhs, Op op)
{
    if (lhs.size() == 1)
    {
        return QuantizationScales(op(lhs[0], static_cast<const std::valarray<float>&>(rhs)));
    }
    else if (rhs.size() == 1)
    {
        return QuantizationScales(op(static_cast<const std::valarray<float>&>(lhs), rhs[0]));
    }
    else
    {
        return QuantizationScales(
            op(static_cast<const std::valarray<float>&>(lhs), static_cast<const std::valarray<float>&>(rhs)));
    }
}
}    // namespace

QuantizationScales operator/(const QuantizationScales& lhs, const QuantizationScales& rhs)
{
    return ApplyWithBroadcast(lhs, rhs, [](auto& x, auto& y) { return x / y; });
}

QuantizationScales operator*(const QuantizationScales& lhs, const QuantizationScales& rhs)
{
    return ApplyWithBroadcast(lhs, rhs, [](auto& x, auto& y) { return x * y; });
}

bool operator==(const QuantizationScales& lhs, const QuantizationScales& rhs)
{
    return std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs), std::end(rhs));
}

bool operator!=(const QuantizationScales& lhs, const QuantizationScales& rhs)
{
    return !(lhs == rhs);
}

InternalErrorException::InternalErrorException(const char* msg)
    : std::runtime_error(std::string("Internal compiler error: ") + msg)
{}

}    // namespace support_library
}    // namespace ethosn
