//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"

#include "CapabilitiesInternal.hpp"
#include "Compiler.hpp"
#include "Graph.hpp"
#include "Network.hpp"
#include "Pass.hpp"

#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ethosn
{
namespace support_library
{

namespace
{

TensorAndId<Operand> GetSingleOutputResult(const std::shared_ptr<Network>& network, Operation& op)
{
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

template <typename T>
struct QuotedT
{
    explicit constexpr QuotedT(const T& value)
        : m_Value(value)
    {}

    const T& m_Value;
};

template <typename T>
QuotedT<T> Quoted(const T& value)
{
    return QuotedT<T>(value);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const QuotedT<T>& field)
{
    return os << '"' << field.m_Value << '"';
}

template <typename T>
struct JsonFieldT
{
    explicit constexpr JsonFieldT(const T& value)
        : m_Value(value)
    {}

    const T& m_Value;
};

template <typename T>
JsonFieldT<T> JsonField(const T& value)
{
    return JsonFieldT<T>(value);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const JsonFieldT<T>& field)
{
    return os << Quoted(field.m_Value) << ':';
}

struct Indent
{
    explicit constexpr Indent(const size_t depth)
        : m_Depth(depth)
    {}

    constexpr operator size_t&()
    {
        return m_Depth;
    }

    constexpr operator size_t() const
    {
        return m_Depth;
    }

    size_t m_Depth;
};

std::ostream& operator<<(std::ostream& os, const Indent& indent)
{
    for (size_t i = 0; i < indent; ++i)
    {
        os << '\t';
    }

    return os;
}

template <typename T>
struct JsonArrayT
{
    explicit constexpr JsonArrayT(const T& value)
        : m_Value(value)
    {}

    const T& m_Value;
};

template <typename T>
JsonArrayT<T> JsonArray(const T& value)
{
    return JsonArrayT<T>(value);
}

template <typename T, typename PrintFn>
std::ostream& Print(
    std::ostream& os, const Indent indent, const JsonArrayT<T>& array, PrintFn&& printFn, const bool multiline = false)
{
    const char sep = multiline ? '\n' : ' ';

    os << indent << '[' << sep;

    for (auto it = array.m_Value.begin(); it != array.m_Value.end(); ++it)
    {
        printFn(os, *it);

        if (it != std::prev(array.m_Value.end()))
        {
            os << ',';
        }

        os << sep;
    }

    if (multiline)
    {
        os << indent;
    }

    os << ']';

    return os;
}

template <typename T>
std::ostream& Print(std::ostream& os, const Indent indent, const JsonArrayT<T>& array, const bool multiline = false)
{
    return Print(os, indent, array, [](std::ostream& os, const auto& value) { os << value; }, multiline);
}

std::ostream& Print(std::ostream& os, Indent indent, const MemoryStats& stats)
{
    os << indent << JsonField("DramParallelBytes") << ' ' << stats.m_DramParallel << ",\n";
    os << indent << JsonField("DramNonParallelBytes") << ' ' << stats.m_DramNonParallel << ",\n";
    os << indent << JsonField("SramBytes") << ' ' << stats.m_Sram;
    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const StripesStats& stats)
{
    os << indent << JsonField("NumCentralStripes") << ' ' << stats.m_NumCentralStripes << ",\n";
    os << indent << JsonField("NumBoundaryStripes") << ' ' << stats.m_NumBoundaryStripes << ",\n";
    os << indent << JsonField("NumReloads") << ' ' << stats.m_NumReloads;
    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const InputStats& stats)
{
    os << indent << "{\n";

    ++indent;

    Print(os, indent, stats.m_MemoryStats);
    os << ",\n";
    Print(os, indent, stats.m_StripesStats);
    os << "\n";

    --indent;

    os << indent << "}";

    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const WeightsStats& stats)
{
    os << indent << "{\n";

    ++indent;

    Print(os, indent, stats.m_MemoryStats);
    os << ",\n";
    Print(os, indent, stats.m_StripesStats);
    os << ",\n";
    os << indent << JsonField("CompressionSavings") << ' ' << stats.m_WeightCompressionSavings << "\n";

    --indent;

    os << indent << "}";

    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const MceStats& mceStats)
{
    os << indent << "{\n";

    ++indent;

    os << indent << JsonField("Operations") << ' ' << mceStats.m_Operations << ",\n";
    os << indent << JsonField("CycleCount") << ' ' << mceStats.m_CycleCount << "\n";

    --indent;

    os << indent << "}";

    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const PleStats& pleStats)
{
    os << indent << "{\n";

    ++indent;

    os << indent << JsonField("NumOfPatches") << ' ' << pleStats.m_NumOfPatches << ",\n";
    os << indent << JsonField("Operation") << ' ' << pleStats.m_Operation << "\n";

    --indent;

    os << indent << "}";

    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const PassPerformanceData& pass)
{
    os << indent << "{\n";

    ++indent;

    os << indent << JsonField("OperationIds") << ' ';
    Print(os, Indent(0), JsonArray(pass.m_OperationIds)) << ",\n";

    os << indent << JsonField("ParentIds") << ' ' << pass.m_ParentIds << ",\n";

    os << indent << JsonField("Input") << '\n';
    Print(os, indent, pass.m_Stats.m_Input) << ",\n";

    os << indent << JsonField("Output") << '\n';
    Print(os, indent, pass.m_Stats.m_Output) << ",\n";

    os << indent << JsonField("Weights") << '\n';
    Print(os, indent, pass.m_Stats.m_Weights) << ",\n";

    os << indent << JsonField("Mce") << '\n';
    Print(os, indent, pass.m_Stats.m_Mce) << ",\n";

    os << indent << JsonField("Ple") << '\n';
    Print(os, indent, pass.m_Stats.m_Ple) << "\n";

    --indent;

    os << indent << "}";

    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const std::map<uint32_t, std::string>& failureReasons)
{
    os << indent << "{\n";

    ++indent;

    for (auto it = failureReasons.begin(); it != failureReasons.end(); ++it)
    {
        os << indent << JsonField(it->first) << ' ' << Quoted(it->second);

        if (it != std::prev(failureReasons.end()))
        {
            os << ",\n";
        }
        else
        {
            os << "\n";
        }
    }

    --indent;

    os << indent << "}";

    return os;
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

std::vector<char> GetPerformanceEstimatorFwAndHwCapabilities(EthosNVariant variant, uint32_t sramSizeBytes)
{
    FirmwareAndHardwareCapabilities capabilities = GetEthosN77FwHwCapabilities();

    if (variant == EthosNVariant::ETHOS_N57)
    {
        capabilities = GetEthosN57FwHwCapabilities();
    }
    else if (variant == EthosNVariant::ETHOS_N37)
    {
        capabilities = GetEthosN37FwHwCapabilities();
    }

    if (sramSizeBytes > 0)
    {
        capabilities.m_TotalSramSize = sramSizeBytes;
    }

    std::vector<char> ret;
    ret.resize(sizeof(capabilities));
    std::copy_n(reinterpret_cast<const char*>(&capabilities), sizeof(capabilities), &ret[0]);

    return ret;
}

std::shared_ptr<Network> CreateNetwork()
{
    return std::make_shared<Network>();
}

std::shared_ptr<Network> CreateEstimationNetwork()
{
    return std::make_shared<Network>(true);
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

TensorAndId<Operand> AddFullyConnected(const std::shared_ptr<Network>& network,
                                       Operand& input,
                                       Constant& bias,
                                       Constant& weights,
                                       const FullyConnectedInfo fullyConnectedInfo)
{
    return GetSingleOutputResult(network, network->AddFullyConnected(input, bias, weights, fullyConnectedInfo));
}

TensorAndId<Operand> AddRelu(const std::shared_ptr<Network>& network, Operand& input, const ReluInfo& reluInfo)
{
    return GetSingleOutputResult(network, network->AddRelu(input, reluInfo));
}

TensorAndId<Operand> AddSoftmax(const std::shared_ptr<Network>& network, Operand& input)
{
    return GetSingleOutputResult(network, network->AddSoftmax(input));
}

TensorAndId<Operand> AddSigmoid(const std::shared_ptr<Network>& network, Operand& input)
{
    return GetSingleOutputResult(network, network->AddSigmoid(input));
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

bool ValidateCapabilities(const CompilationOptions& options, FirmwareAndHardwareCapabilities& caps)
{
    // Decode the capabilities struct by looking first at the header
    if (options.m_FwAndHwCapabilities.size() < sizeof(FirmwareAndHardwareCapabilitiesHeader))
    {
        // Invalid size.
        return false;
    }
    FirmwareAndHardwareCapabilitiesHeader header;
    memcpy(&header, options.m_FwAndHwCapabilities.data(), sizeof(FirmwareAndHardwareCapabilitiesHeader));
    // For now we support only the current version.
    if (header.m_Size != sizeof(FirmwareAndHardwareCapabilities) || header.m_Version != FW_AND_HW_CAPABILITIES_VERSION)
    {
        // Unsupported version.
        return false;
    }
    // Now we can decode the full struct.
    memcpy(&caps, options.m_FwAndHwCapabilities.data(), sizeof(FirmwareAndHardwareCapabilities));

    return true;
}

std::vector<std::unique_ptr<CompiledNetwork>> Compile(const Network& network, const CompilationOptions& options)
{
    std::vector<std::unique_ptr<CompiledNetwork>> allSupportedSubgraphs;

    FirmwareAndHardwareCapabilities caps;
    if (!ValidateCapabilities(options, caps))
    {
        throw VersionMismatchException("m_FwAndHwCapabilities is not valid");
    }
    EstimationOptions estimationOptions;
    Compiler compiler(network, caps, options, estimationOptions);

    // Here we will loop between all supported subgraphs and call Compile() on them
    //      then add the results to allSupportedSubgraphs.
    std::unique_ptr<CompiledNetwork> compiledNetwork = compiler.Compile();

    // compiler.Compile() can fail in which case we will return nothing and skip adding to the returned subgraphs
    if (compiledNetwork)
    {
        allSupportedSubgraphs.push_back(std::move(compiledNetwork));
    }

    return allSupportedSubgraphs;
}

NetworkPerformanceData EstimatePerformance(const Network& network,
                                           const CompilationOptions& compilationOptions,
                                           const EstimationOptions& estimationOptions)
{
    FirmwareAndHardwareCapabilities caps;
    if (!ValidateCapabilities(compilationOptions, caps))
    {
        throw VersionMismatchException("m_FwAndHwCapabilities is not valid");
    }

    Compiler compiler(network, caps, compilationOptions, estimationOptions);

    return compiler.EstimatePerformance();
}

void PrintNetworkPerformanceDataJson(std::ostream& os, uint32_t indentNumTabs, const NetworkPerformanceData& perfData)
{
    Indent indent(indentNumTabs);

    os << indent << "{\n";
    ++indent;

    const auto printPass = [indent](std::ostream& os, const PassPerformanceData& pass) {
        Print(os, Indent(indent + 1), pass);
    };

    os << indent << JsonField("Stream") << '\n';
    Print(os, indent, JsonArray(perfData.m_Stream), printPass, true) << ",\n";

    os << indent << JsonField("Issues") << '\n';
    Print(os, indent, perfData.m_OperationIdFailureReasons) << '\n';

    --indent;
    os << indent << "}\n";
}

std::unique_ptr<CompiledNetwork> DeserializeCompiledNetwork(std::istream& in)
{
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = std::make_unique<CompiledNetworkImpl>();
    compiledNetwork->Deserialize(in);
    return compiledNetwork;
}

const char* EthosNVariantAsString(EthosNVariant npuType)
{
    switch (npuType)
    {
        case EthosNVariant::ETHOS_N77:
            return "Ethos-N77";
        case EthosNVariant::ETHOS_N57:
            return "Ethos-N57";
        case EthosNVariant::ETHOS_N37:
            return "Ethos-N37";
        default:
            return "Unknown NPU type";
    }
}

EthosNVariant EthosNVariantFromString(const char* npuType)
{
    if (!std::strcmp(npuType, EthosNVariantAsString(EthosNVariant::ETHOS_N77)))
    {
        return EthosNVariant::ETHOS_N77;
    }
    else if (!std::strcmp(npuType, EthosNVariantAsString(EthosNVariant::ETHOS_N57)))
    {
        return EthosNVariant::ETHOS_N57;
    }
    else if (!std::strcmp(npuType, EthosNVariantAsString(EthosNVariant::ETHOS_N37)))
    {
        return EthosNVariant::ETHOS_N37;
    }
    else
    {
        throw std::invalid_argument("Unknown NPU type");
    }
}

namespace debug
{
std::ostream& operator<<(std::ostream& os, Network& network)
{
    os << "Network (" << &network << ")" << std::endl;
    for (const auto& operation : network)
    {
        operation->Print(os);
    }
    os << std::endl;

    return os;
}
}    // namespace debug

}    // namespace support_library
}    // namespace ethosn
