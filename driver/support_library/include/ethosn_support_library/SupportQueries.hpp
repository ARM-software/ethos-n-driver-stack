//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Support.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>

// Declares class API to check if a layer known by the library is supported with the given
// set of inputs and configuration parameters.

namespace ethosn
{
namespace support_library
{

namespace
{
constexpr size_t g_ReasonMaxLength = 1024;
}    // namespace

// IsSupported checks return a class which provides an overloaded operator bool for backwards compatibility
// and also allows for extensions with methods.
class SupportedLevel
{
private:
    enum class InternalSupportedLevel
    {
        Unsupported,
        EstimateOnly,
        Supported,
    };

public:
    explicit operator bool() const
    {
        return m_SupportedLevel == InternalSupportedLevel::Supported;
    }

    bool operator==(const SupportedLevel rhs) const
    {
        return this->m_SupportedLevel == rhs.m_SupportedLevel;
    }

    // Static members used for comparison
    static const SupportedLevel Unsupported;
    static const SupportedLevel EstimateOnly;
    static const SupportedLevel Supported;

private:
    SupportedLevel(InternalSupportedLevel level)
        : m_SupportedLevel(level)
    {}
    InternalSupportedLevel m_SupportedLevel;

    // Work around for issue in gcc for -Werror=ctor-dtor-privacy
    friend class unused;
};

/// Support Queries class API
//
/// If the given configuration is not supported then a 'reason'
/// string can optionally be returned, with a human-readable description of the reason.
///
/// If an 'outputInfo' pointer is provided then it will be updated with the TensorInfo that the output of the layer
/// will have. If the provided TensorInfo is already valid (i.e. all of its shape elements are non-zero), then
/// it will be validated against the internally calculated outputInfo and cause the function to return false if it
/// does not match.
///
/// For operations which have an array of outputs (e.g. Split), a pointer to an *array* of TensorInfos can be provided.
/// If provided, each element of this array will be updated or validated according to the above rules.
class SupportQueries
{
private:
    // Hardware capabilities
    std::vector<char> m_Capabilities;
    bool m_ForceExperimentalCompiler;

public:
    /// Create an instance of SupportQueries for the given capabilities
    ///
    /// @param caps The capabilities vector
    /// @exception Throws ethosn::ethosn_library::VersionMismatchException if capabilities are invalid.
    SupportQueries(const std::vector<char>& caps);
    SupportQueries(const std::vector<char>& caps, bool forceExperimentalCompiler);

    /// Get capabilities vector
    const std::vector<char>& GetCapabilities() const
    {
        return m_Capabilities;
    }

    /// Checks whether a specific input operation configuration is supported by the NPU.
    /// @param inputInfo The TensorInfo of the tensor that this Input operation will produce.
    ///                  This is the size of the Tensor that must be provided to the driver library at inference time.
    ///                  It is the equivalent of the convInfo parameter for IsConvolutionSupported(),
    ///                  NOT the equivalent of the inputInfo parameter for IsConvolutionSupported() -
    ///                  i.e. it contains information to derive the result of the Operation and
    ///                  does not describe the input to the Operation (as the Input operation does not have an input).
    /// @param outputInfo (Optional) See comment at top of this file.
    ///                   This is the equivalent of the outputInfo parameter for IsConvolutionSupported()
    ///                   (whose output is derived from the convInfo, weightsInfo etc.), but for an Input operation the
    ///                   output info is exactly what is configured by inputInfo and therefore this parameter is redundant.
    ///                   However this is included for consistency with the other operations.
    SupportedLevel IsInputSupported(const TensorInfo& inputInfo,
                                    TensorInfo* outputInfo = nullptr,
                                    char* reason           = nullptr,
                                    size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific output operation configuration is supported by the NPU
    /// @param inputInfo The TensorInfo of the tensor that this Output will expose.
    ///                  This is the size of the Tensor that will be provided back to the user of the driver library
    ///                  at inference time.
    /// @param format   The data format of the Output.
    SupportedLevel IsOutputSupported(const TensorInfo& inputInfo,
                                     const DataFormat format,
                                     char* reason           = nullptr,
                                     size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific constant operation configuration is supported by the NPU
    SupportedLevel IsConstantSupported(const TensorInfo& info,
                                       char* reason           = nullptr,
                                       size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific convolution operation configuration is supported by the NPU
    SupportedLevel IsConvolutionSupported(const TensorInfo& biasInfo,
                                          const TensorInfo& weightsInfo,
                                          const ConvolutionInfo& convInfo,
                                          const TensorInfo& inputInfo,
                                          TensorInfo* outputInfo = nullptr,
                                          char* reason           = nullptr,
                                          size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific depthwise convolution operation configuration is supported by the NPU
    SupportedLevel IsDepthwiseConvolutionSupported(const TensorInfo& biasInfo,
                                                   const TensorInfo& weightsInfo,
                                                   const ConvolutionInfo& convInfo,
                                                   const TensorInfo& inputInfo,
                                                   TensorInfo* outputInfo = nullptr,
                                                   char* reason           = nullptr,
                                                   size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific transpose convolution operation configuration is supported by the NPU
    SupportedLevel IsTransposeConvolutionSupported(const TensorInfo& biasInfo,
                                                   const TensorInfo& weightsInfo,
                                                   const ConvolutionInfo& convInfo,
                                                   const TensorInfo& inputInfo,
                                                   TensorInfo* outputInfo = nullptr,
                                                   char* reason           = nullptr,
                                                   size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific concatenation operation configuration is supported by the NPU
    SupportedLevel IsConcatenationSupported(const std::vector<TensorInfo>& inputInfos,
                                            const ConcatenationInfo& concatInfo,
                                            TensorInfo* outputInfo = nullptr,
                                            char* reason           = nullptr,
                                            size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific split operation configuration is supported by the NPU
    SupportedLevel IsSplitSupported(const TensorInfo& inputInfo,
                                    const SplitInfo& splitInfo,
                                    std::vector<TensorInfo>* outputInfos = nullptr,
                                    char* reason                         = nullptr,
                                    size_t reasonMaxLength               = g_ReasonMaxLength) const;

    // Checks whether a specific addition (tensor + tensor) operation configuration is supported by the NPU
    SupportedLevel IsAdditionSupported(const TensorInfo& inputInfo0,
                                       const TensorInfo& inputInfo1,
                                       const QuantizationInfo& outputQuantizationInfo,
                                       TensorInfo* outputInfo = nullptr,
                                       char* reason           = nullptr,
                                       size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific fully-connected operation configuration is supported by the NPU
    SupportedLevel IsFullyConnectedSupported(const TensorInfo& biasInfo,
                                             const TensorInfo& weightsInfo,
                                             const FullyConnectedInfo& fullyConnectedInfo,
                                             const TensorInfo& inputInfo,
                                             TensorInfo* outputInfo = nullptr,
                                             char* reason           = nullptr,
                                             size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific relu operation configuration is supported by the NPU
    SupportedLevel IsReluSupported(const ReluInfo& reluInfo,
                                   const TensorInfo& inputInfo,
                                   TensorInfo* outputInfo = nullptr,
                                   char* reason           = nullptr,
                                   size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific leaky relu operation configuration is supported by the NPU
    SupportedLevel IsLeakyReluSupported(const LeakyReluInfo& leakyReluInfo,
                                        const TensorInfo& inputInfo,
                                        TensorInfo* outputInfo = nullptr,
                                        char* reason           = nullptr,
                                        size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific requantize operation configuration is supported by the NPU
    SupportedLevel IsRequantizeSupported(const RequantizeInfo& requantizeInfo,
                                         const TensorInfo& inputInfo,
                                         TensorInfo* outputInfo = nullptr,
                                         char* reason           = nullptr,
                                         size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Check whether a specific reinterpret quantization operation configuration is supported by the NPU
    SupportedLevel IsReinterpretQuantizationSupported(const ReinterpretQuantizationInfo& reinterpretQuantizationInfo,
                                                      const TensorInfo& inputInfo,
                                                      TensorInfo* outputInfo = nullptr,
                                                      char* reason           = nullptr,
                                                      size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific sigmoid operation configuration is supported by the NPU
    SupportedLevel IsSigmoidSupported(const TensorInfo& inputInfo,
                                      TensorInfo* outputInfo = nullptr,
                                      char* reason           = nullptr,
                                      size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific tanh operation configuration is supported by the NPU
    SupportedLevel IsTanhSupported(const TensorInfo& inputInfo,
                                   TensorInfo* outputInfo = nullptr,
                                   char* reason           = nullptr,
                                   size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific mean operation (across width and height) configuration is supported by the NPU
    SupportedLevel IsMeanXySupported(const TensorInfo& inputInfo,
                                     TensorInfo* outputInfo = nullptr,
                                     char* reason           = nullptr,
                                     size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific pooling operation configuration is supported by the NPU
    SupportedLevel IsPoolingSupported(const PoolingInfo& poolingInfo,
                                      const TensorInfo& inputInfo,
                                      TensorInfo* outputInfo = nullptr,
                                      char* reason           = nullptr,
                                      size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific reshape operation configuration is supported by the NPU
    SupportedLevel IsReshapeSupported(const TensorShape& newDimensions,
                                      const TensorInfo& inputInfo,
                                      TensorInfo* outputInfo = nullptr,
                                      char* reason           = nullptr,
                                      size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific DepthToSpace operation configuration is supported by the NPU
    SupportedLevel IsDepthToSpaceSupported(const TensorInfo& inputInfo,
                                           const DepthToSpaceInfo& depthToSpaceInfo,
                                           TensorInfo* outputInfo = nullptr,
                                           char* reason           = nullptr,
                                           size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific SpaceToDepth operation configuration is supported by the NPU
    SupportedLevel IsSpaceToDepthSupported(const TensorInfo& inputInfo,
                                           const SpaceToDepthInfo& spaceToDepthInfo,
                                           TensorInfo* outputInfo = nullptr,
                                           char* reason           = nullptr,
                                           size_t reasonMaxLength = g_ReasonMaxLength) const;

    // Checks whether a specific EstimateOnly operation configuration is supported by the NPU
    SupportedLevel IsEstimateOnlySupported(const std::vector<TensorInfo>& inputInfos,
                                           const EstimateOnlyInfo& estimateOnlyInfo,
                                           std::vector<TensorInfo>* outputInfos = nullptr,
                                           char* reason                         = nullptr,
                                           size_t reasonMaxLength               = g_ReasonMaxLength) const;

    // Checks whether a specific Transpose operation configuration is supported by the NPU
    SupportedLevel IsTransposeSupported(const TransposeInfo& transposeInfo,
                                        const TensorInfo& inputInfo,
                                        TensorInfo* outputInfo = nullptr,
                                        char* reason           = nullptr,
                                        size_t reasonMaxLength = g_ReasonMaxLength) const;

    SupportedLevel IsResizeSupported(const ResizeInfo& resizeInfo,
                                     const TensorInfo& inputInfo,
                                     TensorInfo* outputInfo = nullptr,
                                     char* reason           = nullptr,
                                     size_t reasonMaxLength = g_ReasonMaxLength) const;
};

}    // namespace support_library
}    // namespace ethosn
