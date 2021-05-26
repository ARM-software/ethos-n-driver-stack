//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNConfig.hpp"
#include "EthosNMapping.hpp"

#include <armnn/backends/ILayerSupport.hpp>
#include <ethosn_support_library/SupportQueries.hpp>

namespace armnn
{

// In performance estimation mode we want to support operations which don't "exist" in the support library,
// so that they can be mapped to operations that we do support.
// Inheriting ILayerSupport instead of LayerSupportBase causes a compilation error if we don't overload one
// of the methods, providing a nice check that we're handling everything properly.
class EthosNLayerSupport : public ILayerSupport
{
public:
    enum class AdditionSupportedMode
    {
        None,      ///< Addition cannot be supported by this backend at all.
        Native,    ///< Addition can be supported by this backend, by using an Addition operation in the support library.
        ReplaceWithDepthwise,    //< Addition can be supported by this backend, by using a DepthwiseConvolution operation in the support library.
        ReplaceWithReinterpretQuantize,    //< Scalar Addition can be supported by this backend, using a Reinterpret Quantization operation in the support library.
    };

    enum class MultiplicationSupportedMode
    {
        None,                    ///<Multiplication cannot be supported by this backend at all.
        ReplaceWithDepthwise,    ///<Multiplication can be supported by this backend, by replacing Multiplication operation with a DepthwiseConvolution operation in the support library.
        ReplaceWithReinterpretQuantize,    ///<Multiplication can be supported by this backend, by replacing Multiplication operation with a ReinterpretQuantize operation in the Support Library.
        EstimateOnly,                      //<Estimate only support
    };

    EthosNLayerSupport(const EthosNConfig& config,
                       const EthosNMappings& mappings,
                       const std::vector<char>& capabilities);

    bool IsActivationSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               const ActivationDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsAdditionSupported(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    /// Provides more detail than IsAdditionSupported(), by stating *how* the addition can be supported
    /// (native vs depthwise replacement)
    AdditionSupportedMode GetAdditionSupportedMode(const TensorInfo& input0,
                                                   const TensorInfo& input1,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                           const TensorInfo& output,
                           const OriginsDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsConstantSupported(const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsConvolution2dSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const Convolution2dDescriptor& descriptor,
                                  const TensorInfo& weights,
                                  const Optional<TensorInfo>& biases,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const DepthwiseConvolution2dDescriptor& descriptor,
                                         const TensorInfo& weights,
                                         const Optional<TensorInfo>& biases,
                                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsTransposeConvolution2dSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const TransposeConvolution2dDescriptor& descriptor,
                                           const TensorInfo& weights,
                                           const Optional<TensorInfo>& biases,
                                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsFullyConnectedSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& weights,
                                   const TensorInfo& biases,
                                   const FullyConnectedDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsInputSupported(const TensorInfo& input,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsMemCopySupported(const TensorInfo& input,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsOutputSupported(const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsPooling2dSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const Pooling2dDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsPreCompiledSupported(const TensorInfo& input,
                                const PreCompiledDescriptor& descriptor,
                                Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsRankSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsReduceSupported(const TensorInfo& input,
                           const TensorInfo& output,
                           const ReduceDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsReshapeSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const ReshapeDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsSoftmaxSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const SoftmaxDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsSplitterSupported(const TensorInfo& input,
                             const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                             const ViewsDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsDepthToSpaceSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const DepthToSpaceDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    // We return true for all these and we substitute estimate only operations later.
    bool IsAbsSupported(const TensorInfo& input,
                        const TensorInfo& output,
                        Optional<std::string&> reasonIfUnsupported) const override;

    bool IsArgMinMaxSupported(const armnn::TensorInfo& input,
                              const armnn::TensorInfo& output,
                              const armnn::ArgMinMaxDescriptor& descriptor,
                              armnn::Optional<std::string&> reasonIfUnsupported) const override;

    bool IsBatchNormalizationSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const TensorInfo& mean,
                                       const TensorInfo& var,
                                       const TensorInfo& beta,
                                       const TensorInfo& gamma,
                                       const BatchNormalizationDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const override;

    bool IsBatchToSpaceNdSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const BatchToSpaceNdDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported) const override;

    bool IsCastSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsComparisonSupported(const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
                               const ComparisonDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported) const override;

    bool IsConvertBf16ToFp32Supported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsConvertFp32ToBf16Supported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported) const override;

    bool IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported) const override;

    bool IsDebugSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported) const override;

    bool IsDequantizeSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               Optional<std::string&> reasonIfUnsupported) const override;

    bool IsDetectionPostProcessSupported(const TensorInfo& boxEncodings,
                                         const TensorInfo& scores,
                                         const TensorInfo& anchors,
                                         const TensorInfo& detectionBoxes,
                                         const TensorInfo& detectionClasses,
                                         const TensorInfo& detectionScores,
                                         const TensorInfo& numDetections,
                                         const DetectionPostProcessDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsDilatedDepthwiseConvolutionSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const DepthwiseConvolution2dDescriptor& descriptor,
                                                const TensorInfo& weights,
                                                const Optional<TensorInfo>& biases,
                                                Optional<std::string&> reasonIfUnsupported) const override;

    bool IsDivisionSupported(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported) const override;

    bool IsElementwiseUnarySupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const ElementwiseUnaryDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsEqualSupported(const armnn::TensorInfo& input0,
                          const armnn::TensorInfo& input1,
                          const armnn::TensorInfo& output,
                          armnn::Optional<std::string&> reasonIfUnsupported) const override;

    bool IsFakeQuantizationSupported(const TensorInfo& input,
                                     const FakeQuantizationDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const override;

    bool IsFillSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         const FillDescriptor& descriptor,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsFloorSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported) const override;

    bool IsGatherSupported(const armnn::TensorInfo& input0,
                           const armnn::TensorInfo& input1,
                           const armnn::TensorInfo& output,
                           armnn::Optional<std::string&> reasonIfUnsupported) const override;

    bool IsGatherSupported(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output,
                           const GatherDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsGreaterSupported(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported) const override;

    bool IsInstanceNormalizationSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const InstanceNormalizationDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const override;

    bool IsL2NormalizationSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const L2NormalizationDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported) const override;

    bool IsLogicalBinarySupported(const TensorInfo& input0,
                                  const TensorInfo& input1,
                                  const TensorInfo& output,
                                  const LogicalBinaryDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsLogicalUnarySupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const ElementwiseUnaryDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsLogSoftmaxSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               const LogSoftmaxDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported) const override;

    bool IsLstmSupported(const TensorInfo& input,
                         const TensorInfo& outputStateIn,
                         const TensorInfo& cellStateIn,
                         const TensorInfo& scratchBuffer,
                         const TensorInfo& outputStateOut,
                         const TensorInfo& cellStateOut,
                         const TensorInfo& output,
                         const LstmDescriptor& descriptor,
                         const LstmInputParamsInfo& paramsInfo,
                         Optional<std::string&> reasonIfUnsupported) const override;

    bool IsMaximumSupported(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported) const override;

    bool IsMeanSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         const MeanDescriptor& descriptor,
                         Optional<std::string&> reasonIfUnsupported) const override;

    bool IsMemImportSupported(const armnn::TensorInfo& input,
                              const armnn::TensorInfo& output,
                              armnn::Optional<std::string&> reasonIfUnsupported) const override;

    bool IsMergeSupported(const TensorInfo& input0,
                          const TensorInfo& input1,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported) const override;

    bool IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                           const TensorInfo& output,
                           const OriginsDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported) const override;

    bool IsMinimumSupported(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported) const override;

    bool IsMultiplicationSupported(const TensorInfo& input0,
                                   const TensorInfo& input1,
                                   const TensorInfo& output,
                                   Optional<std::string&> reasonIfUnsupported) const override;

    /// Provides more detail than IsMultiplicationSupported(), by stating *how* the multiplication can be supported
    /// (ReinterpretQuantize vs Depthwise replacement)
    MultiplicationSupportedMode
        GetMultiplicationSupportedMode(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsNormalizationSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const NormalizationDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported) const override;

    bool IsPadSupported(const TensorInfo& input,
                        const TensorInfo& output,
                        const PadDescriptor& descriptor,
                        Optional<std::string&> reasonIfUnsupported) const override;

    bool IsPermuteSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const PermuteDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported) const override;

    bool IsPreluSupported(const TensorInfo& input,
                          const TensorInfo& alpha,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported) const override;

    bool IsQuantizeSupported(const armnn::TensorInfo& input,
                             const armnn::TensorInfo& output,
                             armnn::Optional<std::string&> reasonIfUnsupported) const override;

    bool IsQLstmSupported(const TensorInfo& input,
                          const TensorInfo& previousOutputIn,
                          const TensorInfo& previousCellStateIn,
                          const TensorInfo& outputStateOut,
                          const TensorInfo& cellStateOut,
                          const TensorInfo& output,
                          const QLstmDescriptor& descriptor,
                          const LstmInputParamsInfo& paramsInfo,
                          Optional<std::string&> reasonIfUnsupported) const override;

    bool IsQuantizedLstmSupported(const TensorInfo& input,
                                  const TensorInfo& previousCellStateIn,
                                  const TensorInfo& previousOutputIn,
                                  const TensorInfo& cellStateOut,
                                  const TensorInfo& output,
                                  const QuantizedLstmInputParamsInfo& paramsInfo,
                                  Optional<std::string&> reasonIfUnsupported) const override;

    bool IsResizeBilinearSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   Optional<std::string&> reasonIfUnsupported) const override;

    bool IsResizeSupported(const TensorInfo& input,
                           const TensorInfo& output,
                           const ResizeDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported) const override;

    bool IsRsqrtSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported) const override;

    bool IsShapeSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsSliceSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          const SliceDescriptor& descriptor,
                          Optional<std::string&> reasonIfUnsupported) const override;

    bool IsSpaceToBatchNdSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const SpaceToBatchNdDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported) const override;

    bool IsSpaceToDepthSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const SpaceToDepthDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported) const override;

    bool IsSplitterSupported(const TensorInfo& input,
                             const ViewsDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported) const override;

    bool IsStackSupported(const std::vector<const TensorInfo*>& inputs,
                          const TensorInfo& output,
                          const StackDescriptor& descriptor,
                          Optional<std::string&> reasonIfUnsupported) const override;

    bool IsStandInSupported(const std::vector<const TensorInfo*>& inputs,
                            const std::vector<const TensorInfo*>& outputs,
                            const StandInDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported) const override;

    bool IsStridedSliceSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const StridedSliceDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported) const override;

    bool IsSubtractionSupported(const TensorInfo& input0,
                                const TensorInfo& input1,
                                const TensorInfo& output,
                                Optional<std::string&> reasonIfUnsupported) const override;

    bool IsSwitchSupported(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output0,
                           const TensorInfo& output1,
                           Optional<std::string&> reasonIfUnsupported) const override;

    bool IsTransposeSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const TransposeDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported) const override;

private:
    bool CheckEstimateOnlySupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    Optional<std::string&> reasonIfUnsupported) const;
    bool CheckEstimateOnlySupported(const std::vector<TensorInfo>& inputs,
                                    const std::vector<TensorInfo>& outputs,
                                    Optional<std::string&> reasonIfUnsupported) const;

    bool IsAdditionSupportedByDepthwiseReplacement(const TensorInfo& input0,
                                                   const TensorInfo& input1,
                                                   const TensorInfo& output,
                                                   const ethosn::support_library::TensorInfo& ethosnInput0,
                                                   const ethosn::support_library::TensorInfo& ethosnInput1,
                                                   Optional<std::string&> reasonIfUnsupported) const;

    bool IsAdditionSupportedByReinterpretQuantization(const TensorInfo& input0,
                                                      const TensorInfo& input1,
                                                      const TensorInfo& output,
                                                      const ethosn::support_library::TensorInfo& ethosnInput0,
                                                      const ethosn::support_library::TensorInfo& ethosnInput1,
                                                      Optional<std::string&> reasonIfUnsupported) const;

    bool IsMultiplicationSupportedByDepthwiseReplacement(const TensorInfo& input0,
                                                         const TensorInfo& input1,
                                                         const TensorInfo& output,
                                                         Optional<std::string&> reasonIfUnsupported) const;

    bool
        IsMultiplicationSupportedByReinterpretQuantizationReplacement(const TensorInfo& input0,
                                                                      const TensorInfo& input1,
                                                                      const TensorInfo& output,
                                                                      Optional<std::string&> reasonIfUnsupported) const;

    EthosNConfig m_Config;
    EthosNMappings m_Mappings;
    ethosn::support_library::SupportQueries m_Queries;
};

}    // namespace armnn
