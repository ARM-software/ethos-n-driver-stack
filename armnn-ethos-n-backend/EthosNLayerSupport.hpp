//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNConfig.hpp"

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
        None,    ///<Multiplication cannot be supported by this backend at all.
        Native,    ///<Multiplication can be supported by this backend, by using an Addition operation in the support library.
        ReplaceWithDepthwise,    ///<Multiplication can be supported by this backend, by replacing Multiplication operation with a DepthwiseConvolution operation in the support library.
        ReplaceWithReinterpretQuantize,    ///<Multiplication can be supported by this backend, by replacing Multiplication operation with a ReinterpretQuantize operation in the Support Library.
        EstimateOnly,                      //<Estimate only support
    };

    EthosNLayerSupport(const EthosNConfig& config, const std::vector<char>& capabilities);
    bool IsLayerSupported(const LayerType& type,
                          const std::vector<TensorInfo>& infos,
                          const BaseDescriptor& descriptor,
                          const Optional<LstmInputParamsInfo>& lstmParamsInfo,
                          const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmParamsInfo,
                          Optional<std::string&> reasonIfUnsupported) const override;

    /// Provides more detail than IsAdditionSupported(), by stating *how* the addition can be supported
    /// (native vs depthwise replacement)
    AdditionSupportedMode GetAdditionSupportedMode(const TensorInfo& input0,
                                                   const TensorInfo& input1,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    /// Provides more detail than IsMultiplicationSupported(), by stating *how* the multiplication can be supported
    /// (ReinterpretQuantize vs Depthwise replacement)
    MultiplicationSupportedMode
        GetMultiplicationSupportedMode(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

private:
    bool IsActivationSupportedImpl(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const ActivationDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;
    bool IsAdditionSupportedImpl(const TensorInfo& input0,
                                 const TensorInfo& input1,
                                 const TensorInfo& output,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsConcatSupportedImpl(const std::vector<const TensorInfo*>& inputs,
                               const TensorInfo& output,
                               const OriginsDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsConstantSupportedImpl(const TensorInfo& output,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsConvolution2dSupportedImpl(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const Convolution2dDescriptor& descriptor,
                                      const TensorInfo& weights,
                                      const Optional<TensorInfo>& biases,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsDepthwiseConvolutionSupportedImpl(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const DepthwiseConvolution2dDescriptor& descriptor,
                                             const TensorInfo& weights,
                                             const Optional<TensorInfo>& biases,
                                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsTransposeConvolution2dSupportedImpl(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const TransposeConvolution2dDescriptor& descriptor,
                                               const TensorInfo& weights,
                                               const Optional<TensorInfo>& biases,
                                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsFullyConnectedSupportedImpl(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const TensorInfo& weights,
                                       const TensorInfo& biases,
                                       const FullyConnectedDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsInputSupportedImpl(const TensorInfo& input,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsMemCopySupportedImpl(const TensorInfo& input,
                                const TensorInfo& output,
                                Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsOutputSupportedImpl(const TensorInfo& output,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsPooling2dSupportedImpl(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const Pooling2dDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsPreCompiledSupportedImpl(const TensorInfo& input,
                                    const PreCompiledDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsRankSupportedImpl(const TensorInfo& input,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsReduceSupportedImpl(const TensorInfo& input,
                               const TensorInfo& output,
                               const ReduceDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsReshapeSupportedImpl(const TensorInfo& input,
                                const TensorInfo& output,
                                const ReshapeDescriptor& descriptor,
                                Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsSplitterSupportedImpl(const TensorInfo& input,
                                 const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                 const ViewsDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsDepthToSpaceSupportedImpl(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const DepthToSpaceDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsArgMinMaxSupportedImpl(const armnn::TensorInfo& input,
                                  const armnn::TensorInfo& output,
                                  const armnn::ArgMinMaxDescriptor& descriptor,
                                  armnn::Optional<std::string&> reasonIfUnsupported) const;

    bool IsBatchNormalizationSupportedImpl(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const TensorInfo& mean,
                                           const TensorInfo& var,
                                           const TensorInfo& beta,
                                           const TensorInfo& gamma,
                                           const BatchNormalizationDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const;

    bool IsBatchToSpaceNdSupportedImpl(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const BatchToSpaceNdDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const;

    bool IsCastSupportedImpl(const TensorInfo& input,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsComparisonSupportedImpl(const TensorInfo& input0,
                                   const TensorInfo& input1,
                                   const TensorInfo& output,
                                   const ComparisonDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported) const;

    bool IsDilatedDepthwiseConvolutionSupportedImpl(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const DepthwiseConvolution2dDescriptor& descriptor,
                                                    const TensorInfo& weights,
                                                    const Optional<TensorInfo>& biases,
                                                    Optional<std::string&> reasonIfUnsupported) const;

    bool IsDivisionSupportedImpl(const TensorInfo& input0,
                                 const TensorInfo& input1,
                                 const TensorInfo& output,
                                 Optional<std::string&> reasonIfUnsupported) const;

    bool IsElementwiseUnarySupportedImpl(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const ElementwiseUnaryDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsFakeQuantizationSupportedImpl(const TensorInfo& input,
                                         const FakeQuantizationDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const;

    bool IsFillSupportedImpl(const TensorInfo& input,
                             const TensorInfo& output,
                             const FillDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsFloorSupportedImpl(const TensorInfo& input,
                              const TensorInfo& output,
                              Optional<std::string&> reasonIfUnsupported) const;

    bool IsGatherSupportedImpl(const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
                               const GatherDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsInstanceNormalizationSupportedImpl(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const InstanceNormalizationDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const;

    bool IsL2NormalizationSupportedImpl(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const L2NormalizationDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const;

    bool IsLogicalBinarySupportedImpl(const TensorInfo& input0,
                                      const TensorInfo& input1,
                                      const TensorInfo& output,
                                      const LogicalBinaryDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsLogicalUnarySupportedImpl(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const ElementwiseUnaryDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsLstmSupportedImpl(const TensorInfo& input,
                             const TensorInfo& outputStateIn,
                             const TensorInfo& cellStateIn,
                             const TensorInfo& scratchBuffer,
                             const TensorInfo& outputStateOut,
                             const TensorInfo& cellStateOut,
                             const TensorInfo& output,
                             const LstmDescriptor& descriptor,
                             const LstmInputParamsInfo& paramsInfo,
                             Optional<std::string&> reasonIfUnsupported) const;

    bool IsMaximumSupportedImpl(const TensorInfo& input0,
                                const TensorInfo& input1,
                                const TensorInfo& output,
                                Optional<std::string&> reasonIfUnsupported) const;

    bool IsMeanSupportedImpl(const TensorInfo& input,
                             const TensorInfo& output,
                             const MeanDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported) const;

    bool IsMergeSupportedImpl(const TensorInfo& input0,
                              const TensorInfo& input1,
                              const TensorInfo& output,
                              Optional<std::string&> reasonIfUnsupported) const;

    bool IsMinimumSupportedImpl(const TensorInfo& input0,
                                const TensorInfo& input1,
                                const TensorInfo& output,
                                Optional<std::string&> reasonIfUnsupported) const;

    bool IsMultiplicationSupportedImpl(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const;

    bool IsNormalizationSupportedImpl(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const NormalizationDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const;

    bool IsPadSupportedImpl(const TensorInfo& input,
                            const TensorInfo& output,
                            const PadDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported) const;

    bool IsPermuteSupportedImpl(const TensorInfo& input,
                                const TensorInfo& output,
                                const PermuteDescriptor& descriptor,
                                Optional<std::string&> reasonIfUnsupported) const;

    bool IsPreluSupportedImpl(const TensorInfo& input,
                              const TensorInfo& alpha,
                              const TensorInfo& output,
                              Optional<std::string&> reasonIfUnsupported) const;

    bool IsQuantizeSupportedImpl(const armnn::TensorInfo& input,
                                 const armnn::TensorInfo& output,
                                 armnn::Optional<std::string&> reasonIfUnsupported) const;

    bool IsQLstmSupportedImpl(const TensorInfo& input,
                              const TensorInfo& previousOutputIn,
                              const TensorInfo& previousCellStateIn,
                              const TensorInfo& outputStateOut,
                              const TensorInfo& cellStateOut,
                              const TensorInfo& output,
                              const QLstmDescriptor& descriptor,
                              const LstmInputParamsInfo& paramsInfo,
                              Optional<std::string&> reasonIfUnsupported) const;

    bool IsQuantizedLstmSupportedImpl(const TensorInfo& input,
                                      const TensorInfo& previousCellStateIn,
                                      const TensorInfo& previousOutputIn,
                                      const TensorInfo& cellStateOut,
                                      const TensorInfo& output,
                                      const QuantizedLstmInputParamsInfo& paramsInfo,
                                      Optional<std::string&> reasonIfUnsupported) const;

    bool IsResizeSupportedImpl(const TensorInfo& input,
                               const TensorInfo& output,
                               const ResizeDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported) const;

    bool IsShapeSupportedImpl(const TensorInfo& input,
                              const TensorInfo& output,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsSliceSupportedImpl(const TensorInfo& input,
                              const TensorInfo& output,
                              const SliceDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported) const;

    bool IsSpaceToBatchNdSupportedImpl(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const SpaceToBatchNdDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const;

    bool IsSpaceToDepthSupportedImpl(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const SpaceToDepthDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const;

    bool IsStackSupportedImpl(const std::vector<const TensorInfo*>& inputs,
                              const TensorInfo& output,
                              const StackDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported) const;

    bool IsStandInSupportedImpl(const std::vector<const TensorInfo*>& inputs,
                                const std::vector<const TensorInfo*>& outputs,
                                const StandInDescriptor& descriptor,
                                Optional<std::string&> reasonIfUnsupported) const;

    bool IsStridedSliceSupportedImpl(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const StridedSliceDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const;

    bool IsSubtractionSupportedImpl(const TensorInfo& input0,
                                    const TensorInfo& input1,
                                    const TensorInfo& output,
                                    Optional<std::string&> reasonIfUnsupported) const;

    bool IsSwitchSupportedImpl(const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output0,
                               const TensorInfo& output1,
                               Optional<std::string&> reasonIfUnsupported) const;

    bool IsTransposeSupportedImpl(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const TransposeDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported) const;

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
    ethosn::support_library::SupportQueries m_Queries;
};

}    // namespace armnn
