# Changelog for Arm® Ethos™-N Driver Stack

## 21.08

### New Features

- Support for Arm NN 21.08
- Support for Linux® kernel version 5.10
- Support for new networks:
  - SRGAN
  - U-Net
- Support for new operations:
  - Tanh
- Improved support for operations:
  - 'Constant' operations are now supported as inputs to all other operation types
  - Multiplication of a variable by a scalar constant is now supported by the Arm NN Ethos-N backend in cases where the quantized values in the output are the same as the input
  - Addition of a variable with a scalar constant is now supported by the Arm NN Ethos-N backend in cases where the quantized values in the output are the same as the input
- Support for Power Management:
  - Suspend: Once the suspend command has been given, the NPU is brought to a low power state and the NPU's state is kept in RAM. This feature is sometimes referred to as 'suspend to RAM'.
  - Sleep: This is a runtime power management feature that dynamically puts the NPU in a low-power state when it is not being used; the rest of the system still functions normally.

### Public API Changes
- The versions of the following libraries were updated:
  - Support Library version updated to 1.1.0
  - Driver Library version updated to 1.2.0
  - Kernel module version updated to 1.1.0
- The Support Library's GetFwAndHwCapabilities() function will now throw an exception if the capabilities for Ethos-N37, Ethos-N57 or Ethos-N77 are requested
- The Support Library will now throw an exception if compilation or performance estimation is attempted for Ethos-N37, Ethos-N57 or Ethos-N77
- Added power management profiling counters to kernel module and Driver Library
- Added an IsReinterpretQuantizationSupported method to the Support Library to check if the given operation is supported

### Other Changes
- This driver release supports only the Ethos-N78 NPU
- The previously deprecated Ethos-N77 support has been removed
- The Support Library no longer directly supports broadcasted addition between a constant and non-constant tensor. For users of Arm NN, most cases of this are now handled by the Arm NN Ethos-N backend instead by performing a graph replacement. Other users of the Support Library can handle this similarly.
- Improved weight compression for zero weights
- Improved network compilation performance by speeding up weight encoder compression parameter selection
- Improved robustness of supported checks for Space To Depth and Transpose. The support library now rejects any Transpose or Space To Depth operations that cannot fit into SRAM,
- Improved robustness of supported checks for average pooling with wide or tall input tensor. Average pooling is now rejected by the support library if it cannot fit into SRAM.
- Fixed a crash in the Support Library's IsAdditionSupported function when the two input data types are different
- Fixed a crash in the support library when compiling a network with a Transpose operation with permutation vector 0123
- Fixed an issue where a network with only Requantize operations fails to compile
- Fixed an issue where some networks with Concatenation operations failed to compile
- Fixed an issue where the output of a network could contain uninitialized data outside of the valid tensor region when the output format is NHWCB
- Fixed an issue where consecutive inferences could overwrite each other's outputs

### Known Issues
- The output of a network can contain non-zero data outside of the valid tensor region when the output format is NHWCB
- The network pattern Constant → Reshape → Addition, where the Addition is broadcasting this input along width and height, is no longer supported by the support library or the Arm NN backend.

## 21.05

### New Features

- Support for Arm NN 21.05
- Support for new operators:
  - Max Pooling (1x1 and stride 2,2)
  - Transpose
  - Space To Depth
  - MeanXy (for input 7x7 and 8x8)
- Support for ResNet v2-50 added
- All software components are versioned
- Support for Linux® kernel version 4.19 and 5.4 added
- The Arm NN backend now supports a boolean option to disable the use of Winograd when compiling a network

### Public API changes

- The version of the following libraries was updated:
  - Support Library version updated to 1.0.0
  - Driver Library version updated to 1.1.0
  - Kernel module version updated to 1.0.0
- The Support Library is no longer required at inference-time
- Cleaned up QuantizationInfo constructor signatures in the Support library
- The Driver Library's Network object is now constructed from a byte array rather than a Support Library CompiledNetwork object. This byte array can be obtained by calling Serialize() on the Support Library's CompiledNetwork object.
- The Support Library's DeserializeCompiledNetwork function has been removed. Serialized Compiled Networks can now be loaded by the Driver Library instead.
- The serialized format of a Support Library Compiled Network has changed and is not backwards compatible.
- Methods that provided access to buffer information intended for internal use have been removed from the Support Library CompiledNetwork object
- The Support Library's function for retrieving the buffer infos will return in the same order as the buffers were added by the user.
- Added compatibility checks between various components.

### Other changes

- This driver release only supports the Ethos-N77 and Ethos-N78 NPUs
- The Ethos-N77 support is deprecated and will be removed in 21.08.
- Improved performance of the Support Library's weight encoder, leading to reduced network compilation time, up to 2x in some cases.
- Users can new turn off warnings as errors (-Werror) by specifying werror=0 in scons.
- An XML representation of the command stream can be dumped from the Driver Library by enabling a debug option

### Known issues

- None

## 21.02

### New Features

- Support for the Arm® Ethos™-N78 EAC release.
- Support for additional padding configurations for convolutions.
- Public unit tests for user space libraries.
- Support for running in secure mode.
- Interaction with Trusted Firmware to delegate some registers to non-secure mode and restrict others.

### Public API changes

- Driver Library version updated to 0.1.2.
- Support Library version updated to 0.1.2.

### Other changes

- 3x performance improvement on ADDITION layers, giving ~25% inference performance improvement on Yolo V3 Backbone, depending on the configuration.
- Improved streaming strategies, giving improved performance across all supported networks for smaller configurations, up to 25%.
- The kernel module is built by default for an NPU running in secure mode. Extra build flags are needed to build for the NPU in non-secure mode.

### Known issues

- None

## 20.11

### New Features

- Support for Arm NN 20.11.
- Support for Quantize operator.
- Support for Leaky Relu on Ethos-N78.
- Remove restriction on tensor size for transpose convolution operator.
- Support for concatenation with shared inputs, e.g. used in Yolo v3

### Public API changes

- Support Library: IsSupported APIs changed from pure functions to class methods APIs to hold context of SwHw capabilities.
- Support Library: CompileOptions constructor does not take argument any longer.
- Support Library: Capabilities passed through the Network object to Compile and Estimate.

### Other changes

- Improve accuracy for addition with different input quantization parameters.
- Recommended GNU C/C++ compiler version updated to 7.5.0 in README.
- Fixed bug that could lead to incorrect output when using Relu with signed inputs.
- Support for no op network. e.g. Input → Output or Input → Reshape → Output
- Improved support for layers with large widths or depths.
- Added IsSupported checks for max depth of input and output tensors.

### Known issues

- None

## 20.08

### New Features

- Support for Arm NN 20.08.
- Support for Leaky ReLU activation operator.
- Support for Resize operator
- Support for Requantize operator (only using Support Library API directly, not through Arm NN backend)
- Support for Addition operator with larger tensor shapes
- Support for Multi-core NPU (including SMMU support for multi NPU)
- Support for 8-bit signed weights and activations.
- Support for per-channel quantisation of weights for convolution
- Support for YOLO v3 detection subgraph (backbone).
- Some small parts of the network are currently not supported (in particular the Quantize and Concat layers). This results in Arm NN splitting the network into several smaller subgraphs, of which the supported subgraphs will be run by the NPU. This will be addressed in a future release of the driver stack.
The output of the network does not exactly match the Arm NN reference backend, as with other supported networks. Future releases of the driver stack may improve the accuracy.
- Improved profiling information from firmware.
- Improved performance by better memory streaming strategies.

### Public API Changes

- Support Library's QuantizationInfo struct has been refactored in order to support per-channel quantisation. Please see Support.hpp for details of the new interface. The included Arm NN backend has been updated accordingly.

### Other Changes

- None

### Known Issues

- Space to depth and Transpose operations not supported and may return failure.

## 20.05

### New Features

- Support for Ethos-N78.
- Support for Arm NN 20.05.
- Improved performance for larger layers.
- Support for Inception v4 on Ethos-N57 and Ethos-N37.

### Public API Changes

- None

### Other Changes

- None

### Known Issues

- None

## 20.02.1

### New Features

- First public release (please note that some licenses have changed, see the README.md file for more details).
- Resolved race condition when loading the kernel module with profiling enabled.
- Improved detection of unsupported cases for operator Concat.

### Public API Changes

- None

### Other Changes

- None

### Known Issues

- None

## 20.02

### New Features

- Compatible with Arm NN 20.02
- Limited support for Profiling in the Driver Library, Kernel module and the Control Unit
  - Support for Counters and timeline events
- Support for SMMU in the Kernel module for kernel version 4.14
  - Users can define an SMMU stream id in their dts file
- The kernel module now can be built for linux kernel version 4.14

### Public API Changes

- Change the namespace in the support library and driver library from npu to ethosn
- Renamed NpuVariant to EthosNVariant in Support.hpp
- Renamed NPU_SUPPORT_LIBRARY_VERSION_* to ETHOSN_SUPPORT_LIBRARY_VERSION_*
- Renamed NPU_DRIVER_LIBRARY_VERSION_* to ETHOSN_DRIVER_LIBRARY_VERSION_*
- Renamed anpu.ko to ethosn.ko
- Renamed arm-npu.bin to ethosn.bin

### Other Changes

- Added contact information for security related issues in the README.md
- Fixed bug with concatenation with different quantization scale inputs

### Known Issues

- None

## 19.11

### New Features

- Support for TransposeConvolution in stand-alone and Arm NN backend
- Support for the Split operator
- Support for the DepthToSpace operator with a block size of 2x2
- Support for the FSRCNN network
- Support for a variety of different interrupt configurations that can be specified the Linux kernel driver device tree. See the example .dts file shipped with the kernel driver for details.
- Support for depthwise multiplier > 1 when the number of input channels is 1 in depthwise convolutions
- Support a bias add operation after a convolution
- Support for Ethos-N77 with 4MB SRAM
- Initial framework for collecting event and counter based profiling has been added to the Driver Library and Kernel Driver. This feature is not yet complete and will - be expanded upon in a future release.

### Public API Changes

- Added EstimatePerformance() and supporting types to the support library
- Arm NN backend runs in performance estimation mode if enabled in a config file which location is specified via enviroment variable ARMNN_NPU_BACKEND_CONFIG_FILE

### Other Changes

- Bug Fixes
  - Fixed a resource leak that could lead to not being able to unload the kernel driver module
  - Fixed a hang that could occur during an inference with certain configurations of large tensors
  - Fixed potential kernel panic condition on insmod
- Updated to Arm NN 19.11 release
- Improved error messages from the Support Library when a network fails to compile

### Known Issues

- None

## 19.08

### New Features

- Support for the Inception V3 network for the Ethos-N57 and Ethos-N37 configurations
- Support for the Inception V4 network for the Ethos-N77 configuration
- Support for the SSD Mobilenet V1 network
- Support for the Sigmoid operation
- Improved support for the Concatenation operation
- Improved support for operations with large inputs
- Improved inference performance for networks with branching
- Arm NN NPU backend compatible with Arm NN 19.08
- Improved support for Arm NN Reshape operations with non-4D shapes

### Public API Changes

- Intermediate compression is now enabled by default (CompilationOptions::m_EnableIntermediateCompression)
- Support Library's CompilationOptions now requires an opaque blob of "capabilities" data, provided by the Driver Library.

### Other Changes

- Improved stability, especially when queuing multiple inferences
- Improved performance, especially for multiple layer networks
- Fixed intermittent incorrect results or crash when running an inference with multiple inputs or outputs.

### Known Issues

- None

# Trademarks and copyrights

Arm and Ethos are registered trademarks or trademarks of Arm Limited (or its subsidiaries) in the US and/or elsewhere.

Linux® is the registered trademark of Linus Torvalds in the U.S. and other countries.
