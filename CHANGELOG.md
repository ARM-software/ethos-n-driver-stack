# Changelog

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

- Support for TransposeConvolution in stand-alone and ArmNN backend
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
- ArmNN backend runs in performance estimation mode if enabled in a config file which location is specified via enviroment variable ARMNN_NPU_BACKEND_CONFIG_FILE

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
- ArmNN NPU backend compatible with ArmNN 19.08
- Improved support for ArmNN Reshape operations with non-4D shapes

### Public API Changes

- Intermediate compression is now enabled by default (CompilationOptions::m_EnableIntermediateCompression)
- Support Library's CompilationOptions now requires an opaque blob of "capabilities" data, provided by the Driver Library.

### Other Changes

- Improved stability, especially when queuing multiple inferences
- Improved performance, especially for multiple layer networks
- Fixed intermittent incorrect results or crash when running an inference with multiple inputs or outputs.

### Known Issues

- None
