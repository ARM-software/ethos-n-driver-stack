# Changelog

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
