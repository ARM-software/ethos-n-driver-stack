# Changelog for Arm® Ethos™-N Driver Stack

## 24.07

### New features

- Support for Arm NN 24.05

### Public API changes

- Add a possibility to get total vertical / horizontal padding from a padding structure.

### Other changes

- Fix a cyclic dependency issue when using standalone padding for certain dimensions.
- Fix a precision issue when using standalone padding for certain dimensions.
- Move debug formatting data usage in estimation utilities to improve compilation time.

### Known issues

## 24.05

### New features

- Support for Arm NN 24.02

### Public API changes

- None

### Other changes

- Fix a potential crash when the device power is cut prematurely by the kernel module.
- Disable dynamic read allocate mode in the NCU MCU.
- Fix a crash where the sleeping function is called from an invalid context in the kernel module.
- Fix int8 bilinear resizing by introducing a support library workaround.
- Fix multiplication tensor addressing for 2d and 1d tensors in the Ethos-N Arm NN backend.
- Fix a memory leak in the Ethos-N Arm NN backend.

### Known issues

- Standalone padding and convolution layers of certain dimensions with padding might trigger a cyclic dependency during graph compilation.

## 23.11

### New features

- Add the ability to use llvm-embedded toolchain:
  - Please specify the following SCons parameters or place them in your options.py:
    - use_llvm_embedded=1
    - llvm_embedded_toolchain_path=<path_to_llvm_binaries>
  - Add support for elementwise multiplication.

### Public API changes

- Remove the 'cascading' prefix from the driver stack components.
- Make firmware, model interface, and PLE source code public.
- Command stream version increased to 7.
- Driver Library version increased to 8.
- Support Library version increased to 5.

### Other changes

- Remove the '-block-inferences-debug' option from system tests.
- Improve the ordering of commands in the control unit controller.
- Improve performance under higher system latencies.
- Fix bugs related to:
  - Handling relative paths when installing the library.
  - Caching networks with multiple subgraphs.
- Some IRQ flags are now fetched from the device tree instead of being hardcoded.
- Make sure weight streams payload size fit in the Ethos-N hardware struct.
- Add some useful devloper tools to the repository.

### Known issues

- Standalone padding and convolution layers of certain dimensions with padding might trigger a cyclic dependency during graph compilation.

## 23.08

### New features

- Off device compilation is now supported through Arm NN.
  - Please set OFFLINE = 1 and PERFORMANCE_VARIANT and PERFORMANCE_SRAM_SIZE_BYTES_OVERRIDE in the Ethos-N Arm NN config file when compiling. This will cause the backend to only compile the network. Use this in conjunction with the caching feature to generate a cached compiled network.
  - Copy this cached compiled network to the target device and use it as normal.
- Improved multithreaded performance in the support library.
  - Configurable with the ETHOSN_SUPPORT_LIBRARY_NUM_THREADS environment variable. If unset the number of threads is automatically chosen.
  - One can use a different allocator e.g. mimalloc for even better performance.
- Runtime performance improvements.
  - Cascading can now be performed over branches.
  - Preloading weights from later layers.
  - Improved allocation to minimise the overlap of buffers in SRAM.
  - Sigmoid PLE kernel sped up.
- Support up to 7 padding in Convolution type operators.
- Support Max pooling Stride 1x1.
- Reduce compilation memory requirements in the weight encoder.
- Reduce cached compiled network memory requirements.
- Better error messages reporting from the hardware in the kernel log.

### Public API changes

- Command stream version bumped to 6.
- Driver Library version bumped to 7.
- Firmware version bumped to 15.

### Other changes

- Fixed a crash in the kernel module when mapping smaller regions of virtual memory.
- Fixed other bugs

### Known issues

## 23.05

### New features

- Compiler flag to disable winograd for 7x7 kernels and larger
  - Set the following in the scons command line: disable_large_winograd=true
- The cascading compiler is now the default and only compiler

### Public API changes

- Command stream major version 4 -> 5
- Support library major version 3 -> 4
- Driver library major version 5 -> 6

### Other changes

- No longer using deprecated Arm NN functions in the Arm NN backend
- Network compilation time performance improvements in cascading
- Inference performance improvements in cascading
- Compiled network caching with multiple subgraphs in the Arm NN backend has been fixed

### Known issues

- Temporary performance regression on some networks with heavy branching. Performance improvements currently in progress

## 23.02

### New features

- TZMP1 Support
- Per-Process Memory Isolation (SMMU only)

### Public API changes

- Kernel supports creating process memory allocator in protected context
- Kernel UAPI changed to use __kernel_size_t to ensure consistent type size
- ProcMemAllocator std::string constructor changed to const char *
- Version number updates:
  - Driver library version 4.0.0 → 5.0.0
  - Kernel module version 5.0.0 → 6.0.0
  - Firmware version 6.0.0 → 11.0.0

### Other changes

- Updated list of supported models to a higher performance MobileNet variant
- Suggested development platform changed from Ubuntu 18.04 to Ubuntu 20.04
- Support for Arm NN backend option ProtectedContentAllocation
- Kernel module
  - Fix a crash in the kernel module caused by the shared interrupt handler getting triggered during NPU reset
  - Buffers are now zeroed out before being freed to not leave any data in the memory
  - Kernel module only supports a single binary in the firmware
  - Kernel module will only accept an NPU with a matching security level that it was built for
  - Kernel module now sets mailbox size to be the nearest power of 2
  - Improved error handling in network creation

### Known issues

- Dual core with carveout is not supported

### Notes

- A workaround for erratum 2838783 is available in Trusted Firmware-A: <https://review.trustedfirmware.org/plugins/gitiles/TF-A/trusted-firmware-a/+/00af8f4a7dd75cbbbb597996439233614badd04e>

## 22.11

### New features

- None

### Public API changes

- Driver Library supports importing an intermediate buffer
- Kernel supports importing intermediate buffers
  - Inferences based on networks with imported intermediate buffers cannot run simultaneously on multiple cores, so they will be queued until the previous inference is completed
- Driver library and kernel has new process memory allocator APIs to create buffers and register networks. Support for old APIs is removed. The new APIs are not backwards compatible
- Version number updates:
  - Driver Library version 3.0.0 → 4.0.0
  - Kernel Module version 4.0.0 → 5.0.0
  - Firmware version 5.0.0 → 6.0.0

### Other changes

- Public architecture header files make use of assert instead of truncation in set_XXX functions
- Improvements to SmallVector type
- Kernel module
  - Fix a crash in the kernel module when the firmware binary changes
  - Fix kernel module not picking up a new firmware binary after failing to load a previous one
  - Kernel module will now only load firmware binaries that contain an identifying magic number

### Known issues

- None

## 22.08.1

### New features

- Estimation mode for split now supports multiple outputs
- Support has been added to use separate SMMU streams for different memory assets e.g. firmware, input/output, command stream
  - Device tree layout has been changed to support having multiple SMMU streams
    - Multiple asset allocators may be defined in the device tree, however only the first one is currently used
  - The Ethos-N NPU kernel module and Trusted Firmware-A driver have been updated to support the new device tree and the use of separate SMMU streams

### Public API changes

- The Support Library's CompiledNetwork class now has a function to get how much intermediate buffer memory a network requires
- Version number updates:
  - Support Library version 3.1.0 → 3.2.0
  - Kernel Module version 3.0.0 → 4.0.0

### Other changes

- Fixed the Support Library's compiler allowing an output buffer to be used as an intermediate buffer
- Improvements to SmallVector constructor and operator support

### Known issues

- Refer to 22.08 changelog for more details

## 22.08

### New features

- Split operation now supported (see SUPPORTED.md for more information)

### Public API changes

- Version number updates:
  - Driver Library version 2.0.0 → 3.0.0
  - Command Stream version 3.0.0 → 3.1.0
  - Support Library version 3.0.1 → 3.1.0
  - Kernel Module version 2.0.0 → 3.0.0
  - Firmware version 4.0.0 → 5.0.0

### Other changes

- Bias quantization fixes
- Input quantization documentation fixes
- Fixed issues not using the correct data formats during cascading for the following:
  - Fully connected
  - Branching
  - Concatenation
- PLE kernels are now mapped as read-only when a SMMU is available
- Fixed power surge issue when clearing SRAM at the beginning of each inference

### Known issues

- Some Resize Billinear configurations (align_corners=True, half_pixel_centres=True when heights and widths are not both even or both odd) produce inaccurate results
- Warnings from Arm NN for using a deprecated 'ConstTensorsAsInputs' API call

## 22.05

### New features

- Zero copy support:
  - Support added for using the Import API with Arm NN using dma_buf.

### Public API changes

- Added new API for importing a dma_buf file descriptor.

### Other changes

- Extended the range of OFM multiplier.
- Added support for deallocation of SRAM buffers in the Cascading Support Library.
- Limited the maximum size of a Section generated by the Cascading Support Library to the corresponding size supported by the Firmware.
- Reduced inference latency.
- Reduced estimation time for networks with Parts without weights.

### Known issues

- None

## 22.02

### New features

- Support for Arm NN 22.02
- DRAM usage for intermediate tensors has been decreased. Depending on the network and hardware configuration, this can lead to a 90% reduction in DRAM usage for intermediate tensors.

### Public API changes

- Temporarily disabled Split, SpaceToDepth and Transpose operations. They will be re-enabled in a future release. These operations are now only supported at the "EstimateOnly" level. They can be used in SPA and will contribute zero performance impact, but cannot be compiled for execution on the actual hardware.
- Removed deprecated method GetMappedBuffer in driver library.
- Changes to the Arm NN Ethos-N NPU backend configuration file:
  - The "COMPILER_ALGORITHM" flag has been removed.
  - The estimation approach used when "PERFORMANCE_CURRENT=0" has changed to be closer to what we expect from a future release of the driver stack. The "PERFORMANCE_CURRENT" is used as follows:
    - 0 is used to report the estimation numbers of the possible future performance of the driver stack.
    - 1 is used to report the estimation numbers of the current performance of the driver stack.

### Other changes

- Cascading support:
  - Added support for performance estimation, producing estimation numbers that are closer to what we expect from a future release of the driver stack.
  - Improved compilation time for some networks.
- Arm NN Ethos-N NPU backend:
  - Added caching functionality via model options to improve compilation time.
    - Saving: Set "SaveCachedNetwork" to true and supply a valid file path to an empty file for "CachedNetworkFilePath".
    - Loading: Set "SaveCachedNetwork" to false and supply a valid file path to the previously saved network for "CachedNetworkFilePath".
  - Added StrictPrecision via model options. When enabled the network is more precise as the Re-quantize operations aren't fused, but it is slower to compile as there will be additional operations. This is currently only supported for the Concat operation.
  - The backend no longer uses a deprecated method to access weights tensors from Fully Connected layers. This removes a warning from the Arm NN log.
- Support library:
  - Fixed a bug where the order of output buffers in the CompiledNetwork wouldn't match the order the outputs were added to the Network.
- Initial Android build instructions added in the README.md
- Kernel module:
  - Enable SMMU translation table entries pre-loading by default.
  - Improved mailbox and firmware error handling and reporting.

### Known issues

- None

## 21.11

### New features

- Support for Arm NN 21.11.
- Support for EfficientNet Lite.
- Support of user-controlled workload scheduling on multiple separated devices.
- Support for hibernation in SMMU configurations.

### Public API changes

- Version number updates:
  - Command stream version updated to 3.0.0.
  - Driver library version updated to 1.3.0.
  - Firmware version updated to 3.0.0.
  - Kernel module version updated to 2.0.0.
  - Support library version updated to 3.0.0.
- Driver library:
  - GetMappedBuffer is deprecated and support will be removed in next release.

### Other changes

- Performance improvement of 2.5x for Leaky ReLU.
- Kernel module:
  - Use of DMA API for buffer allocation.
  - An additional task stack is now allocated for use by the firmware.
  - Memory regions and streams have been decoupled to allow multiple regions to be associated with the same stream with more fine grained memory restrictions.
  - Support use of reserved regions in SMMU configurations.
  - Linux version 4.14 and 4.19 are deprecated in SMMU configurations and support will be removed in next release.
  - Improved SMC error handling to support errors from both 32-bit and 64-bit Trusted Firmware-A.
  - Additional SMC logging.
  - Build in release mode by default.
  - Fixed some resources not being cleaned up correctly in error handling.
  - Removed 512MB limit for DMA buffers when SMMU is present.
- Support library:
  - Improve zero point checks in support queries.
  - Fixed some resources not being cleaned up correctly in error handling.
  - Fixed overflow in direct convolution when estimating SRGAN performance.
  - Improve error handling by replacing some asserts with throwing exceptions.

### Known issues

- None

## 21.08

### New features

- Support for Arm NN 21.08.
- Support for Linux® kernel version 5.10.
- Support for new networks:
  - SRGAN
  - U-Net
- Support for new operations:
  - tanh
- Improved support for operations:
  - 'Constant' operations are now supported as inputs to all other operation types.
  - Multiplication of a variable by a scalar constant is now supported by the Arm NN Ethos-N backend in cases where the quantized values in the output are the same as the input.
  - Addition of a variable with a scalar constant is now supported by the Arm NN Ethos-N backend in cases where the quantized values in the output are the same as the input.
- Support for power management:
  - Suspend: Once the suspend command has been given, the NPU is brought to a low-power state and the state of the NPU is kept in RAM. This feature is sometimes referred to as 'suspend to RAM'.
  - Sleep: This is a runtime power management feature that dynamically puts the NPU in a low-power state when it is not being used. The rest of the system still functions normally.

### Public API changes

- The versions of the following libraries were updated:
  - Support library version updated to 1.1.0.
  - Driver library version updated to 1.2.0.
  - Kernel module version updated to 1.1.0.
- If the capabilities for Ethos-N37, Ethos-N57, or Ethos-N77 are requested, the GetFwAndHwCapabilities() function of the support library now throws an exception.
- If compilation or performance estimation is attempted for Ethos-N37, Ethos-N57, or Ethos-N77, the support library now throws an exception.
- Added power management profiling counters to the kernel module and driver library.
- Added an IsReinterpretQuantizationSupported method to the support library to check if the given operation is supported.

### Other changes

- This driver release supports only the Ethos-N78 NPU.
- The previously deprecated Ethos-N77 support has been removed.
- The support library no longer directly supports broadcasted addition between a constant and non-constant tensor. For users of Arm NN, most cases of this are now handled by the Arm NN Ethos-N backend instead by performing a graph replacement. Other users of the support library can handle this similarly.
- Improved weight compression for zero weights.
- Improved network compilation performance by speeding up weight encoder compression parameter selection.
- Improved robustness of supported checks for space to depth and transpose. The support library now rejects any transpose or space to depth operations that cannot fit into SRAM.
- Improved robustness of supported checks for average pooling with wide or tall input tensor. The support library now rejects average pooling if it cannot fit into SRAM.
- Fixed a crash in the IsAdditionSupported function of the support library when the two input data types are different.
- Fixed a crash in the support library when compiling a network with a transpose operation with permutation vector 0123.
- Fixed an issue where a network with only requantize operations fails to compile.
- Fixed an issue where some networks with concatenation operations failed to compile.
- Fixed an issue where the output of a network could contain uninitialized data outside of the valid tensor region when the output format is NHWCB.
- Fixed an issue where consecutive inferences could overwrite each other's outputs.

### Known issues

- The output of a network can contain nonzero data outside of the valid tensor region when the output format is NHWCB.
- The network pattern Constant → Reshape → Addition, where the Addition is broadcasting this input along width and height, is no longer supported by the support library or the Arm NN backend.

## 21.05

### New features

- Support for Arm NN 21.05.
- Support for new operators:
  - Max pooling (1 x 1 and stride 2, 2)
  - Transpose
  - Space to depth
  - MeanXy (for input 7 x 7 and 8 x 8)
- Support for ResNet v2-50 added.
- All software components are versioned.
- Support for Linux® kernel version 4.19 and 5.4 added.
- The Arm NN backend now supports a Boolean option to disable the use of Winograd when compiling a network.

### Public API changes

- The version of the following libraries was updated:
  - Support library version updated to 1.0.0.
  - Driver library version updated to 1.1.0.
  - Kernel module version updated to 1.0.0.
- The support library is no longer required at inference-time.
- Cleaned up QuantizationInfo constructor signatures in the support library.
- The driver library's network object is now constructed from a byte array rather than a support library CompiledNetwork object. This byte array can be obtained by calling Serialize() on the support library's CompiledNetwork object.
- The support library's DeserializeCompiledNetwork function has been removed. Serialized compiled networks can now be loaded by the driver library instead.
- The serialized format of a support library compiled network has changed and is not backwards compatible.
- Methods that provided access to buffer information intended for internal use have been removed from the support library CompiledNetwork object.
- The support library's function for retrieving the buffer information returns in the order that the buffers were added by the user.
- Added compatibility checks between various components.

### Other changes

- This driver release only supports the Ethos-N77 and Ethos-N78 NPUs.
- The Ethos-N77 support is deprecated and was removed in 21.08.
- Improved performance of the support library's weight encoder, leading to reduced network compilation time, up to 2x in some cases.
- Users can new turn off warnings as errors (-Werror) by specifying werror=0 in scons.
- An XML representation of the command stream can be dumped from the driver library by enabling a debug option

### Known issues

- None

## 21.02

### New features

- Support for the Arm® Ethos™-N78 EAC release.
- Support for extra padding configurations for convolutions.
- Public unit tests for user-space libraries.
- Support for running in Secure mode.
- Interaction with trusted firmware to delegate some registers to Non-secure mode and restrict others.

### Public API changes

- Driver library version updated to 0.1.2.
- Support library version updated to 0.1.2.

### Other changes

- 3x performance improvement on ADDITION layers, giving ~25% inference performance improvement on Yolo V3 Backbone, depending on the configuration.
- Improved streaming strategies, giving improved performance across all supported networks for smaller configurations, up to 25%.
- The kernel module is built by default for an NPU running in Secure mode. Extra build flags are needed to build for the NPU in Non-secure mode.

### Known issues

- None

## 20.11

### New features

- Support for Arm NN 20.11.
- Support for quantize operator.
- Support for Leaky ReLU on Ethos-N78.
- Remove restriction on tensor size for transpose convolution operator.
- Support for concatenation with shared inputs, for example, in Yolo v3.

### Public API changes

- Support library: IsSupported APIs changed from pure functions to class methods APIs to hold context of SwHw capabilities.
- Support library: CompileOptions constructor does not take argument any longer.
- Support library: Capabilities passed through the network object to compile and estimate.

### Other changes

- Improve accuracy for addition with different input quantization parameters.
- Recommended GNU C/C++ compiler version updated to 7.5.0 in the README.md file.
- Fixed bug that could lead to incorrect output when using ReLU with signed inputs.
- Support for no op network, for example Input → Output or Input → Reshape → Output
- Improved support for layers with large widths or depths.
- Added IsSupported checks for maximum depth of input and output tensors.

### Known issues

- None

## 20.08

### New features

- Support for Arm NN 20.08.
- Support for Leaky ReLU activation operator.
- Support for resize operator.
- Support for requantize operator (only using support library API directly, not through Arm NN backend).
- Support for addition operator with larger tensor shapes.
- Support for multi-core NPU (including SMMU support for multi NPU).
- Support for 8-bit signed weights and activations.
- Support for per-channel quantization of weights for convolution.
- Support for YOLO v3 detection subgraph (backbone).
- Some small parts of the network are currently not supported (in particular the quantize and concatenation layers). This results in Arm NN splitting the network into several smaller subgraphs, of which the supported subgraphs will be run by the NPU. This will be addressed in a future release of the driver stack.
The output of the network does not exactly match the Arm NN reference backend, as with other supported networks. Future releases of the driver stack may improve the accuracy.
- Improved profiling information from firmware.
- Improved performance by better memory streaming strategies.

### Public API changes

- The support library's QuantizationInfo struct has been refactored to support per-channel quantization. For details of the new interface, see `Support.hpp`. The included Arm NN backend has been updated accordingly.

### Other changes

- None

### Known issues

- Space to depth and transpose operations not supported and may return failure.

## 20.05

### New features

- Support for Ethos-N78.
- Support for Arm NN 20.05.
- Improved performance for larger layers.
- Support for Inception v4 on Ethos-N57 and Ethos-N37.

### Public API changes

- None

### Other changes

- None

### Known issues

- None

## 20.02.1

### New features

- First public release (please note that some licenses have changed, see the README.md file for more details).
- Resolved race condition when loading the kernel module with profiling enabled.
- Improved detection of unsupported cases for operator concatenation.

### Public API changes

- None

### Other changes

- None

### Known issues

- None

## 20.02

### New features

- Compatible with Arm NN 20.02.
- Limited support for profiling in the driver library, kernel module, and the control unit.
  - Support for counters and timeline events.
- Support for SMMU in the kernel module for kernel version 4.14.
  - Users can define an SMMU StreamID in their dts file.
- The kernel module now can be built for Linux kernel version 4.14.

### Public API changes

- Change the namespace in the support library and driver library from npu to ethosn.
- Renamed NpuVariant to EthosNVariant in Support.hpp.
- Renamed NPU_SUPPORT_LIBRARY_VERSION_* to ETHOSN_SUPPORT_LIBRARY_VERSION_*.
- Renamed NPU_DRIVER_LIBRARY_VERSION_* to ETHOSN_DRIVER_LIBRARY_VERSION_*.
- Renamed anpu.ko to ethosn.ko.
- Renamed arm-npu.bin to ethosn.bin.

### Other changes

- Added contact information for security-related issues in the README.md file.
- Fixed bug with concatenation with different quantization scale inputs.

### Known issues

- None

## 19.11

### New features

- Support for TransposeConvolution in stand-alone and Arm NN backend.
- Support for the split operator.
- Support for the DepthToSpace operator with a block size of 2 x 2.
- Support for the FSRCNN network.
- Support for various different interrupt configurations that can be specified in the Linux kernel driver Device tree. See the example .dts file shipped with the kernel driver for details.
- Support for depthwise multiplier >1 when the number of input channels is 1 in depthwise convolutions.
- Support a bias add operation after a convolution.
- Support for Ethos-N77 with 4MB SRAM.
- Initial framework for collecting event and counter based profiling has been added to the driver library and kernel driver. This feature is not yet complete and will be expanded on in a future release.

### Public API changes

- Added EstimatePerformance() and supporting types to the support library.
- Arm NN backend runs in performance estimation mode if enabled in a config file which location is specified in the environment variable ARMNN_NPU_BACKEND_CONFIG_FILE.

### Other changes

- Bug fixes:
  - Fixed a resource leak that could lead to not being able to unload the kernel driver module.
  - Fixed a hang that could occur during an inference with certain configurations of large tensors.
  - Fixed potential kernel panic condition on insmod.
- Updated to Arm NN 19.11 release.
- Improved error messages from the support library when a network fails to compile.

### Known issues

- None

## 19.08

### New features

- Support for the Inception V3 network for the Ethos-N57 and Ethos-N37 configurations.
- Support for the Inception V4 network for the Ethos-N77 configuration.
- Support for the SSD Mobilenet V1 network.
- Support for the sigmoid operation.
- Improved support for the concatenation operation.
- Improved support for operations with large inputs.
- Improved inference performance for networks with branching.
- Arm NN NPU backend compatible with Arm NN 19.08.
- Improved support for Arm NN reshape operations with non-4D shapes.

### Public API changes

- Intermediate compression is now enabled by default (CompilationOptions::m_EnableIntermediateCompression).
- The support library's CompilationOptions now requires an opaque blob of "capabilities" data, provided by the driver library.

### Other changes

- Improved stability, especially when queuing multiple inferences.
- Improved performance, especially for multiple layer networks.
- Fixed intermittent incorrect results or crash when running an inference with multiple inputs or outputs.

### Known issues

- None

# Trademarks and copyrights

Arm and Ethos are registered trademarks or trademarks of Arm Limited (or its subsidiaries) in the US and/or elsewhere.

Linux® is the registered trademark of Linus Torvalds in the U.S. and other countries.
