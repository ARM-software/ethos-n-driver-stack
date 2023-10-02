# Arm® Ethos™-N Driver Stack - Firmware

This repo contains the firmware code that runs on the Ethos-N NPU hardware.

## Folder Structure

This section provides a very brief overview of the folder structure.

* `firmware` contains the prebuilt binary "ethosn.bin" for control unit hardware and the folders below.

   * `control_unit` contains the firmware code for control unit hardware.
   * `ple` contains the programmable layer engine code.
   * `tpip-licenses` contains the appropriate license text applicable for this code.

## Control Unit (`/control_unit`)

This is the firmware code for the Ethos-N Control Unit. It uses a HAL to support running directly on the hardware.

## PLE (`/ple`)

This directory contains code for post-processing operations. These are referred to as "kernels".

## Dependencies

_Last known working versions of each component are stated where possible_

  * One of the following two compiler toolchains:
    * [`6.16.1LTS`] Arm Compiler toolchain (`armclang`) (https://developer.arm.com/downloads/view/ACOMP616).
      * A license may be required to use Arm Compiler, please consult the Arm Compiler documentation for more details.
    * The LLVM Embedded Toolchain for Arm. See the [section](#using-the-llvm-embedded-toolchain-for-arm) below for more details
  * [`V5.9.0`] CMSIS (https://github.com/ARM-software/CMSIS_5/releases/tag/5.9.0)

## Building the firmware and PLE

Assuming ethos-n-driver-stack is in the `<local_location>` folder, run the following command to build the control unit firmware and PLE in release mode. Building the firmware for control unit hardware will also build the PLE kernels as a dependency.
Also note that the PLE kernels are embedded in the firmware binary.

```sh
cd <local_location>/ethos-n-driver-stack
scons -C driver -j `nproc` cmsis_dir="<local_location>/cmsis/CMSIS/Core/Include" PATH="<local_location>/armclang/bin" control_unit-hardware
```
* For the build command above, the binary gets generated in `<local_location>/ethos-n-driver-stack/firmware/control_unit/build/release_hardware/ethosn.bin`

* The build also supports various command line parameters. Below are a few frequently used ones.

  * `debug=1` - Build in debug mode. By default, it is set to 0, i.e. release.
  * `logging=1` - Enable logging support.
  * `profiling=1` - Enable performance profiling of firmware.

* For a list of command line parameters and more information, refer to the descriptions in `<local_location>/ethos-n-driver-stack/driver/SConstruct` file.

## Using the LLVM Embedded Toolchain for Arm

The LLVM Embedded Toolchain for Arm can be used to build the PLE kernels and control unit firmware instead of Armclang if a license hasn't yet been acquired. Please note that limited testing has been performed with this toolchain and the performance is worse.

In order to use it download the release (https://github.com/ARM-software/LLVM-embedded-toolchain-for-Arm) and set the following command line options (or add them to your options.py) when building the control unit firmware or ple kernels.

 * `use_llvm_embedded=1`
 * `llvm_embedded_toolchain_path=<path_to_llvm_embedded_toolchain_root>`

The version of the toolchain that has been tested is 16.0.0.