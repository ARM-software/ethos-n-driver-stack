# Arm® Ethos™-N Driver Stack

## About the Arm Ethos-N neural processing unit (NPU)

The Arm Ethos-N NPUs improve the inference performance of neural networks. The NPUs target 8-bit integer quantized Convolutional Neural Networks (CNN). However, the NPUs also improve the performance of 16-bit integer CNNs and Recurrent Neural Networks (RNN). Please note that 16-bit integer and RNN support are not part of this driver stack release.

For more information, please refer to:
<https://www.arm.com/solutions/artificial-intelligence>

## About the Ethos-N driver stack

The Ethos-N driver stack targets the Ethos-N78 set of NPUs.

The Ethos-N driver stack consists of several components.

The list of open source components are:

* **Arm NN:** A software library that enables machine learning workloads on power efficient devices. On Linux®, applications can directly link to Arm NN. On Android™, you can use Arm NN as a backend for the Android NNAPI or applications can directly link to Arm NN.
* **Arm NN Android neural networks driver:** Supports the Android NNAPI on the NPU. The Arm NN Android neural networks driver is optional.
* **Ethos-N NPU driver:** Contains the user space component of the driver.
* **Ethos-N NPU kernel module:** Contains the kernel space component of the driver.
* **Arm NN Ethos-N NPU backend:** Contains the Ethos-N NPU backend for Arm NN.

The following software component is available under an Arm proprietary license:

* **Ethos-N NPU firmware binaries file:** Contains the firmware that runs on the NPU.

Arm NN and the Arm NN Android neural networks driver are external downloads and links are provided below. All other components are part of this driver stack release.

## Platform requirements

Your (target) platform must meet specific requirements to run the Ethos-N NPU driver. Your platform must have:

* An Armv8-A application processor.
* An Arm Ethos-N NPU.
* At least 4GB of RAM.
* At least 5GB of free storage space.

## Secure mode

The Arm Ethos-N NPU will boot up in either secure or non-secure mode depending on how the hardware has been configured. For more information, see Arm Ethos-N78 NPU Technical Reference Manual.

To use the NPU in secure mode, the target platform must have a [Trusted Firmware-A (TF-A)](https://www.trustedfirmware.org/projects/tf-a/) that includes the Arm Ethos-N NPU SiP service.

The Arm Ethos-N NPU SiP service can be found in the TF-A source tree, along with a reference implementation for how to use it on the Arm Juno platform.
For instructions on how to get the TF-A source and how to build it, see [TF-A documentation](https://trustedfirmware-a.readthedocs.io/en/latest/index.html). _Note that TF-A version v2.5 or later must be used._
The build flag needed to enable the Arm Ethos-N NPU SiP service for the Arm Juno platform can be found here [TF-A Arm Platform Build Options](https://trustedfirmware-a.readthedocs.io/en/latest/plat/arm/arm-build-options.html).

For information about how to port TF-A to another platform, see [TF-A Porting Guide](https://trustedfirmware-a.readthedocs.io/en/latest/getting_started/porting-guide.html).

## Build tools

To build the Ethos-N NPU software, you require some tools. You must install the following tools on your development platform:

* A Linux distribution.  An open-source operating system.
* [Git](https://git-scm.com/) [Recommended: `2.17.1`].  A version control system that software developers use for source code management.
* [SCons](https://scons.org/) [Recommended: `v3.0.1`].  An open-source software construction tool.
* [Make](https://www.gnu.org/software/make/) [Recommended: `4.1`].  A build automation tool.
* [Sparse](git://git.kernel.org/pub/scm/devel/sparse/sparse.git) [Recommended: `v0.6.3`].  A semantic parser for C.
* [GNU C and C++ and compilers](https://gcc.gnu.org/) [Recommended: `7.5.0`].  Open-source tools for Arm processors.

### Install the build tools

You must use specific tools to build the Ethos-N NPU driver. You can use a package manager to install the build tools. For example, enter the following commands to install the build tools on `Ubuntu 18.04` in order to cross compile:

```sh
sudo apt install git \
    scons \
    make \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    gcc \
    bison \
    flex \
    libssl-dev \
    bc
```

You will need to build and install sparse [Recommended: `v0.6.3`]:
```sh
git clone git://git.kernel.org/pub/scm/devel/sparse/sparse.git <path_to_sparse>/sparse --branch v0.6.3
cd <path_to_sparse>/sparse
sudo make PREFIX=/usr install
```

Additionally if you want to build unit tests for the Ethos-N NPU user space libraries, you need Catch2 [Recommended: `v2.13.0`]:

```sh
git clone --depth 1 https://github.com/catchorg/Catch2.git --branch v2.13.0 <path_to_catch>/Catch2
```

## Install the Linux source tree

The Ethos-N driver stack source code depends on the Linux source tree to build the kernel module. You must configure the kernel to build the kernel module.
Arm has tested version `4.9` of the Linux source tree in non-SMMU configurations and version `4.14`, `4.19`, `5.4`, `5.10` in SMMU configurations.

1. Download version `4.9`, `4.14`, `4.19`, `5.4` or `5.10` of the Linux source tree from [www.kernel.org](http://www.kernel.org).
2. How you compile the driver affects how you configure the Linux kernel source tree:

    * If you compile the driver natively, enter the following commands to configure the Linux kernel source tree:

        ```sh
        make -C <path_to_kernel> defconfig
        make -C <path_to_kernel> modules_prepare
        ```

    * If you cross compile the driver, enter the following commands to configure the Linux kernel source tree:

        ```sh
        make -C <path_to_kernel> ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- defconfig
        make -C <path_to_kernel> ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules_prepare
        ```

    _Note that `<path_to_kernel>` is the directory where the Linux kernel tree is stored._

## Building the Ethos-N driver stack

The Ethos-N driver stack is written using portable `C++14` and the build system uses `scons` so it is possible to build for a wide variety of target platforms, from a wide variety of host environments.

### Download the Ethos-N driver stack

You must download the different components of the driver stack to build the driver. The different components of the driver stack are available for download in different ways.

Enter the following commands to download Arm NN, the Ethos-N NPU driver, kernel module, and other components you require:

```sh
mkdir driver_stack
cd driver_stack
git clone https://github.com/Arm-software/armnn --branch v21.08
git clone https://github.com/Arm-software/ethos-n-driver-stack --branch 21.08
```

## Configure SMMU support

Arm recommends that you configure the Linux kernel with Input/Output Memory Management Unit (IOMMU) support for use as one of the dependencies of the kernel driver.
Arm has tested versions `4.14`, `4.19`, `5.4` and `5.10` of the Linux kernel with IOMMU support.

Add the following flag to your Linux configuration to include all the dependencies the kernel module needs:
```make
CONFIG_ARM_SMMU_V3=y
```

Ensure to comment out the following one since the SMMU driver cannot handle the SMMU v1 or v2 and the SMMU v3 both enabled at the same time:
```make
# CONFIG_ARM_SMMU is not set
```

If you run the NPU without an IOMMU, you must create a reserved memory area. This is used to store working data for the NPU, for example the firmware code and network data. The size of the reserved memory area should be chosen based on your specific use case. The amount of memory needed depends on several factors, including the number of NPU cores and the size of the networks being used. We recommend that you test to ensure the chosen size suits your needs. There are several restrictions on the properties of the reserved memory area. If these are not met then the kernel module will not load successfully or the NPU will not behave as expected:

1. The reserved memory area must begin on a 512MB aligned address
2. The reserved memory area must not be larger than 512MB
3. The reserved memory area must not be smaller than 4MB
4. The size of the reserved memory area must be a power-of-two
5. If the reserved memory area is smaller than 512MB, the NPU may still perform speculative memory reads to addresses up to 512MB from the starting address, which must not fail. The values returned from these speculative reads will not affect the behaviour of the NPU. This means that the NPU must have read access to a full 512MB region, however the portion of the 512MB region which is not in the reserved memory area does not need to be backed by physical memory.

## Build the Ethos-N NPU driver

You must follow specific steps to build the Ethos-N NPU driver. You must build the Ethos-N NPU driver, Ethos-N NPU kernel module, and Arm NN.  Depending on your system, you must run some of the following steps with appropriate privileges.

1. Copy the `<path_to>/driver_stack/ethos-n-driver-stack/firmware/ethosn.bin` file into the `/lib/firmware/` folder of the target system that runs the Ethos-N NPU driver.

    _Note that `<path_to>` is the directory where the `driver_stack` directory is stored._

2. How you compile the driver affects how you build the Ethos-N NPU kernel module:

    *Note:* By default, the kernel module is built for an NPU running in secure mode. If non-secure mode is needed, the following flag `EXTRA_CCFLAGS=" -DETHOSN_NS"` must be added to the make commands below. It is not possible to use a kernel module built for non-secure mode with an NPU running in secure mode.

    * If you compile the driver natively, enter the following commands to build the Ethos-N NPU kernel module:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/kernel-module
        make -C <path_to_kernel> M=$PWD modules
        ```

    * If you cross compile the driver, enter the following commands to build the Ethos-N NPU kernel module:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/kernel-module
        make -C <path_to_kernel> M=$PWD ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules
        ```

    _Note that `<path_to_kernel>` is the directory where the Linux kernel tree is stored._

    It is recommended to strip unnecessary symbols from the driver.

    * If you compile the driver natively:

        ```sh
        strip --strip-unneeded <path_to>/driver_stack/ethos-n-driver-stack/kernel-module/ethosn.ko
        ```

    * If you cross compile the driver:

        ```sh
        aarch64-linux-gnu-strip --strip-unneeded <path_to>/driver_stack/ethos-n-driver-stack/kernel-module/ethosn.ko
        ```


3. Copy the kernel module `<path_to>/driver_stack/ethos-n-driver-stack/kernel-module/ethosn.ko` to the system that runs the Ethos-N NPU driver.

4. Enter the following command to load the kernel module on the target system:

    ```sh
    insmod ethosn.ko
    ```

5. Enter the following commands to build the user space libraries of the Ethos-N NPU driver:

    * If you compile the driver natively:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/driver
        scons
        ```

    * If you cross compile the driver:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/driver
        scons platform=aarch64
        ```

    _Use the configuration options to include dependencies from non-standard locations and to install files into non-standard locations. Enter the following command to see all configuration options:_

    ```sh
    scons --help
    ```

6. Enter the following command to install the user space libraries of the Ethos-N NPU driver:

    * If you compile the driver natively:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/driver
        scons install_prefix=<install_directory> install
        ```

    * If you cross compile the driver:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/driver
        scons platform=aarch64 install_prefix=<install_directory> install
        ```

7. Enter the following commands to link the Ethos-N NPU backend to the Arm NN source tree:

    ```sh
    cd <path_to>/driver_stack/armnn/src/backends
    ln -s <path_to>/driver_stack/ethos-n-driver-stack/armnn-ethos-n-backend ethos-n
    ```

8. Build Arm NN for TensorFlow Lite. For instructions on building Arm NN, see <https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/configure-the-arm-nn-sdk-build-environment>

    The following build options are required to the CMake call in the [**Build Arm NN**](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/configure-the-arm-nn-sdk-build-environment/build-arm-nn) section of the guide:

    ```cmake
    -DBUILD_TESTS=1
    -DARMNNREF=1
    -DETHOSN_SUPPORT=1
    -DETHOSN_ROOT=<install_directory>
    ```

    For cross compilation, please refer to <https://github.com/ARM-software/armnn/blob/master/BuildGuideCrossCompilation.md>

    As part of the Arm NN build, the process automatically builds the Ethos-N NPU driver plug-in for Arm NN.

    _Arm uses TensorFlow Lite as an example. You can also build Arm NN for [ONNX](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/configure-the-arm-nn-sdk-build-environment/generate-the-build-dependencies-for-onnx)._

9. If you require Android NNAPI support, see [the instructions](https://github.com/Arm-software/android-nn-driver#armnn-android-neural-networks-driver) for how to build the Arm NN Android NNAPI driver.

## Running the Ethos-N NPU driver

There are multiple ways to exercise the Ethos-N NPU driver.

1. Running the Arm NN Ethos-N NPU backend unit tests. You need to have built Arm NN and the Ethos-N NPU driver.

    If you have cross compiled you will need to copy the following files onto the target platform:
    * All `*.so*` files built from Arm NN
    * `UnitTests` built from Arm NN
    * `libEthosNSupport.so` built from the Ethos-N NPU driver inside `<install_directory>/lib/`
    * `libEthosNDriver.so` built from the Ethos-N NPU driver inside `<install_directory>/lib/`

    *Note:* You may need to copy additional .so files depending on your toolchain and its runtime dependencies.

    Some tests require data files as input. These can be found in the folders `<path_to>/driver_stack/ethos-n-driver-stack/armnn-ethos-n-backend/test/replacement-tests`
    and `<path_to>/driver_stack/ethos-n-driver-stack/armnn-ethos-n-backend/test/mapping-tests`. These two folders (and their contents) must be available to the `UnitTests`
    executable, under the paths `armnn-ethos-n-backend/test/replacement-tests` and `armnn-ethos-n-backend/test/mapping-tests`, respectively,
    relative to the current directory that you run `UnitTests` from.
    If you have cross compiled you will therefore need to copy these folders onto the target platform.

    Set `LD_LIBRARY_PATH` so the supplied libraries can be found and run the **UnitTests for the Ethos-N NPU**.

    ```sh
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./UnitTests --test-suite=*Ethos*
    ```

2. Running the Ethos-N NPU driver user space unit tests.

    You need to have built the driver with testing enabled:
    * Add `tests=1` to your scons commands to build the user space component unit tests.
    * Make sure your CPATH scons variable (specified on the command-line) points to
      <path_to_catch>/Catch2/single_include/catch2/

    If you have cross compiled you will need to copy the following files onto the target platform:
    * `UnitTests` built for the Ethos-N NPU support library inside `<path_to>/driver_stack/ethos-n-driver-stack/driver/support_library/build/release_<platform>/tests`
    * `UnitTests` built for the Ethos-N NPU command stream inside `<path_to>/driver_stack/ethos-n-driver-stack/driver/support_library/command_stream/build/release_<platform>/tests`
    * `UnitTests` built for the Ethos-N NPU driver library inside `<path_to>/driver_stack/ethos-n-driver-stack/driver/driver_library/build/release_<platform>_kmod/tests`
    * `libEthosNSupport.so` built from the Ethos-N NPU driver inside `<install_directory>/lib/`
    * `libEthosNDriver.so` built from the Ethos-N NPU driver inside `<install_directory>/lib/`
    * `ethosn.ko` built from the Ethos-N NPU driver inside `<path_to>/driver_stack/ethos-n-driver-stack/kernel-module/`

    *Note:* You may need to copy additional .so files depending on your toolchain and its runtime dependencies.

    Set `LD_LIBRARY_PATH` so the supplied libraries can be found and run the **UnitTests for the Ethos-N NPU**.

    For each Ethos-N NPU driver user space component unit tests:

    ```sh
    LD_LIBRARY_PATH=<path_to_ethosn_libraries> ./UnitTests
    ```

3. Running the `ExecuteNetwork` program provided by Arm NN. This supports running of TfLite models.

    If you have cross compiled you will need to copy the following files onto the target platform:
    * All `*.so*` files built from Arm NN
    * `ExecuteNetwork` built from Arm NN in the `tests/` folder
    * `libEthosNSupport.so` built from the Ethos-N NPU driver inside `<install_directory>/lib/`
    * `libEthosNDriver.so` built from the Ethos-N NPU driver inside `<install_directory>/lib/`

    *Note:* You may need to copy additional .so files depending on your toolchain and its runtime dependencies.

    The `ExecuteNetwork` program requires parameters passed in. Detail about these can be found by running:

    ```sh
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./tests/ExecuteNetwork --help
    ```

    The minimum set of parameters which need to be provided are:

    ```less
    -f The format of the model provided in e.g. 'tflite-binary'
    -i A comma separated list of input tensor names in the model provided
    -y A comma separated list of data types for the input tensors
    -o A comma separated list of output tensor names in the model provided
    -z A comma separated list of data types for the output tensors
    -d Path to a whitespace separated list of values for the input
    -m Path to the model to be run
    -c The device you wish to run this inference on
    ```

    An example of running **Mobilenet_v1_1.0_224_quant.tflite** on the Ethos-N NPU:

    ```sh
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./tests/ExecuteNetwork -f tflite-binary -i input -y qasymm8 -o MobilenetV1/Predictions/Reshape_1 -z qasymm8 -d input_data.txt -m mobilenet_v1_1.0_224_quant.tflite -c EthosNAcc -c CpuRef
    ```

## Power Management

The Ethos-N NPU supports the following power management features to allow efficient power usage:
* Suspend: Once the suspend command has been given, the NPU is brought to a low power state and the NPU's state is kept in RAM. This feature is sometimes referred to as 'suspend to RAM'.
* Sleep: This is a runtime power management feature that dynamically puts the NPU in a low-power state when it is not being used; the rest of the system still functions normally.

The Ethos-N NPU kernel module implements the Linux Power Management (PM) callbacks. This gives the flexibility to the system integrator to integrate the Ethos-N NPU in any power domain.

## Firmware Binary

The `ethosn.bin` has been compiled with the following security related flags:

```sh
-Werror -Wall -Wextra -Wformat=2 -Wno-format-nonliteral -Wctor-dtor-privacy -Woverloaded-virtual -Wsign-promo -Wstrict-overflow=2 -Wswitch-default -Wconversion -Wold-style-cast -Wsign-conversion -Wno-missing-braces
```

## Limitations

### Functional limitations

The following features and feature combinations have known limitations in this Ethos-N driver stack release.

* This release has only been tested with specific networks

    * VGG16
    * MobileNet v1-1-224
    * SSD MobileNet v1
    * InceptionV3
    * InceptionV4
    * FSRCNN
    * Yolo V3
    * ResNet v2-50
    * SRGAN

    _Running other networks may result in parts of the network being run by the Arm NN CPU reference backend._

### Memory limitations

The driver expects that the minimum amount of memory available for an Arm Ethos-N NPU is 512 MB for systems that do not implement Arm® System Memory Management Unit (SMMU). The NPU might have problems booting up with less memory available.

Systems that implement Arm SMMU require a memory footprint of 3 MB to create all the page translations for the NPU memory accesses.

For more information on memory requirements and limitations, please see the documentation for your SoC.

### Power management limitations

Hibernate (sometimes referred to as 'suspend to disk') is not supported.

## License

The Ethos-N driver stack is composed of multiple components, each with their own license. The components are:

* The Arm Ethos-N NPU driver, which is the collection of user space libraries.
* The Arm NN backend, which interfaces the Ethos-N NPU driver to Arm NN.
* The Arm Ethos-N NPU kernel module, to be used with the Linux kernel.
* The Arm Ethos-N NPU firmware, which will be loaded by the kernel module onto the NPU.

### License for the Ethos-N NPU driver and Arm NN backend

The Arm Ethos-N NPU driver and the Arm NN backend are provided under the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license. See [LICENSE](driver/LICENSE) and [LICENSE](armnn-ethos-n-backend/LICENSE) for more information. Contributions to this project are accepted under the same license.

```less
Copyright 2018-2021 Arm Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

Individual files contain the following tag instead of the full license text.

```less
SPDX-License-Identifier: Apache-2.0
```

This enables machine processing of license information based on the SPDX License Identifiers that are available here: <http://spdx.org/licenses/>

### License for the Ethos-N NPU kernel module

The Arm Ethos-N NPU kernel module is provided under the [GPL v2 only](https://spdx.org/licenses/GPL-2.0-only.html) license.
See [LICENSE](kernel-module/LICENSE) for more information. Contributions to this project are accepted under the same license.

Individual files contain the following tag instead of the full license text.

```less
SPDX-License-Identifier: GPL-2.0-only
```

### EULA and TPIP for the Ethos-N NPU firmware

The Ethos-N NPU firmware binary is released under an [EULA](firmware/LES-PRE-21755.pdf).

The Ethos-N NPU firmware binary was compiled against the CMSIS library, which is released under the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license. See [apache-2.0.txt](firmware/tpip-licenses/apache-2.0.txt) for more information.


### Trademarks and copyrights

Arm and Ethos are registered trademarks or trademarks of Arm Limited (or its subsidiaries) in the US and/or elsewhere.

Android is a trademark of Google LLC.

Linux® is the registered trademark of Linus Torvalds in the U.S. and other countries.
