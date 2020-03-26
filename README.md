
# Arm® Ethos™-N series Driver Stack

## About the Arm Ethos-N Processors (NPU)

The Arm® Ethos-N NPU improves the inference performance of neural networks. The NPU targets 8-bit integer quantized Convolutional Neural Networks (CNN). However, the NPU also improves the performance of 16-bit integer CNNs and Recurrent Neural Networks (RNN). Please note that 16-bit integer and RNN support are not part of this driver stack release.

For more information, please refer to:
<https://www.arm.com/solutions/artificial-intelligence>

## About the Ethos-N driver stack

The Ethos-N driver stack consists of several components.

The list of open source components are:

* **Arm NN:** A set of Linux software that enables machine learning workloads on power efficient devices. On Linux, applications can directly link to Arm NN. On Android, you can use Arm NN as a backend for the Android NNAPI or applications can directly link to Arm NN.
* **Arm NN Android neural networks driver:** Supports the Android NNAPI on the NPU. The Arm NN Android neural networks driver is optional.
* **Ethos-N driver:** Contains the user space component of the driver.
* **Ethos-N kernel module:** Contains the kernel space component of the driver.
* **Arm NN Ethos-N backend:** Contains the Ethos-N backend for Arm NN.

The following software component is available under an Arm proprietary license:

* **Ethos-N firmware binaries package:** Contains the firmware that runs on the NPU.

Arm NN and the Arm NN Android neural networks driver are external downloads and links are provided below. All other components are part of this driver stack release.

## Platform requirements

Your (target) platform must meet specific requirements to run the Ethos-N driver. Your platform must have:

* An Armv8-A application processor.
* An Arm Ethos-N series NPU.
* At least 4GB of RAM.
* At least 5GB of free storage space.

## Build tools

To build the Ethos-N software, you require some tools. You must install the following tools on your development platform:

* A Linux distribution.  An open-source operating system.
* [Git](https://git-scm.com/) [Recommended: `2.17.1`].  A version control system that software developers use for source code management.
* [SCons](https://scons.org/) [Recommended: `v3.0.1`].  An open-source software construction tool.
* [Make](https://www.gnu.org/software/make/) [Recommended: `4.1`].  A build automation tool.
* [Sparse](https://www.kernel.org/doc/html/v4.12/dev-tools/sparse.html) [Recommended: `0.5.1`].  A semantic parser for C.
* [GNU C and C++ and compilers](https://gcc.gnu.org/) [Recommended: `5.3.1 20160413`].  Open-source tools for Arm processors.

### Install the build tools

You must use specific tools to build the Ethos-N driver. You can use a package manager to install the build tools. For example, enter the following commands to install the build tools on `Ubuntu 18.04` in order to cross compile:

```sh
sudo apt install git \
    scons \
    make \
    sparse \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    gcc \
    bc
```

## Install the Linux source tree

The Ethos-N driver stack source code depends on the Linux source tree to build the kernel module. You must configure the kernel to build the kernel module. Arm has tested version `4.9` of the Linux source tree.

1. Download version `4.9` or later of the Linux source tree from www.kernel.org.
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

    It is recommended to strip unnecessary symbols from the driver.

    * If you compile the driver natively:

        ```sh
        strip --strip-unneeded ethosn.ko
        ```

    * If you cross compile the driver:

        ```sh
        aarch64-linux-gnu-strip --strip-unneeded ethosn.ko
        ```

## Building the Ethos-N driver stack

The Ethos-N driver stack is written using portable `C++14` and the build system uses `scons` so it is possible to build for a wide variety of target platforms, from a wide variety of host environments.

### Download the Ethos-N driver stack

You must download the different components of the driver stack to build the driver. The different components of the driver stack are available for download in different ways.

Enter the following commands to download Arm NN, the Ethos-N driver, kernel module, and other components you require:

```sh
mkdir driver_stack
cd driver_stack
git clone https://github.com/Arm-software/armnn
git clone https://github.com/Arm-software/ethos-n-driver-stack ethosn-driver
```

## Configure SMMU support

Arm recommends that you configure the Linux kernel with Input/Output Memory Management Unit (IOMMU) support for use as one of the dependencies of the kernel driver.

Add the following flag to your Linux configuration to include all the dependencies the kernel module needs:

```make
CONFIG_ARM_SMMU_V3=y
```

If you run the NPU without an IOMMU, you must create a reserved memory area. The reserved memory area must begin on a 512MB aligned address and must not be larger than 512MB.

## Build the Ethos-N driver

You must follow specific steps to build the Ethos-N driver. You must build the Ethos-N driver, Ethos-N kernel module, and Arm NN.  Depending on your system, you must run some of the following steps with appropriate privileges.

1. Copy the `<path_to>/driver_stack/ethosn-driver/firmware/ethosn.bin` file into the `/lib/firmware/` folder of the target system that runs the Ethos-N driver.

    _Note that `<path_to>` is the directory where the `driver_stack` directory is stored._

2. How you compile the driver affects how you build the Ethos-N kernel module:

    * If you compile the driver natively, enter the following commands to build the Ethos-N kernel module:

        ```sh
        cd <path_to>/driver_stack/ethosn-driver/kernel-module
        make -C <path_to_kernel> M=$PWD modules
        ```

    * If you cross compile the driver, enter the following commands to build the Ethos-N kernel module:

        ```sh
        cd <path_to>/driver_stack/ethosn-driver/kernel-module
        make -C <path_to_kernel> M=$PWD ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules
        ```

    _Note that `<path_to_kernel>` is the directory where the Linux kernel tree is stored._

3. Copy the kernel module `ethosn.ko` to the system that runs the Ethos-N driver.

4. Enter the following command to load the kernel module on the target system:

    ```sh
    insmod ethosn.ko
    ```

5. Enter the following commands to build the user-space libraries of the Ethos-N driver:

    * If you compile the driver natively:

        ```sh
        cd <path_to>/driver_stack/ethosn-driver/driver
        scons
        ```

    * If you cross compile the driver:

        ```sh
        cd <path_to>/driver_stack/ethosn-driver/driver
        scons platform=aarch64
        ```

    _Use the configuration options to include dependencies from non-standard locations and to install files into non-standard locations. Enter the following command to see all configuration options:_

    ```sh
    scons --help
    ```

6. Enter the following command to install the user-space libraries of the Ethos-N driver:

    * If you compile the driver natively:

        ```sh
        cd <path_to>/driver_stack/ethosn-driver/driver
        scons install_prefix=<install_directory> install
        ```

    * If you cross compile the driver:

        ```sh
        cd <path_to>/driver_stack/ethosn-driver/driver
        scons platform=aarch64 install_prefix=<install_directory> install
        ```

7. Enter the following commands to link the Ethos-N backend to the Arm NN source tree:

    ```sh
    cd <path_to>/driver_stack/armnn/src/backends
    ln -s <path_to>/driver_stack/ethosn-driver/armnn-ethos-n-backend ethos-n
    ```

8. Build Arm NN for TensorFlow Lite. For instructions on building Arm NN, see <https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/configuring-the-arm-nn-sdk-build-environment-for-tensorflow-lite.>

    The following build options are required to the CMake call in the **Build Arm NN** section of the guide:

    ```cmake
    -DBUILD_TESTS=1
    -DARMNNREF=1
    -DETHOSN_SUPPORT=1
    -DETHOSN_ROOT=<install_directory>
    ```

    As part of the Arm NN build, the process automatically builds the Ethos-N driver plug-in for Arm NN.

    _Arm uses TensorFlow Lite as an example. You can also build Arm NN for [TensorFlow](https://developer.arm.com/technologies/machine-learning-on-arm/developer-material/how-to-guides/configuring-the-arm-nn-sdk-build-environment-for-tensorflow), [Caffe](https://developer.arm.com/technologies/machine-learning-on-arm/developer-material/how-to-guides/configuring-the-arm-nn-sdk-build-environment-for-caffe) or [ONNX](https://developer.arm.com/technologies/machine-learning-on-arm/developer-material/how-to-guides/configuring-the-arm-nn-sdk-build-environment-for-onnx)._

9. If you require Android NNAPI support, see [the instructions](https://github.com/Arm-software/android-nn-driver#armnn-android-neural-networks-driver) for how to build the Arm NN Android NNAPI driver.

## Running the Ethos-N driver

There are mutliple ways to exercise the Ethos-N driver.

1. Running the Arm NN Ethos-N backend unit tests. You need to have built Arm NN and the Ethos-N driver.

    If you have cross compiled you will need to copy the following files onto the target platform:
    * All `*.so` files built from Arm NN
    * `UnitTests` built from Arm NN
    * `libEthosNSupport.so` built from the Ethos-N driver inside `<install_directory>/lib/`
    * `libEthosNDriver.so` built from the Ethos-N driver inside `<install_directory>/lib/`

    Set `LD_LIBRARY_PATH` so the supplied libraries can be found and run the **UnitTests for the Ethos-N**.

    ```sh
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./UnitTests --run_test=*EthosN*
    ```

2. Running the `ExecuteNetwork` program provided by Arm NN. This supports running of TfLite models.

    If you have cross compiled you will need to copy the following files onto the target platform
    * All `*.so` files built from Arm NN
    * `ExecuteNetwork` built from Arm NN in the `tests/` folder
    * `libEthosNSupport.so` built from the Ethos-N driver inside `<install_directory>/lib/`
    * `libEthosNDriver.so` built from the Ethos-N driver inside `<install_directory>/lib/`

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

    An example of running **Mobilenet_v1_1.0_224_quant.tflite** on the Ethos-N:

    ```sh
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./tests/ExecuteNetwork -f tflite-binary -i input -y qasymm8 -o MobilenetV1/Predictions/Reshape_1 -z qasymm8 -d input_data.txt -m mobilenet_v1_1.0_224_quant.tflite -c EthosNAcc -c CpuRef
    ```

## Firmware Binary

The `ethosn.bin` has been compiled with the following security related flags:

```sh
-Werror -Wall -Wextra -Wformat=2 -Wno-format-nonliteral -Wctor-dtor-privacy -Woverloaded-virtual -Wsign-promo -Wstrict-overflow=2 -Wswitch-default -Wconversion -Wold-style-cast -Wsign-conversion -Wno-missing-braces
```

## Limitations

### Functional limitations

The following features and feature combinations have known limitations in this Arm Ethos-N driver stack release.

* This release has only been tested with specific networks

  * **Arm Ethos-N77:**
    * VGG16
    * MobileNet v1-1-224
    * SSD MobileNet v1
    * InceptionV3
    * InceptionV4
    * FSRCNN
  * **Arm Ethos-N57:**
    * VGG16
    * MobileNet v1-1-224
    * SSD MobileNet v1
    * InceptionV3
    * FSRCNN
  * **Arm Ethos-N37:**
    * VGG16
    * MobileNet v1-1-224
    * SSD MobileNet v1
    * InceptionV3
    * FSRCNN

    _Running other networks may result in parts of the network being run by the Arm NN CPU reference backend._

### Memory limitations

The driver expects that the minimum amount of memory available for an Arm Ethos-N NPU is 512 MB for systems that do not implement Arm® System Memory Management Unit (SMMU). The NPU might have problems booting up with less memory available.

Systems that implement Arm SMMU require a memory footprint of 3 MB to create all the page translations for the NPU memory accesses.

For more information on memory requirements and limitations, please see the documentation for your SoC.

## Security Issues

If you believe you have discovered a security issue please contact <MLG-Security@arm.com>

## License

The Arm Ethos-N driver stack is composed of multiple components, each with their own license. The components are:

* The Arm Ethos-N driver, which is the collection of user space libraries.
* The Arm NN backend, which interfaces the Ethos-N driver to Arm NN.
* The Arm Ethos-N kernel module, to be used with the Linux kernel.
* The Arm Ethos-N firmware, which will be loaded by the kernel module onto the NPU.

### License for the Ethos-N driver and Arm NN backend

The Arm Ethos-N driver and the Arm NN backend are provided under the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license. See [LICENSE](driver/LICENSE) and [LICENSE](armnn-ethos-n-backend/LICENSE) for more information. Contributions to this project are accepted under the same license.

```less
Copyright 2018-2020 Arm Limited

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

### License for the Ethos-N kernel module

The Arm Ethos-N kernel module is provided under the [GPL v2 only](https://spdx.org/licenses/GPL-2.0-only.html) license.
See [LICENSE](kernel-module/LICENSE) for more information. Contributions to this project are accepted under the same license.

Individual files contain the following tag instead of the full license text.

```less
SPDX-License-Identifier: GPL-2.0-only
```

This enables machine processing of license information based on the SPDX License Identifiers that are available here: <http://spdx.org/licenses/>

### EULA and TPIP for the Ethos-N firmware

The Ethos-N firmware binary is released under an [EULA](firmware/LES-PRE-21755.pdf).

The Ethos-N firmware binary was compiled against the CMSIS library, which is released under the [Apache 2.0 license](firmware/tpip-licenses/apache-2.0.txt). See [apache-2.0.txt](firmware/tpip-licenses/apache-2.0.txt) for more information.
