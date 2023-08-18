# Arm® Ethos™-N Driver Stack

## About the Arm Ethos-N Neural Processing Unit (NPU)

The Arm Ethos-N NPUs improve the inference performance of neural networks. The NPUs target 8-bit integer quantized Convolutional Neural Networks (CNN).

For more information, see <https://www.arm.com/solutions/artificial-intelligence>.

## Supported networks

* We have tested this release with the following networks:

    * VGG16
    * MobileNet v1 0.25 224
    * SSD MobileNet v1
    * InceptionV3
    * InceptionV4
    * FSRCNN
    * Yolo V3
    * ResNet v2-50
    * SRGAN
    * U-Net
    * EfficientNet Lite

    _Note: Running other networks may result in parts of the network being run by the Arm NN CPU reference backend._

## About the Ethos-N driver stack

The Ethos-N driver stack targets the Ethos-N78 NPU and consists of open-source and other software components.

The open-source components are:

* **Arm NN:** A software library that enables machine learning workloads on power efficient devices. On Linux®, applications can link directly to Arm NN. On Android™, you can use Arm NN as a backend for the Android NNAPI or applications can link directly to Arm NN.
* **Arm NN Android neural network driver:** Supports the Android NNAPI on the NPU. The Arm NN Android neural network driver is optional.
* **Ethos-N NPU driver:** Contains the user-space component of the driver.
* **Ethos-N NPU kernel module:** Contains the kernel space component of the driver.
* **Arm NN Ethos-N NPU backend:** Contains the Ethos-N NPU backend for Arm NN.

The software component that is available under an Arm proprietary license is:

* **Ethos-N NPU firmware binaries file:** Contains the firmware that runs on the NPU.

Arm NN and the Arm NN Android neural network driver are external downloads and links are provided in this README file. All other components are part of this driver stack release.

## Target platform requirements

Your target platform must meet specific requirements to run the Ethos-N NPU driver. Your platform must have:

* An Armv8-A application processor
* An Arm Ethos-N NPU
* At least 4GB of RAM
* At least 16GB of free storage space

## Secure mode and TZMP1

Depending on how the hardware has been configured, the NPU boots up in either Secure or Non-secure mode.

To use the NPU in Secure mode, the target platform must have a [Trusted Firmware-A (TF-A)](https://www.trustedfirmware.org/projects/tf-a/) that is built with Arm Ethos-N NPU support.

NPU support is available in the TF-A source tree, along with a reference implementation showing how to use it on the Arm Juno platform.
For instructions on how to get and build the TF-A source, see the [TF-A documentation](https://trustedfirmware-a.readthedocs.io/en/latest/index.html). The following commit of TF-A has been tested:
[4796d2d9bb4a1c0ccaffa4f6b49dbb0f0304d1d1](https://review.trustedfirmware.org/plugins/gitiles/TF-A/trusted-firmware-a/+/4796d2d9bb4a1c0ccaffa4f6b49dbb0f0304d1d1).

The build flag required to enable NPU support for the Arm Juno platform is available at [TF-A Arm Platform Build Options](https://review.trustedfirmware.org/plugins/gitiles/TF-A/trusted-firmware-a/+/4796d2d9bb4a1c0ccaffa4f6b49dbb0f0304d1d1/docs/plat/arm/arm-build-options.rst).

For information about boot up in Secure or Non-secure modes, see the Arm Ethos-N78 NPU Technical Reference Manual.

For information about how to port TF-A to another platform, see the [TF-A Porting Guide](https://review.trustedfirmware.org/plugins/gitiles/TF-A/trusted-firmware-a/+/4796d2d9bb4a1c0ccaffa4f6b49dbb0f0304d1d1/docs/porting-guide.rst).

If TZMP1 support is required, the TF-A must be built with the TZMP1 build option as well as the NPU build option.

The kernel module must also be built with appropriate flags depending on the security level required.

Please refer to the following table for the supported configurations:

| System security level | NPU hardware configuration | TF-A build configuration | Kernel module build configuration |
| --------------------- | --------------------------- | ------------------------ | --------------------------------- |
| **Non-secure**        | Non-secure                  | No flags                 | Non-secure                        |
| **Secure**            | Secure                      | NPU                      | Secure                            |
| **TZMP1**             | Secure + SMMU + Single-core | NPU + TZMP1              | TZMP1                             |

All other combinations of configurations are not supported and may lead to errors or unexpected behaviour.

## Build tools

To build the Ethos-N NPU software, some tools must be installed on the platform used for the compilation. The platform can be either your target platform or the host platform when cross compiling. We have only tested building the driver stack on the `Ubuntu 20.04 LTS x86 64-bit` Linux distribution. The required tools are:

* A Linux distribution. An open-source operating system.
* [Git](https://git-scm.com/) A version control system that software developers use for source code management. We recommend version `2.25.1`.
* [SCons](https://scons.org/) An open-source software construction tool. We recommend `v3.1.2`.
* [Make](https://www.gnu.org/software/make/) A build automation tool. We recommend version `4.2.1`.
* [Sparse](https://git.kernel.org/pub/scm/devel/sparse/sparse.git) A semantic parser for C. We recommend `v0.6.3`.
* [GNU C and C++ and compilers](https://gcc.gnu.org/) Open-source tools for Arm processors. We recommend version `9.4.0`.

The build platform must have at least:
* 8GB of RAM
* 4GB of free storage space

### Install the build tools

You must use specific tools to build the Ethos-N NPU driver. To install these build tools, you can use a package manager. For example, to install the build tools on `Ubuntu 20.04` to cross compile, enter the following commands:

```sh
sudo apt install git \
    scons \
    make \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    gcc \
    g++ \
    bison \
    flex \
    libssl-dev \
    bc \
    rsync \
    python3
```

You must build and install sparse. We recommend `v0.6.3`. To build and install sparse, enter the following commands:

```sh
git clone git://git.kernel.org/pub/scm/devel/sparse/sparse.git <path_to_sparse>/sparse --branch v0.6.3
cd <path_to_sparse>/sparse
sudo make PREFIX=/usr install
```

Also, if you want to build unit tests for the Ethos-N NPU user-space libraries, you must install Catch2. We recommend `v2.13.8`. To install Catch2, enter the following commands:

```sh
git clone --depth 1 https://github.com/catchorg/Catch2.git --branch v2.13.8 <path_to_catch>/Catch2
```

## Install the Linux source tree

The Ethos-N driver stack source code depends on the Linux source tree to build the kernel module. You must configure the kernel to build the kernel module.
Arm has tested versions `5.4`, and `5.10` of the Linux source tree in System Memory Management Unit (SMMU) configurations.

To configure the kernel:

1. Download version `5.4`, or `5.10` of the Linux source tree from [www.kernel.org](http://www.kernel.org).
2. Configure the memory system for the NPU:

    * If you run the NPU with an SMMU, because the SMMU driver cannot simultaneously enable the SMMU v1 or v2 and the SMMU v3, you must disable the CONFIG_ARM_SMMU configuration key and enable the CONFIG_ARM_SMMU_V3 configuration key:

        ```make
        CONFIG_ARM_SMMU_V3=y
        CONFIG_ARM_SMMU=n
        ```

    * If you run the NPU without an SMMU, you must create a reserved memory area. The reserved memory area stores working data for the NPU, for example the firmware code and network data. The size of the reserved memory area depends on your specific use case and several factors. These factors include the number of NPU cores and the size of the networks being used. We recommend that you test to ensure the chosen size is suitable.

    There are restrictions on using the reserved memory area. These restrictions are detailed in [Limitations](#limitations).


3. How you compile the driver affects how you configure the Linux kernel source tree:

    * If you compile the driver natively, enter the following commands:

        ```sh
        make -C <path_to_kernel> defconfig
        make -C <path_to_kernel> modules_prepare
        ```

    * If you cross compile the driver, enter the following commands:

        ```sh
        make -C <path_to_kernel> ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- defconfig
        make -C <path_to_kernel> ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules_prepare
        ```

    _Note: The `<path_to_kernel>` directory is where the Linux kernel tree is stored._

## Build the Ethos-N driver stack

The Ethos-N driver stack is written using portable `C++14` and the build system uses `scons`. It is therefore possible to build for a wide variety of target platforms, from a wide variety of host environments.

### Download the Ethos-N driver stack

To build the Ethos-N driver stack, you must download Arm NN and the Ethos-N driver stack components. The Ethos-N driver stack download contains the Ethos-N NPU driver, kernel module, backend, and firmware binaries file.

To download the components, enter the following commands:

```sh
mkdir driver_stack
cd driver_stack
git clone https://github.com/Arm-software/armnn --branch v23.08
git clone https://github.com/Arm-software/ethos-n-driver-stack --branch 23.05
```

    _Note: The default branch on GitHub has changed to main._

### Build the Ethos-N NPU driver

You must follow specific steps to build the Ethos-N NPU driver. You must build the Ethos-N NPU driver, Ethos-N NPU kernel module, and Arm NN. Depending on your system, appropriate privileges are required to run some of the following commands:

1. Copy the `<path_to>/driver_stack/ethos-n-driver-stack/firmware/ethosn.bin` file into the `/lib/firmware/` folder of the target system that runs the Ethos-N NPU driver.

    _Note: The `<path_to>` directory is where the `driver_stack` directory is stored._

2. Build the Ethos-N NPU kernel module.

    _Note: By default, the kernel module is built for an NPU running in Secure mode. If Non-secure mode is required, add the `EXTRA_CCFLAGS=" -DETHOSN_NS"` flag to the following make commands. If TZMP1 mode is required, add the `EXTRA_CCFLAGS=" -DETHOSN_TZMP1"` flag to the following make commands. If disabling all SMC calls is required when using using Non-secure mode, add the `EXTRA_CCFLAGS=" -DETHOSN_NO_SMC"` flag to the following make commands. Secure mode still makes use of SMC calls regardless of this flag. Please see the above [Secure mode and TZMP1](#secure-mode-and-tzmp1) section for more details._

    How you compile the driver affects how you build the Ethos-N NPU kernel module:

    * If you compile the driver natively, enter the following commands:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/kernel-module
        make -C <path_to_kernel> M=$PWD modules
        ```

    * If you cross compile the driver, enter the following commands:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/kernel-module
        make -C <path_to_kernel> M=$PWD ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules
        ```

    _Note: The `<path_to_kernel>` directory is where the Linux kernel tree is stored._

    We recommend you strip unnecessary symbols from the driver:

    * If you compile the driver natively, enter the following command:

        ```sh
        strip --strip-unneeded <path_to>/driver_stack/ethos-n-driver-stack/kernel-module/ethosn.ko
        ```

    * If you cross compile the driver, enter the following command:

        ```sh
        aarch64-linux-gnu-strip --strip-unneeded <path_to>/driver_stack/ethos-n-driver-stack/kernel-module/ethosn.ko
        ```


3. Copy the kernel module `<path_to>/driver_stack/ethos-n-driver-stack/kernel-module/ethosn.ko` to the system that runs the Ethos-N NPU driver.

4. To load the kernel module on the target system, enter the following command:

    ```sh
    insmod ethosn.ko
    ```

5. To build the user-space libraries of the Ethos-N NPU driver:

    * If you compile the driver natively, enter the following command:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/driver
        scons
        ```

    * If you cross compile the driver, enter the following command:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/driver
        scons platform=aarch64
        ```

    To include dependencies from non-standard locations and to install files into non-standard locations, use the configuration options. To see all configuration options, enter the following command:

    ```sh
    scons --help
    ```

6. To install the user-space libraries of the Ethos-N NPU driver:

    * If you compile the driver natively, enter the following command:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/driver
        scons install_prefix=<install_directory> install
        ```

    * If you cross compile the driver, enter the following command:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/driver
        scons platform=aarch64 install_prefix=<install_directory> install
        ```

    _Note: The `<install_directory>` directory is where files built from the Ethos-N NPU driver will be stored._

7. Set up the Arm NN directory structure and link the Ethos-N NPU backend to the Arm NN source tree:

    ```sh
    cd <path_to>/driver_stack
    mkdir armnn_build
    cd armnn_build
    mkdir source
    mkdir build
    cd ..
    mv armnn armnn_build/source
    cd <path_to>/driver_stack/armnn_build/source/armnn/src/backends
    ln -s <path_to>/driver_stack/ethos-n-driver-stack/armnn-ethos-n-backend ethos-n
    ```

8. Build Arm NN for TensorFlow Lite. For instructions about building Arm NN, see the README.md file in the armnn/build-tool directory.

    ```sh
    cd <path_to>/driver_stack/armnn_build
    sudo ./source/armnn/build-tool/scripts/install-packages.sh
    ./source/armnn/build-tool/scripts/setup-armnn.sh --tflite-parser --target-arch=aarch64
    ```

    The following build options are required to use the Ethos-N backend:

    ```cmake
    -DBUILD_TESTS=1
    -DARMNNREF=1
    -DETHOSN_SUPPORT=1
    -DETHOSN_ROOT=<install_directory>
    ```

    ```sh
    ./source/armnn/build-tool/scripts/build-armnn.sh  --target-arch=aarch64 --tflite-parser --ref-backend --armnn-cmake-args="-DBUILD_TESTS=1,-DARMNNREF=1,-DETHOSN_SUPPORT=1,-DETHOSN_ROOT=<install_directory>"
    ```

    Running the above command will produce a compressed folder containing all the files required to use Arm NN.

## Exercise the Ethos-N NPU driver

There are multiple ways to exercise the Ethos-N NPU driver:

1. Running the Arm NN Ethos-N NPU backend unit tests.

    _Note: You must have built Arm NN and the Ethos-N NPU driver before running the backend unit tests._

    If you have cross compiled the driver, copy the following files onto the target platform:

    * All `*.so*` files built from Arm NN.
    * `UnitTests` built from Arm NN.
    * `libEthosNSupport.so` built from the Ethos-N NPU driver inside the `<install_directory>/lib/` directory.
    * `libEthosNDriver.so` built from the Ethos-N NPU driver inside the `<install_directory>/lib/` directory.

    _Note: You may need to copy extra `.so` files depending on your toolchain and its runtime dependencies._

    _Note: You may need to set the library path so that the supplied libraries can be found._

    To set the library path and run the backend unit tests for the Ethos-N NPU, enter the following command:

    ```sh
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./UnitTests --test-suite=*Ethos*
    ```

2. Running the Ethos-N NPU driver user-space unit tests.

    Before running the user-space unit tests, you must have built the driver with testing enabled. To enable testing:
    * Add `tests=1` to your scons commands to build the user-space component unit tests.
    * Make sure your CPATH scons variable, which is specified on the command-line, points to
      `<path_to_catch>/Catch2/single_include/catch2/`.
    * Make sure that your unit_test_kernel_dir scons variable, which is specified on the command-line, points to the Linux kernel source tree used to build the Ethos-N NPU Linux kernel module. The Linux kernel's user-space headers will be used when building and running the unit tests to determine what features that are supported and should be tested.
    _Note: In order to support all the unit tests, the Linux kernel source tree used must be version 5.6 or higher. The Linux kernel source tree should be the same as the one used to compile the Linux kernel module._
    _Note: If the Linux kernel source tree does not contain the Linux kernel's generated user-space headers, scons will generate them using the Linux kernel's `headers_install` make target._

    If you have cross compiled the driver, copy the following files onto the target platform:
    * `UnitTests` built for the Ethos-N NPU support library inside `<path_to>/driver_stack/ethos-n-driver-stack/driver/support_library/build/release_<platform>/tests`.
    * `UnitTests` built for the Ethos-N NPU command stream inside `<path_to>/driver_stack/ethos-n-driver-stack/driver/support_library/command_stream/build/release_<platform>/tests`.
    * `UnitTests` built for the Ethos-N NPU driver library inside `<path_to>/driver_stack/ethos-n-driver-stack/driver/driver_library/build/release_<platform>_kmod/tests`.
    * `libEthosNSupport.so` built from the Ethos-N NPU driver inside `<install_directory>/lib/`.
    * `libEthosNDriver.so` built from the Ethos-N NPU driver inside `<install_directory>/lib/`.
    * `ethosn.ko` built from the Ethos-N NPU driver inside `<path_to>/driver_stack/ethos-n-driver-stack/kernel-module/`.

    _Note: You may need to copy additional `.so` files depending on your toolchain and its runtime dependencies._

    _Note: You may need to set the library path so that the supplied libraries can be found._

    To set the library path and run the user-space unit tests for the Ethos-N NPU, enter the following command:

    ```sh
    LD_LIBRARY_PATH=<path_to_ethosn_libraries> ./UnitTests
    ```

3. Run the `ExecuteNetwork` program provided by Arm NN. This program supports the running of TensorFlow Lite models.

    If you have cross compiled the driver, you must copy the following files onto the target platform:
    * All `*.so*` files built from Arm NN.
    * `ExecuteNetwork` built from Arm NN in the `tests/` folder.
    * `libEthosNSupport.so` built from the Ethos-N NPU driver inside `<install_directory>/lib/`.
    * `libEthosNDriver.so` built from the Ethos-N NPU driver inside `<install_directory>/lib/`.

    _Note: You may need to copy additional `.so` files depending on your toolchain and its runtime dependencies._

    The `ExecuteNetwork` program requires parameters passed in. To find details about these parameters, enter the following command:

    ```sh
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./tests/ExecuteNetwork --help
    ```

    The minimum set of required parameters are:

   * `-d` The path to a whitespace separated list of values for the input.
   * `-m` The path to the model to be run.
   * `-c` The device you wish to run this inference on.

    An example of running **mobilenet_v1_0.25_224_default_minmax.tflite** on the Ethos-N NPU is:

    ```sh
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./tests/ExecuteNetwork -f tflite-binary -i input -y qasymmu8 -o MobilenetV1/Predictions/Reshape_1 -z qasymmu8 -d input_data.txt -m mobilenet_v1_0.25_224_default_minmax.tflite -c EthosNAcc -c CpuRef
    ```

5. Run the Ethos-N NPU System Tests. This program supports the running of TensorFlow Lite models, networks described in Ggf format, and contains a suite of built-in tests.

    To build the `system-tests` executable, use the following command:

        ```sh
        cd <path_to>/driver_stack/ethos-n-driver-stack/driver
        scons tests=1 ../tools/system_tests
        ```

    * If you are cross compiling the system tests, add `platform=aarch64` to the scons command.
    * Make sure your CPATH scons variable, which is specified on the command-line, points to `<path_to_catch>/Catch2/single_include/catch2/`.
    * Make sure that your armnn_dir scons variable, which is specified on the command line, points to `<path_to>/driver_stack/armnn_build/source/armnn/`.
    * Make sure that your unit_test_kernel_dir scons variable, which is specified on the command-line, points to the Linux kernel source tree used to build the Ethos-N NPU Linux kernel module. The Linux kernel's user-space headers will be used when building and running the unit tests to determine what features are supported and should be tested.
    _Note: In order to support all the unit tests, the Linux kernel source tree used must be version 5.6 or higher. The Linux kernel source tree should be the same as the one used to compile the Linux kernel module._
    _Note: If the Linux kernel source tree does not contain the Linux kernel's generated user-space headers, scons will generate them using the Linux kernel's `headers_install` make target._

    This generates an executable for system tests and system test unit tests:
    * `system-tests` in `<path_to>/driver_stack/ethos-n-driver-stack/tools/system_tests/build/release_<platform>/`
    * `UnitTests` in `<path_to>/driver_stack/ethos-n-driver-stack/tools/system_tests/build/release_<platform>/tests`

    If you have cross compiled the driver, you must copy the following files onto the target platform:
    * All `*.so*` files built from Arm NN.
    * `system-tests` and `UnitTests` from the Ethos-N NPU driver mentioned above.
    * `libEthosNSupport.so` built from the Ethos-N NPU driver inside `<install_directory>/lib/`.
    * `libEthosNDriver.so` built from the Ethos-N NPU driver inside `<install_directory>/lib/`.

    The `system-tests` program requires parameters passed in. To find details about these parameters, enter the following command:

    ```sh
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./system-tests -?
    ```

    _Note: As system tests is built using the Catch2 framework, this will also display the built-in options for the Catch2 framework._

    To run networks using system tests, you can use `TfLiteRunner` which accepts TensorFlow Lite models, or `GgfRunner` which accepts networks described in a text format. For examples of how `.ggf` files are laid out, see `<path_to>/driver_stack/ethos-n-driver-stack/tools/system_tests/graphs`. For more information, check `GgfParser.cpp`.

    An example of using `TfLiteRunner` to run **mobilenet_v1_0.25_224_default_minmax.tflite** in system tests is:

    ```sh
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./system-tests TfLiteRunner --tflite-file mobilenet_v1_0.25_224_default_minmax.tflite
    ```

    An example of using `GgfRunner` to run  **conv1x1_nhwc.ggf** in system tests is:

    ```
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./system-tests GgfRunner --ggf-file conv1x1_nhwc.ggf
    ```

    _Note: `TfLiteRunner` and `GgfRunner` compare the inference result against a reference result generated using Arm NN with the CpuRef backend. To skip this check, use the `--skip-ref` option._

    To run other tests within system tests, use the following command:

    ```
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./system-tests <test_case>
    ```

    If `<test_case>` is empty, system tests will run every test.

    _Note: Some test cases are dependant on specific Ethos-N NPU configurations to run successfully._

    To run the unit tests for system tests, use the following command:

    ```
    LD_LIBRARY_PATH=<path_to_ethosn_libraries>:<path_to_armnn_libs> ./UnitTests <test_case>
    ```

## Power Management

To allow efficient power usage, the Ethos-N NPU supports the following power management features:

* Suspend: Puts the NPU in a low-power state. The state of the NPU is stored in RAM. This feature is sometimes referred to as 'suspend to RAM'.
* Hibernate: Puts the NPU in a low-power state. The state of the NPU is saved to disk. This feature is sometimes referred to as 'suspend to disk'.
* Sleep: This is a runtime power management feature that dynamically puts the NPU in a low-power state when the NPU is not being used. The rest of the system still functions normally.

The Ethos-N NPU kernel module implements the Linux Power Management (PM) callbacks. This gives the flexibility to the system integrator to integrate the Ethos-N NPU in any power domain.

There are some restrictions on using the power management features. These restrictions are detailed in [Limitations](#limitations).

## Firmware Binary

The `ethosn.bin` has been compiled with the following security-related flags:

```sh
-Werror -Wall -Wextra -Wformat=2 -Wno-format-nonliteral -Wctor-dtor-privacy -Woverloaded-virtual -Wsign-promo -Wstrict-overflow=2 -Wswitch-default -Wconversion -Wold-style-cast -Wsign-conversion -Wno-missing-braces
```
## Build the Ethos-N driver stack in Android

To build the Ethos-N driver stack in Android, you must first install the Android NN driver with Arm NN in Android. To install the Android NN driver in Android, follow the instructions: [https://github.com/ARM-software/android-nn-driver/blob/main/docs/IntegratorGuide.md](https://github.com/ARM-software/android-nn-driver/blob/main/docs/IntegratorGuide.md). You must then put the downloaded Ethos-N project folder in the Android build folder.

 _Note: The Android build folder is the same folder that you put the Android NN driver._

After you install the Android NN driver with Arm NN in Android, install the Ethos-N driver stack using the setup_android.sh script. You can edit the paths in the script and execute it, or use the script as reference if you create your own script. The script performs the following steps:

1. The code in armnn-ethos-n-backend is an Arm NN backend which is built by Arm NN. For Arm NN to build the backend, a symbolic link is needed from the Arm NN src backend folder to the Ethos-N NPU backend folder:

    ```sh
    ln -s ../../../../ethos-n-driver-stack/armnn-ethos-n-backend ../android-nn-driver/armnn/src/backends/ethos-n
    ```

2. Build the Ethos-N kernel module, which will be treated as a pre-built by the Android build.

3. Update device.mk with instructions to include Arm NN Ethos-N backend and add Ethos-N kernel module and firmware

    We need to add a few config lines in device.mk. Arm NN needs a build flag to include the Ethos-N driver.
    * ARMNN_ETHOSN_ENABLE := 1

    In an Android build, we recommend adding the vendor-specific drivers to the /vendor file system so that the /system and /vendor file systems can be updated independently. This can be achieved by adding the kernel module and firmware to your device.mk, for example:
     * BOARD_VENDOR_KERNEL_MODULES += vendor/arm/ethos-n-driver-stack/kernel-module/ethosn.ko
     * PRODUCT_COPY_FILES += vendor/arm/ethos-n-driver-stack/firmware/ethosn.bin:$(TARGET_COPY_OUT_VENDOR)/lib/firmware/ethosn.bin

4. When booting, the kernel module must be started. To load the kernel module at boot, add the following to your init board file (for example, init.juno.rc):

    ```sh
    on post-fs
        insmod /vendor/lib/modules/ethosn.ko
    ```

5. To ensure the device tree contains the correct nodes for your Ethos-N hardware, the Ethos-N dts files must be merged with your dts file.

Build the Android image. Flash and run the image according to your normal target instructions.

    _Note: If you get "No firmware found." in the kernel log, make sure your kernel searches for firmware files in /vendor/lib/firmware. For example, this can be done by adding the kernel argument "firmware_class.path=/vendor/lib/firmware/" when booting. If you want to use a different folder for your firmware files, you can also edit the build to make sure the ethonsn.bin file is moved to the different folder._

To use the Android NN driver and Arm NN, please see the instructions in the "Testing" section of the integration guide: https://github.com/ARM-software/android-nn-driver/blob/main/docs/IntegratorGuide.md#testing.

To specify the Ethos-N backend, use "-c EthosNAcc", for example:

```sh
adb shell /vendor/bin/hw/android.hardware.neuralnetworks@1.3-service-armnn -v -c EthosNAcc
```

## Limitations

The following features and feature combinations have known limitations in this Ethos-N driver stack release.

### Backend limitations

Custom allocators can only be used for importing intermediate tensors, not inputs and outputs.

### Memory limitations

For systems that do not implement an Arm SMMU, the driver expects a reserved memory area to be associated with the NPU device. There are several restrictions on the properties of the reserved memory area. If these restrictions are not met, then the kernel module will not load successfully or the NPU will behave unexpectedly. The restrictions are:

1. The reserved memory area must begin on a 512MB aligned address.
2. The reserved memory area must not be larger than 512MB.
3. The reserved memory area must not be smaller than 4MB.
4. The size of the reserved memory area must be a power-of-two.
5. If the reserved memory area is smaller than 512MB, the NPU may still perform speculative memory reads to addresses up to 512MB from the starting address, which must not fail. The values returned from these speculative reads will not affect the behavior of the NPU. This means that the NPU must have read access to a full 512MB region, however the portion of the 512MB region which is not in the reserved memory area does not need to be backed by physical memory.

For systems that do implement a SMMU, a memory footprint of 3MB is required to create the page translations for the NPU memory accesses. This is to guarantee that speculative memory accesses to unmapped memory addresses cannot occur. During a short interval, when the SMMU page tables are reconfigured to map new buffers' address ranges, speculative accesses to unmapped addresses will result in a bus error, hence an error interrupt and any running inference will fail.

For more information on memory requirements and limitations, see the documentation for your SoC.

### Power Management limitations

The hibernate power management feature is only supported with SMMU configurations.

### Security limitations

The security guarantees provided by the NPU hardware and software stack depend on the configuration of the system. In particular, configurations
that do not use an SMMU (and therefore use a reserved memory area) have weaker security because there is no SMMU that can be used to prevent
the NPU from accessing memory that it should not be accessing (e.g. that belonging to inferences from other processes). A malicious userspace process might be able to cause the NPU to read or write to such memory by sending a maliciously crafted command stream. If security is of concern, we recommend
that you use an SMMU.

## License

The Ethos-N driver stack is composed of multiple components, each with their own license. The components are:

* The Arm Ethos-N NPU driver, which is the collection of user-space libraries.
* The Arm NN backend, which interfaces the Ethos-N NPU driver to Arm NN.
* The Arm Ethos-N NPU kernel module, to be used with the Linux kernel.
* The Arm Ethos-N NPU firmware, which will be loaded by the kernel module onto the NPU.

### License for the Ethos-N NPU driver and Arm NN backend

The Arm Ethos-N NPU driver and the Arm NN backend are provided under the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license. For more information, see [LICENSE](driver/LICENSE) and [LICENSE](armnn-ethos-n-backend/LICENSE). Contributions to this project are accepted under the same license.

```less
Copyright 2018-2023 Arm Limited

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

Individual files contain the following tag instead of the full license text:

```less
SPDX-License-Identifier: Apache-2.0
```

This enables machine processing of license information based on the SPDX License Identifiers that are available here: <http://spdx.org/licenses/>.

### License for the Ethos-N NPU kernel module

The Arm Ethos-N NPU kernel module is provided under the [GPL v2 only](https://spdx.org/licenses/GPL-2.0-only.html) license.
For more information, see [LICENSE](kernel-module/LICENSE). Contributions to this project are accepted under the same license.

Individual files contain the following tag instead of the full license text:

```less
SPDX-License-Identifier: GPL-2.0-only
```

### EULA and TPIP for the Ethos-N NPU firmware

The Ethos-N NPU firmware binary is released under an [EULA](firmware/LES-PRE-21755.pdf).

The Ethos-N NPU firmware binary was compiled against the CMSIS library, which is released under the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license. For more information, see [apache-2.0.txt](firmware/tpip-licenses/apache-2.0.txt).


### Trademarks and copyrights

Arm and Ethos are registered trademarks or trademarks of Arm Limited (or its subsidiaries) in the US and/or elsewhere.

Android is a trademark of Google LLC.

Linux is the registered trademark of Linus Torvalds in the U.S. and other countries.
