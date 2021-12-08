#!/bin/sh

# Script to use when setting up and doing the needed changed to add Ethos-N drivers to a Android build
# Please change <YOUR_KERNEL_SOURCE_FOLDER>, <YOUR_KERNEL_BUILD_FLAGS> and <YOUR_DEVICE_BOARD_FOLDER>
# to match your setup.
# The script is intended be executed from vendor/arm/ethos-n-driver-stack/ and requires that 
# the Android NN driver has already been installed in vendor/arm/android-nn-driver/ according to it's documentation.

# 1. Create symbolic link from Arm NN backends to build Ethos-N backend

ln -s ../../../../ethos-n-driver-stack/armnn-ethos-n-backend ../android-nn-driver/armnn/src/backends/ethos-n

# 2. Build Ethos-N kernel module

# This idea here is to build the kernel module, you might have some other ways to solve this in your setup.
# Please edit or uncomment the lines below depending on your setup.
#
# <YOUR_KERNEL_SOURCE_FOLDER> is the folder where you keep the gki_kernel source code
# <YOUR_KERNEL_BUILD_FLAGS> could be something like O=../out/android12-5.10/common EXTRA_CCFLAGS="-DDEBUG" ARCH=arm64 LLVM=1 LLVM_IAS=1 CROSS_COMPILE=aarch64-linux-gnu- DEPMOD=depmod DTC=dtc

cd kernel-module
    # Depending on what you do in other build steps you might need to uncomment this two lines
    #make -C <YOUR_KERNEL_SOURCE_FOLDER> <YOUR_KERNEL_BUILD_FLAGS> defconfig
    #make -C <YOUR_KERNEL_SOURCE_FOLDER> <YOUR_KERNEL_BUILD_FLAGS> modules_prepare
    make -C <YOUR_KERNEL_SOURCE_FOLDER> <YOUR_KERNEL_BUILD_FLAGS> M=$PWD modules
cd ..

# 3. Make sure Ethos-N kernel module end up in vendor kernel module folder

# The idea here is to make ethosn.ko to be placed in the vendor/lib/modules folder
# when creating your vendor image. You might have some other ways to solve this in your setup.
# Please edit the lines below depending on your setup.

cp kernel-module/ethosn.ko <YOUR_DEVICE_BOARD_FOLDER>/5.10/
cp kernel-module/ethosn.ko <YOUR_KERNEL_SOURCE_FOLDER>/../out/android12-5.10/common/dist/
cp kernel-module/ethosn.ko <YOUR_KERNEL_SOURCE_FOLDER>/../out/android12-5.10/dist/

# 4. Make sure Ethos-N dts files get merged with your dts file

# The idea here is to make the device tree contain Ethos-N.
# You might have some other ways to solve this in your setup.
# Please edit the lines below depending on your setup.

mkdir -p <YOUR_KERNEL_SOURCE_FOLDER>/arch/arm64/boot/dts/arm/  # If folder dont exist, create it
cp kernel-module/*.dts <YOUR_KERNEL_SOURCE_FOLDER>/arch/arm64/boot/dts/arm/
cp kernel-module/*.dtsi <YOUR_KERNEL_SOURCE_FOLDER>/arch/arm64/boot/dts/arm/
echo 'dtb-$(CONFIG_ARCH_VEXPRESS) += juno-r2-ethosn.dts' >> <YOUR_KERNEL_SOURCE_FOLDER>/arch/arm64/boot/dts/arm/Makefile

# 5. Add firmware and kernel module to PRODUCT_PACKAGES to make sure it is part of vendor file system image

# e.g. you need to add this
#     PRODUCT_PACKAGES += ethosn.ko ethosn.bin
# to your device.mk
# For juno.r2 boards you can use this line:
#     echo 'PRODUCT_PACKAGES += ethosn.ko ethosn.bin' >> ../../../device/linaro/juno/device.mk

echo 'PRODUCT_PACKAGES += ethosn.ko ethosn.bin' >> <YOUR_DEVICE_BOARD_FOLDER>/device.mk
