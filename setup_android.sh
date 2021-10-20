#!/bin/sh

# Script to use when setting up and doing the needed changed to add Ethos-N drivers to a Android build
# Please change <YOUR_KERNEL_SOURCE_FOLDER> and <YOUR_DEVICE_BOARD_FOLDER> to match your setup
# The script is intended be executed from vendor/arm/ethos-n-driver-stack/ and requires that 
# the Android NN driver has already been installed in vendor/arm/android-nn-driver/

# 1. Create symbolic link from Arm NN backends to build Ethos-N backend

ln -s ../../../../ethos-n-driver-stack/armnn-ethos-n-backend ../android-nn-driver/armnn/src/backends/ethos-n

# 2. Build Ethos-N kernel module

cd kernel-module
    make -C <YOUR_KERNEL_SOURCE_FOLDER> ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- defconfig
    make -C <YOUR_KERNEL_SOURCE_FOLDER> ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules_prepare
    make -C <YOUR_KERNEL_SOURCE_FOLDER> ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- M=$PWD modules
cd ..

# 3. Make sure Ethos-N kernel module end up in vendor kernel module folder

# The idea here is to make ethosn.ko to be placed in the vendor/lib/modules folder
# when creating your vendor image.

cp kernel-module/ethosn.ko <YOUR_DEVICE_BOARD_FOLDER>/5.10/
cp kernel-module/ethosn.ko <YOUR_KERNEL_SOURCE_FOLDER>/../out/android12-5.10/common/dist/
cp kernel-module/ethosn.ko <YOUR_KERNEL_SOURCE_FOLDER>/../out/android12-5.10/dist/

# 4. Add firmware and kernel module to PRODUCT_PACKAGES to make sure it is part of vendor file system image
# e.g. you need to add this
#     PRODUCT_PACKAGES += ethosn.ko ethosn.bin
# to your device.mk
# For juno.r2 boards you can use this line:
#     echo 'PRODUCT_PACKAGES += ethosn.ko ethosn.bin' >> ../../../device/linaro/juno/device.mk

echo 'PRODUCT_PACKAGES += ethosn.ko ethosn.bin' >> <YOUR_DEVICE_BOARD_FOLDER>/device.mk