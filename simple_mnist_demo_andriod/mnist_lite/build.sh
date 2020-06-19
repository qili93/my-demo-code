#!/bin/bash
# Change it to your NDK path
# ANDROID_NDK=/opt/android-ndk-r17c # For paddlelite docker
ANDROID_NDK=/Users/liqi27/Library/android-ndk-r17c # For macOS with ndk-bundle
USE_FULL_API=TRUE

ANDROID_ABI=arm64-v8a # for armv8
# ANDROID_ABI=armeabi-v7a # for armv7
if [ "x$1" != "x" ]; then
    ANDROID_ABI=$1
fi

ANDROID_NATIVE_API_LEVEL=android-23
if [ $ANDROID_ABI == "armeabi-v7a" ]; then
    ANDROID_NATIVE_API_LEVEL=android-24
fi

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

rm -rf build
mkdir build
cd build
cmake -DUSE_FULL_API=${USE_FULL_API} \
      -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
      -DANDROID_NDK=${ANDROID_NDK} \
      -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL} \
      -DANDROID_STL=c++_shared \
      -DANDROID_ABI=${ANDROID_ABI} \
      -DANDROID_ARM_NEON=TRUE \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make

# list build files
cd -
echo "ls -l build"
ls -l build