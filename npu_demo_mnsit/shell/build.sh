#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################
# ANDROID_NDK=/opt/android-ndk-r17c # For paddlelite docker
ANDROID_NDK=/Users/liqi27/Library/android-ndk-r17c # For macOS with ndk-bundle
USE_FULL_API=TRUE

ANDROID_ABI=armeabi-v7a # arm64-v8a
ANDROID_NATIVE_API_LEVEL=android-23
if [ $ANDROID_ABI == "armeabi-v7a" ]; then
    ANDROID_NATIVE_API_LEVEL=android-24
fi

PADDLE_LITE_DIR=$(readlinkf ../../npu_lite_libs/armeabi-v7a-log)

#######################################
# Build commands, do not change them
#######################################
build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} \
           -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
           -DANDROID_NDK=${ANDROID_NDK} \
           -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL} \
           -DANDROID_STL=c++_shared \
           -DANDROID_ABI=${ANDROID_ABI} \
           -DANDROID_ARM_NEON=TRUE \
           -DUSE_FULL_API=${USE_FULL_API} \
           -DCMAKE_VERBOSE_MAKEFILE=ON \
           -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
           ..
make

cd -
echo "ls -l $build_dir"
ls -l $build_dir