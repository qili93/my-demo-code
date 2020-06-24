#!/bin/bash
# set -ex
#######################################################################
# global variables, you can change them according to your requirements
#######################################################################
# andriod ndk path
ANDROID_NDK=/Users/liqi27/Library/android-ndk-r17c
# arm64-v8a or armeabi-v7a
# ANDROID_ABI=arm64-v8a
ANDROID_ABI=armeabi-v7a
# arm64-v8a => 23 or armeabi-v7a => 24
ANDROID_NATIVE_API_LEVEL=android-24 # armeabi-v7a
# ANDROID_NATIVE_API_LEVEL=android-23 # arm64-v8a 
# full api or tiny api
USE_FULL_API=TRUE
#######################################################################

#######################################################################
# compiling functions
#######################################################################
# absolute path current
function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

function main {
    cur_dir=$(pwd)

    build_dir=$cur_dir/build
    mkdir -p $build_dir
    cd $build_dir

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

    cd -
    echo "ls -l $build_dir"
    ls -l $build_dir
}
main $@