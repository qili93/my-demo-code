#!/bin/bash
# Change it to your NDK path
ANDROID_NDK=/opt/android-ndk-r17c # For paddlelite docker
USE_FULL_API=TRUE

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

rm -rf build-x86
mkdir build-x86
cd build-x86
cmake -DCMAKE_VERBOSE_MAKEFILE=ON \
            -DLIB_DIR_PATH="$(readlinkf ../../x86)" \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DCMAKE_BUILD_TYPE=Release \
            ..
make