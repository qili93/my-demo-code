#!/bin/bash
# set -ex
#######################################################################
# global variables, you can change them according to your requirements
#######################################################################
# absolute path current
function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}
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
# default with log
PADDLE_LITE_DIR=$(readlinkf ../../../libs/npu/${ANDROID_ABI}-log)
#######################################################################

#######################################################################
# pring usage
#######################################################################
function print_usage {
    set +x
    echo -e "\nUSAGE:"
    echo
    echo "----------------------------------------"
    echo -e "compile tiny publish so lib:"
    echo -e "   ./build.sh --arm_abi=<abi> --with_log=<ON|OFF> tiny_build"
    echo
    echo -e "compile full publish so lib:"
    echo -e "   ./build.sh --arm_abi=<abi> --with_log=<ON|OFF> full_build"
    echo
    echo -e "optional argument:"
    echo -e "--with_log: (OFF|ON); controls whether to print log information, default is ON"
    echo
    echo -e "argument choices:"
    echo -e "--arm_abi:\t armv8|armv7; default is armv7"
    echo
    echo -e "tasks:"
    echo -e "tiny_build: build with tiny publish library."
    echo -e "full_build: build with full publish library."
    echo "----------------------------------------"
    echo
}

#######################################################################
# compiling functions
#######################################################################
function make_build {
    local abi=$1
    local log=$2
    local pub=$3

    cur_dir=$(pwd)
    build_dir=$cur_dir/build
    rm -rf $build_dir
    mkdir -p $build_dir
    cd $build_dir

    if [ ${pub} == "full" ]; then
        USE_FULL_API=TRUE
    else
        USE_FULL_API=FALSE
    fi
    echo "Building api: USE_FULL_API=<${USE_FULL_API}>"

    if [ ${abi} == "armv8" ]; then
        ANDROID_ABI=arm64-v8a
        ANDROID_NATIVE_API_LEVEL=android-23
    else
        ANDROID_ABI=armeabi-v7a
        ANDROID_NATIVE_API_LEVEL=android-24
    fi
    echo "Building target: ANDROID_ABI=<${ANDROID_ABI}> ANDROID_NATIVE_API_LEVEL=<${ANDROID_NATIVE_API_LEVEL}>"

    if [ ${log} == "ON" ]; then
        PADDLE_LITE_DIR=$(readlinkf ../../../libs/npu/${ANDROID_ABI}-log)
    else
        PADDLE_LITE_DIR=$(readlinkf ../../../libs/npu/${ANDROID_ABI}-nolog)
    fi
    echo "Linking library: PADDLE_LITE_DIR=<${PADDLE_LITE_DIR}>"

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
        -DPADDLE_LITE_DIR="${PADDLE_LITE_DIR}" \
        ..
    make

    cd -
    echo "ls -l $build_dir"
    ls -l $build_dir
}
#######################################################################
# main functions
#######################################################################
function main {
    if [ -z "$1" ]; then
        print_usage
        exit -1
    fi

    # Parse command line.
    for i in "$@"; do
        case $i in
            --arm_abi=*)
                ARM_ABI="${i#*=}"
                shift
                ;;
            --with_log=*)
                WITH_LOG="${i#*=}"
                shift
                ;;
            tiny_build)
                make_build $ARM_ABI $WITH_LOG tiny
                shift
                ;;
            full_build)
                make_build $ARM_ABI $WITH_LOG full
                shift
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
    done
}

main $@