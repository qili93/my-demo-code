#!/bin/bash
# set -ex
#######################################################################
# global variables, you can change them according to your requirements
#######################################################################
# andriod ndk path
ANDROID_NDK=/Users/liqi27/Library/android-ndk-r17c
# arm64-v8a or armeabi-v7a
ANDROID_ABI=arm64-v8a
# arm64-v8a => 23 or armeabi-v7a => 24
ANDROID_NATIVE_API_LEVEL=android-23
# full api or tiny api
USE_FULL_API=TRUE
#######################################################################


#######################################################################
# function of print help information
#######################################################################
function print_usage {
    echo -e "\nUSAGE:"
    echo "./build <options> full | tiny"
    echo "-------------------------------------------------"
    echo -e "--sys=<os> mac or x86, default is mac"
    echo -e "--abi=<abi> v8, v7, default is v8"
    echo "-------------------------------------------------"
    echo
}

#######################################################################
# compiling functions
#######################################################################
# absolute path current
function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

function build_exe {
    cur_dir=$(pwd)

    if [ "${ARM_ABI}" == "v7" ]; then
        ANDROID_ABI=armeabi-v7a
        ANDROID_NATIVE_API_LEVEL=android-24
    fi
    if [ "${SYS_OS}" == "x86" ]; then
        ANDROID_NDK=/opt/android-ndk-r17c
    fi

    local publish_dir
    if [[ "${USE_FULL_API}" == "TRUE" ]]; then
        publish_dir="full"
    else
        publish_dir="tiny"
    fi

    build_dir=$cur_dir/build.arm${ARM_ABI}.${publish_dir}
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

#######################################################################
# main functions: choose compiling method according to input argument
#######################################################################
function main {
    # Parse command line.
    for i in "$@"; do
        case $i in
            --abi=*)
                ARM_ABI="${i#*=}"
                shift
                ;;
            --sys=*)
                SYS_OS="${i#*=}"
                shift
                ;;
            full)
                USE_FULL_API=TRUE
                build_exe
                exit 0
                ;;
            tiny)
                USE_FULL_API=FALSE
                build_exe
                exit 0
                ;;
            help)
            # print help info
                print_usage
                exit 0
                ;;
            *)
                # unknown option
                echo "Error: unsupported argument \"${i#*=}\""
                print_usage
                exit 1
                ;;
        esac
    done
}
main $@