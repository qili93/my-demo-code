#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

# for armv8
arm_abi=armv8
arm_arch=arm64-v8a
# # for armv7
# arm_abi=armv7
# arm_arch=armeabi-v7a

# change together with run_demo.sh
BUILD_DIR_NAME=build.lite.android.${arm_abi}.gcc

# paddle repo dir
BASE_REPO_PATH=$(readlinkf ../../../Paddle-Lite)
PADDLE_LITE_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/inference_lite_lib.android.${arm_abi}

# Andriod NDK dir
ANDROID_NDK=~/Library/android-ndk-r17c
ANDROID_LIB_DIR=${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${arm_arch}

# define lib name
LITE_FULL_LIB_NAME="libpaddle_full_api_shared.so"
LITE_TINY_LIB_NAME="libpaddle_light_api_shared.so"
NDK_CPP_LIB_NAME="libc++_shared.so"


# paddle full lib
LITE_FULL_LIB=$PADDLE_LITE_DIR/cxx/lib/$LITE_FULL_LIB_NAME
LITE_TINY_LIB=$PADDLE_LITE_DIR/cxx/lib/$LITE_TINY_LIB_NAME
NDK_CPP_LIB=${ANDROID_LIB_DIR}/${NDK_CPP_LIB_NAME}

# paddld include dir
LITE_INC_DIR=$PADDLE_LITE_DIR/cxx/include

# target dirs
target_dir=$(pwd)
target_lib=$target_dir/lib/
target_inc=$target_dir/include

echo "---------------Prepare target dirs-----------------"
if [ -d "$target_lib" ]; then
  rm -rf "$target_lib"
  echo "$target_lib is deleted"
fi
if [ -d "$target_inc" ]; then
  rm -rf "$target_inc"
  echo "$target_inc is deleted"
fi
mkdir -p "$target_lib"
echo "$target_lib created"

echo "---------------COPY Paddle-Lite Full Libs-----------------"
echo "copy from == $LITE_FULL_LIB"
echo "copy to ==== $target_lib"
cp $LITE_FULL_LIB $target_lib

echo "---------------COPY Paddle-Lite Tiny Libs-----------------"
echo "copy from == $LITE_TINY_LIB"
echo "copy to ==== $target_lib"
cp $LITE_TINY_LIB $target_lib

echo "---------------COPY Paddle-Lite Headers-----------------"
echo "copy from == $LITE_INC_DIR"
echo "copy to ==== $target_inc"
cp -r $LITE_INC_DIR $target_dir

echo "---------------COPY SHARED C++ Libs-----------------"
echo "copy from == $NDK_CPP_LIB"
echo "copy to ==== $target_lib"
cp $NDK_CPP_LIB $target_lib

echo "---------------List Files-----------------"
tree ${target_dir}
