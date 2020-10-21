#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

# change together with run_demo.sh
BUILD_DIR_NAME=build-v2.7-mkl-on-log-of-tailer

# paddle repo dir
BASE_REPO_PATH=$(readlinkf ../../../Paddle-Lite)
PADDLE_LITE_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/build.lite.x86/inference_lite_lib

# define lib name
if [[ "$OSTYPE" == "darwin"*  ]]; then
  LITE_FULL_LIB_NAME="libpaddle_full_api_shared.dylib"
  LITE_TINY_LIB_NAME="libpaddle_light_api_shared.dylib"
  LITE_IOMP5_LIB_NAME="libiomp5.dylib"
  LITE_MKLML_LIB_NAME="libmklml.dylib"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  LITE_FULL_LIB_NAME="libpaddle_full_api_shared.so"
  LITE_TINY_LIB_NAME="libpaddle_light_api_shared.so"
  LITE_IOMP5_LIB_NAME="libiomp5.so"
  LITE_MKLML_LIB_NAME="libmklml*.so"
fi

# paddle full lib
LITE_FULL_LIB=$PADDLE_LITE_DIR/cxx/lib/$LITE_FULL_LIB_NAME
LITE_TINY_LIB=$PADDLE_LITE_DIR/cxx/lib/$LITE_TINY_LIB_NAME

# paddld include dir
LITE_INC_DIR=$PADDLE_LITE_DIR/cxx/include

# MKLML Lib
LITE_IOMP5_LIB=$PADDLE_LITE_DIR/third_party/mklml/lib/$LITE_IOMP5_LIB_NAME
LITE_MKLML_LIB=$PADDLE_LITE_DIR/third_party/mklml/lib/$LITE_MKLML_LIB_NAME

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

echo "---------------COPY MKLML Libs-----------------"
echo "copy from == $LITE_IOMP5_LIB"
echo "copy from == $LITE_MKLML_LIB"
echo "copy to ==== $target_lib"
cp $LITE_IOMP5_LIB $target_lib
cp $LITE_MKLML_LIB $target_lib


echo "---------------List Files-----------------"
tree ${target_dir}