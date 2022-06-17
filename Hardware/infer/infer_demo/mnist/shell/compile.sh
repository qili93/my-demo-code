#!/bin/bash
set +x
set -e

work_path=${PWD}

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

DEMO_NAME=mnist_test

WITH_MKL=ON
WITH_ONNXRUNTIME=OFF
WITH_ARM=OFF
WITH_MIPS=OFF
WITH_SW=OFF

# LIB_DIR=$(readlinkf ../../paddle_inference)
LIB_DIR=/workspace/Paddle/build_custom_cpu_infer/paddle_inference_install_dir

#######################################
# Build commands, do not change them
#######################################

build_dir=$work_path/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_MKL=${WITH_MKL} \
  -DWITH_ONNXRUNTIME=${WITH_ONNXRUNTIME} \
  -DWITH_ARM=${WITH_ARM} \
  -DWITH_MIPS=${WITH_MIPS} \
  -DWITH_SW=${WITH_SW} \
  -DWITH_STATIC_LIB=OFF

if [ "$WITH_ARM" == "ON" ];then
  make TARGET=ARMV8 -j
else
  make -j
fi

cd -
echo "ls -l $build_dir"
ls -l $build_dir
