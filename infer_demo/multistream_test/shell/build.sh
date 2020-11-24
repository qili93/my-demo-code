#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

# paddle repo dir
BASE_REPO_PATH=$(readlinkf ../../../../Paddle)
BUILD_DIR_NAME=build_infer
PADDLE_INFER_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/paddle_inference_install_dir

# local lib dir
# PADDLE_INFER_DIR=$(readlinkf ../../paddle_inference_install_dir)

#######################################
# Build commands, do not change them
#######################################
build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake -DPADDLE_INFER_DIR=${PADDLE_INFER_DIR} \
      -DWITH_GPU=OFF \
      -DUSE_TENSORRT=OFF \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make


cd -
echo "ls -l $build_dir"
ls -l $build_dir
