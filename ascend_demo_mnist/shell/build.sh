#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################
ASCEND_PATH=/usr/local/Ascend

BASE_REPO_PATH=/workspace/Github-qili93/Paddle-Lite
PADDLE_LITE_DIR=$BASE_REPO_PATH/build.lite.huawei_ascend/inference_lite_lib

USE_FULL_API=TRUE

#######################################
# Build commands, do not change them
#######################################
build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} \
      -DASCEND_PATH=${ASCEND_PATH} \
      -DUSE_FULL_API=${USE_FULL_API} \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_SKIP_RPATH=TRUE \
      ..
make

cd -
echo "ls -l $build_dir"
ls -l $build_dir
