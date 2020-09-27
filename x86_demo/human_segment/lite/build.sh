#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

# BASE_REPO_PATH=/Users/liqi27/Documents/Github-qili93/Paddle-Lite
# PADDLE_LITE_DIR=$BASE_REPO_PATH/build.lite.x86/inference_lite_lib

BASE_REPO_PATH=/workspace/Github-qili93/Paddle-Lite
PADDLE_LITE_DIR=$BASE_REPO_PATH/build.lite.x86/inference_lite_lib

USE_FULL_API=TRUE # x86 is always TRUE to use full api
# USE_FULL_API=FALSE
#######################################
# Build commands, do not change them
#######################################
build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} \
      -DUSE_FULL_API=${USE_FULL_API} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make

cd -
echo "ls -l $build_dir"
ls -l $build_dir
