#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

# change together with run_demo.sh
BUILD_DIR_NAME=build-v2.7-mkl-on-log-of-tailer

# paddle repo dir
# BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite)
# PADDLE_LITE_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/build.lite.x86/inference_lite_lib

# local sync lib dir
PADDLE_LITE_DIR=$(readlinkf ../../x86_lite_libs)

# USE_FULL_API=TRUE
USE_FULL_API=FALSE
#######################################
# Build commands, do not change them
#######################################
build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake .. -DWITH_MKL=ON \
      -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} \
      -DUSE_FULL_API=${USE_FULL_API} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Release

make

cd -
echo "ls -l $build_dir"
ls -l $build_dir
