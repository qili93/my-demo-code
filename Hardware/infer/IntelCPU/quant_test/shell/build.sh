#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

# paddle repo dir => NOTE: change together with run_demo.sh
BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite)
BUILD_DIR_NAME=build-dev-debug
PADDLE_LITE_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/build.lite.x86/inference_lite_lib

# local sync lib dir => NOTE: change together with CMakeLists.txt
# PADDLE_LITE_DIR=$(readlinkf ../../x86_lite_libs)

USE_FULL_API=TRUE
# USE_FULL_API=FALSE
#######################################
# Build commands, do not change themq
#######################################
build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake .. \
      -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} \
      -DUSE_FULL_API=${USE_FULL_API} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Debug

make

cd -
echo "ls -l $build_dir"
ls -l $build_dir