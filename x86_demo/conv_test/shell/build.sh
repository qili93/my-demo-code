#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

# # paddle repo dir
BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite/build-v2.7-release)
# BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite/conv_fix_v27-release)
# BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite/conv_fix_v27-debuging)
PADDLE_LITE_DIR=${BASE_REPO_PATH}/build.lite.x86/inference_lite_lib

# local sync lib dir
# PADDLE_LITE_DIR=$(readlinkf ../../inference_lite_lib)

USE_FULL_API=TRUE
# USE_FULL_API=FALSE
USE_SHARED_API=TRUE
# USE_SHARED_API=FALSE
#######################################
# Build commands, do not change them
#######################################

build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake .. -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} \
      -DUSE_FULL_API=${USE_FULL_API} \
      -DUSE_SHARED_API=${USE_SHARED_API} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Debug

make

cd -
echo "ls -l $build_dir"
ls -l $build_dir
