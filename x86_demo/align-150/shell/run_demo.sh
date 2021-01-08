#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# # paddle repo dir
BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite/build-dev-ruliu)
PADDLE_LITE_DIR=${BASE_REPO_PATH}/build.lite.x86/inference_lite_lib
PADDLE_LITE_DIR=/Users/liqi27/Documents/Github-qili93/Paddle-Lite/build-dev-ruliu/build.lite.x86/inference_lite_lib
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# local sync lib dir
# PADDLE_LITE_DIR=$(readlinkf ../../inference_lite_lib)
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/lib:$LD_LIBRARY_PATH

# run demo
export GLOG_v=5
./build/model_test
