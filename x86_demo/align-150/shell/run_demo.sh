#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# paddle repo dir
# BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite)
# BUILD_DIR_NAME=build-v2.7-test
# PADDLE_LITE_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/build.lite.x86/inference_lite_lib
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# local sync lib dir
PADDLE_LITE_DIR=$(readlinkf ../../inference_lite_lib)
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/lib:$LD_LIBRARY_PATH

# set model dir
MODEL_DIR=$(readlinkf ../assets/models)
MODEL_NAME=align150-fp32

# run demo
export GLOG_v=5
./build/model_test $MODEL_DIR/$MODEL_NAME
