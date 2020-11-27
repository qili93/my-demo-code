#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# # paddle repo dir
BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite/build-v2.7-debuging)
PADDLE_LITE_DIR=${BASE_REPO_PATH}/build.lite.x86/inference_lite_lib
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# local sync lib dir
# PADDLE_LITE_DIR=$(readlinkf ../../x86_lite_libs)
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/lib:$LD_LIBRARY_PATH

# set model dir
MODEL_DIR=$(readlinkf ../train)
MODEL_NAME=model_group1
# MODEL_NAME=model_group3

# run demo
export GLOG_v=0
./build/model_test $MODEL_DIR/$MODEL_NAME
