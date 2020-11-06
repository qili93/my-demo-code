#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# change together with run_demo.sh
BUILD_DIR_NAME=build-v2.7-debug

# # paddle repo dir
# BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite)
# PADDLE_LITE_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/build.lite.x86/inference_lite_lib
# # export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${LD_LIBRARY_PATH}
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# local sync lib dir
PADDLE_LITE_DIR=$(readlinkf ../../x86_lite_libs)
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/lib:$LD_LIBRARY_PATH

# set model dir
MODEL_DIR=$(readlinkf ../assets/models)

# MODEL_NAME=pc-seg-float-model
MODEL_NAME=pc-seg-float-model-dst

# run demo
export GLOG_v=5
./build/model_test $MODEL_DIR/$MODEL_NAME
