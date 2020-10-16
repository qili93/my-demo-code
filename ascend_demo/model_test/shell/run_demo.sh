#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite)
PADDLE_LITE_DIR=$BASE_REPO_PATH/build.lite.huawei_ascend_npu/inference_lite_lib
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# for local lib
# PADDLE_LITE_DIR=$(readlinkf ../../x86_lite_libs)
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/lib:$LD_LIBRARY_PATH

# set model dir
MODEL_DIR=$(readlinkf ../assets)
MODEL_TYPE=1 # 0 uncombined; 1 combined paddle fluid model
MODEL_NAME=yolov3

# run demo
export GLOG_v=0
./build/model_test $MODEL_DIR $MODEL_NAME $MODEL_TYPE