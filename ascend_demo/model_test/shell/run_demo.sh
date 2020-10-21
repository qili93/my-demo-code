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

# set model dir
MODEL_DIR=$(readlinkf ../assets/models)
MODEL_TYPE=0 # 0 uncombined; 1 combined paddle fluid model

MODEL_NAME=mobilenet_v1
MODEL_NAME=mobilenet_v2
# MODEL_NAME=mobilenet_v1_fp32_224_fluid
# MODEL_NAME=mobilenet_v2_fp32_224_fluid
# MODEL_NAME=resnet18_fp32_224_fluid
# MODEL_NAME=resnet50_fp32_224_fluid
# MODEL_NAME=mnasnet_fp32_224_fluid

# run demo
export GLOG_v=0
./build/model_test $MODEL_DIR $MODEL_NAME $MODEL_TYPE