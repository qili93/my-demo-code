#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# BASE_REPO_PATH=/workspace/Github-qili93/Paddle-Lite
# PADDLE_LITE_DIR=$BASE_REPO_PATH/log-off-build.lite.x86/inference_lite_lib
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

PADDLE_LITE_DIR=$(readlinkf ../../x86_lite_libs)
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/lib:$LD_LIBRARY_PATH

# set model dir
MODEL_DIR=$(readlinkf ../assets/models)
MODEL_TYPE=1 # 0 uncombined; 1 combined paddle fluid model

# MODEL_NAME=align150-fp32
# MODEL_NAME=angle-fp32
# MODEL_NAME=detect_rgb-fp16
# MODEL_NAME=detect_rgb-int8
# MODEL_NAME=eyes_position-fp32
# MODEL_NAME=iris_position-fp32
# MODEL_NAME=mouth_position-fp32
MODEL_NAME=seg-model-int8

# run demo
export GLOG_v=5
./build/human_seg_demo $MODEL_DIR/$MODEL_NAME $MODEL_TYPE