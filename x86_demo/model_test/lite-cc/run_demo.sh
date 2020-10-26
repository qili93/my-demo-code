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

# paddle repo dir
BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite)
PADDLE_LITE_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/build.lite.x86/inference_lite_lib
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# local sync lib dir
# PADDLE_LITE_DIR=$(readlinkf ../../x86_lite_libs)
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/lib:$LD_LIBRARY_PATH

# set model dir
ASSETS_DIR=$(readlinkf ../assets)
MODEL_TYPE=1 # 0 uncombined; 1 combined paddle fluid model

# MODEL_NAME=align150-fp32
# MODEL_NAME=angle-fp32
# MODEL_NAME=detect_rgb-fp32
# MODEL_NAME=eyes_position-fp32
# MODEL_NAME=iris_position-fp32
# MODEL_NAME=mouth_position-fp32
# MODEL_NAME=pc-seg-float-model

MODEL_NAME=face_detect_fp32
IMAGE_NAME=face.raw

# run demo
export GLOG_v=0
./build/model_test $ASSETS_DIR $MODEL_NAME $IMAGE_NAME
