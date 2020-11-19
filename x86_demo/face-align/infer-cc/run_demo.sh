#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# paddle repo dir
# BASE_REPO_PATH=$(readlinkf ../../../../Paddle)
# BUILD_DIR_NAME=build.infer.debug
# PADDLE_LIB_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/paddle_inference_install_dir

# local lib dir
PADDLE_LIB_DIR=$(readlinkf ../../x86_lite_libs/fluid_inference/fluid_inference_install_dir)

export LD_LIBRARY_PATH=${PADDLE_LIB_DIR}/paddle/lib:${PADDLE_LIB_DIR}/third_party/install/mklml/lib:$LD_LIBRARY_PATH

# set model dir
MODEL_DIR=$(readlinkf ../assets)
MODEL_TYPE=1 # 0 uncombined; 1 combined paddle fluid model

# MODEL_NAME=align150-fp32
# MODEL_NAME=angle-fp32
# MODEL_NAME=detect_rgb-fp32
# MODEL_NAME=eyes_position-fp32
# MODEL_NAME=iris_position-fp32
# MODEL_NAME=mouth_position-fp32

MODEL_NAME=face_detect_fp32
IMAGE_NAME=face.raw

# run demo
export GLOG_v=0
./build/model_test ${MODEL_DIR} ${MODEL_NAME} ${IMAGE_NAME}