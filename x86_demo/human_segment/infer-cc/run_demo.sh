#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# set paddle-lite environment
# BASE_REPO_PATH=/workspace/Github-qili93/Paddle/build-infer-mkl
# PADDLE_LIB_DIR=${BASE_REPO_PATH}/paddle_inference_install_dir/

PADDLE_LIB_DIR=$(readlinkf ../../fluid_inference/fluid_inference_install_dir)
export LD_LIBRARY_PATH=${PADDLE_LIB_DIR}/paddle/lib:${PADDLE_LIB_DIR}/third_party/install/mklml/lib:$LD_LIBRARY_PATH

# set model dir
MODEL_DIR=$(readlinkf ../assets/models)
MODEL_TYPE=1 # 0 uncombined; 1 combined paddle fluid model

# MODEL_NAME=align150-fp32
# MODEL_NAME=angle-fp32
# MODEL_NAME=detect_rgb-fp32
# MODEL_NAME=detect_rgb-int8
# MODEL_NAME=eyes_position-fp32
# MODEL_NAME=iris_position-fp32
# MODEL_NAME=mouth_position-fp32
MODEL_NAME=seg-model-int8

# run demo
export GLOG_v=0
./build/human_seg_demo ${MODEL_DIR}/${MODEL_NAME} ${MODEL_TYPE}