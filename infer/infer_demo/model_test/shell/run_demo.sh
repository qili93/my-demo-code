#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# paddle repo dir
BASE_REPO_PATH=$(readlinkf ../../../../Paddle)
BUILD_DIR_NAME=build_infer
PADDLE_INFER_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/paddle_inference_install_dir

# local lib dir
# PADDLE_INFER_DIR=$(readlinkf ../../paddle_inference_install_dir)

export LD_LIBRARY_PATH=${PADDLE_INFER_DIR}/paddle/lib:${PADDLE_INFER_DIR}/third_party/install/mklml/lib:$LD_LIBRARY_PATH

# set model dir
MODEL_DIR=$(readlinkf ../assets/models)
MODEL_NAME=mobilenet_v1_fp32_224_fluid
# MODEL_NAME=mobilenet_v1_fp32_224_combined

# run demo
export GLOG_v=0
./build/model_test ${MODEL_DIR}/${MODEL_NAME}