#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# set paddle-lite environment
BASE_REPO_PATH=/workspace/Github-qili93/Paddle/build-infer-mkl
PADDLE_LIB_DIR=${BASE_REPO_PATH}/paddle_inference_install_dir/
export LD_LIBRARY_PATH=${PADDLE_LIB_DIR}/cxx/lib:${PADDLE_LIB_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# set model dir
MODEL_DIR=$(readlinkf ../assets/models)
# MODEL_NAME=PC-quant-seg-model
MODEL_NAME=mobilenet_v1

# run demo
export GLOG_v=0
./build/human_seg_demo ${MODEL_DIR}/${MODEL_NAME}