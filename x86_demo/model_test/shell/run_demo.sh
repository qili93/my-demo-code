#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# change together with build.sh
BUILD_DIR_NAME=build-v2.7-mkl-on-log-of-tailer

# paddle repo dir
BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite)
PADDLE_LITE_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/build.lite.x86/inference_lite_lib
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# local sync lib dir
# PADDLE_LITE_DIR=$(readlinkf ../../x86_lite_libs)
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/lib:$LD_LIBRARY_PATH

# set model dir
MODEL_DIR=$(readlinkf ../assets/models)
MODEL_TYPE=1 # 0 uncombined; 1 combined paddle fluid model

# MODEL_NAME=align150-fp32
# MODEL_NAME=angle-fp32
# MODEL_NAME=detect_rgb-fp32
# MODEL_NAME=eyes_position-fp32
# MODEL_NAME=iris_position-fp32
# MODEL_NAME=mouth_position-fp32
MODEL_NAME=pc-seg-float-model

# run demo
export GLOG_v=0
./build/model_test $MODEL_DIR $MODEL_NAME $MODEL_TYPE
