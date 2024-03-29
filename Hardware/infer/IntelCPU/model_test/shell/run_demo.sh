#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# paddle repo dir
# BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite)
#BUILD_DIR_NAME=build-v2.7-relse-static-openmp
#PADDLE_LITE_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/build.lite.x86/inference_lite_lib
#export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# local sync lib dir
PADDLE_LITE_DIR=$(readlinkf ../../x86_lite_libs)
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/lib:$LD_LIBRARY_PATH

# set model dir
ASSETS_DIR=$(readlinkf ../assets/models)

# MODEL_NAME=align150-fp32
# MODEL_NAME=angle-fp32
# MODEL_NAME=detect_rgb-fp32
# MODEL_NAME=face_detect_fp32
# MODEL_NAME=eyes_position-fp32
# MODEL_NAME=iris_position-fp32
# MODEL_NAME=mouth_position-fp32
MODEL_NAME=pc-seg-float-model
# MODEL_NAME=pc-seg-float-model-new
# MODEL_NAME=pc_seg_float_x86_1016

# run demo
export GLOG_v=5
./build/model_test $ASSETS_DIR/$MODEL_NAME
