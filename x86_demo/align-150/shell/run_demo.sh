#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# # paddle repo dir
# BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite/build-v2.7-release)
# BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite/conv_fix_v27-release)
# BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite/conv_fix_v27-debuging)
BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite/build-v2.8-ruliu-logon)
PADDLE_LITE_DIR=${BASE_REPO_PATH}/build.lite.x86/inference_lite_lib
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# local sync lib dir
# PADDLE_LITE_DIR=$(readlinkf ../../inference_lite_lib)
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/lib:$LD_LIBRARY_PATH

# run demo
export GLOG_v=0
./build/model_test
