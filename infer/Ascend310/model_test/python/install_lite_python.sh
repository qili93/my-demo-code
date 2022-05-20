#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

# BASE_REPO_PATH=/workspace/Github-qili93/Paddle-Lite
BASE_REPO_PATH=/workspace/temp_repo/Paddle-Lite
PADDLE_LITE_DIR=$BASE_REPO_PATH/build.lite.huawei_ascend_npu/inference_lite_lib/python/install/dist

#######################################
# Install commands, do not change them
#######################################
pip3.7.5 install -U $PADDLE_LITE_DIR/paddlelite*.whl
pip3.7.5 list | grep paddlelite
