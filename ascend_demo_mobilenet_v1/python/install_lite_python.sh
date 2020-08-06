#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################
export HUAWEI_ASCEND_NPU_DDK_ROOT=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux_gcc4.8.5
echo "export HUAWEI_ASCEND_NPU_DDK_ROOT=$HUAWEI_ASCEND_NPU_DDK_ROOT"

BASE_REPO_PATH=/workspace/Github-qili93/Paddle-Lite
PADDLE_LITE_DIR=$BASE_REPO_PATH/build.lite.huawei_ascend_npu/inference_lite_lib/python/install/dist

#######################################
# Install commands, do not change them
#######################################
pip3.7.5 install -U $PADDLE_LITE_DIR/paddlelite-*.whl
pip3.7.5 list | grep paddlelite
