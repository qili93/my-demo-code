#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

# set ascend environment
export ASCEND_PATH=/usr/local/Ascend
export ASCEND_ATC_PATH=$ASCEND_PATH/atc
export ASCEND_OPP_PATH=$ASCEND_PATH/opp
export PATH=/usr/local/python3.7.5/bin:${ASCEND_ATC_PATH}/ccec_compiler/bin:${ASCEND_ATC_PATH}/bin:${ASCEND_PATH}/toolkit/tools/ide_daemon/bin:$PATH
export PYTHONPATH=${ASCEND_ATC_PATH}/python/site-packages/te:${ASCEND_ATC_PATH}/python/site-packages/topi:${ASCEND_ATC_PATH}/python/site-packages/auto_tune.egg:${ASCEND_ATC_PATH}/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${ASCEND_PATH}/acllib/lib64:${ASCEND_ATC_PATH}/lib64:${ASCEND_PATH}/toolkit/lib64:${ASCEND_PATH}/add-ons:${ASCEND_PATH}/opp/op_proto/built-in:$LD_LIBRARY_PATH
export SOC_VERSION=Ascend310

# set paddle-lite environment
BASE_REPO_PATH=/home/liqi27/Github-qili93/Paddle-Lite
PADDLE_LITE_DIR=$BASE_REPO_PATH/build.lite.x86/inference_lite_lib
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# run demo
export GLOG_v=5
./build/relu_demo
