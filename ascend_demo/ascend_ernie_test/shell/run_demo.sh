#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: ascend environements
#######################################

export HUAWEI_ASCEND_NPU_DDK_ROOT=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux_gcc4.8.5
echo "export HUAWEI_ASCEND_NPU_DDK_ROOT=$HUAWEI_ASCEND_NPU_DDK_ROOT"

export PATH=/usr/local/python3.7.5/bin:$PATH
export PATH=${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/ccec_compiler/bin:$PATH
export PATH=${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/bin:$PATH
echo "export PATH=$PATH"

export PYTHONPATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/atc/python/site-packages/te:$PYTHONPATH
export PYTHONPATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/atc/python/site-packages/topi:$PYTHONPATH
export PYTHONPATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/atc/python/site-packages/auto_tune.egg/auto_tune:$PYTHONPATH
export PYTHONPATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
echo "export PYTHONPATH=$PYTHONPATH"

export LD_LIBRARY_PATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/acllib/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/atc/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/opp/op_proto/built-in:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/toolkit/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/driver/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

export ASCEND_OPP_PATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/opp
echo "export ASCEND_OPP_PATH=$ASCEND_OPP_PATH"

export SOC_VERSION=Ascend310
echo "export SOC_VERSION=$SOC_VERSION"

#######################################
# Local Settings: paddle-lite envs
#######################################

# set paddle-lite environment
BASE_REPO_PATH=/workspace/Github-qili93/Paddle-Lite
# BASE_REPO_PATH=/workspace/temp_repo/Paddle-Lite
PADDLE_LITE_DIR=$BASE_REPO_PATH/build.lite.huawei_ascend_npu/inference_lite_lib
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# BASE_REPO_PATH=/workspace/Github-qili93/Paddle-Lite
# PADDLE_LITE_DIR=$BASE_REPO_PATH/build.lite.huawei_ascend_npu_test
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/third_party/install/mklml/lib:$LD_LIBRARY_PATH

# set model dir
MODEL_DIR=$(readlinkf ../assets/models)
MODEL_NAME=ernie_fp32
MODEL_TYPE=0 # 0 uncombined; 1 combined paddle fluid model

# run demo
export GLOG_v=5
./build/ernie_test $MODEL_DIR/$MODEL_NAME $MODEL_TYPE
