#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: ascend environements
#######################################
export ASCEND_HOME=/usr/local/Ascend
echo "export ASCEND_HOME=$ASCEND_HOME"

# export HUAWEI_ASCEND_NPU_DDK_ROOT=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux_gcc4.8.5
export HUAWEI_ASCEND_NPU_DDK_ROOT=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux_gcc7.3.0
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
BASE_REPO_PATH=/home/HwHiAiUser/Github-qili93/Paddle-Lite
PADDLE_LITE_DIR=$BASE_REPO_PATH/build.lite.huawei_ascend_npu/inference_lite_lib
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/cxx/lib:${PADDLE_LITE_DIR}/third_party/mklml/lib:$LD_LIBRARY_PATH

# PADDLE_LITE_DIR=/workspace/Paddle-Lite/build.lite.huawei_ascend_npu_test
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/third_party/install/mklml/lib:$LD_LIBRARY_PATH

# set model dir
MODEL_DIR=$(readlinkf ../assets/models)
echo "MODEL_DIR is $MODEL_DIR"
MODEL_NAME=mnist_model
rm -rf $MODEL_DIR/*.om
rm -rf $MODEL_DIR/*.cfg

# run demo
export GLOG_v=5
./build/mnist_demo $MODEL_DIR/$MODEL_NAME
