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

# set model dir
MODEL_DIR=$(readlinkf ./models)
MODEL_NAME=ebf3a8a1981c0c22588ee92c02b38971

# Run mode. 0(default): model => framework model; 1: framework model => json; 3: only pre-check; 5: pbtxt => json
atc --mode=1 --om=$MODEL_DIR/$MODEL_NAME.om  --json=$MODEL_DIR/$MODEL_NAME.json