#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

# setting atc tool env
export HUAWEI_ASCEND_NPU_DDK_ROOT=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux_gcc4.8.5
echo "export HUAWEI_ASCEND_NPU_DDK_ROOT=$HUAWEI_ASCEND_NPU_DDK_ROOT"

# export PATH=/usr/local/python3.7.5/bin:$PATH
# export PATH=${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/ccec_compiler/bin:$PATH
# export PATH=${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/bin:$PATH
# echo "export PATH=$PATH"

export PYTHONPATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/atc/python/site-packages/te:$PYTHONPATH
export PYTHONPATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/atc/python/site-packages/topi:$PYTHONPATH
export PYTHONPATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/atc/python/site-packages/auto_tune.egg/auto_tune:$PYTHONPATH
export PYTHONPATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
echo "export PYTHONPATH=$PYTHONPATH"

export LD_LIBRARY_PATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/atc/lib64:$LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

export ASCEND_OPP_PATH=$HUAWEI_ASCEND_NPU_DDK_ROOT/opp
echo "export ASCEND_OPP_PATH=$ASCEND_OPP_PATH"

export SOC_VERSION=Ascend310
echo "export SOC_VERSION=$SOC_VERSION"

cd out
./ir_build Ascend310