
#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

TARGET_ARCH_ABI=x86_64-linux_gcc7.3.0 # x86_64-linux_gcc7.3.0 or x86_64-linux_gcc4.8.5
if [ "x$1" != "x" ]; then
    TARGET_ARCH_ABI=$1
fi

#######################################
# Huawei Ascend NPU DDK Environments
#######################################
export HUAWEI_ASCEND_NPU_DDK_ROOT=/usr/local/Ascend/ascend-toolkit/latest/${TARGET_ARCH_ABI}
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
# Paddle-Lite Demo Run Scripts Settings
#######################################
PADDLE_LITE_DIR="$(readlinkf ../../libs/PaddleLite)"

MODEL_NAME=mobilenet_v1_fp32_224_fluid
LABEL_NAME=synset_words.txt
IMAGE_NAME=tabby_cat.raw

export GLOG_v=2
export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/${TARGET_ARCH_ABI}/lib:$LD_LIBRARY_PATH
./build/image_classification_demo ../assets/models/${MODEL_NAME} ../assets/labels/${LABEL_NAME} ../assets/images/${IMAGE_NAME}
