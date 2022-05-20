#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite envs
#######################################

# # # ncnn repo dir
# MNN_REPO_PATH=$(readlinkf ../../../../mnn/build-debug)
# MNN_INSTALL_DIR=${MNN_REPO_PATH}/install

# local sync lib dir
MNN_INSTALL_DIR=$(readlinkf ../../inference_mnn_lib)
export LD_LIBRARY_PATH=${MNN_INSTALL_DIR}/lib:$LD_LIBRARY_PATH

# run demo
export GLOG_v=5
./build/model_test
