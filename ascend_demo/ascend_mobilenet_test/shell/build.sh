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

# BASE_REPO_PATH=/workspace/Github-qili93/Paddle-Lite
BASE_REPO_PATH=/workspace/temp_repo/Paddle-Lite
PADDLE_LITE_DIR=$BASE_REPO_PATH/build.lite.huawei_ascend_npu/inference_lite_lib

USE_FULL_API=TRUE

#######################################
# Build commands, do not change them
#######################################
build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} \
      -DHUAWEI_ASCEND_NPU_DDK_ROOT=${HUAWEI_ASCEND_NPU_DDK_ROOT} \
      -DUSE_FULL_API=${USE_FULL_API} \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_SKIP_RPATH=TRUE \
      ..
make

cd -
echo "ls -l $build_dir"
ls -l $build_dir
