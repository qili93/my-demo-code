#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

ASCEND_PATH=/usr/local/Ascend
BASE_REPO_PATH=/home/liqi27/Github-qili93/Paddle-Lite
#PADDLE_LITE_DIR="$(readlinkf ../build.lite.x86/inference_lite_lib)"
#PADDLE_LITE_DIR="$(readlinkf ../build.lite.ascend/inference_lite_lib.ascend)"
PADDLE_LITE_DIR=$BASE_REPO_PATH/build.lite.ascend/inference_lite_lib.ascend

build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} \
      -DASCEND_PATH=${ASCEND_PATH} \
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
