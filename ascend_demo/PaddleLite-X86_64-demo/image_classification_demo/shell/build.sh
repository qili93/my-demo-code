#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

TARGET_ARCH_ABI=x86_64-linux_gcc7.3.0 # x86_64-linux_gcc7.3.0 or x86_64-linux_gcc4.8.5
if [ "x$1" != "x" ]; then
    TARGET_ARCH_ABI=$1
fi

HUAWEI_ASCEND_NPU_DDK_ROOT=/usr/local/Ascend/ascend-toolkit/latest/${TARGET_ARCH_ABI}
PADDLE_LITE_DIR="$(readlinkf ../../libs/PaddleLite)"

rm -rf build
mkdir build
cd build
# export CXX=g++ # Huawei Ascend NPU need g++ both on CentOS and Ubuntu
# export LD_LIBRARY_PATH=${PADDLE_LITE_DIR}/${TARGET_ARCH_ABI}/lib:$LD_LIBRARY_PATH

cmake -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} \
      -DTARGET_ARCH_ABI=${TARGET_ARCH_ABI} \
      -DHUAWEI_ASCEND_NPU_DDK_ROOT=${HUAWEI_ASCEND_NPU_DDK_ROOT} \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_SKIP_RPATH=TRUE \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..
make

cd -
