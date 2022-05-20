#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

# # paddle repo dir
BASE_REPO_PATH=$(readlinkf ../../../../Paddle-Lite)
PADDLE_LITE_DIR=${BASE_REPO_PATH}/build.lite.x86/inference_lite_lib

# local sync lib dir
# PADDLE_LITE_DIR=$(readlinkf ../../inference_lite_lib)

USE_FULL_API=TRUE
# USE_FULL_API=FALSE
#######################################
# Build commands, do not change them
#######################################
build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

# export LDFLAGS="-L/usr/local/opt/opencv@2/lib"
# export CPPFLAGS="-I/usr/local/opt/opencv@2/include"
# export PKG_CONFIG_PATH="/usr/local/opt/opencv@2/lib/pkgconfig"

cmake .. -DWITH_MKL=ON -DWITH_OPENCV=OFF \
      -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} \
      -DUSE_FULL_API=${USE_FULL_API} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Debug

make

cd -
echo "ls -l $build_dir"
ls -l $build_dir
