#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

# # ncnn repo dir
NCNN_REPO_PATH=$(readlinkf ../../../../ncnn/build-debug)
NCNN_INSTALL_DIR=${NCNN_REPO_PATH}/install

# local sync lib dir
# NCNN_INSTALL_DIR=$(readlinkf ../../inference_ncnn_lib)

#######################################
# Build commands, do not change them
#######################################

build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake .. \
      -DNCNN_INSTALL_DIR=${NCNN_INSTALL_DIR} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Debug

make

cd -
echo "ls -l $build_dir"
ls -l $build_dir