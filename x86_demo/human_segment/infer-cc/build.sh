#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

PADDLE_LIB_DIR=$(readlinkf ../../fluid_inference)

#######################################
# Build commands, do not change them
#######################################
build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

# cmake .. -DPADDLE_LIB=${PADDLE_LIB_DIR} \
#   -DWITH_MKL=ONN \
#   -DWITH_GPU=OFF \
#   -DWITH_STATIC_LIB=OFF \
#   -DUSE_TENSORRT=OFF \
#   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
#   -DCMAKE_BUILD_TYPE=Debug

# make -j


cmake -DPADDLE_LIB=${PADDLE_LIB_DIR} \
      -DWITH_MKL=ONN \
      -DWITH_GPU=OFF \
      -DWITH_STATIC_LIB=OFF \
      -DUSE_TENSORRT=OFF \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      ..
make


cd -
echo "ls -l $build_dir"
ls -l $build_dir
