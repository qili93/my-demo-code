#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

# paddle repo dir
# BASE_REPO_PATH=$(readlinkf ../../../../Paddle)
# BUILD_DIR_NAME=build.infer.debug
# PADDLE_LIB_DIR=${BASE_REPO_PATH}/${BUILD_DIR_NAME}/paddle_inference_install_dir

# local lib dir
PADDLE_LIB_DIR=$(readlinkf ../../x86_lite_libs/fluid_inference/fluid_inference_install_dir)

#######################################
# Build commands, do not change them
#######################################
build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

# mkl is disabled on MAC
if [[ "$OSTYPE" == "darwin"*  ]]; then
  WITH_MKL=OFF
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  WITH_MKL=ON
fi

cmake -DPADDLE_LIB=${PADDLE_LIB_DIR} \
      -DWITH_MKL=${WITH_MKL} \
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
