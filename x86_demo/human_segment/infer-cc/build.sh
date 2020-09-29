#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

BASE_REPO_PATH=/workspace/Github-qili93/Paddle
PADDLE_LIB_DIR=${BASE_REPO_PATH}/build-infer/fluid_inference_install_dir

#######################################
# Build commands, do not change them
#######################################
build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake .. -DPADDLE_LIB=${PADDLE_LIB_DIR} \
  -DWITH_MKL=ONN \
  -DWITH_GPU=OFF \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  
  # -DCUDNN_LIB=${CUDNN_LIB} \
  # -DCUDA_LIB=${CUDA_LIB} \
  # -DTENSORRT_ROOT=${TENSORRT_ROOT}

make -j

cd -
echo "ls -l $build_dir"
ls -l $build_dir
