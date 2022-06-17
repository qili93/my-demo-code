#!/bin/bash
set +x
set -e

work_path=${PWD}

# 2. Prepare build directory
build_dir=$work_path/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

# 3. Configure options
ON_INFER=OFF # if ON, please set LIB_DIR to Paddle Inference C++ Lib
LIB_DIR=/workspace/Paddle/build_custom_cpu_infer/paddle_inference_install_dir

# 4. CMake command
cmake .. \
  -DPython_EXECUTABLE=`which python3` \
  -DON_INFER=${ON_INFER} \
  -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_STATIC_LIB=OFF

# 5. Make command
make -j
