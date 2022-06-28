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
# LIB_DIR=${work_path}/../../Paddle/build/paddle_inference_install_dir
# LIB_DIR=/workspace/Paddle/build_custom_cpu_infer/paddle_inference_install_dir
WITH_TESTING=OFF

# 4. Configure based on system arch
arch=$(uname -i)
if [[ $arch == x86_64* ]]; then
    WITH_ARM=OFF
elif  [[ $arch == aarch64* ]]; then
    WITH_ARM=ON
fi

# 5. CMake command
cmake .. \
  -DPython_EXECUTABLE=`which python3` \
  -DWITH_ARM=${WITH_ARM} \
  -DON_INFER=${ON_INFER} \
  -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_TESTING=${WITH_TESTING} \
  -DWITH_STATIC_LIB=OFF

# 6. Make command
if [ "$WITH_ARM" == "ON" ];then
  make TARGET=ARMV8 -j
else
  make -j
fi
