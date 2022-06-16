#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################

# # # ncnn repo dir
MNN_LIB_DIR=$(readlinkf ../../../../MNN/build-debug)
MNN_INC_DIR=$(readlinkf ../../../../MNN/include)

# local sync lib dir
# MNN_LIB_DIR=$(readlinkf ../../inference_mnn_lib/lib)
# MNN_INC_DIR=$(readlinkf ../../inference_mnn_lib/include)

#######################################
# Build commands, do not change them
#######################################

build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake .. \
      -DMNN_LIB_DIR=${MNN_LIB_DIR} \
      -DMNN_INC_DIR=${MNN_INC_DIR} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Debug

make

cd -
echo "ls -l $build_dir"
ls -l $build_dir