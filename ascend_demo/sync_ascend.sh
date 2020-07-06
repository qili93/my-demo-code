#!/bin/bash

# paddle repo dir
# base_repo_dir=/workspace/Github-qili93/Paddle-Lite
base_repo_dir=/home/liqi27/Paddle-Lite

# cd $base_repo_dir
# # build full
# ./lite/tools/build_ascend.sh build
# cd -

# paddle full lib
base_full_dir=$base_repo_dir/build.lite.ascend/inference_lite_lib.ascend
base_full_lib=$base_full_dir/cxx/lib
base_full_include=$base_full_dir/cxx/include
echo "base_full_dir=$base_full_dir"

# paddle tiny lib
# base_tiny_dir=$base_repo_dir/build.lite.android.armv8.gcc_tiny_log/inference_lite_lib.android.armv8.npu
# base_tiny_lib=$base_tiny_dir/cxx/lib
# base_tiny_include=$base_tiny_dir/cxx/include
# echo "base_tiny_dir=$base_tiny_dir"

# target
cur_dir=$(pwd)
target_dir=$cur_dir/libs
target_lib=$target_dir/lib
target_inc=$target_dir/include
echo "target_dir=$target_dir"

# delete target_arch dir
if [ -d "$target_dir" ]; then
  rm -rf "$target_dir"
  echo "$target_dir is deleted"
fi

# create target_arch/lib dir
mkdir -p "$target_lib"
echo "$target_lib created"s

# copy full so
full_so_name=libpaddle_full_api_shared.so
cp $base_full_lib/$full_so_name $target_lib

# copy tiny so
tiny_so_name=libpaddle_light_api_shared.so
cp $base_full_lib/$tiny_so_name $target_lib

# copy target_arch/include dir
cp -r $base_full_include $target_dir

# list include files
echo "ls -l $target_inc"
ls -l $target_inc

# list lib files
echo "ls -l $target_lib"
ls -lh $target_lib
