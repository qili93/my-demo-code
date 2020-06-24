#!/bin/bash

base_repo=/workspace/Github-qili93/Paddle-Lite
base_dir=$base_repo/build.lite.x86/inference_lite_lib
# x86 lib based on base_dir
base_lib=$base_dir/cxx/lib
base_include=$base_dir/cxx/include
base_mklml_dir=$base_dir/third_party/mklml
echo "base_dir=$base_dir"

cur_dir=$(pwd)
target_dir=$cur_dir
target_lib=$target_dir/lib
target_inc=$target_dir/include
target_mkl=$target_dir/mklml
echo "target_dir=$target_dir"

# delete x86 lib dir
if [ -d "$target_lib" ]; then
  rm -rf "$target_lib"
  echo "$target_lib is deleted"
fi

# create x86/lib dir
mkdir -p "$target_lib"
echo "$target_lib created"

# copy full so
full_so_name="libpaddle_full_api_shared.so"
cp "$base_lib/$full_so_name" "$target_lib"
# copy tiny so
tiny_so_name="libpaddle_light_api_shared.so"
cp "$base_lib/$tiny_so_name" "$target_lib"

# delete x86/include dir
if [ -d "$target_inc" ]; then
  rm -rf "$target_inc"
  echo "$target_inc is deleted"
fi
# copy x86/include
cp -r "$base_include" "$target_dir"

# delete x86/mklml dir
if [ -d "$target_mkl" ]; then
  rm -rf "$target_mkl"
  echo "$target_mkl is deleted"
fi
# copy mklml dir
cp -r "$base_mklml_dir" "$target_dir"