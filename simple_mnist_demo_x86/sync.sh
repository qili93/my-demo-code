#!/bin/bash

base_dir="/workspace/Github-qili93/Paddle-Lite/build.lite.x86/inference_lite_lib"
base_lib=$base_dir"/cxx/lib"
base_include=$base_dir"/cxx/include"
base_mklml_dir=$base_dir"/third_party/mklml"
echo "base_dir=$base_dir"s

target_dir=$PWD"/x86"
target_lib=$target_dir"/lib"
echo "target_dir=$target_dir"

# delete x86 dir
if [ -d "$target_dir" ]; then
  rm -rf "$target_dir"
  echo "$target_dir is deleted"
fi

# create x86/lib dir
mkdir -p "$target_lib"
echo "$target_lib created"

# copy full so
full_so_name="libpaddle_full_api_shared.so"
# if [ -f "$target_lib/$full_so_name" ]; then
#   rm -rf "$target_lib/$full_so_name"
#   echo "$target_lib/$full_so_name is deleted"
# fi
cp "$base_lib/$full_so_name" "$target_lib"
if [ $? -ne 0 ]; then
  echo "copy form $base_lib/$full_so_name to $target_lib failed"
else
  echo "copy form $base_lib/$full_so_name to $target_lib succeed"
fi

# copy tiny so
tiny_so_name="libpaddle_light_api_shared.so"
# if [ -f "$target_lib/$tiny_so_name" ]; then
#   rm -rf "$target_lib/$tiny_so_name"
#   echo "$target_lib/$tiny_so_name is deleted"
# fi
cp "$base_lib/$tiny_so_name" "$target_lib"
if [ $? -ne 0 ]; then
  echo "copy form $base_lib/$tiny_so_name to $target_lib failed"
else
  echo "copy form $base_lib/$tiny_so_name to $target_lib succeed"
fi

# if [ -d "$target_include" ]; then
#   rm -rf "$target_include"
#   echo "$target_include is deleted"
# fi

# copy x86/include dir
cp -r "$base_include" "$target_dir"
if [ $? -ne 0 ]; then
  echo "copy form $base_include to $target_dir failed"
else
  echo "copy form $base_include to $target_dir succeed"
fi

# copy mklml dir
cp -r "$base_mklml_dir" "$target_dir"
if [ $? -ne 0 ]; then
  echo "copy form $base_mklml_dir to $target_dir failed"
else
  echo "copy form $base_mklml_dir to $target_dir succeed"
fi