#!/bin/bash

target_arch="armeabi-v7a"

# paddle full lib
base_repo_dir="/workspace/Github-qili93/Paddle-Lite/"
base_full_dir=$base_repo_dir"build.lite.npu.android.armv7.gcc.cxx_shared.full_publish/inference_lite_lib.android.armv7.npu"
base_full_lib=$base_full_dir"/cxx/lib"
base_full_include=$base_full_dir"/cxx/include"
if [ ! -d "$base_full_lib" ]; then
  echo "$base_full_lib NOT exist, pleaes build Paddle-Lite FULL inference library first !!!"
  exit 1
else
  echo "base_full_dir=$base_full_dir"
fi

# paddle tiny lib
base_tiny_dir=$base_repo_dir"build.lite.npu.android.armv7.gcc.cxx_shared.tiny_publish/inference_lite_lib.android.armv7.npu"
base_tiny_lib=$base_tiny_dir"/cxx/lib"
base_tiny_include=$base_tiny_dir"/cxx/include"
if [ ! -d "$base_tiny_lib" ]; then
  echo "$base_tiny_lib NOT exist, pleaes build Paddle-Lite TINY inference library first !!!"
  exit 1
else
  echo "base_tiny_dir=$base_tiny_dir"
fi

# target
target_dir="$PWD/$target_arch"
target_lib=$target_dir"/lib"
echo "target_dir=$target_dir"

# delete target_arch dir
if [ -d "$target_dir" ]; then
  rm -rf "$target_dir"
  echo "$target_dir is deleted"
fi

# create target_arch/lib dir
mkdir -p "$target_lib"
echo "$target_lib created"

# hiai
base_hiai_dir=$base_repo_dir"ai_ddk_lib/lib/"
cp $base_hiai_dir"libhiai.so" $target_lib
cp $base_hiai_dir"libhiai_ir.so" $target_lib
cp $base_hiai_dir"libhiai_ir_build.so" $target_lib
echo "copy form $base_hiai_dir to $target_lib succeed"

# ndk - c++_shared
base_nkd_dir="/opt/android-ndk-r17c"
base_shared_lib=$base_nkd_dir"/sources/cxx-stl/llvm-libc++/libs/"$target_arch"/libc++_shared.so"
cp $base_shared_lib $target_lib
echo "copy form $base_shared_lib to $target_lib succeed"

# copy full so
full_so_name="libpaddle_full_api_shared.so"
cp "$base_full_lib/$full_so_name" "$target_lib"
if [ $? -ne 0 ]; then
  echo "copy form $base_full_lib/$full_so_name to $target_lib failed"
else
  echo "copy form $base_full_lib/$full_so_name to $target_lib succeed"
fi

# copy tiny so
tiny_so_name="libpaddle_light_api_shared.so"
cp "$base_tiny_lib/$tiny_so_name" "$target_lib"
if [ $? -ne 0 ]; then
  echo "copy form $base_tiny_lib/$tiny_so_name to $target_lib failed"
else
  echo "copy form $base_tiny_lib/$tiny_so_name to $target_lib succeed"
fi

# copy target_arch/include dir
cp -r "$base_full_include" "$target_dir"
if [ $? -ne 0 ]; then
  echo "copy form $base_full_include to $target_dir failed"
else
  echo "copy form $base_full_include to $target_dir succeed"
fi

# list include files
echo "ls -l $target_dir/include"
ls -l $target_dir"/include"

# list lib files
echo "ls -l $target_lib"
ls -l $target_lib