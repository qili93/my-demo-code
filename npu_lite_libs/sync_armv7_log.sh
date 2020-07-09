#!/bin/bash

target_arch=armeabi-v7a

# paddle repo dir
base_repo_dir=/workspace/Github-qili93/Paddle-Lite
# base_repo_dir=/workspace/tempcode/Paddle-Lite

# cd $base_repo_dir
# # build full
# ./lite/tools/build.sh --arm_os=android --arm_abi=armv7 --arm_lang=gcc --android_stl=c++_shared \
#                       --build_npu=ON --npu_ddk_root=./ai_ddk_lib \
#                       --build_extra=ON --with_log=ON \
#                       full_publish
# mv build.lite.android.armv7.gcc/ build.lite.android.armv7.gcc_full_log/

# # build tiny
# ./lite/tools/build.sh --arm_os=android --arm_abi=armv7 --arm_lang=gcc --android_stl=c++_shared \
#                       --build_npu=ON --npu_ddk_root=./ai_ddk_lib \
#                       --build_extra=ON --with_log=ON \
#                       tiny_publish
# mv build.lite.android.armv7.gcc/ build.lite.android.armv7.gcc_tiny_log/
# cd -

# paddle full lib
base_full_dir=$base_repo_dir/build.lite.android.armv7.gcc_full_log/inference_lite_lib.android.armv7.npu
base_full_lib=$base_full_dir/cxx/lib
base_full_include=$base_full_dir/cxx/include
echo "base_full_dir=$base_full_dir"

# paddle tiny lib
base_tiny_dir=$base_repo_dir/build.lite.android.armv7.gcc_tiny_log/inference_lite_lib.android.armv7.npu
base_tiny_lib=$base_tiny_dir/cxx/lib
base_tiny_include=$base_tiny_dir/cxx/include
echo "base_tiny_dir=$base_tiny_dir"

# target
cur_dir=$(pwd)
target_dir=$cur_dir/$target_arch-log
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
echo "$target_lib created"

# copy hiai
echo "---------------COPY HW HIAI DDK Libs-----------------"
base_hiai_dir=$base_repo_dir/ai_ddk_lib_610/lib
echo "copy form == $base_hiai_dir"
echo "copy to ==== $target_lib"
cp $base_hiai_dir/libhiai.so $target_lib
cp $base_hiai_dir/libhiai_ir.so $target_lib
cp $base_hiai_dir/libhiai_ir_build.so $target_lib
cp $base_hiai_dir/libhcl.so $target_lib
cp $base_hiai_dir/libcpucl.so $target_lib

echo "---------------COPY NDK C++ Shared Libs-----------------"
base_nkd_dir=/opt/android-ndk-r17c
base_shared_lib=$base_nkd_dir/sources/cxx-stl/llvm-libc++/libs/$target_arch/libc++_shared.so
echo "copy form == $base_shared_lib"
echo "copy to ==== $target_lib"
cp $base_shared_lib $target_lib

# copy full so
echo "---------------COPY Paddle-Lite Full Libs-----------------"
full_so_name=libpaddle_full_api_shared.so
echo "copy from == $base_full_lib/$full_so_name"
echo "copy to ==== $target_lib"
cp $base_full_lib/$full_so_name $target_lib

# copy tiny so
echo "---------------COPY Paddle-Lite Tiny Libs-----------------"
tiny_so_name=libpaddle_light_api_shared.so
echo "copy from == $base_full_lib/$tiny_so_name"
echo "copy to ==== $target_lib"
cp $base_tiny_lib/$tiny_so_name $target_lib

# copy target_arch/include dir
echo "---------------COPY Paddle-Lite Headers-----------------"
echo "copy from == $base_full_include"
echo "copy to ==== $target_lib"
cp -r $base_full_include $target_dir

# list include files
echo "ls -l $target_inc"
ls -l $target_inc

# list lib files
echo "ls -l $target_lib"
ls -lh $target_lib

# compress libs
tar -zcvf armeabi-v7a-log.tar.gz ./armeabi-v7a-log
