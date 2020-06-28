#!/bin/bash
cur_dir=$(pwd)

build_dir=$cur_dir/build
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake -DASCEND_PATH=/usr/local/Ascend \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=g++ \
      ..
make

cd -
echo "ls -l $build_dir"
ls -l $build_dir