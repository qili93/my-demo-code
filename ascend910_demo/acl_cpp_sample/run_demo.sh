#!/bin/bash

cd src/

build_dir="$(pwd)/build"
if [ -d ${build_dir} ];then
  rm -rf ${build_dir}
fi

# build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make

# run
echo "-------------- start running --------------"
./main
