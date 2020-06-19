#!/bin/bash

USE_FULL_API=TRUE

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

rm -rf build-x86
mkdir build-x86
cd build-x86
cmake -DUSE_FULL_API=${USE_FULL_API} \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make