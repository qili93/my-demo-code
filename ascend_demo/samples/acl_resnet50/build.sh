#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux_gcc4.8.5
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux_gcc4.8.5/acllib/lib64/stub

#######################################
# Build commands, do not change them
#######################################
build_dir=$cur_dir/build/intermediates/host
rm -rf $build_dir
mkdir -p $build_dir
cd $build_dir

cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make

cd -
echo "ls -l $cur_dir/out"
ls -l $cur_dir/out
