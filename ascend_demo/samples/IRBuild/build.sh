#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################
export ASCEND_PATH=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux_gcc4.8.5

#######################################
# Build commands, do not change them
#######################################
make -j8

cd -
echo "ls -l $cur_dir/out"
ls -l $cur_dir/out
