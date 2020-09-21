#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}


#######################################
# Step1: Export DDK Environment
#######################################
export LD_LIBRARY_PATH=/usr/local/Ascend/nnrt/latest/x86_64-linux_gcc7.3.0/acllib/lib64:/usr/local/Ascend/add-ons


#######################################
# Step2: Run Demo
#######################################
cd out/
./main
cd -