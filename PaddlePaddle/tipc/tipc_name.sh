#!/bin/bash
set -ex

function func_model_name() {
    strs=$1
    IFS="/"
    array=(${strs})
    tmp=${array[3]}
    echo ${tmp}
}

mkdir -p "./tipc_log/"

config_files=$(find ./test_tipc/configs -name "train_infer_python.txt")
for file_path in ${config_files}; do
  model_name=$(func_model_name ${file_path})
  echo "$model_name" >> tipc_name.txt
done

