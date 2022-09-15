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
  echo -e "\n" >> tipc_run.sh
  echo "echo '-------------------------- $file_path ----------------------------------'" >> tipc_run.sh
  echo "timeout 60m bash test_tipc/prepare.sh $file_path 'lite_train_lite_infer'" >> tipc_run.sh
  echo "timeout 60m bash test_tipc/test_train_inference_python_xpu.sh $file_path 'lite_train_lite_infer' > tipc_log/$model_name.log 2>&1" >> tipc_run.sh
done

