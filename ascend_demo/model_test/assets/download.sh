#!/bin/bash

function download_model() {
  local model_name=$1

  cur_dir=$(pwd)
  target_dir=$cur_dir/models/$model_name
  if [ -d ${target_dir} ]
  then
    rm -rf ${target_dir}
  fi
  mkdir -p ${target_dir}

  wget https://paddlelite-demo.bj.bcebos.com/models/${model_name}.tar.gz
  tar zxf ${model_name}.tar.gz -C ${target_dir}
  rm -rf ${model_name}.tar.gz
}

# mobilenet_v1_fp32_224_fluid
MODEL_NAME=mobilenet_v1_fp32_224_fluid
echo "---------------Download: ${MODEL_NAME}-----------------"
download_model ${MODEL_NAME}

# mobilenet_v2_fp32_224_fluid
MODEL_NAME=mobilenet_v2_fp32_224_fluid
echo "---------------Download: ${MODEL_NAME}-----------------"
download_model ${MODEL_NAME}

# resnet18_fp32_224_fluid
MODEL_NAME=resnet18_fp32_224_fluid
echo "---------------Download: ${MODEL_NAME}-----------------"
download_model ${MODEL_NAME}

# resnet50_fp32_224_fluid
MODEL_NAME=resnet50_fp32_224_fluid
echo "---------------Download: ${MODEL_NAME}-----------------"
download_model ${MODEL_NAME}

# mnasnet_fp32_224_fluid
MODEL_NAME=mnasnet_fp32_224_fluid
echo "---------------Download: ${MODEL_NAME}-----------------"
download_model ${MODEL_NAME}