#!/bin/bash

function download_model() {
  local model_http=$1
  local model_name=$2
  local dir_exists=$3

  cur_dir=$(pwd)
  if [ ${dir_exists} == "TRUE" ]; then
    target_dir=$cur_dir/models
  else
    target_dir=$cur_dir/models/$model_name
  fi

  echo "target_dir="$target_dir

  model_dir=$cur_dir/models/$model_name
  if [ -d ${model_dir} ]; then
    rm -rf ${model_dir}
    echo "Deleted: ${model_dir}"
  fi
  if [ ! ${dir_exists} == "TRUE" ]; then
    mkdir -p ${model_dir}
    echo "Created: ${model_dir}"
  fi

  wget ${model_http}/${model_name}.tar.gz
  tar zxf ${model_name}.tar.gz -C ${target_dir}
  rm -rf ${model_name}.tar.gz
}

# mobilenet_v1
MODEL_NAME=mobilenet_v1
LITE_URL="http://paddle-inference-dist.bj.bcebos.com"
echo "---------------Download: ${MODEL_NAME}-----------------"
download_model ${LITE_URL} ${MODEL_NAME} TRUE

# mobilenet_v2
MODEL_NAME=mobilenet_v2
LITE_URL="http://paddle-inference-dist.bj.bcebos.com"
echo "---------------Download: ${MODEL_NAME}-----------------"
download_model ${LITE_URL} ${MODEL_NAME} TRUE

# mobilenet_v1_fp32_224_fluid
MODEL_NAME=mobilenet_v1_fp32_224_fluid
LITE_URL="https://paddlelite-demo.bj.bcebos.com/models"
echo "---------------Download: ${MODEL_NAME}-----------------"
download_model ${LITE_URL} ${MODEL_NAME} FALSE

# mobilenet_v2_fp32_224_fluid
MODEL_NAME=mobilenet_v2_fp32_224_fluid
LITE_URL="https://paddlelite-demo.bj.bcebos.com/models"
echo "---------------Download: ${MODEL_NAME}-----------------"
download_model ${LITE_URL} ${MODEL_NAME} FALSE

# resnet18_fp32_224_fluid
MODEL_NAME=resnet18_fp32_224_fluid
LITE_URL="https://paddlelite-demo.bj.bcebos.com/models"
echo "---------------Download: ${MODEL_NAME}-----------------"
download_model ${LITE_URL} ${MODEL_NAME} FALSE

# resnet50_fp32_224_fluid
MODEL_NAME=resnet50_fp32_224_fluid
LITE_URL="https://paddlelite-demo.bj.bcebos.com/models"
echo "---------------Download: ${MODEL_NAME}-----------------"
download_model ${LITE_URL} ${MODEL_NAME} FALSE

# mnasnet_fp32_224_fluid
MODEL_NAME=mnasnet_fp32_224_fluid
LITE_URL="https://paddlelite-demo.bj.bcebos.com/models"
echo "---------------Download: ${MODEL_NAME}-----------------"
download_model ${LITE_URL} ${MODEL_NAME} FALSE
