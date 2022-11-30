#!/bin/bash
set -xe

##### global environment #####

export WORKSPACE=/home/liqi27/
export CACHE_ROOT=/home/liqi27
export cache_dir="${CACHE_ROOT}/.cache"
export ccache_dir="${CACHE_ROOT}/.ccache"

export PADDLE_BRANCH=develop
export PADDLE_VERSION=0.0.0
export PADDLE_DEV_NAME=registry.baidubce.com/device/paddle-npu:cann504-x86_64-gcc82
export PADDLE_DEV_NAME=registry.baidubce.com/device/paddle-npu:cann504-aarch64-gcc82

export whl_package=paddle-device/develop/npu
export tgz_package=paddle-device/develop/npu


docker pull ${PADDLE_DEV_NAME}
docker run --rm -it \
  --privileged --pids-limit 409600 --network=host --shm-size=128G \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v ${cache_dir}:/root/.cache \
  -v ${ccache_dir}:/root/.ccache \
  -v ${WORKSPACE}:/workspace -w /workspace \
  -e "http_proxy=${proxy}" \
  -e "https_proxy=${proxy}" \
  -e "no_proxy=bcebos.com" \
  ${PADDLE_DEV_NAME}  /bin/bash

bash -x scripts/paddle_ci.sh custom_npu # cice test
  
# 或者直接在现有容器中执行如下脚本，输出如下环境变量

bash -x scripts/paddle_ci.sh custom_npu
