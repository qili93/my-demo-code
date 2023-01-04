#!/bin/bash
set -ex

##### global environment, delete from scirpt and add into ci config #####

export WORKSPACE=/workspace/npu-dev
export CACHE_ROOT=/workspace/npu-dev

export PADDLE_BRANCH=develop
export PADDLE_COMMIT=develop
export PADDLE_VERSION=0.0.0

export PADDLE_DEV_NAME=registry.baidubce.com/device/paddle-npu:cann600-$(uname -m)-gcc82

export whl_package=paddle-device/develop/npu
export tgz_package=paddle-device/develop/npu

##### local environment #####

set +x
export http_proxy=${proxy}
export https_proxy=${proxy}
export ftp_proxy=${proxy}
export no_proxy=bcebos.com
set -x

mkdir -p ${WORKSPACE}
mkdir -p ${CACHE_ROOT}

cd ${WORKSPACE}
sleep 10s
rm -rf Paddle*
rm -rf output*

git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"
git clone -b ${PADDLE_BRANCH} https://github.com/PaddlePaddle/PaddleCustomDevice.git
cd PaddleCustomDevice/backends/npu/tools/dockerfile

# get md5 cache
export md5_content=$(cat * |md5sum | awk '{print $1}')
md5_cache_dir="${CACHE_ROOT}/md5_cache"
md5_cache_file=${md5_cache_dir}/${md5_content}.txt

if [ ! -d ${md5_cache_dir} ];then
  mkdir -p ${md5_cache_dir}
fi

# prepare dockerfile
echo "FROM registry.baidubce.com/device/paddle-npu:cann600-$(uname -m)-gcc82"  > Dockerfile.npu
echo -e "MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>\n" >> Dockerfile.npu
echo "RUN wget -q https://paddle-device.bj.bcebos.com/develop/cpu/paddlepaddle-0.0.0-cp37-cp37m-linux_$(uname -m).whl" >> Dockerfile.npu
echo "RUN pip install -U --no-deps --force-reinstall paddlepaddle-0.0.0-cp37-cp37m-linux_$(uname -m).whl" >> Dockerfile.npu
echo "RUN rm -rf paddlepaddle-0.0.0-cp37-cp37m-linux_$(uname -m).whl" >> Dockerfile.npu
echo -e "\nEXPOSE 22" >> Dockerfile.npu

# docker system prune -a -f

docker pull registry.baidubce.com/device/paddle-cpu:ubuntu18-$(uname -m)-gcc82
docker pull registry.baidubce.com/device/paddle-npu:cann600-$(uname -m)-gcc82

if [ -f ${md5_cache_file} ];then
  docker build --no-cache --network=host -f Dockerfile.npu -t ${PADDLE_DEV_NAME} .
  docker push ${PADDLE_DEV_NAME}
else
  rm -rf ${md5_cache_dir}/*
  bash -x build-$(uname -m).sh
fi

if [ $? -eq 0 ];then
  touch -f ${md5_cache_file}
fi
