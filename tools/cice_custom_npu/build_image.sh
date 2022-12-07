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

export http_proxy=${proxy}
export https_proxy=${proxy}
export ftp_proxy=${proxy}
export no_proxy=bcebos.com

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

# build docker image 
docker system prune -a -f
bash -x build-$(uname -m).sh
