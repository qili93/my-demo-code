#!/bin/bash
set -ex

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
bash -x build-aarch64.sh
