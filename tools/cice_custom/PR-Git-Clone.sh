#!/bin/bash
work_dir=/home/work
mkdir -p ${work_dir}
rm -rf ${work_dir}/*
cd ${work_dir}

# set repo name
REPO_NAME=PadleTecoBackend

set +xe
unset http_proxy
unset https_proxy
set -xe

# Skip here to update source code always
# wget -q --no-proxy -O ${REPO_NAME}.tar.gz https://paddle-device.bj.bcebos.com/PR/${REPO_NAME}/${AGILE_PULL_ID}/${AGILE_REVISION}/${REPO_NAME}.tar.gz
# if [ $? -eq 0 ];then
#   exit 0
# fi

# Git clone
if [ -d ${REPO_NAME} ]; then rm -rf ${REPO_NAME}; fi 
git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"
git clone --depth=200 --recursive https://github.com/PaddlePaddle/${REPO_NAME}.git
cd ${REPO_NAME}
git fetch origin pull/${AGILE_PULL_ID}/head
git checkout -b test FETCH_HEAD
git merge --no-edit develop

cd ${work_dir}
tar -zcf ${REPO_NAME}.tar.gz ${REPO_NAME}
file_tgz=${REPO_NAME}.tar.gz

# Push BOS
# pip install pycrypto
push_dir=/home
push_file=${push_dir}/bce-python-sdk-0.8.27/BosClient.py
if [ ! -f ${push_file} ];then
    set +x
    wget -q --no-proxy -O /home/bce_whl.tar.gz  https://paddle-docker-tar.bj.bcebos.com/home/bce_whl.tar.gz --no-check-certificate
    set -x
    tar xf /home/bce_whl.tar.gz -C ${push_dir}
fi

cd ${work_dir}
python ${push_file}  ${file_tgz} paddle-device/PR/${REPO_NAME}/${AGILE_PULL_ID}/${AGILE_REVISION}
