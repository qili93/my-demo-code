#!/bin/bash
set -xe

##### global environment #####

set +x
export proxy=http:xxxxxxx
export WORKSPACE=/workspace/codestyle
export PADDLE_BRANCH=develop
export PADDLE_VERSION=0.0.0
export PADDLE_TAG=v0.0.0
export PADDLE_COMMIT=develop
set -x

##### local environment #####

set +x
export http_proxy=${proxy}
export https_proxy=${proxy}
export no_proxy=bcebos.com
set -x

mkdir -p ${WORKSPACE}
cd ${WORKSPACE}
sleep 10s
rm -rf Paddle*
rm -rf output*

# PaddleCustomDevice
git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"
git clone -b ${PADDLE_BRANCH} https://github.com/PaddlePaddle/PaddleCustomDevice.git
cd PaddleCustomDevice

# --- pull request ---
git fetch origin pull/${AGILE_PULL_ID}/head
git checkout -b test FETCH_HEAD
git merge --no-edit ${PADDLE_BRANCH}
# --- merge upstream ---
git remote add upstream https://github.com/PaddlePaddle/PaddleCustomDevice.git
git checkout ${PADDLE_BRANCH}
git --no-pager pull upstream ${PADDLE_BRANCH}
git checkout test 1>nul
git merge ${PADDLE_BRANCH} --no-edit
# --- show history ---
git log --pretty=oneline -20
# --- submodule ---
git submodule sync
git submodule update --init --recursive

# prepare cache dir
source_dir="${WORKSPACE}/PaddleCustomDevice"
cache_dir="${WORKSPACE}/.cache"
ccache_dir="${WORKSPACE}/.ccache"
mkdir -p "${cache_dir}"
mkdir -p "${ccache_dir}"

# start ci test in container
set +x
PADDLE_DEV_NAME=registry.baidubce.com/device/paddle-cpu:ubuntu18-$(uname -m)-gcc82
docker pull ${PADDLE_DEV_NAME}
docker run --rm -i \
  --privileged --network=host --shm-size=128G \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v ${cache_dir}:/root/.cache \
  -v ${ccache_dir}:/root/.ccache \
  -v ${source_dir}:/paddle -w /paddle \
  -e "PADDLE_BRANCH=${PADDLE_BRANCH}" \
  -e "PADDLE_VERSION=${PADDLE_VERSION}" \
  -e "http_proxy=${http_proxy}" \
  -e "https_proxy=${https_proxy}" \
  -e "no_proxy=${no_proxy}" \
  ${PADDLE_DEV_NAME} \
  /bin/bash -c -x '
pre-commit install
tools/codestyle/pre_commit.sh;EXCODE=$?
echo "ipipe_log_param_EXCODE: $EXCODE"
exit $EXCODE
'
