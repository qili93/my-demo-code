#!/bin/bash
set -xe

export AGILE_PULL_ID=288

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
rm -rf ${WORKSPACE}/output/*

cd ${WORKSPACE}
sleep 10s
rm -rf Paddle*
rm -rf output*

# PaddleCustomDevice
export BRANCH=${PADDLE_BRANCH}
git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"
git clone --depth=200 https://github.com/PaddlePaddle/PaddleCustomDevice.git
cd PaddleCustomDevice

# pull pr code and switch to test branch
git fetch origin pull/${AGILE_PULL_ID}/head
git checkout -b test FETCH_HEAD
git merge --no-edit ${BRANCH}

# setup remote get develop branch code
git remote add upstream https://github.com/PaddlePaddle/PaddleCustomDevice.git
git checkout ${BRANCH}
git --no-pager pull upstream ${BRANCH}
git checkout test 1>nul
git merge ${BRANCH} --no-edit

# show git log history
git log --pretty=oneline -20

# prepare cache dir
source_dir="${WORKSPACE}/PaddleCustomDevice"
cache_dir="${CACHE_ROOT}/.cache"
ccache_dir="${CACHE_ROOT}/.ccache"

# start ci test in container
set +x
docker pull ${PADDLE_DEV_NAME}
docker run --rm -i \
  --privileged --pids-limit 409600 --network=host --shm-size=128G \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v ${cache_dir}:/root/.cache \
  -v ${ccache_dir}:/root/.ccache \
  -v ${source_dir}:/paddle -w /paddle \
  -e "http_proxy=${proxy}" \
  -e "https_proxy=${proxy}" \
  -e "no_proxy=bcebos.com" \
  ${PADDLE_DEV_NAME} \
  /bin/bash -c -x '
python -c "import paddle; print(paddle.__version__)"
python -c "import paddle; print(paddle.version.commit)"

bash -x tools/codestyle/pre_commit.sh;EXCODE=$?

echo "ipipe_log_param_EXCODE: $EXCODE"

exit $EXCODE
'


