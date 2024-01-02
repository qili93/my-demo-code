#!/bin/bash
set -xe

##### global environment #####

set +x
export proxy=http:xxxxxxx
export WORKSPACE=/workspace/npu
export PADDLE_BRANCH=develop
export PADDLE_VERSION=0.0.0
export PADDLE_TAG=v0.0.0
export PADDLE_COMMIT=develop
set -x

##### local environment #####

set +x
export http_proxy=${proxy}
export https_proxy=${proxy}
export ftp_proxy=${proxy}
export no_proxy=bcebos.com
set -x

mkdir -p ${WORKSPACE}
rm -rf ${WORKSPACE}/output/*

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
# --- show history ---
git log --pretty=oneline -20

# !!!!! SKIP IF NO NPU CHANGE !!!!
echo "=========== Checking PR Changes If NPU FULL CI Needed ==========="
change_numbers=$(git diff --name-only remotes/origin/develop | wc -l)
change_backend=$(git diff --name-only remotes/origin/develop | grep "backends/"| wc -l)
change_npu_only=$(git diff --name-only remotes/origin/develop | grep "backends/npu"| wc -l)
if [ $change_numbers -ne $change_backend ]; then
  echo "Common file changed, continue to run NPU FULL CI test ..."
elif [ $change_npu_only -eq 0 ] ; then
  echo "NO NPU backend changes found, skip NPU FULL CI ...."
  exit 0
fi

# --- submodule ---
# git submodule sync
# git submodule update --init --recursive

# prepare cache dir
source_dir="${WORKSPACE}/PaddleCustomDevice"
cache_dir="${WORKSPACE}/.cache"
ccache_dir="${WORKSPACE}/.ccache"
mkdir -p "${cache_dir}"
mkdir -p "${ccache_dir}"

# start ci test in container
set +x
PADDLE_DEV_NAME=registry.baidubce.com/device/paddle-npu:cann601-ubuntu18-$(uname -m)-gcc82
docker pull ${PADDLE_DEV_NAME}
docker run --rm -i \
  --privileged --network=host --shm-size=128G \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -e ASCEND_RT_VISIBLE_DEVICES="0,1" \
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
echo "============ CANN Version ============="
ls -l /usr/local/Ascend/ascend-toolkit/

echo "============ Install PaddlePaddle CPU ============="
wget -q https://paddle-device.bj.bcebos.com/${PADDLE_VERSION}/cpu/paddlepaddle-${PADDLE_VERSION}-cp39-cp39-linux_$(uname -m).whl
pip install paddlepaddle-*.whl && rm -rf paddlepaddle-*.whl
python -c "import paddle; print(paddle.__version__)"
python -c "import paddle; print(paddle.version.commit)"

bash backends/npu/tools/pr_ci_npu.sh;EXCODE=$?

if [[ $EXCODE -eq 0 ]];then
    echo "Congratulations!  Your PR passed the CI."
elif [[ $EXCODE -eq 4 ]];then
    echo "Sorry, your code style check failed."
elif [[ $EXCODE -eq 6 ]];then
    echo "Sorry, your pr need to be approved."
elif [[ $EXCODE -eq 7 ]];then
    echo "Sorry, build failed."
elif [[ $EXCODE -eq 8 ]];then
    echo "Sorry, some tests failed."
elif [[ $EXCODE -eq 9 ]];then
    echo "Sorry, coverage check failed."
fi

exit $EXCODE
'
