#!/bin/bash
set -xe

##### global environment, delete from scirpt and add into ci config #####

export WORKSPACE=/workspace/custom-device-npu
export CACHE_ROOT=/workspace/custom-device-npu

export PADDLE_BRANCH=develop
export PADDLE_VERSION=0.0.0
export PADDLE_DEV_NAME=registry.baidubce.com/device/paddle-npu:cann504-x86_64-gcc82

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

# git clone PaddleCustomDevice
git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"
git clone --depth=200 --recursive https://github.com/PaddlePaddle/PaddleCustomDevice.git
cd PaddleCustomDevice
# sync submodule
git submodule sync
git submodule update --init --recursive
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
scripts/paddle_ci.sh custom_npu;EXCODE=$?

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


mkdir -p ${WORKSPACE}/output
cp ${source_dir}/backends/npu/build/dist/paddle_custom_npu*.whl ${WORKSPACE}/output

wget -q --no-proxy -O ${WORKSPACE}/bce_whl.tar.gz  https://paddle-docker-tar.bj.bcebos.com/home/bce_whl.tar.gz --no-check-certificate
tar xf ${WORKSPACE}/bce_whl.tar.gz -C ${WORKSPACE}/output
push_file=${WORKSPACE}/output/bce-python-sdk-0.8.27/BosClient.py

# Install dependency
/usr/bin/python2 -m pip install pycrypto

# Upload paddlepaddle-rocm whl package to paddle-device/develop/dcu1
cd ${WORKSPACE}/output
for file_whl in `ls *.whl` ;do
  /usr/bin/python2 ${push_file} ${file_whl} ${whl_package}
done
