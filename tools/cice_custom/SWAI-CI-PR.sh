#!/bin/bash
set -xe

##### global environment, delete from scirpt and add into ci config #####

export WORKSPACE=/root/PR-CI-SWAI
export CACHE_ROOT=/root/PR-CI-SWAI/.cache/BUILD_CI_ROCM

export PADDLE_BARNCH=develop
export PADDLE_VERSION=0.0.0

export whl_package=paddle-device/cpu/sunway/
export tgz_package=paddle-device/cpu/sunway/

#### login to qemu environment ####
sshpass -p 123456 ssh -p 8111 root@127.0.0.1<<EOT

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
rm -rf paddlepaddle*

# install latest paddlepaddle cpu whl 
wget -q https://paddle-device.bj.bcebos.com/cpu/sunway/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl
pip install -U --no-deps --force-reinstall paddlepaddle-*.whl

# download and update swdnn
rm -rf swdnn*
wget -q https://paddle-device.bj.bcebos.com/cpu/sunway/swdnn-v1.4.5.tar.gz
tar -xf swdnn-v1.4.5.tar.gz
mv swdnn-v1.4.5 swdnn

# download and update tecoblas
rm -rf tecoblas*
wget -q https://paddle-device.bj.bcebos.com/cpu/sunway/tecoblas-v1.1.0-rc2.tar.gz
tar -xf tecoblas-v1.1.0-rc2.tar.gz
mv tecoblas-v1.1.0-rc2 tecoblas

# download and update extend_ops
rm -rf extend_ops*
wget -q https://paddle-device.bj.bcebos.com/cpu/sunway/extend_ops.tar.gz
tar -xf extend_ops.tar.gz

# git clone PaddleTecoBackend
git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"
git clone --depth=200 --recursive https://github.com/PaddlePaddle/PaddleTecoBackend.git
cd PaddleTecoBackend
# sync submodule
git submodule sync
git submodule update --init --recursive
# pull pr code
git fetch origin pull/${AGILE_PULL_ID}/head
git checkout -b test FETCH_HEAD
git merge --no-edit develop
# show git log history
git log --pretty=oneline -10

# build and test
bash scripts/paddle_ci.sh custom_swai

EOT

EXCODE=$?

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
else
    echo "Sorry, unknown error."
fi

exit $EXCODE