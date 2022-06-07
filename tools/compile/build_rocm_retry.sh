#!/bin/bash

set -ex

export http_proxy=http://172.19.57.45:3128
export https_proxy=http://172.19.57.45:3128
export ftp_proxy=http://172.19.57.45:3128
export no_proxy=bcebos.com,gitee.com

cd /workspace/Paddle

# save all changes to stash 
git stash save -u "$(date)"
git stash list

# pull develop branch - retry 3 times
for i in {1..3}
do
    git pull upstream develop && pull_error=0 && break || pull_error=$? && sleep 5
done

if [ $pull_error -ne 0 ]; then
    echo "Fail to pull latest code from develop branch after retrying 3 times"
    exit $pull_error
fi

# prepare build directory
BUILD_DIR="/workspace/Paddle/build_rocm"
if [ ! -d ${BUILD_DIR} ];then
    mkdir -p ${BUILD_DIR}
fi

# cmake
cd ${BUILD_DIR}
cmake .. -DPY_VERSION=3.7 -DWITH_ROCM=ON -DWITH_TESTING=ON -DWITH_DISTRIBUTE=ON \
         -DWITH_PSCORE=OFF -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release; cmake_error=$?

if [ "$cmake_error" != 0 ];then
    echo "Fail to generate cmake"
    exit $cmake_error
fi

# retry 5 times
for i in {1..3}
do
    make -j8 && make_error=0 && break || make_error=$? && sleep 15
done


if [ "$make_error" != 0 ];then
    echo "Fail to compile after retry 5 times"
    exit $make_error
fi
