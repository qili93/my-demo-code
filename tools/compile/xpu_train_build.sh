#!/bin/bash

set -ex

export http_proxy=http://172.19.57.45:3128
export https_proxy=http://172.19.57.45:3128
export ftp_proxy=http://172.19.57.45:3128
export no_proxy=bcebos.com

# export http_proxy=http://172.19.56.199:3128
# export https_proxy=http://172.19.56.199:3128
# export ftp_proxy=http://172.19.56.199:3128
# export no_proxy=bcebos.com

arch=$(uname -i)
if [[ $arch == x86_64* ]]; then
    WITH_ARM=OFF
elif  [[ $arch == aarch64* ]]; then
    WITH_ARM=ON
fi
echo "Compiling with WITH_ARM=${WITH_ARM}"

cd /workspace/Paddle

# save all changes to stash 
git stash save -u "$(date +'%Y-%m-%d-%H-%M-%S')"
git stash list

# checkout to develop branch
git checkout develop

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
BUILD_DIR="/workspace/Paddle/build_xpu_train"
if [ ! -d ${BUILD_DIR} ];then
    mkdir -p ${BUILD_DIR}
fi

# cmake
cd ${BUILD_DIR}
cmake .. -DPY_VERSION=3.7 -DON_INFER=OFF -DWITH_DISTRIBUTE=ON -DWITH_ARM=${WITH_ARM} -DWITH_AARCH64=${WITH_ARM} \
         -DWITH_XPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-error -w" \
         -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_VERBOSE_MAKEFILE=OFF; cmake_error=$?

if [ "$cmake_error" != 0 ];then
    echo "Fail to generate cmake"
    exit $cmake_error
fi

# retry 3 times
for i in {1..3}
do
    if [ "$WITH_ARM" == "ON" ];then
        make TARGET=ARMV8 -j  && make_error=0 && break || make_error=$? && sleep 15
    else
        make -j8 && make_error=0 && break || make_error=$? && sleep 15
    fi
done


if [ "$make_error" != 0 ];then
    echo "Fail to compile after retry 5 times"
    exit $make_error
fi
