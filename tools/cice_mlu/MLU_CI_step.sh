#!/bin/bash
set -xe

export proxy=http://172.19.57.45:3128

export WORKSPACE=/home/liqi27/develop/mlu
export CACHE_ROOT=/home/liqi27/develop/mlu/.cache/BUILD_CI_MLU

export PADDLE_BRANCH=develop
export PADDLE_VERSION=0.0.0
export PADDLE_DEV_NAME=registry.baidubce.com/device/paddle-mlu:neuware

export whl_package=paddle-device/develop/mlu
export tgz_package=paddle-device/develop/mlu

export PADDLE_DIR="${WORKSPACE}/Paddle"

docker run  --rm -it --network=host --shm-size=128G \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/cambricon_ctl --device=/dev/cambricon_dev0 \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  -v ${cache_dir}:/root/.cache \
  -v ${ccache_dir}:/root/.ccache \
  -v ${PADDLE_DIR}:/paddle \
  -w /paddle \
  -e "WITH_GPU=OFF" \
  -e "WITH_MLU=ON" \
  -e "WITH_CNCL=ON" \
  -e "WITH_TENSORRT=OFF" \
  -e "WITH_COVERAGE=OFF" \
  -e "COVERALLS_UPLOAD=OFF" \
  -e "CMAKE_BUILD_TYPE=Release" \
  -e "WITH_MKL=ON" \
  -e "WITH_AVX=ON" \
  -e "WITH_CACHE=ON" \
  -e "PADDLE_VERSION=${PADDLE_VERSION}" \
  -e "PADDLE_BRANCH=${PADDLE_BRANCH}" \
  -e "BRANCH=${PADDLE_BRANCH}" \
  -e "WITH_TEST=ON" \
  -e "RUN_TEST=ON" \
  -e "WITH_TESTING=ON" \
  -e "WITH_DISTRIBUTE=ON" \
  -e "CTEST_PARALLEL_LEVEL=ON" \
  -e "PYTHON_ABI=conda-python3.7" \
  -e "PY_VERSION=3.7" \
  -e "http_proxy=${proxy}" \
  -e "https_proxy=${proxy}" \
  ${PADDLE_DEV_NAME} /bin/bash

bash -x paddle/scripts/paddle_build.sh build_only 8 # compile only
bash -x paddle/scripts/paddle_build.sh test # test only

# 或者输出如下环境变量

export proxy=http://172.19.57.45:3128
export PADDLE_BRANCH=develop
export PADDLE_VERSION=0.0.0
export PADDLE_DEV_NAME=registry.baidubce.com/device/paddle-mlu:neuware


export WITH_GPU=OFF
export WITH_MLU=ON
export WITH_CNCL=ON
export WITH_TENSORRT=OFF
export WITH_COVERAGE=OFF
export COVERALLS_UPLOAD=OFF
export CMAKE_BUILD_TYPE=Release
export WITH_MKL=ON
export WITH_AVX=ON
export WITH_CACHE=ON
export PADDLE_VERSION=${PADDLE_VERSION}
export PADDLE_BRANCH=${PADDLE_BRANCH}
export BRANCH=${PADDLE_BRANCH}
export WITH_TEST=ON
export RUN_TEST=ON
export WITH_TESTING=ON
export WITH_DISTRIBUTE=ON
export CTEST_PARALLEL_LEVEL=ON
export PYTHON_ABI=conda-python3.7
export PY_VERSION=3.7
export http_proxy=${proxy}
export https_proxy=${proxy}

bash -x paddle/scripts/paddle_build.sh build_only 8 # compile only
bash -x paddle/scripts/paddle_build.sh test # test only
bash -x paddle/scripts/paddle_build.sh gpu_cicheck_py35 # test only