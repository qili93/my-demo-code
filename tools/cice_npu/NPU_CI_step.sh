#!/bin/bash
set -xe

export proxy=http://172.19.57.45:3128

export WORKSPACE=/workspace/npu_dev
export CACHE_ROOT=/workspace/npu_dev/.cache/PR_CI_NPU
export cache_dir="${CACHE_ROOT}/.cache"
export ccache_dir="${CACHE_ROOT}/.ccache"

export PADDLE_BRANCH=develop
export PADDLE_VERSION=0.0.0
export PADDLE_DEV_NAME=registry.baidubce.com/qili93/paddle:latest-dev-cann5.0.2.alpha005-gcc82-x86_64

export whl_package=paddle-wheel/develop-npu-cann5.0.2-x86_64
export tgz_package=paddle-inference-lib/latest-npu-cann5.0.2-x86_64

export PADDLE_DIR="${WORKSPACE}/Paddle"

docker run --rm -it \
  --pids-limit 409600 --network=host --shm-size=128G \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v ${cache_dir}:/root/.cache \
  -v ${ccache_dir}:/root/.ccache \
  -v ${PADDLE_DIR}:${PADDLE_DIR} \
  -w ${PADDLE_DIR} \
  -e "WITH_GPU=OFF" \
  -e "WITH_ASCEND=OFF" \
  -e "WITH_ASCEND_CL=ON" \
  -e "WITH_INFERENCE_API_TEST=OFF" \
  -e "WITH_ASCEND_INT64=ON" \
  -e "WITH_ARM=OFF" \
  -e "WITH_TENSORRT=OFF" \
  -e "WITH_COVERAGE=ON" \
  -e "COVERALLS_UPLOAD=OFF" \
  -e "CMAKE_BUILD_TYPE=Release" \
  -e "WITH_MKL=ON" \
  -e "WITH_CACHE=ON" \
  -e "PADDLE_VERSION=${PADDLE_VERSION}" \
  -e "PADDLE_BRANCH=${PADDLE_BRANCH}" \
  -e "BRANCH=${PADDLE_BRANCH}" \
  -e "WITH_TEST=ON" \
  -e "RUN_TEST=ON" \
  -e "WITH_TESTING=ON" \
  -e "WITH_DISTRIBUTE=ON" \
  -e "ON_INFER=ON" \
  -e "PYTHON_ABI=conda-python3.7" \
  -e "CTEST_PARALLEL_LEVEL=1" \
  -e "PY_VERSION=3.7" \
  -e "http_proxy=${proxy}" \
  -e "https_proxy=${proxy}" \
  -e "GIT_PR_ID=${AGILE_PULL_ID}" \
  -e "AGILE_JOB_BUILD_ID=${AGILE_JOB_BUILD_ID}" \
  -e "GITHUB_API_TOKEN=${GITHUB_API_TOKEN}" \
  ${PADDLE_DEV_NAME} /bin/bash

bash -x paddle/scripts/paddle_build.sh build_only 8 # compile only
bash -x paddle/scripts/paddle_build.sh test # test only

# 或者输出如下环境变量

export proxy=http://172.19.57.45:3128
export PADDLE_BRANCH=develop
export PADDLE_VERSION=0.0.0

export WITH_GPU=OFF
export WITH_ASCEND=OFF
export WITH_ASCEND_CL=ON
export WITH_ASCEND_INT64=ON
export WITH_ARM=OFF
export WITH_TENSORRT=OFF
export WITH_COVERAGE=OFF
export COVERALLS_UPLOAD=OFF
export CMAKE_BUILD_TYPE=Release
export WITH_MKL=ON
export WITH_CACHE=ON
export PADDLE_VERSION=${PADDLE_VERSION}
export PADDLE_BRANCH=${PADDLE_BRANCH}
export BRANCH=${PADDLE_BRANCH}
export WITH_TEST=ON
export RUN_TEST=ON
export WITH_TESTING=ON
export WITH_DISTRIBUTE=ON
export ON_INFER=ON
export PYTHON_ABI=conda-python3.7
export CTEST_PARALLEL_LEVEL=1
export PY_VERSION=3.7
export CI_SKIP_CPP_TEST=OFF
export http_proxy=${proxy}
export https_proxy=${proxy}

bash -x paddle/scripts/paddle_build.sh build_only 8 # compile only
bash -x paddle/scripts/paddle_build.sh test # test only
bash -x paddle/scripts/paddle_build.sh gpu_cicheck_py35 # test only