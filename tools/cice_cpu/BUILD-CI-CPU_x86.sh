#!/bin/bash
set -xe

export proxy=http:xxxxxxx

##### global environment #####

export WORKSPACE=/workspace/cpu-dev
export CACHE_ROOT=/workspace/cpu-dev

export PADDLE_BRANCH=develop
export PADDLE_VERSION=0.0.0
export PADDLE_TAG=v0.0.0
export PADDLE_COMMIT=develop

##### local environment #####

set +x
export http_proxy=${proxy}
export https_proxy=${proxy}
export no_proxy=bcebos.com
set -x

mkdir -p ${WORKSPACE}
mkdir -p ${CACHE_ROOT}
rm -rf ${WORKSPACE}/output/*

cd ${WORKSPACE}
sleep 10s
rm -rf Paddle*
rm -rf output*

# Paddle
git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"
git clone -b ${PADDLE_BRANCH} https://github.com/PaddlePaddle/Paddle.git
cd Paddle
# --- release info ---
# git checkout tags/${PADDLE_TAG}
# git checkout ${PADDLE_COMMIT}
# git pull origin pull/51244/head
# --- submodule ---
# git submodule sync
# git submodule update --init --recursive
# --- show history ---
git log --pretty=oneline -20

# prepare cache dir
source_dir="${WORKSPACE}/Paddle"
cache_dir="${CACHE_ROOT}/.cache"
ccache_dir="${CACHE_ROOT}/.ccache"
mkdir -p "${cache_dir}"
mkdir -p "${ccache_dir}"

# cache third_party
export WITH_CACHE=ON
export md5_content=$(cat \
            ${source_dir}/cmake/external/*.cmake \
            |md5sum | awk '{print $1}')
tp_cache_dir="${CACHE_ROOT}/third_party"
tp_cache_file_tar=${tp_cache_dir}/${md5_content}.tar
tp_cache_file=${tp_cache_file_tar}.xz

if [[ "${WITH_CACHE}" == "ON" ]]; then
  if [ ! -d ${tp_cache_dir} ];then
      mkdir -p ${tp_cache_dir}
  fi
  if [ -f ${tp_cache_file} ];then
      mkdir -p ${PWD}/build
      set +e
      tar xpf ${tp_cache_file} -C $PWD/build
      if [ $? -ne 0 ]; then
        rm ${tp_cache_file}
        rm -rf ${PWD}/build
      fi
      set -e
  else
      # clear the older tar files if MD5 has chanaged.
      update_cached_package=ON
      echo "cached thirdparty pacakge: FAILED"
      rm -rf ${tp_cache_dir}/*
  fi
fi

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
  -e "WITH_DOC=OFF" \
  -e "WITH_GPU=OFF" \
  -e "WITH_ROCM=OFF" \
  -e "WITH_TENSORRT=OFF" \
  -e "WITH_COVERAGE=OFF" \
  -e "COVERALLS_UPLOAD=OFF" \
  -e "CMAKE_BUILD_TYPE=Release" \
  -e "WITH_MKL=ON" \
  -e "WITH_AVX=ON" \
  -e "WITH_ARM=OFF" \
  -e "WITH_CACHE=ON" \
  -e "WITH_TEST=OFF" \
  -e "RUN_TEST=OFF" \
  -e "WITH_TESTING=OFF" \
  -e "WITH_DISTRIBUTE=ON" \
  -e "BRANCH=${PADDLE_BRANCH}" \
  -e "PADDLE_BRANCH=${PADDLE_BRANCH}" \
  -e "PADDLE_VERSION=${PADDLE_VERSION}" \
  -e "CMAKE_EXPORT_COMPILE_COMMANDS=ON" \
  -e "PY_VERSION=3.9" \
  -e "http_proxy=${proxy}" \
  -e "https_proxy=${proxy}" \
  -e "no_proxy=${no_proxy}" \
  ${PADDLE_DEV_NAME} \
  /bin/bash -c -x '
bash paddle/scripts/paddle_build.sh build_only;EXCODE=$?

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
cp ${source_dir}/dist/paddlepaddle*.whl ${WORKSPACE}/output

wget -q --no-proxy https://xly-devops.bj.bcebos.com/home/bos_new.tar.gz --no-check-certificate
tar xf bos_new.tar.gz -C ${WORKSPACE}/output

# Install dependency
python3 -m pip install bce-python-sdk==0.8.73 -i http://mirror.baidu.com/pypi/simple --trusted-host mirror.baidu.com

# Upload whl package to bos
cd ${WORKSPACE}/output
for file_whl in `ls *.whl` ;do
  python3 BosClient.py ${file_whl} paddle-device/${PADDLE_VERSION}/cpu
done

echo "Successfully uploaded to https://paddle-device.bj.bcebos.com/${PADDLE_VERSION}/cpu/${file_whl}"

set -ex
# local save third-paty directory if build success
if [ $? -eq 0 ] && [ "${WITH_CACHE}" == "ON" ] && [ "${update_cached_package}" == "ON" ];then
    cd ${source_dir}
    tar cf ${tp_cache_file_tar} -C build  third_party
    cd ${tp_cache_dir}
    xz -T `nproc` -0 ${tp_cache_file_tar}
fi
