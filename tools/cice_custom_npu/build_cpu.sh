#!/bin/bash
set -xe

##### global environment, delete from scirpt and add into ci config #####

export WORKSPACE=/workspace/cpu-dev
export CACHE_ROOT=/workspace/cpu-dev

export PADDLE_BRANCH=develop
export PADDLE_COMMIT=develop
export PADDLE_VERSION=0.0.0

export PADDLE_DEV_NAME=registry.baidubce.com/device/paddle-npu:cann600-$(uname -m)-gcc82

export whl_package=paddle-device/develop/cpu
export tgz_package=paddle-device/develop/cpu

##### local environment #####

set +x
export http_proxy=${proxy}
export https_proxy=${proxy}
export ftp_proxy=${proxy}
export no_proxy=bcebos.com
set -x

mkdir -p ${WORKSPACE}
mkdir -p ${CACHE_ROOT}

cd ${WORKSPACE}
sleep 10s
rm -rf Paddle*
rm -rf output*

git clone -b ${PADDLE_BRANCH} https://github.com/PaddlePaddle/Paddle.git
cd Paddle
# git checkout tags/${PADDLE_TAG}
# git checkout ${PADDLE_COMMIT}
# git pull origin pull/48774/head
git log --oneline -20

export PADDLE_DIR="${WORKSPACE}/Paddle"
export WITH_CACHE=ON
export md5_content=$(cat \
            ${PADDLE_DIR}/cmake/external/*.cmake \
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

cache_dir="${CACHE_ROOT}/.cache"
ccache_dir="${CACHE_ROOT}/.ccache"

if [ ! -d "${cache_dir}" ];then
    mkdir -p "${cache_dir}"
fi
if [ ! -d "${ccache_dir}" ];then
    mkdir -p "${ccache_dir}"
fi

docker pull ${PADDLE_DEV_NAME}

# py37
echo "Start build python37 whl "
set -ex
docker run --network=host --rm -i \
  -v ${cache_dir}:/root/.cache \
  -v ${ccache_dir}:/root/.ccache \
  -v ${PADDLE_DIR}:/paddle \
  -w /paddle \
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
  -e "PADDLE_VERSION=${PADDLE_VERSION}" \
  -e "PADDLE_BRANCH=${PADDLE_BRANCH}" \
  -e "BRANCH=${PADDLE_BRANCH}" \
  -e "WITH_TEST=OFF" \
  -e "RUN_TEST=OFF" \
  -e "WITH_TESTING=OFF" \
  -e "WITH_DISTRIBUTE=ON" \
  -e "CMAKE_EXPORT_COMPILE_COMMANDS=ON" \
  -e "PY_VERSION=3.7" \
  -e "http_proxy=${proxy}" \
  -e "https_proxy=${proxy}" \
  -e "no_proxy=bcebos.com" \
  ${PADDLE_DEV_NAME} \
  /bin/bash -c -x '
paddle/scripts/paddle_build.sh build_only;EXCODE=$?

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
cp ${PADDLE_DIR}/build/python/dist/paddlepaddle*.whl ${WORKSPACE}/output

wget -q --no-proxy https://xly-devops.bj.bcebos.com/home/bos_new.tar.gz --no-check-certificate
tar xf bos_new.tar.gz -C ${WORKSPACE}/output

# Install dependency
python3 -m pip install bce-python-sdk -i http://mirror.baidu.com/pypi/simple --trusted-host mirror.baidu.com

# Upload paddlepaddle-rocm whl package to paddle-device/develop/dcu1
cd ${WORKSPACE}/output
for file_whl in `ls *.whl` ;do
  python3 BosClient.py ${file_whl} ${whl_package}
done

set -ex
# local save third-paty directory if build success
if [ $? -eq 0 ] && [ "${WITH_CACHE}" == "ON" ] && [ "${update_cached_package}" == "ON" ];then
    cd ${PADDLE_DIR}
    tar cf ${tp_cache_file_tar} -C build  third_party
    cd ${tp_cache_dir}
    xz -T `nproc` -0 ${tp_cache_file_tar}
fi
