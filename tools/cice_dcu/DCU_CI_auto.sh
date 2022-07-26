#!/bin/bash
set -xe

##### global environment #####
export proxy=http://172.19.57.45:3128

export WORKSPACE=/workspace/rocm-dev
export CACHE_ROOT=/workspace/rocm-dev/.cache/BUILD_CI_ROCM

export PADDLE_BRANCH=develop
export PADDLE_VERSION=0.0.0
export PADDLE_DEV_NAME=registry.baidubce.com/device/paddle-dcu:rocm4.0.1

export whl_package=paddle-device/develop/dcu1
export tgz_package=paddle-device/develop/dcu1

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

git clone -b ${PADDLE_BRANCH} https://github.com/PaddlePaddle/Paddle.git
cd Paddle
# git checkout tags/${PADDLE_TAG}
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

# build only
echo "Start build rocm whl"
set +x
docker run --privileged --network=host --rm -i \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v ${cache_dir}:/root/.cache \
  -v ${ccache_dir}:/root/.ccache \
  -v ${PADDLE_DIR}:/paddle \
  -w /paddle \
  -e "WITH_GPU=OFF" \
  -e "WITH_ROCM=ON" \
  -e "WITH_TENSORRT=OFF" \
  -e "WITH_COVERAGE=OFF" \
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
  -e "PYTHON_ABI=conda-python3.7" \
  -e "PY_VERSION=3.7" \
  -e "http_proxy=${proxy}" \
  -e "https_proxy=${proxy}" \
  -e "no_proxy=bcebos.com" \
  ${PADDLE_DEV_NAME} \
  /bin/bash -c -x '
paddle/scripts/paddle_build.sh build_only 8;EXCODE=$?

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

wget -q --no-proxy -O ${WORKSPACE}/bce_whl.tar.gz  https://paddle-docker-tar.bj.bcebos.com/home/bce_whl.tar.gz --no-check-certificate
tar xf ${WORKSPACE}/bce_whl.tar.gz -C ${WORKSPACE}/output
push_file=${WORKSPACE}/output/bce-python-sdk-0.8.27/BosClient.py

# Install dependency
/workspace/conda/envs/py27env/bin/python -m pip install pycrypto

# Upload paddlepaddle-rocm whl package to paddle-device/develop/dcu1
cd ${WORKSPACE}/output
for file_whl in `ls *.whl` ;do
  /workspace/conda/envs/py27env/bin/python ${push_file} ${file_whl} ${whl_package}
done

# prepare dockerfile
echo "FROM registry.baidubce.com/device/paddle-dcu:rocm4.0.1"  > Dockerfile.dcu401
echo "MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>" >> Dockerfile.dcu401
echo -e "\nWORKDIR /opt\nCOPY paddlepaddle*whl /opt" >> Dockerfile.dcu401
echo "RUN /opt/conda/bin/pip install paddlepaddle*whl" >> Dockerfile.dcu401
echo "RUN rm -rf /opt/paddlepaddle*whl" >> Dockerfile.dcu401
echo -e "\nEXPOSE 22" >> Dockerfile.dcu401
# build new image with paddlepaddle installed
docker build --network=host -f Dockerfile.dcu401 \
       -t registry.baidubce.com/device/paddlepaddle:develop-dcu-rocm4.0.1 .
docker push registry.baidubce.com/device/paddlepaddle:develop-dcu-rocm4.0.1

set +x
# local save third-paty directory if build success
if [ $? -eq 0 ] && [ "${WITH_CACHE}" == "ON" ] && [ "${update_cached_package}" == "ON" ];then
    cd ${PADDLE_DIR}
    tar cf ${tp_cache_file_tar} -C build  third_party
    cd ${tp_cache_dir}
    xz -T `nproc` -0 ${tp_cache_file_tar}
fi
