#!/bin/bash
set -xe

##### global environment #####
export proxy=http://172.19.57.45:3128

export WORKSPACE=/home/liqi27/develop/ipu
export CACHE_ROOT=/home/liqi27/develop/ipu/.cache/BUILD_CI_IPU

export PADDLE_BRANCH=develop
export PADDLE_VERSION=0.0.0
export PADDLE_DEV_NAME=registry.baidubce.com/device/paddle-ipu:poplar260

export whl_package=paddle-device/develop/ipu
export tgz_package=paddle-device/develop/ipu

##### local environment #####

set +x
export http_proxy=${proxy}
export https_proxy=${proxy}
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

set +x
export IPUOF_CONFIG_PATH=/opt/ipuof.conf
docker run  --rm -i \
  --shm-size=128G --network=host --ulimit memlock=-1:-1 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --cap-add=IPC_LOCK --device=/dev/infiniband/ --ipc=host \
  -v ${IPUOF_CONFIG_PATH}:/ipuof.conf -e IPUOF_CONFIG_PATH=/ipuof.conf \
  -v ${cache_dir}:/root/.cache \
  -v ${ccache_dir}:/root/.ccache \
  -v ${PADDLE_DIR}:/paddle \
  -w /paddle \
  -e "WITH_GPU=OFF" \
  -e "WITH_IPU=ON" \
  -e "POPLAR_DIR=/opt/poplar" \
  -e "POPART_DIR=/opt/popart" \
  -e "POPLAR_IPUMODEL=1" \
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
  ${PADDLE_DEV_NAME} \
  /bin/bash -c -x '
bash paddle/scripts/paddle_build.sh check_ipu_coverage 8;EXCODE=$?

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

# 安装依赖库
/home/liqi27/conda/envs/py27env/bin/python -m pip install pycrypto

cd ${WORKSPACE}/output
for file_whl in `ls *.whl` ;do
  /home/liqi27/conda/envs/py27env/bin/python ${push_file}  ${file_whl} ${whl_package}
done

set +x
# 如果执行成功，并且开启缓存，则本地保存第三方库
if [ $? -eq 0 ] && [ "${WITH_CACHE}" == "ON" ] && [ "${update_cached_package}" == "ON" ];then
    cd ${PADDLE_DIR}
    tar cf ${tp_cache_file_tar} -C build  third_party
    cd ${tp_cache_dir}
    xz -T `nproc` -0 ${tp_cache_file_tar}
fi
