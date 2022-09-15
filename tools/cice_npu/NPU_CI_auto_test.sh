#!/bin/bash
set -xe

set +x
export http_proxy=${proxy}
export https_proxy=${proxy}
set -x

cd ${WORKSPACE}
rm -rf Paddle*
#yum install curl pigz -y 

# Check build.tar.gz in BOS
url="https://xly-devops.bj.bcebos.com/PR/build_npu_whl/${AGILE_PULL_ID}/${AGILE_REVISION}/npu.txt"
url_return=`curl -o /dev/null -s -w %{http_code} $url` 
if [ "$url_return" = '200' ];then
    echo "NO NPU operators files changed and no '[NPU]' found in PR title, skip NPU unit tests!" 
    EXCODE=0
    exit $EXCODE
else
    wget -q -O Paddle.tar.gz https://xly-devops.bj.bcebos.com/PR/build_npu_whl/${AGILE_PULL_ID}/${AGILE_REVISION}/build.tar.gz --no-check-certificate
    tar --use-compress-program=pigz -xpf Paddle.tar.gz
    cd Paddle
fi
export PADDLE_DIR="${WORKSPACE}/Paddle"

set -x
docker run --rm -i \
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
  ${PADDLE_DEV_NAME} \
  /bin/bash -c -x '
bash /workspace/npu_dev/Paddle/paddle/scripts/paddle_build.sh test'

EXCODE=$?

set -x
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
