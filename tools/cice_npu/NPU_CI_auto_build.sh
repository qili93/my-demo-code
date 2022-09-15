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

# --------------------------------------------------

mkdir -p ${WORKSPACE}

cd ${WORKSPACE}
sleep 10s
rm -rf Paddle*

# wget paddle
set -x
wget -q -O Paddle.tar.gz https://xly-devops.bj.bcebos.com/PR/Paddle/${AGILE_PULL_ID}/${AGILE_REVISION}/Paddle.tar.gz --no-check-certificate
set -x
tar xf Paddle.tar.gz
cd Paddle
git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"

set -xe

git remote add upstream https://github.com/PaddlePaddle/Paddle.git
git pull upstream ${PADDLE_BRANCH}
git log --oneline -20

set -x
export "WITH_GPU=OFF" 
export "WITH_ASCEND=OFF" 
export "WITH_ASCEND_CL=ON" 
export "WITH_ASCEND_INT64=ON" 
export "WITH_ARM=OFF" 
export "WITH_TENSORRT=OFF" 
export "WITH_COVERAGE=OFF" 
export "COVERALLS_UPLOAD=OFF" 
export "CMAKE_BUILD_TYPE=Release" 
export "WITH_MKL=ON" 
export "WITH_CACHE=ON" 
export "PADDLE_VERSION=${PADDLE_VERSION}" 
export "PADDLE_BRANCH=${PADDLE_BRANCH}" 
export "BRANCH=${PADDLE_BRANCH}" 
export "WITH_TEST=ON" 
export "RUN_TEST=ON" 
export "WITH_TESTING=ON" 
export "WITH_DISTRIBUTE=ON" 
export "ON_INFER=ON" 
export "PYTHON_ABI=conda-python3.7" 
export "CTEST_PARALLEL_LEVEL=1" 
export "PY_VERSION=3.7" 
export "CI_SKIP_CPP_TEST=OFF"
export "http_proxy=${proxy}" 
export "https_proxy=${proxy}" 
export "GIT_PR_ID=${AGILE_PULL_ID}" 
export "GITHUB_API_TOKEN=${GITHUB_API_TOKEN}"

export work_dir="${WORKSPACE}/Paddle"
mkdir -p /home/data/cfs/.cache/npu-build
export CACHE_DIR=/home/data/cfs/.cache/npu-build
mkdir -p /home/data/cfs/.ccache/npu-build
if [ -f '/home/data/gzcfs/gz.txt' ];then
    export CCACHE_DIR=/home/data/gzcfs/.ccache/npu-build
else
    export CCACHE_DIR=/home/data/cfs/.ccache/npu-build
fi

export WITH_INCREMENTAL_COVERAGE=OFF
export CCACHE_STATSLOG=/paddle/build/.stats.log
export INFERENCE_DEMO_INSTALL_DIR=/home/data/cfs/.cache/build

# PR_CI_Coverage 流水线因为打开了 WITH_TESTING 和 WITH_COVERAGE，
# 所以需要编译更多目标文件，也比较大。
# ccache 默认空间大小是 5GB，需要扩容至30GB。
# 当执行 ccache -s的结果中，cleanups performed一项很高时，就可以考虑扩容。
export CCACHE_MAXSIZE=150G
export CCACHE_LIMIT_MULTIPLE=0.8


# md5_content 所有cmake的md5
# tar_dir 存放third_party目录
# bce_file bce路径
# file_tar 缓存路径

#git branch
#cat ${work_dir}/cmake/external/gloo.cmake
export md5_content=$(cat \
            ${work_dir}/cmake/external/*.cmake cmake/third_party.cmake\
            |md5sum | awk '{print $1}')
tar_dir="/home/data/cfs/third_party/PR_CI_NPU"
file_tar="${tar_dir}/${md5_content}.tar.gz"
#bce_file="/home/bce-python-sdk-0.8.33/BosClient.py"

cd ${WORKSPACE}/Paddle

if [ ! -d "$tar_dir" ];then
    mkdir -p ${tar_dir}
fi 
 
    #判断本地有没有third_party缓存，没有就去bos拉，如果拉下来就使用，没有拉下来就设置update_cached_package=on，执行成功后会判断这个变量为on就会往bos上推送third_party缓存。
if [ ! -f "${file_tar}" ];then
    update_cached_package=ON 
else
    mkdir -p ${work_dir}/build
    set +e
    tar -xpf ${file_tar} -C ${work_dir}/build || export update_cached_package=ON
    set -e
fi

pip config set global.cache-dir "/home/data/cfs/.cache/pip"
pip install --upgrade pip
pip install -r "${work_dir}/python/requirements.txt"

bash -x ${work_dir}/paddle/scripts/paddle_build.sh build_only;EXCODE=$?|| true

EXCODE=0
#如果执行成功，并且开启缓存，并且bos上没有缓存的数据，则打包third_party目录，并且推送到bos上
if [ ${EXCODE} -eq 0 ];then
    set +x
    cd ${WORKSPACE}
    wget -q --no-proxy https://xly-devops.bj.bcebos.com/home/bos_new.tar.gz
    tar xf bos_new.tar.gz
    set -x

    cd ${WORKSPACE}/Paddle
    pip install bce-python-sdk

    # get npu py or npu cc file changes
    git diff --name-only remotes/upstream/$BRANCH
    npu_cc_changes=$(git diff --name-only remotes/upstream/$BRANCH | grep "op_npu.cc" || true)
    npu_py_changes=$(git diff --name-only remotes/upstream/$BRANCH | grep "op_npu.py" || true)
    # get PR name
    npu_pr_tile=$(curl https://github.com/PaddlePaddle/Paddle/pull/${GIT_PR_ID} 2>/dev/null | grep "<title>" | grep "NPU" || true)
    if [ -z "${npu_cc_changes}" ] && [ -z "${npu_py_changes}" ] && [ -z "${npu_pr_tile}" ]; then
        echo "NO NPU operators files changed and no '[NPU]' found in PR title, skip NPU unit tests!"
        cd ${WORKSPACE}
        echo "this is npu test" > npu.txt
        python BosClient.py npu.txt xly-devops/PR/build_npu_whl/${AGILE_PULL_ID}/${AGILE_REVISION}

        if [ "${update_cached_package}" == ON ];then
            cd ${work_dir}
            tar --use-compress-program=pigz -cpf ${file_tar} -C build  third_party 
        fi
        exit 0
    fi
    
    cd ${WORKSPACE}
    tar --use-compress-program="pigz -1" -cpPf build.tar.gz ./Paddle
    python BosClient.py build.tar.gz xly-devops/PR/build_npu_whl/${AGILE_PULL_ID}/${AGILE_REVISION}
fi

# 输出退出码代表的错误信息
set +x
if [[ $EXCODE -eq 0 ]];then
    echo "Congratulations!  Your PR passed the paddle-build."
elif [[ $EXCODE -eq 4 ]];then
    echo "Sorry, your code style check failed."
elif [[ $EXCODE -eq 5 ]];then
    echo "Sorry, API's example code check failed."
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
