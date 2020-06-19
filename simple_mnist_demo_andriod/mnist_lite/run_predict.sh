#!/bin/bash
MODEL_NAME=simple_mnist
WORK_SPACE=/data/local/tmp

ANDROID_ABI=arm64-v8a #armeabi-v7a

EXE_SHELL="cd ${WORK_SPACE}; "
EXE_SHELL+="export GLOG_v=5;"
EXE_SHELL+="LD_LIBRARY_PATH=. ./mnist_save ./${MODEL_NAME}"
echo ${EXE_SHELL}

adb shell "rm -r ${WORK_SPACE}/*"
adb push ../arm64-v8a/lib/. ${WORK_SPACE}
adb push ../model/. ${WORK_SPACE}
adb push build/mnist_save ${WORK_SPACE}
adb shell chmod +x "${WORK_SPACE}/mnist_save"
adb shell ${EXE_SHELL}
#adb shell "cd ${WORK_SPACE}; export GLOG_v=5; export SUBGRAPH_DISABLE_ONLINE_MODE=true; export SUBGRAPH_CUSTOM_PARTITION_CONFIG_FILE=./subgraph_custom_partition_config_file.txt; LD_LIBRARY_PATH=. ./image_classification_demo ./${MODEL_NAME} ${MODEL_TYPE} ./${LABEL_NAME} ./${IMAGE_NAME}"
adb pull ${WORK_SPACE}/${MODEL_NAME}.nb ../model/



#!/bin/bash
MODEL_NAME=simple_mnist
WORK_SPACE=/data/local/tmp

ANDROID_ABI=arm64-v8a # for armv8
# ANDROID_ABI=armeabi-v7a # for armv7

LITE_DIR=../${ANDROID_ABI}

# delete model.nb before save
MODEL_DIR="../../assets/models"
MODEL_PATH="$MODEL_DIR/simple_mnist.nb"
if [ ! -f "$MODEL_PATH" ]; then
  echo "$MODEL_PATH NOT exist!!!"
  exit 1
fi

# push to device work space
adb shell     rm -r ${WORK_SPACE}/*
adb push    ${LITE_DIR}/lib/.          ${WORK_SPACE}
adb push    ${MODEL_PATH}         ${WORK_SPACE}
adb push    build/mnist_lite           ${WORK_SPACE}
adb shell     chmod +x "${WORK_SPACE}/mnist_lite"

# define exe commands
EXE_SHELL="cd ${WORK_SPACE}; "
EXE_SHELL+="export GLOG_v=5;"
EXE_SHELL+="LD_LIBRARY_PATH=. ./mnist_lite ./${MODEL_NAME}.nb predict"
echo ${EXE_SHELL}
# run
adb shell ${EXE_SHELL}
adb shell ls -l ${WORK_SPACE}