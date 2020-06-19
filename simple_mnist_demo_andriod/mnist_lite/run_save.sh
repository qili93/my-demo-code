#!/bin/bash
MODEL_NAME=simple_mnist
WORK_SPACE=/data/local/tmp

ANDROID_ABI=arm64-v8a # for armv8
# ANDROID_ABI=armeabi-v7a # for armv7

LITE_DIR=../${ANDROID_ABI}

# delete model.nb before save
MODEL_DIR="../../assets/models"
MODEL_PATH="$MODEL_DIR/simple_mnist"
rm -rf "$MODEL_PATH.nb"
if [ $? -ne 0 ]; then
  echo "failed to delete $MODEL_PATH.nb"
else
  echo "succeed to delete $MODEL_PATH.nb"
fi

# push to device work space
adb shell     rm -r ${WORK_SPACE}/*
adb push    ${LITE_DIR}/lib/.                          ${WORK_SPACE}
adb push    ${MODEL_DIR}/simple_mnist     ${WORK_SPACE}
adb push    build/mnist_lite                          ${WORK_SPACE}
adb shell    chmod +x "${WORK_SPACE}/mnist_lite"

# define exe commands
EXE_SHELL="cd ${WORK_SPACE}; "
EXE_SHELL+="export GLOG_v=5;"
EXE_SHELL+="LD_LIBRARY_PATH=. ./mnist_lite ./${MODEL_NAME} save"
echo ${EXE_SHELL}
# run
adb shell ${EXE_SHELL}
adb shell ls -l ${WORK_SPACE}

# pull optimized model
adb pull ${WORK_SPACE}/${MODEL_NAME}.nb ${MODEL_DIR}

# list models files
echo "ls -l ${MODEL_DIR}"
ls -l ${MODEL_DIR}