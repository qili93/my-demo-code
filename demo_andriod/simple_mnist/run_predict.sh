#!/bin/bash
MODEL_NAME=simple_mnist
WORK_SPACE=/data/local/tmp

ANDROID_ABI=arm64-v8a # for armv8
# ANDROID_ABI=armeabi-v7a # for armv7

LITE_DIR=../${ANDROID_ABI}

# delete model.nb before save
MODEL_DIR="../../assets/models"
MODEL_PATH="$MODEL_DIR/simple_mnist.nb"
if [ ! -f "$MODEL_PATH" ]; thens
  echo "$MODEL_PATH NOT exist!!!"
  exit 1
fi

# push to device work space
adb shell   rm -r ${WORK_SPACE}/*
adb push    ${LITE_DIR}/lib/.     ${WORK_SPACE}
adb push    ${MODEL_PATH}         ${WORK_SPACE}
adb push    build/simple_mnist    ${WORK_SPACE}
adb shell   chmod +x "${WORK_SPACE}/simple_mnist"

# define exe commands
EXE_SHELL="cd ${WORK_SPACE}; "
EXE_SHELL+="export GLOG_v=5;"
EXE_SHELL+="LD_LIBRARY_PATH=. ./simple_mnist ./${MODEL_NAME}.nb predict"
echo ${EXE_SHELL}
# run
adb shell ${EXE_SHELL}
adb shell ls -l ${WORK_SPACE}