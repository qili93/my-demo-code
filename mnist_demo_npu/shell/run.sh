#!/bin/bash
TARGET_EXE=mnist_demo
MODEL_NAME=mnist_model

#ANDROID_ABI=arm64-v8a # for armv8
ANDROID_ABI=armeabi-v7a # for armv7

LITE_DIR=../../libs/npu/${ANDROID_ABI}
MODEL_DIR="../../assets/models"
MODEL_PATH="$MODEL_DIR/${MODEL_NAME}"

WORK_SPACE=/data/local/tmp
# push to device work space
adb shell   rm -r ${WORK_SPACE}/*
adb push   ${LITE_DIR}/lib/.     ${WORK_SPACE}
adb push   ${MODEL_PATH}         ${WORK_SPACE}
adb push   build/${TARGET_EXE}    ${WORK_SPACE}
adb shell   chmod +x "${WORK_SPACE}/${TARGET_EXE}"

# define exe commands
EXE_SHELL="cd ${WORK_SPACE}; "
EXE_SHELL+="export GLOG_v=5;"
EXE_SHELL+="LD_LIBRARY_PATH=. ./${TARGET_EXE} ./${MODEL_NAME}"
echo ${EXE_SHELL}
# run
adb shell ${EXE_SHELL}
adb shell ls -l ${WORK_SPACE}

# pull optimized model
adb pull ${WORK_SPACE}/${MODEL_NAME}.nb ${MODEL_DIR}

# list models files
echo "ls -l ${MODEL_DIR}"
ls -l ${MODEL_DIR}