#!/bin/bash
MODEL_FILE=simple_mnist.nb
WORK_SPACE=/data/local/tmp

ANDROID_ABI=arm64-v8a #armeabi-v7a

EXE_SHELL="cd ${WORK_SPACE}; "
EXE_SHELL+="export GLOG_v=5;"
EXE_SHELL+="LD_LIBRARY_PATH=. ./mnist_predict ./${MODEL_FILE}"
echo ${EXE_SHELL}

adb shell "rm -r ${WORK_SPACE}/*"
adb push ../arm64-v8a/lib/. ${WORK_SPACE}
adb push ../model/${MODEL_FILE} ${WORK_SPACE}
adb push build/mnist_predict ${WORK_SPACE}
adb shell chmod +x "${WORK_SPACE}/mnist_predict"
adb shell ${EXE_SHELL}