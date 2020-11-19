#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: please change accrodingly
#######################################
ANDROID_NDK=~/Library/android-ndk-r17c
ANDROID_ABI=arm64-v8a
# ANDROID_ABI=armeabi-v7a
ANDROID_NATIVE_API_LEVEL=android-23
if [ $ANDROID_ABI == "armeabi-v7a" ]; then
    ANDROID_NATIVE_API_LEVEL=android-24
fi

# paddle lite dir
PADDLE_LITE_DIR=$(readlinkf ../../inference_lite_lib.android.armv8)

#######################################
# Model variables, do not change them
#######################################
# build target
TARGET_EXE=model_test
# model name
MODEL_NAME=face_detect_fp32
# model dir
MODEL_DIR=$(readlinkf ../assets/models)
# workspace
WORK_SPACE=/data/local/tmp

#######################################
# Running commands, do not change
#######################################
# push to device work space
adb shell  "rm -r ${WORK_SPACE}/*"
adb push   ${PADDLE_LITE_DIR}/lib/.     ${WORK_SPACE}
adb push   ${MODEL_DIR}/.               ${WORK_SPACE}
adb push   build/${TARGET_EXE}          ${WORK_SPACE}
adb shell  chmod +x "${WORK_SPACE}/${TARGET_EXE}"
# define exe commands
EXE_SHELL="cd ${WORK_SPACE}; "
EXE_SHELL+="export GLOG_v=5;"
EXE_SHELL+="LD_LIBRARY_PATH=. ./${TARGET_EXE} ./${MODEL_NAME}"
echo ${EXE_SHELL}
# run
adb shell ${EXE_SHELL}

#######################################
# Post processing commands, do not change
#######################################
# show files of /data/local/tmp
echo ""
echo "ls -l ${WORK_SPACE}"
adb shell ls -l ${WORK_SPACE}
echo ""

# pull optimized model
adb pull ${WORK_SPACE}/${MODEL_NAME}.nb ${MODEL_DIR}

# list models files
echo ""
echo "ls -l ${MODEL_DIR}"
ls -l ${MODEL_DIR}