#!/bin/bash
# set -ex
#######################################################################
# global variables, you can change them according to your requirements
#######################################################################
# absolute path current
function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}
# arm64-v8a or armeabi-v7a
# ANDROID_ABI=arm64-v8a
ANDROID_ABI=armeabi-v7a
# arm64-v8a => 23 or armeabi-v7a => 24
ANDROID_NATIVE_API_LEVEL=android-24 # armeabi-v7a
# ANDROID_NATIVE_API_LEVEL=android-23 # arm64-v8a 
# default with log
PADDLE_LITE_DIR=$(readlinkf ../../../libs/npu/${ANDROID_ABI}-log)
#######################################################################
# local variables, do not change them
#######################################################################
# build target
TARGET_EXE=long_sentence_prediction_demo
# model name
MODEL_NAME=tiny_long_sentence_prediction_fp32_fluid
# MODEL_TYPE=0 
MODEL_TYPE=0
# test name
TEST_NAME=tiny_input.txt
# model dir
MODEL_DIR="../assets/models"
# model path
MODEL_PATH=${MODEL_DIR}/${MODEL_NAME}
TEST_PATH="../assets/tests"/${TEST_NAME}
# workspace
WORK_SPACE=/data/local/tmp
#######################################################################
# pring usage
#######################################################################
function print_usage {
    set +x
    echo -e "\nUSAGE:"
    echo
    echo "----------------------------------------"
    echo -e "compile tiny publish so lib:"
    echo -e "   ./run_demo.sh --arm_abi=<abi> --with_log=<ON|OFF> tiny_demo"
    echo
    echo -e "compile full publish so lib:"
    echo -e "   ./run_demo.sh --arm_abi=<abi> --with_log=<ON|OFF> full_demo"
    echo
    echo -e "optional argument:"
    echo -e "--with_log: (OFF|ON); controls whether to print log information, default is ON"
    echo
    echo -e "argument choices:"
    echo -e "--arm_abi:\t armv8|armv7; default is armv7"
    echo
    echo -e "tasks:"
    echo -e "tiny_demo: demo with tiny publish library."
    echo -e "tiny_demo: demo with full publish library."
    echo "----------------------------------------"
    echo
}

#######################################################################
# compiling functions
#######################################################################
function run_demo {
    local abi=$1
    local log=$2
    local pub=$3

    if [ ${pub} == "tiny" ]; then
        MODEL_PATH=${MODEL_PATH}.nb
    fi
    echo "Building api: USE_FULL_API=<${USE_FULL_API}>"

    if [ ${abi} == "armv8" ]; then
        ANDROID_ABI=arm64-v8a
        ANDROID_NATIVE_API_LEVEL=android-23
    else
        ANDROID_ABI=armeabi-v7a
        ANDROID_NATIVE_API_LEVEL=android-24
    fi
    echo "Running target: ANDROID_ABI=<${ANDROID_ABI}> ANDROID_NATIVE_API_LEVEL=<${ANDROID_NATIVE_API_LEVEL}>"

    if [ ${log} == "ON" ]; then
        PADDLE_LITE_DIR_LIB=$(readlinkf ../../libs/npu/${ANDROID_ABI}-log/lib)
    else
        PADDLE_LITE_DIR_LIB=$(readlinkf ../../libs/npu/${ANDROID_ABI}-nolog/lib)
    fi
    echo "Running with library: PADDLE_LITE_DIR_LIB=<${PADDLE_LITE_DIR_LIB}>"

    # push to device work space
    # adb shell   "rm -r ${WORK_SPACE}/*"
    adb push   ${PADDLE_LITE_DIR_LIB}/.    ${WORK_SPACE}
    #adb push ../assets/models/. ${WORK_SPACE}
    adb push   ${MODEL_PATH}                  ${WORK_SPACE}
    #adb push   ${MODEL_PATH}.nb             ${WORK_SPACE}
    adb push   ${TEST_PATH}                       ${WORK_SPACE}
    adb push ./subgraph_custom_partition_config_file.txt ${WORK_SPACE}
    adb push   build/${TARGET_EXE}           ${WORK_SPACE}
    adb shell   chmod +x "${WORK_SPACE}/${TARGET_EXE}"
    # define exe commands
    EXE_SHELL="cd ${WORK_SPACE}; "
    EXE_SHELL+="export GLOG_v=5;"
    EXE_SHELL+="export SUBGRAPH_DISABLE_ONLINE_MODE=true;"
    EXE_SHELL+="export SUBGRAPH_CUSTOM_PARTITION_CONFIG_FILE=./subgraph_custom_partition_config_file.txt;"
    EXE_SHELL+="LD_LIBRARY_PATH=. ./${TARGET_EXE} ./${MODEL_NAME} ${MODEL_TYPE} ./${TEST_NAME}"
    echo ${EXE_SHELL}
    # run
    adb shell ${EXE_SHELL}
    adb shell ls -l ${WORK_SPACE}

    # pull optimized model
    adb pull ${WORK_SPACE}/${MODEL_NAME}.nb ${MODEL_DIR}

    # list models files
    echo "ls -l ${MODEL_DIR}"
    ls -l ${MODEL_DIR}
}
#######################################################################
# main functions
#######################################################################
function main {
    if [ -z "$1" ]; then
        print_usage
        exit -1
    fi

    # Parse command line.
    for i in "$@"; do
        case $i in
            --arm_abi=*)
                ARM_ABI="${i#*=}"
                shift
                ;;
            --with_log=*)
                WITH_LOG="${i#*=}"
                shift
                ;;
            tiny_demo)
                run_demo $ARM_ABI $WITH_LOG tiny
                shift
                ;;
            full_demo)
                run_demo $ARM_ABI $WITH_LOG full
                shift
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
    done
}

main $@