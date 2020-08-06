#!/bin/bash
cur_dir=$(pwd)

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

#######################################
# Local Settings: paddle-lite env
#######################################
BASE_REPO_PATH=/workspace/Paddle-Lite
PADDLE_LITE_DIR=$BASE_REPO_PATH/build.opt/lite/api

#######################################
# Local Settings: model info
#######################################
MODEL_DIR=$cur_dir/models
MODEL_NAME=yolov3.fruits

MODEL_FILE=$MODEL_DIR/$MODEL_NAME/model
PARAM_FILE=$MODEL_DIR/$MODEL_NAME/params


#######################################
# Build commands, do not change them
#######################################
cd $PADDLE_LITE_DIR
./opt \
    --model_file=$MODEL_FILE \
    --param_file=$PARAM_FILE \
    --optimize_out_type=protobuf \
    --optimize_out=$MODEL_DIR \
    --valid_targets=x86,huawei_ascend_npu \
    --record_tailoring_info=true

# ./opt --print_model_ops=true --model_file=$MODEL_FILE --param_file=$PARAM_FILE --valid_targets=huawei_ascend_npu
# ./opt --print_model_ops=true --model_file=$MODEL_FILE --param_file=$PARAM_FILE --valid_targets=x86,huawei_ascend_npu
cd -

