#!/bin/bash
cur_dir=$(pwd)

# set environment
export ASCEND_PATH=/usr/local/Ascend
export ASCEND_ATC_PATH=$ASCEND_PATH/atc
export ASCEND_OPP_PATH=$ASCEND_PATH/opp
export PATH=/usr/local/python3.7.5/bin:${ASCEND_ATC_PATH}/ccec_compiler/bin:${ASCEND_ATC_PATH}/bin:${ASCEND_PATH}/toolkit/tools/ide_daemon/bin:${ASCEND_PATH}/driver/tools:$PATH
export PYTHONPATH=${ASCEND_ATC_PATH}/python/site-packages/te:${ASCEND_ATC_PATH}/python/site-packages/topi:${ASCEND_ATC_PATH}/python/site-packages/auto_tune.egg:${ASCEND_ATC_PATH}/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${ASCEND_PATH}/acllib/lib64:${ASCEND_ATC_PATH}/lib64:${ASCEND_PATH}/toolkit/lib64:${ASCEND_PATH}/add-ons:$LD_LIBRARY_PATH
export SOC_VERSION=Ascend310

# run demo
./build/bias_add_run