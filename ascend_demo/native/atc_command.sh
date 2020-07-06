#!/bin/bash
cur_dir=$(pwd)

# set environment
export ASCEND_PATH=/usr/local/Ascend
export ASCEND_ATC_PATH=$ASCEND_PATH/atc
export PATH=/usr/local/python3.7.5/bin:${ASCEND_PATH}/atc/ccec_compiler/bin:${ASCEND_PATH}/atc/bin:$PATH
export PYTHONPATH=${ASCEND_PATH}/atc/python/site-packages/te:${ASCEND_PATH}/atc/python/site-packages/topi:${ASCEND_PATH}/atc/python/site-packages/auto_tune.egg/auto_tune:${ASCEND_PATH}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${ASCEND_PATH}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${ASCEND_PATH}/opp

atc --mode=1 --om=../assets/test.om  --json=../assets/test.json