# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# target
set(CUSTOM_OBJECT_NAME "main")

# project information
project(acl_op)

# Compile options
add_compile_options(-std=c++11)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../../outputs")
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../outputs")
# set(CMAKE_INSTALL_PREFIX "../../../run")
# set(CMAKE_OUTPUT_DIR "out")

# Header path
set(ACLLIB_ROOT_PATH /usr/local/Ascend/ascend-toolkit/latest)
message(STATUS "ACLLIB_ROOT_PATH=${ACLLIB_ROOT_PATH}")
include_directories(
        ${ACLLIB_ROOT_PATH}/acllib/include
        ${ACLLIB_ROOT_PATH}/fwkacllib/include
        ../inc
)

# add host lib path
link_directories(
        ${ACLLIB_ROOT_PATH}/acllib/lib64/stub
        ${ACLLIB_ROOT_PATH}/fwkacllib/lib64/stub
)

add_executable(${CUSTOM_OBJECT_NAME}
        operator_desc.cpp
        op_runner.cpp
        main.cpp)

target_link_libraries(${CUSTOM_OBJECT_NAME}
        ascendcl
        acl_op_compiler
        stdc++)

# install(TARGETS ${CUSTOM_OBJECT_NAME} DESTINATION ${CMAKE_OUTPUT_DIR})

# custom command process om conversion
# [TODO] Please uncomment the following part when your project is ready to convert single-op om.

#[[
message(STATUS "SOC_VERSION=${SOC_VERSION}")
set(ASCEND_HOME $ENV{ADK_PATH})
add_custom_command(TARGET ${CUSTOM_OBJECT_NAME}
    POST_BUILD
    COMMAND export LD_LIBRARY_PATH="${ASCEND_HOME}/atc/python/site-packages/te.egg/lib:${ASCEND_HOME}/atc/lib64:${ASCEND_HOME}/acllib_lhisilinux/acllib/lib64/stub" &&
        export PATH="${PYTHON3_BIN}:${ASCEND_HOME}/atc/ccec_compiler/bin:${ASCEND_HOME}/atc/bin:$ENV{PATH}" &&
        export ASCEND_OPP_PATH="${ASCEND_HOME}/opp" &&
        export PYTHONPATH="$ENV{PYTHONPATH}:${ASCEND_HOME}/atc/python/site-packages/te.egg:${ASCEND_HOME}/atc/python/site-packages/topi.egg" &&
        ${ASCEND_HOME}/atc/bin/atc --singleop=$ENV{PROJECT_PATH}/tbe/testcases/acl_op/run/out/test_data/config/op_list.txt --soc_version=${SOC_VERSION} --output=$ENV{PROJECT_PATH}/tbe/testcases/acl_op/run/out/op_models> atc.log
    USES_TERMINAL
    COMMENT "single-op om conversion, if got failing log below, please check build/intermediates/acl_op/atc.log for detail."
)
]]


