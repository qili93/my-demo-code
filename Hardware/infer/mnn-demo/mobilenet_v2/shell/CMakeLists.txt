cmake_minimum_required(VERSION 3.10)
project(model_test)
set(TARGET_EXE model_test)

# 1. path to MNN lib
if(${MNN_LIB_DIR} STREQUAL "")
  message(FATAL_ERROR "MNN_LIB_DIR not set.")
  return()
endif()
message(STATUS "MNN_LIB_DIR=${MNN_LIB_DIR}")
# 2. path to MNN inc
if(${MNN_INC_DIR} STREQUAL "")
  message(FATAL_ERROR "MNN_INC_DIR not set.")
  return()
endif()
message(STATUS "MNN_INC_DIR=${MNN_INC_DIR}")

add_definitions(-std=c++11 -g -O3 -pthread)

include_directories(${MNN_INC_DIR})

# 3.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

add_library(mnn_lib SHARED IMPORTED GLOBAL)
set_property(TARGET mnn_lib PROPERTY IMPORTED_LOCATION ${MNN_LIB_DIR}/libMNN.dylib)

target_link_libraries(${TARGET_EXE} mnn_lib)

# /Library/Developer/CommandLineTools/usr/bin/c++ -o model-test -std=c++11 -g -O3 model_test.cc -I../include -L../lib/ -lncnn
