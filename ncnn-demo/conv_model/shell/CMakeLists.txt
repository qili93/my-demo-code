cmake_minimum_required(VERSION 3.10)
project(model_test)
set(TARGET_EXE model_test)

# 1. path to NCNN lib
if(${NCNN_INSTALL_DIR} STREQUAL "")
  message(FATAL_ERROR "NCNN_INSTALL_DIR not set.")
  return()
endif()
message(STATUS "NCNN_INSTALL_DIR=${NCNN_INSTALL_DIR}")

add_definitions(-std=c++11 -g -O3 -pthread)

# 2. link Paddle-Lite directory
# link_directories(${NCNN_INSTALL_DIR}/lib)
include_directories(${NCNN_INSTALL_DIR}/include)
# include_directories(${VULKAN_SDK}/include)

# 3.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

add_library(ncnn_lib STATIC IMPORTED GLOBAL)
set_property(TARGET ncnn_lib PROPERTY IMPORTED_LOCATION ${NCNN_INSTALL_DIR}/lib/libncnn.a)

target_link_libraries(${TARGET_EXE} ncnn_lib)

# /Library/Developer/CommandLineTools/usr/bin/c++ -o model-test -std=c++11 -g -O3 model_test.cc -I../include -L../lib/ -lncnn
