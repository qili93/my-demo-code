cmake_minimum_required(VERSION 3.10)

project(human_seg_demo)

set(TARGET_EXE human_seg_demo)

# 1. path to Paddle-Lite lib
if(${PADDLE_LITE_DIR} STREQUAL "")
  message(FATAL_ERROR "PADDLE_LITE_DIR not set.")
  return()
endif()
message(STATUS "PADDLE_LITE_DIR=${PADDLE_LITE_DIR}")

# set(PADDLE_LIB_DIR ${PADDLE_LITE_DIR})
# set(MKLML_LIB_DIR ${PADDLE_LITE_DIR})
set(PADDLE_LIB_DIR ${PADDLE_LITE_DIR}/cxx)
set(MKLML_LIB_DIR ${PADDLE_LITE_DIR}/third_party/mklml)
message(STATUS "PADDLE_LIB_DIR=${PADDLE_LIB_DIR}")
message(STATUS "MKLML_LIB_DIR=${MKLML_LIB_DIR}")

# 2. compile options
add_definitions(-std=c++11 -g -O3 -pthread)
# add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 3. include Paddle-Lite directory
include_directories(${PADDLE_LIB_DIR}/include)

# 4. link paddle-lite libraries
if(NOT DEFINED USE_FULL_API)
  set(USE_FULL_API TRUE)
endif()

if(APPLE)
  set(LITE_FULL_LIB_NAME "libpaddle_full_api_shared.dylib")
  set(LITE_TINY_LIB_NAME "libpaddle_light_api_shared.dylib")
  set(LITE_IOMP5_LIB_NAME "libiomp5.dylib")
else(APPLE)
  set(LITE_FULL_LIB_NAME "libpaddle_full_api_shared.so")
  set(LITE_TINY_LIB_NAME "libpaddle_light_api_shared.so")
  set(LITE_IOMP5_LIB_NAME "libiomp5.so")
endif(APPLE)

message(STATUS "LITE_FULL_LIB_NAME=${LITE_FULL_LIB_NAME}")
message(STATUS "LITE_TINY_LIB_NAME=${LITE_TINY_LIB_NAME}")
message(STATUS "LITE_IOMP5_LIB_NAME=${LITE_IOMP5_LIB_NAME}")
message(STATUS "LITE_MKLML_LIB_NAME=${LITE_MKLML_LIB_NAME}")

if(USE_FULL_API)
  add_definitions(-DUSE_FULL_API)
  add_library(paddle_api_shared SHARED IMPORTED GLOBAL)
  set_property(TARGET paddle_api_shared PROPERTY IMPORTED_LOCATION ${PADDLE_LIB_DIR}/lib/${LITE_FULL_LIB_NAME})
else()
  add_library(paddle_api_shared SHARED IMPORTED GLOBAL)
  set_property(TARGET paddle_api_shared PROPERTY IMPORTED_LOCATION ${PADDLE_LIB_DIR}/lib/${LITE_TINY_LIB_NAME})
endif()

add_library(mkl_iomp5 SHARED IMPORTED GLOBAL)
set_property(TARGET mkl_iomp5 PROPERTY IMPORTED_LOCATION ${MKLML_LIB_DIR}/lib/${LITE_IOMP5_LIB_NAME})

# 5.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

# 6. link libraries
target_link_libraries(${TARGET_EXE} paddle_api_shared mkl_iomp5)
