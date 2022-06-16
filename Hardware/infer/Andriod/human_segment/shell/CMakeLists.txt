cmake_minimum_required(VERSION 3.4.1)
project(human_seg_demo)

set(TARGET_EXE human_seg_demo)

# 1. path to Paddle-Lite lib
if(${PADDLE_LITE_DIR} STREQUAL "")
  message(FATAL_ERROR "PADDLE_LITE_DIR not set.")
  return()
endif()
message(STATUS "PADDLE_LITE_DIR=${PADDLE_LITE_DIR}")

# 2. compile options
add_definitions(-std=c++11 -g -O3 -pthread)
# add_compile_options(-std=c++11)
# set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
# set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 3. include Paddle-Lite directory
include_directories(${PADDLE_LITE_DIR}/cxx/include)
include_directories(${PADDLE_LITE_DIR}/third_party/mklml/include)


# 4. link paddle-lite libraries
if(NOT DEFINED USE_FULL_API)
  set(USE_FULL_API TRUE)
endif()

if(USE_FULL_API)
  add_definitions(-DUSE_FULL_API)
  add_library(paddle_api_shared SHARED IMPORTED GLOBAL)
  set_property(TARGET paddle_api_shared PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/cxx/lib/libpaddle_full_api_shared.so)
else()
  add_library(paddle_api_shared SHARED IMPORTED GLOBAL)
  set_property(TARGET paddle_api_shared PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/cxx/lib/libpaddle_light_api_shared.so)
endif()

add_library(mkl_iomp5 SHARED IMPORTED GLOBAL)
set_property(TARGET mkl_iomp5 PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/third_party/mklml/lib/libiomp5.so)

# 5.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

# 6. link libraries
target_link_libraries(${TARGET_EXE} paddle_api_shared mkl_iomp5)