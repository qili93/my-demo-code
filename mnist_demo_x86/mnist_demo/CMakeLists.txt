cmake_minimum_required(VERSION 3.4.1)
project(mnist_lite)

set(TARGET_EXE mnist_lite)
set(TARGET_ARCH x86)

# 1. path to Paddle-Lite lib
set(LITE_DIR "${PROJECT_SOURCE_DIR}/../${TARGET_ARCH}")
set(MKLML_DIR "${PROJECT_SOURCE_DIR}/../${TARGET_ARCH}/mklml")
message(STATUS "LITE_DIR=${LITE_DIR}")

# 2. link Paddle-Lite directory
link_directories(${LITE_DIR}/lib ${MKLML_DIR}/lib)
include_directories(${LITE_DIR}/include ${MKLML_DIR}/include)

# 3. compile options
add_definitions(-std=c++11 -g -O3 -pthread)

# 4.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

# 5. link libraries
if(NOT DEFINED USE_FULL_API)
  set(USE_FULL_API TRUE)
endif()

if(USE_FULL_API)
  add_definitions(-DUSE_FULL_API)
  target_link_libraries(${TARGET_EXE} -lpaddle_full_api_shared)
else()
  target_link_libraries(${TARGET_EXE} -lpaddle_light_api_shared)
endif()
target_link_libraries(${TARGET_EXE} -liomp5)
target_link_libraries(${TARGET_EXE} -ldl)