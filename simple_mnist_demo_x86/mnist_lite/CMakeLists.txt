cmake_minimum_required(VERSION 3.4.1)
project(mnist_lite)

set(TARGET mnist_lite)

# 1. path to Paddle-Lite lib
set(LITE_DIR "${PROJECT_SOURCE_DIR}/../x86")
set(MKLML_DIR "${PROJECT_SOURCE_DIR}/../x86/mklml")

# 2. link Paddle-Lite directory
link_directories(${LITE_DIR}/lib ${MKLML_DIR}/lib)
include_directories(${LITE_DIR}/include ${MKLML_DIR}/include)

# 3. compile options
add_definitions(-std=c++11 -g -O3 -pthread)

# 4.add executable output
add_executable(${TARGET} ${TARGET}.cc)

# 5. link libraries
if(NOT DEFINED USE_FULL_API)
  set(USE_FULL_API TRUE)
endif()

if(USE_FULL_API)
  target_link_libraries(${TARGET} -lpaddle_full_api_shared)
else()
  target_link_libraries(${TARGET} -lpaddle_light_api_shared)
endif()
target_link_libraries(${TARGET} -liomp5)
target_link_libraries(${TARGET} -ldl)