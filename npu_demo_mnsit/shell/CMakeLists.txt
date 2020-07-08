cmake_minimum_required(VERSION 3.4.1)
project(mnist_demo)

set(TARGET_EXE mnist_demo)

# 1. path to Paddle-Lite lib
if(${PADDLE_LITE_DIR} STREQUAL "")
  message(FATAL_ERROR "PADDLE_LITE_DIR not set.")
  return()
endif()
message(STATUS "PADDLE_LITE_DIR=${PADDLE_LITE_DIR}")

# 2. link Paddle-Lite directory
#link_directories(${PADDLE_LITE_DIR}/lib)
include_directories(${PADDLE_LITE_DIR}/include)

# 3.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

# 4. add andriod ndk log
find_library(log-lib log)
message(STATUS "log-lib=${log-lib}")

# 5. add hiai
add_library(npu_ddk_hiai SHARED IMPORTED GLOBAL)
set_property(TARGET npu_ddk_hiai PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/lib/libhiai.so)

add_library(npu_ddk_ir SHARED IMPORTED GLOBAL)
set_property(TARGET npu_ddk_ir PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/lib/libhiai_ir.so)

add_library(npu_ddk_ir_build SHARED IMPORTED GLOBAL)
set_property(TARGET npu_ddk_ir_build PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/lib/libhiai_ir_build.so)

# for 320 only
# add_library(npu_ddk_hcl SHARED IMPORTED GLOBAL)
# set_property(TARGET npu_ddk_hcl PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/lib/libhcl.so)

# 310
set(npu_runtime_libs npu_ddk_hiai CACHE INTERNAL "npu ddk runtime libs")
# 320
#set(npu_runtime_libs npu_ddk_hiai npu_ddk_hcl CACHE INTERNAL "npu ddk runtime libs")

set(npu_builder_libs npu_ddk_ir npu_ddk_ir_build CACHE INTERNAL "npu ddk builder libs")

# 5. link paddle-lite libraries
if(NOT DEFINED USE_FULL_API)
  set(USE_FULL_API TRUE)
endif()

if(USE_FULL_API)
  add_definitions(-DUSE_FULL_API)
  add_library(paddle_api_shared SHARED IMPORTED GLOBAL)
  set_property(TARGET paddle_api_shared PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/lib/libpaddle_full_api_shared.so)
else()
  add_library(paddle_api_shared SHARED IMPORTED GLOBAL)
  set_property(TARGET paddle_api_shared PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/lib/libpaddle_light_api_shared.so)
endif()

target_link_libraries(${TARGET_EXE} paddle_api_shared ${log-lib} ${npu_builder_libs} ${npu_runtime_libs})