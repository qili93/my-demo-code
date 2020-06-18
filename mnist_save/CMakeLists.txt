cmake_minimum_required(VERSION 3.4.1)

project(mnist_save)

if(${LIB_DIR_PATH} STREQUAL "")
  message(FATAL_ERROR "LIB_DIR_PATH not set.")
  return()
endif()

include_directories(${LIB_DIR_PATH}/include)
add_executable(mnist_save mnist_save.cc)

find_library(log-lib log)
message(STATUS "log-lib=${log-lib}")

add_library(npu_ddk_hiai SHARED IMPORTED GLOBAL)
set_property(TARGET npu_ddk_hiai PROPERTY IMPORTED_LOCATION ${LIB_DIR_PATH}/lib/libhiai.so)

add_library(npu_ddk_ir SHARED IMPORTED GLOBAL)
set_property(TARGET npu_ddk_ir PROPERTY IMPORTED_LOCATION ${LIB_DIR_PATH}/lib/libhiai_ir.so)

add_library(npu_ddk_ir_build SHARED IMPORTED GLOBAL)
set_property(TARGET npu_ddk_ir_build PROPERTY IMPORTED_LOCATION ${LIB_DIR_PATH}/lib/libhiai_ir_build.so)

# 320
# add_library(npu_ddk_hcl SHARED IMPORTED GLOBAL)
# set_property(TARGET npu_ddk_hcl PROPERTY IMPORTED_LOCATION ${LIB_DIR_PATH}/lib/libhcl.so)
# set(npu_runtime_libs npu_ddk_hiai npu_ddk_hcl CACHE INTERNAL "npu ddk runtime libs")

set(npu_runtime_libs npu_ddk_hiai CACHE INTERNAL "npu ddk runtime libs")
set(npu_builder_libs npu_ddk_ir npu_ddk_ir_build CACHE INTERNAL "npu ddk builder libs")

add_library(paddle_full_api_shared SHARED IMPORTED GLOBAL)
set_property(TARGET paddle_full_api_shared PROPERTY IMPORTED_LOCATION ${LIB_DIR_PATH}/lib/libpaddle_full_api_shared.so)

target_link_libraries(mnist_save paddle_full_api_shared ${log-lib} ${npu_builder_libs} ${npu_runtime_libs})