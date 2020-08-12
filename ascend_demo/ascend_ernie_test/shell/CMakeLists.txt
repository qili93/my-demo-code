cmake_minimum_required(VERSION 3.4.1)
project(ernie_test)

set(TARGET_EXE ernie_test)

# 1. path to Paddle-Lite lib
if(${PADDLE_LITE_DIR} STREQUAL "")
  message(FATAL_ERROR "PADDLE_LITE_DIR not set.")
  return()
endif()
message(STATUS "PADDLE_LITE_DIR=${PADDLE_LITE_DIR}")

# 2. path to Ascend path
if(NOT DEFINED HUAWEI_ASCEND_NPU_DDK_ROOT)
    set(HUAWEI_ASCEND_NPU_DDK_ROOT $ENV{HUAWEI_ASCEND_NPU_DDK_ROOT})
    if(NOT HUAWEI_ASCEND_NPU_DDK_ROOT)
        message(FATAL_ERROR "Must set HUAWEI_ASCEND_NPU_DDK_ROOT or env HUAWEI_ASCEND_NPU_DDK_ROOT when LITE_WITH_ASCEND=ON")
    endif()
endif()
message(STATUS "HUAWEI_ASCEND_NPU_DDK_ROOT: ${HUAWEI_ASCEND_NPU_DDK_ROOT}")

# Require C++11
add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 3. include Paddle-Lite directory
include_directories(${PADDLE_LITE_DIR}/cxx/include)
include_directories(${PADDLE_LITE_DIR}/third_party/mklml/include)

# 4. ACL libs
add_library(acl_ascendcl SHARED IMPORTED GLOBAL)
set_property(TARGET acl_ascendcl PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/acllib/lib64/libascendcl.so)

add_library(acl_runtime SHARED IMPORTED GLOBAL)
set_property(TARGET acl_runtime PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/acllib/lib64/libruntime.so)

# 5. ATC libs

add_library(atc_register SHARED IMPORTED GLOBAL)
set_property(TARGET atc_register PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64/libregister.so)

add_library(atc_protobuf SHARED IMPORTED GLOBAL)
set_property(TARGET atc_protobuf PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64/libprotobuf.so.19)

add_library(atc_graph SHARED IMPORTED GLOBAL)
set_property(TARGET atc_graph PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64/libgraph.so)

add_library(atc_ge_compiler SHARED IMPORTED GLOBAL)
set_property(TARGET atc_ge_compiler PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64/libge_compiler.so)

add_library(atc_ge_common SHARED IMPORTED GLOBAL)
set_property(TARGET atc_ge_common PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64/libge_common.so)

add_library(atc_resource SHARED IMPORTED GLOBAL)
set_property(TARGET atc_resource PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64/libresource.so)

# OPP libs
add_library(opp_ops_proto SHARED IMPORTED GLOBAL)
set_property(TARGET opp_ops_proto PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/opp/op_proto/built-in/libopsproto.so)

add_library(atc_error_manager SHARED IMPORTED GLOBAL)
set_property(TARGET atc_error_manager PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64/liberror_manager.so)

set(ascend_builder_libs atc_register atc_protobuf atc_graph opp_ops_proto atc_error_manager 
    atc_ge_compiler atc_ge_common atc_resource CACHE INTERNAL "ascend atc builder libs")

set(ascend_runtime_libs acl_ascendcl acl_runtime CACHE INTERNAL "ascend acllib runtime libs")

# 5. link paddle-lite libraries
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

# 3.add executable output
add_definitions("-DLITE_WITH_HUAWEI_ASCEND_NPU")
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

#target_link_libraries(${TARGET_EXE} paddle_api_shared mkl_iomp5)

target_link_libraries(${TARGET_EXE} paddle_api_shared mkl_iomp5 ${ascend_runtime_libs} ${ascend_builder_libs})