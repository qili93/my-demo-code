cmake_minimum_required(VERSION 3.10)
project(image_classification_demo)

set(TARGET_EXE image_classification_demo)

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
        message(FATAL_ERROR "HUAWEI_ASCEND_NPU_DDK_ROOT not set.")
    endif()
endif()
message(STATUS "HUAWEI_ASCEND_NPU_DDK_ROOT: ${HUAWEI_ASCEND_NPU_DDK_ROOT}")

# 3. Set compile options
add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

if(TARGET_ARCH_ABI STREQUAL "x86_64-linux_gcc7.3.0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
elseif(TARGET_ARCH_ABI STREQUAL "x86_64-linux_gcc4.8.5")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    message(FATAL_ERROR "Unknown arch abi ${TARGET_ARCH_ABI}, only support x86_64-linux_gcc7.3.0 and x86_64-linux_gcc4.8.5.")
    return()
endif()

# 4. include Paddle-Lite directory
include_directories(${PADDLE_LITE_DIR}/${TARGET_ARCH_ABI}/include)

# 5. Huawei Ascend NPU - ACL libs and its dependencies
add_library(acl_ascendcl SHARED IMPORTED GLOBAL)
set_property(TARGET acl_ascendcl PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/acllib/lib64/libascendcl.so)

add_library(acl_runtime SHARED IMPORTED GLOBAL)
set_property(TARGET acl_runtime PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/acllib/lib64/libruntime.so)

# 6. Huawei Ascend NPU - ATC libs and its dependencies

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

# 7. Huawei Ascend NPU - OPP libs and its dependencies
add_library(opp_ops_proto SHARED IMPORTED GLOBAL)
set_property(TARGET opp_ops_proto PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/opp/op_proto/built-in/libopsproto.so)

add_library(atc_error_manager SHARED IMPORTED GLOBAL)
set_property(TARGET atc_error_manager PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64/liberror_manager.so)

# 8. note: huawei_ascend_npu_runtime_libs should before huawei_ascend_npu_builder_libs
set(huawei_ascend_npu_runtime_libs acl_ascendcl acl_runtime CACHE INTERNAL "ascend acllib runtime libs")
set(huawei_ascend_npu_builder_libs atc_register atc_protobuf atc_graph opp_ops_proto atc_error_manager 
    atc_ge_compiler atc_ge_common atc_resource CACHE INTERNAL "ascend atc builder libs")

# 9. Paddle Lite Libs - paddle_api_shared
add_library(paddle_api_shared SHARED IMPORTED GLOBAL)
set_property(TARGET paddle_api_shared PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/${TARGET_ARCH_ABI}/lib/libpaddle_full_api_shared.so)

# 10. Paddle Lite Libs - libiomp5
add_library(mkl_iomp5 SHARED IMPORTED GLOBAL)
set_property(TARGET mkl_iomp5 PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/${TARGET_ARCH_ABI}/lib/libiomp5.so)

# 11. Executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)
target_link_libraries(${TARGET_EXE} paddle_api_shared mkl_iomp5 ${huawei_ascend_npu_runtime_libs} ${huawei_ascend_npu_builder_libs})