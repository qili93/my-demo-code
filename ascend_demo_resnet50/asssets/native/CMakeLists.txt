cmake_minimum_required(VERSION 3.4.1)
project(resnet50_build)

set(TARGET_EXE resnet50_build)

# 1. path to Ascend Path
if(NOT DEFINED ASCEND_PATH)
    set(ASCEND_PATH $ENV{ASCEND_PATH})
    if(NOT ASCEND_PATH)
        message(FATAL_ERROR "Must set ASCEND_PATH or env ASCEND_PATH when LITE_WITH_ASCEND=ON")
    endif()
endif()
message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

# Require C++11
add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 2. Ascend include directory
set(ATC_INCLUDE_DIR "${ASCEND_PATH}/atc/include")
set(OPP_INCLUDE_DIR "${ASCEND_PATH}/opp/op_proto/built-in/inc")
include_directories(${ATC_INCLUDE_DIR})
include_directories(${OPP_INCLUDE_DIR})

# 3. ATC libs
add_library(atc_register SHARED IMPORTED GLOBAL)
set_property(TARGET atc_register PROPERTY IMPORTED_LOCATION ${ASCEND_PATH}/atc/lib64/libregister.so)

add_library(atc_protobuf SHARED IMPORTED GLOBAL)
set_property(TARGET atc_protobuf PROPERTY IMPORTED_LOCATION ${ASCEND_PATH}/atc/lib64/libprotobuf.so.19)

add_library(atc_graph SHARED IMPORTED GLOBAL)
set_property(TARGET atc_graph PROPERTY IMPORTED_LOCATION ${ASCEND_PATH}/atc/lib64/libgraph.so)

add_library(atc_ge_compiler SHARED IMPORTED GLOBAL)
set_property(TARGET atc_ge_compiler PROPERTY IMPORTED_LOCATION ${ASCEND_PATH}/atc/lib64/libge_compiler.so)

add_library(atc_ge_common SHARED IMPORTED GLOBAL)
set_property(TARGET atc_ge_common PROPERTY IMPORTED_LOCATION ${ASCEND_PATH}/atc/lib64/libge_common.so)

add_library(atc_resource SHARED IMPORTED GLOBAL)
set_property(TARGET atc_resource PROPERTY IMPORTED_LOCATION ${ASCEND_PATH}/atc/lib64/libresource.so)

add_library(atc_runtime SHARED IMPORTED GLOBAL)
set_property(TARGET atc_runtime PROPERTY IMPORTED_LOCATION ${ASCEND_PATH}/atc/lib64/libruntime_compile.so)

# OPP libs
add_library(opp_ops_proto SHARED IMPORTED GLOBAL)
set_property(TARGET opp_ops_proto PROPERTY IMPORTED_LOCATION ${ASCEND_PATH}/opp/op_proto/built-in/libopsproto.so)

add_library(atc_error_manager SHARED IMPORTED GLOBAL)
set_property(TARGET atc_error_manager PROPERTY IMPORTED_LOCATION ${ASCEND_PATH}/atc/lib64/liberror_manager.so)

set(ascend_builder_libs atc_register atc_protobuf atc_graph opp_ops_proto atc_error_manager
    atc_ge_compiler atc_ge_common atc_resource atc_runtime CACHE INTERNAL "ascend atc builder libs")

# 3.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cpp)

target_link_libraries(${TARGET_EXE} ${ascend_builder_libs})