# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(ir_build)

set(TARGET_EXE main)

# 1. path to Huawei Ascend Install Path
if(NOT DEFINED HUAWEI_ASCEND_NPU_DDK_ROOT)
    set(HUAWEI_ASCEND_NPU_DDK_ROOT $ENV{HUAWEI_ASCEND_NPU_DDK_ROOT})
    if(NOT HUAWEI_ASCEND_NPU_DDK_ROOT)
        message(FATAL_ERROR "Must set HUAWEI_ASCEND_NPU_DDK_ROOT or env HUAWEI_ASCEND_NPU_DDK_ROOT when LITE_WITH_HUAWEI_ASCEND_NPU=ON")
    endif()
endif()
message(STATUS "HUAWEI_ASCEND_NPU_DDK_ROOT: ${HUAWEI_ASCEND_NPU_DDK_ROOT}")

# 2.  Add compile options
add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

# 3. Huawei Ascend include directory
set(ATC_INCLUDE_DIR "${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/include")
set(OPP_INCLUDE_DIR "${HUAWEI_ASCEND_NPU_DDK_ROOT}/opp")
include_directories(${ATC_INCLUDE_DIR})
include_directories(${OPP_INCLUDE_DIR})

# 4.1 find ATC libs - libregister.so
find_library(ATC_REGISTER_FILE NAMES register
  PATHS ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64
  NO_DEFAULT_PATH)

if(NOT ATC_REGISTER_FILE)
  message(FATAL_ERROR "Can not find ATC_REGISTER_FILE in ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64")
else()
  message(STATUS "Found ATC_REGISTER_FILE Library: ${ATC_REGISTER_FILE}")
  add_library(atc_register SHARED IMPORTED GLOBAL)
  set_property(TARGET atc_register PROPERTY IMPORTED_LOCATION ${ATC_REGISTER_FILE})
endif()

# 4.1.1 dependency of register - libprotobuf.so.19,
find_library(ATC_PROTOBUF_FILE NAMES libprotobuf.so.19
  PATHS ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64
  NO_DEFAULT_PATH)

  if(NOT ATC_REGISTER_FILE)
  message(FATAL_ERROR "Can not find ATC_PROTOBUF_FILE in ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64")
else()
  message(STATUS "Found ATC_PROTOBUF_FILE Library: ${ATC_PROTOBUF_FILE}")
  add_library(atc_protobuf SHARED IMPORTED GLOBAL)
  set_property(TARGET atc_protobuf PROPERTY IMPORTED_LOCATION ${ATC_PROTOBUF_FILE})
endif()

# 4.1.2 dependency of register - libgraph.so
find_library(ATC_GRAPH_FILE NAMES graph
  PATHS ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64
  NO_DEFAULT_PATH)

if(NOT ATC_GRAPH_FILE)
  message(FATAL_ERROR "Can not find ATC_GRAPH_FILE in ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64")
else()
  message(STATUS "Found ATC_GRAPH_FILE Library: ${ATC_GRAPH_FILE}")
  add_library(atc_graph SHARED IMPORTED GLOBAL)
  set_property(TARGET atc_graph PROPERTY IMPORTED_LOCATION ${ATC_GRAPH_FILE})
endif()

# 4.2 find ATC libs - libge_compiler.so
find_library(ATC_GE_COMPILER_FILE NAMES ge_compiler
  PATHS ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64
  NO_DEFAULT_PATH)

if(NOT ATC_GE_COMPILER_FILE)
  message(FATAL_ERROR "Can not find ATC_GE_COMPILER_FILE in ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64")
else()
  message(STATUS "Found ATC_GE_COMPILER_FILE Library: ${ATC_GE_COMPILER_FILE}")
  add_library(atc_ge_compiler SHARED IMPORTED GLOBAL)
  set_property(TARGET atc_ge_compiler PROPERTY IMPORTED_LOCATION ${ATC_GE_COMPILER_FILE})
endif()

# 4.2.1 dependencies of libge_compiler.so - libge_common.so
find_library(ATC_GE_COMMON_FILE NAMES ge_common
  PATHS ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64
  NO_DEFAULT_PATH)

if(NOT ATC_GE_COMMON_FILE)
  message(FATAL_ERROR "Can not find ATC_GE_COMMON_FILE in ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64")
else()
  message(STATUS "Found ATC_GE_COMMON_FILE Library: ${ATC_GE_COMMON_FILE}")
  add_library(atc_ge_common SHARED IMPORTED GLOBAL)
  set_property(TARGET atc_ge_common PROPERTY IMPORTED_LOCATION ${ATC_GE_COMMON_FILE})
endif()

# 4.2.3 dependencies of libge_compiler.so - libresource.so
find_library(ATC_RESOURCE_FILE NAMES resource
  PATHS ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64
  NO_DEFAULT_PATH)

if(NOT ATC_RESOURCE_FILE)
  message(FATAL_ERROR "Can not find ATC_RESOURCE_FILE in ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64")
else()
  message(STATUS "Found ATC_RESOURCE_FILE Library: ${ATC_RESOURCE_FILE}")
  add_library(atc_resource SHARED IMPORTED GLOBAL)
  set_property(TARGET atc_resource PROPERTY IMPORTED_LOCATION ${ATC_RESOURCE_FILE})
endif()

# 4.3 find OPP libs - libopsproto.so
find_library(OPP_OPS_PROTO_FILE NAMES opsproto
  PATHS ${HUAWEI_ASCEND_NPU_DDK_ROOT}/opp/op_proto/built-in
  NO_DEFAULT_PATH)

if(NOT OPP_OPS_PROTO_FILE)
  message(FATAL_ERROR "Can not find OPP_OPS_PROTO_FILE in ${HUAWEI_ASCEND_NPU_DDK_ROOT}/opp/op_proto/built-in")
else()
  message(STATUS "Found OPP_OPS_PROTO_FILE Library: ${OPP_OPS_PROTO_FILE}")
  add_library(opp_ops_proto SHARED IMPORTED GLOBAL)
  set_property(TARGET opp_ops_proto PROPERTY IMPORTED_LOCATION ${OPP_OPS_PROTO_FILE})
endif()

# 4.3.1 dependency of  opp_ops_proto - liberror_manager.so
find_library(ATC_ERROR_MANAGER_FILE NAMES error_manager
  PATHS ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64
  NO_DEFAULT_PATH)

if(NOT ATC_ERROR_MANAGER_FILE)
  message(FATAL_ERROR "Can not find ATC_ERROR_MANAGER_FILE in ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64")
else()
  message(STATUS "Found ATC_ERROR_MANAGER_FILE Library: ${ATC_ERROR_MANAGER_FILE}")
  add_library(atc_error_manager SHARED IMPORTED GLOBAL)
  set_property(TARGET atc_error_manager PROPERTY IMPORTED_LOCATION ${ATC_ERROR_MANAGER_FILE})
endif()

# 4.4 libruntime_compile.so
find_library(ATC_RUNTIME_COMPILE_FILE NAMES runtime_compile
  PATHS ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64
  NO_DEFAULT_PATH)

if(NOT ATC_RUNTIME_COMPILE_FILE)
  message(FATAL_ERROR "Can not find ATC_RUNTIME_COMPILE_FILE in ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64")
else()
  message(STATUS "Found ATC_RUNTIME_COMPILE_FILE Library: ${ATC_RUNTIME_COMPILE_FILE}")
  add_library(atc_runtime_compile SHARED IMPORTED GLOBAL)
  set_property(TARGET atc_runtime_compile PROPERTY IMPORTED_LOCATION ${ATC_RUNTIME_COMPILE_FILE})
endif()

set(huawei_ascend_npu_builder_libs atc_register atc_protobuf atc_graph opp_ops_proto atc_error_manager 
    atc_ge_compiler atc_ge_common atc_resource CACHE INTERNAL "huawei_ascend_npu atc builder libs")
set(huawei_ascend_npu_runtime_libs atc_runtime_compile CACHE INTERNAL "huawei_ascend_npu atc runtime libs")

add_executable(${TARGET_EXE} ${TARGET_EXE}.cpp)
target_link_libraries(${TARGET_EXE} ${huawei_ascend_npu_builder_libs} ${huawei_ascend_npu_runtime_libs} stdc++)
