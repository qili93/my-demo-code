cmake_minimum_required(VERSION 3.4.1)
project(model_test)

set(TARGET_EXE model_test)

# 1. path to Paddle-Lite lib
if(${PADDLE_LITE_DIR} STREQUAL "")
  message(FATAL_ERROR "PADDLE_LITE_DIR not set.")
  return()
endif()
message(STATUS "PADDLE_LITE_DIR: ${PADDLE_LITE_DIR}")

set(PADDLE_INC_DIR ${PADDLE_LITE_DIR}/cxx/include)
set(PADDLE_LIB_DIR ${PADDLE_LITE_DIR}/cxx/lib)
set(MKLML_LIB_DIR ${PADDLE_LITE_DIR}/third_party/mklml/lib)

# message(STATUS "PADDLE_INC_DIR=${PADDLE_INC_DIR}")
# message(STATUS "PADDLE_LIB_DIR=${PADDLE_LIB_DIR}")
# message(STATUS "MKLML_LIB_DIR=${MKLML_LIB_DIR}")

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

# 3. include Paddle-Lite directory
include_directories(${PADDLE_INC_DIR})

# Create demo_deps_library function
set(dep_libs CACHE INTERNAL "dependent libraries")
function(demo_deps_library TARGET_NAME PATH_DIR)
  find_library(${TARGET_NAME}_FILE NAMES ${TARGET_NAME} PATHS "${PATH_DIR}" NO_DEFAULT_PATH)
  if(NOT ${TARGET_NAME}_FILE)
    message(FATAL_ERROR "Can not find ${TARGET_NAME}_FILE Library in ${PATH_DIR}")
  else()
    message(STATUS "Found ${TARGET_NAME}: ${${TARGET_NAME}_FILE}")
    add_library(${TARGET_NAME}_lib_dep SHARED IMPORTED GLOBAL)
    set_property(TARGET ${TARGET_NAME}_lib_dep PROPERTY IMPORTED_LOCATION "${${TARGET_NAME}_FILE}")
    set(dep_libs "${dep_libs};${${TARGET_NAME}_FILE}" CACHE INTERNAL "")
    # list(APPEND DEP_LIBS "${${TARGET_NAME}_FILE}")
  endif()
endfunction()

# 5. Add ACL Libs
demo_deps_library(ascendcl ${HUAWEI_ASCEND_NPU_DDK_ROOT}/acllib/lib64)
demo_deps_library(runtime ${HUAWEI_ASCEND_NPU_DDK_ROOT}/acllib/lib64)
# 5. ATC libs
demo_deps_library(register ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64)
demo_deps_library(graph ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64)
demo_deps_library(ge_compiler ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64)
demo_deps_library(ge_common ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64)
# OPP libs
demo_deps_library(opsproto ${HUAWEI_ASCEND_NPU_DDK_ROOT}/opp/op_proto/built-in)
demo_deps_library(error_manager ${HUAWEI_ASCEND_NPU_DDK_ROOT}/atc/lib64)

# 5. link paddle-lite libraries
if(NOT DEFINED USE_FULL_API)
  set(USE_FULL_API TRUE)
endif()
if(USE_FULL_API)
  add_definitions(-DUSE_FULL_API)
  demo_deps_library(paddle_full_api_shared ${PADDLE_LIB_DIR})
else()
  demo_deps_library(paddle_light_api_shared ${PADDLE_LIB_DIR})
endif()
demo_deps_library(iomp5 ${MKLML_LIB_DIR})

# add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)
target_link_libraries(${TARGET_EXE} ${dep_libs})