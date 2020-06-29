cmake_minimum_required(VERSION 3.5.1)

project(bias_add_run)

set(TARGET_EXE bias_add_run)

# 1. path to Ascend Path
if(NOT DEFINED ASCEND_PATH)
    set(ASCEND_PATH $ENV{ASCEND_PATH})
    if(NOT ASCEND_PATH)
        message(FATAL_ERROR "Must set ASCEND_PATH or env ASCEND_PATH when LITE_WITH_ASCEND=ON")
    endif()
endif()
message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

# require C++11
add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

include_directories(${ASCEND_PATH}/acllib/include)

# 2.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

# 3. Ascend include directory
set(ACL_INCLUDE_DIR "${ASCEND_PATH}/acllib/include")
include_directories(${ACL_INCLUDE_DIR})

# 4. Ascend link directory
link_directories(${ASCEND_PATH}/acllib/lib64/stub)

# 5. Ascend link libraries
find_library(ASCEND_CL NAMES ascendcl
  PATHS ${ASCEND_PATH}/acllib/lib64/stub
  NO_DEFAULT_PATH)

if(NOT ASCEND_CL)
  message(FATAL_ERROR "Can not find ASCEND_CL in ${ASCEND_PATH}/acllib/lib64/stub")
else()
  message(STATUS "Found ASCEND_CL Library: ${ASCEND_CL}")
  add_library(ascendcl SHARED IMPORTED GLOBAL)
  set_property(TARGET ascendcl PROPERTY IMPORTED_LOCATION ${ASCEND_CL})
endif()

set(ascend_runtime_libs ascendcl CACHE INTERNAL "ascend acllib runtime libs")

target_link_libraries(${TARGET_EXE} ${ascend_runtime_libs})