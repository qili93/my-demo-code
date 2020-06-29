cmake_minimum_required(VERSION 3.5.1)

project(bias_add_build)

set(TARGET_EXE bias_add_build)

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

# 2.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

# 3. Ascend include directory
set(ATC_INCLUDE_DIR "${ASCEND_PATH}/atc/include")
set(OPP_INCLUDE_DIR "${ASCEND_PATH}/opp/op_proto/built-in/inc")

include_directories(${OPP_INCLUDE_DIR})
include_directories(${ATC_INCLUDE_DIR})
include_directories(${ATC_INCLUDE_DIR}/graph)
include_directories(${ATC_INCLUDE_DIR}/ge)
include_directories(${ATC_INCLUDE_DIR}/register)

# 4. Ascend link directory
link_directories(${ASCEND_PATH}/atc/lib64/plugin/opskernel)
link_directories(${ASCEND_PATH}/atc/lib64)
link_directories(${ASCEND_PATH}/opp/op_proto/built-in)
link_directories(${ASCEND_PATH}/opp/fusion_pass/built_in)
link_directories(${ASCEND_PATH}/opp/fusion_pass/built_in/vector_core)

# 5. Ascend link libraries
find_library(ASCEND_REGISTER_FILE NAMES register
  PATHS ${ASCEND_PATH}/atc/lib64
  NO_DEFAULT_PATH)

find_library(ASCEND_GRAPH_FILE NAMES graph
  PATHS ${ASCEND_PATH}/atc/lib64
  NO_DEFAULT_PATH)

find_library(ASCEND_GE_COMPILER_FILE NAMES ge_compiler
  PATHS ${ASCEND_PATH}/atc/lib64
  NO_DEFAULT_PATH)

find_library(ASCEND_OPSPROTO_FILE NAMES opsproto
  PATHS ${ASCEND_PATH}/opp/op_proto/built-in
  NO_DEFAULT_PATH)

if(NOT ASCEND_REGISTER_FILE)
  message(FATAL_ERROR "Can not find ASCEND_REGISTER_FILE in ${ASCEND_PATH}/atc/lib64")
else()
  message(STATUS "Found ASCEND REGISTER Library: ${ASCEND_REGISTER_FILE}")
  add_library(ascend_register SHARED IMPORTED GLOBAL)
  set_property(TARGET ascend_register PROPERTY IMPORTED_LOCATION ${ASCEND_REGISTER_FILE})
endif()

if(NOT ASCEND_GRAPH_FILE)
  message(FATAL_ERROR "Can not find ASCEND_GRAPH_FILE in ${ASCEND_PATH}/atc/lib64")
else()
  message(STATUS "Found ASCEND GRAPH Library: ${ASCEND_GRAPH_FILE}")
  add_library(ascend_graph SHARED IMPORTED GLOBAL)
  set_property(TARGET ascend_graph PROPERTY IMPORTED_LOCATION ${ASCEND_GRAPH_FILE})
endif()

if(NOT ASCEND_GE_COMPILER_FILE)
  message(FATAL_ERROR "Can not find ASCEND_GE_COMPILER_FILE in ${ASCEND_PATH}/atc/lib64")
else()
  message(STATUS "Found ASCEND GE COMPILER Library: ${ASCEND_GE_COMPILER_FILE}")
  add_library(ascend_ge_compiler SHARED IMPORTED GLOBAL)
  set_property(TARGET ascend_ge_compiler PROPERTY IMPORTED_LOCATION ${ASCEND_GE_COMPILER_FILE})
endif()

if(NOT ASCEND_OPSPROTO_FILE)
  message(FATAL_ERROR "Can not find ASCEND_OPSPROTO_FILE in ${ASCEND_PATH}/opp/op_proto/built-in")
else()
  message(STATUS "Found ASCEND OPSPROTO Library: ${ASCEND_OPSPROTO_FILE}")
  add_library(ascend_opsproto SHARED IMPORTED GLOBAL)
  set_property(TARGET ascend_opsproto PROPERTY IMPORTED_LOCATION ${ASCEND_OPSPROTO_FILE})
endif()

set(ascend_builder_libs ascend_register ascend_graph ascend_ge_compiler ascend_opsproto CACHE INTERNAL "ascend atc builder libs")

target_link_libraries(${TARGET_EXE} ${ascend_builder_libs})