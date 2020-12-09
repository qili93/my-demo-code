cmake_minimum_required(VERSION 3.10)

project(model_test)

set(TARGET_EXE model_test)

option(WITH_OPENCV "compile with open cv" OFF)
if(WITH_OPENCV)
  find_package(OpenCV REQUIRED PATHS /usr/local/opt/opencv@2 NO_DEFAULT_PATH)
  message(STATUS "OpenCV_FOUND: " ${OpenCV_FOUND})
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# 1. path to Paddle-Lite lib
if(${PADDLE_LITE_DIR} STREQUAL "")
  message(FATAL_ERROR "PADDLE_LITE_DIR not set.")
  return()
endif()
message(STATUS "PADDLE_LITE_DIR=${PADDLE_LITE_DIR}")

# set PADDLE_LIB_DIR to LOCAL LIB DIR
# set(PADDLE_INC_DIR ${PADDLE_LITE_DIR}/include) # for local lib
# set(PADDLE_LIB_DIR ${PADDLE_LITE_DIR}/lib) # for local lib
# set(MKLML_LIB_DIR ${PADDLE_LITE_DIR}/lib) # for local lib

# set PADDLE_LIB_DIR to PADDLE BUILD DIR
set(PADDLE_INC_DIR ${PADDLE_LITE_DIR}/cxx/include)
set(PADDLE_LIB_DIR ${PADDLE_LITE_DIR}/cxx/lib)
set(MKLML_LIB_DIR ${PADDLE_LITE_DIR}/third_party/mklml/lib)

message(STATUS "PADDLE_INC_DIR=${PADDLE_INC_DIR}")
message(STATUS "PADDLE_LIB_DIR=${PADDLE_LIB_DIR}")
message(STATUS "MKLML_LIB_DIR=${MKLML_LIB_DIR}")

# 2. compile options
if (WIN32)
  add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
  option(MSVC_STATIC_CRT "use static C Runtime library by default" OFF)
  if (MSVC_STATIC_CRT)
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
  else(MSVC_STATIC_CRT)
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MDd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MD")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MD")
  endif(MSVC_STATIC_CRT)
endif()

if (NOT WIN32)
  add_definitions(-std=c++11 -g -O3 -pthread)
endif()

# add_definitions(-std=c++11 -g -O3 -pthread)
# # add_compile_options(-std=c++11)
# set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
# set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 3. include Paddle-Lite directory
include_directories(${PADDLE_INC_DIR})

# 4. link paddle-lite libraries
if(APPLE)
  set(LITE_FULL_SHARED_NAME "paddle_full_api_shared")
  set(LITE_TINY_SHARED_NAME "paddle_light_api_shared")
  set(MKLML_IOMP5_SHARED_NAME "iomp5")
elseif(WIN32)
  set(LITE_FULL_STATIC_NAME "libpaddle_api_full_bundled")
  set(LITE_FULL_STATIC_NAME "libpaddle_api_light_bundled")
  set(MKLML_IOMP5_SHARED_NAME "libiomp5md")
else()
  set(LITE_FULL_SHARED_NAME "paddle_full_api_shared")
  set(LITE_TINY_SHARED_NAME "paddle_light_api_shared")
  set(LITE_FULL_STATIC_NAME "paddle_api_full_bundled")
  set(LITE_TINY_STATIC_NAME "paddle_api_light_bundled")
  set(MKLML_IOMP5_SHARED_NAME "iomp5")
endif()

# message("prefix: ${CMAKE_FIND_LIBRARY_PREFIXES}")
# message("suffix: ${CMAKE_FIND_LIBRARY_SUFFIXES}")
# message("shared prefix: ${CMAKE_SHARED_LIBRARY_PREFIX}")
# message("shared suffix: ${CMAKE_SHARED_LIBRARY_SUFFIX}")
# message("static prefix: ${CMAKE_STATIC_LIBRARY_PREFIX}")
# message("static suffix: ${CMAKE_STATIC_LIBRARY_SUFFIX}")

# message(STATUS "LITE_FULL_LIB_NAME=${LITE_FULL_LIB_NAME}")
# message(STATUS "LITE_TINY_LIB_NAME=${LITE_TINY_LIB_NAME}")
# message(STATUS "LITE_MKLML_LIB_NAME=${LITE_MKLML_LIB_NAME}")
# message(STATUS "LITE_IOMP5_LIB_NAME=${LITE_IOMP5_LIB_NAME}")

function(add_lite_lib TARGET_NAME)
  set(oneValueArgs NAME PATH)
  cmake_parse_arguments(add_lite_lib "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  # message(STATUS "TARGET_NAME = ${TARGET_NAME}")
  # message(STATUS "add_lite_lib_NAME = ${add_lite_lib_NAME}")
  # message(STATUS "add_lite_lib_PATH = ${add_lite_lib_PATH}")
  unset(ABS_PATH CACHE)
  find_library(ABS_PATH NAMES ${add_lite_lib_NAME} PATHS ${add_lite_lib_PATH} NO_DEFAULT_PATH)
  if(add_lite_lib_SHARED) # build shared lib
    add_library(${TARGET_NAME} SHARED IMPORTED GLOBAL)
    set_property(TARGET ${TARGET_NAME} PROPERTY IMPORTED_LOCATION ${ABS_PATH})
  else()
    add_library(${TARGET_NAME} STATIC IMPORTED GLOBAL)
    set_property(TARGET ${TARGET_NAME} PROPERTY IMPORTED_LOCATION ${ABS_PATH})
  endif()
  if(NOT ABS_PATH)
    message(FATAL_ERROR "Can not find ${add_lite_lib_NAME} Library in ${add_lite_lib_PATH}")
  else()  
    message(STATUS "Found ${add_lite_lib_NAME}: ${ABS_PATH}")
  endif()
endfunction()

option(USE_FULL_API "use paddle lite full api by default" ON)
option(USE_SHARED_API "use paddle lite shared api by default" ON)
if(USE_FULL_API)
  add_definitions(-DUSE_FULL_API)
  if(USE_SHARED_API)
    add_lite_lib(paddle_api SHARED NAME ${LITE_FULL_SHARED_NAME} PATH ${PADDLE_LIB_DIR})
  else()
    add_lite_lib(paddle_api STATIC NAME ${LITE_FULL_STATIC_NAME} PATH ${PADDLE_LIB_DIR})
  endif()
else()
  if(USE_SHARED_API)
    add_lite_lib(paddle_api SHARED NAME ${LITE_TINY_SHARED_NAME} PATH ${PADDLE_LIB_DIR})
  else()
    add_lite_lib(paddle_api STATIC NAME ${LITE_TINY_STATIC_NAME} PATH ${PADDLE_LIB_DIR})
  endif()
endif()

option(WITH_MKL "use Intel MKLML Library by default" ON)
if(WITH_MKL)
  add_lite_lib(mklml_iomp5 SHARED NAME ${MKLML_IOMP5_SHARED_NAME} PATH ${MKLML_LIB_DIR})
endif()

# 5.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

# 6. link libraries
target_link_libraries(${TARGET_EXE} paddle_api mklml_iomp5)

option(WITH_STATIC_MKL "use Intel MKLML Library by default" OFF)
if(WIN32)
  if (NOT WITH_STATIC_MKL)
    ADD_LIBRARY(mklml_dll SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET mklml_dll PROPERTY IMPORTED_LOCATION ${LITE_MKLML_DLL_FILE})
    SET_PROPERTY(TARGET mklml_dll PROPERTY IMPORTED_IMPLIB ${LITE_MKLML_LIB_FILE})

    target_link_libraries(${TARGET_EXE} mklml_dll)
    add_custom_command(TARGET ${TARGET_EXE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_LIB_DIR}/mklml.dll ${CMAKE_BINARY_DIR}/Release
    )
  endif()
  target_link_libraries(${TARGET_EXE} shlwapi.lib)
  add_custom_command(TARGET ${TARGET_EXE} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_LIB_DIR}/libiomp5md.dll ${CMAKE_BINARY_DIR}/Release
  )
endif()
