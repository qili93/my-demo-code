cmake_minimum_required(VERSION 3.10)

project(lite_lib_test)

# 1. path to Paddle-Lite lib
if(${PADDLE_LITE_DIR} STREQUAL "")
  message(FATAL_ERROR "PADDLE_LITE_DIR not set.")
  return()
endif()
message(STATUS "PADDLE_LITE_DIR=${PADDLE_LITE_DIR}")

# set PADDLE_LIB_DIR to LOCAL LIB DIR
set(PADDLE_INC_DIR ${PADDLE_LITE_DIR}/include) # for local lib
set(PADDLE_LIB_DIR ${PADDLE_LITE_DIR}/lib) # for local lib
set(MKLML_LIB_DIR ${PADDLE_LITE_DIR}/lib) # for local lib

# set PADDLE_LIB_DIR to PADDLE BUILD DIR
# set(PADDLE_INC_DIR ${PADDLE_LITE_DIR}/cxx/include)
# set(PADDLE_LIB_DIR ${PADDLE_LITE_DIR}/cxx/lib)
# set(MKLML_LIB_DIR ${PADDLE_LITE_DIR}/third_party/mklml/lib)

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

# 3. include Paddle-Lite directory
include_directories(${PADDLE_INC_DIR})

# 4. link paddle-lite libraries
if(APPLE)
  set(LITE_FULL_LIB_FILE   ${PADDLE_LIB_DIR}/libpaddle_full_api_shared.dylib)
  set(LITE_TINY_LIB_FILE   ${PADDLE_LIB_DIR}/libpaddle_light_api_shared.dylib)
  set(LITE_IOMP5_DLL_FILE  ${MKLML_LIB_DIR}/libiomp5.dylib)
elseif(WIN32)
  set(LITE_FULL_LIB_FILE   ${PADDLE_LIB_DIR}/libpaddle_api_full_bundled.lib)
  set(LITE_TINY_LIB_FILE   ${PADDLE_LIB_DIR}/libpaddle_api_light_bundled.lib)
  set(LITE_IOMP5_LIB_FILE  ${MKLML_LIB_DIR}/libiomp5md.lib)
  set(LITE_IOMP5_DLL_FILE  ${MKLML_LIB_DIR}/libiomp5md.dll)
  set(LITE_MKLML_LIB_FILE  ${MKLML_LIB_DIR}/mklml.lib)
  set(LITE_MKLML_DLL_FILE  ${MKLML_LIB_DIR}/mklml.dll)
else()
  set(LITE_FULL_LIB_FILE   ${PADDLE_LIB_DIR}/libpaddle_full_api_shared.so)
  set(LITE_TINY_LIB_FILE   ${PADDLE_LIB_DIR}/libpaddle_light_api_shared.so)
  set(LITE_IOMP5_DLL_FILE  ${MKLML_LIB_DIR}/libiomp5.so)
endif()

if(NOT DEFINED USE_FULL_API)
  set(USE_FULL_API TRUE)
endif()

IF(WIN32)
  if(USE_FULL_API)
    add_definitions(-DUSE_FULL_API)
    ADD_LIBRARY(paddle_api STATIC IMPORTED GLOBAL) 
    SET_PROPERTY(TARGET paddle_api PROPERTY IMPORTED_LOCATION ${LITE_FULL_LIB_FILE})
  else()
    ADD_LIBRARY(paddle_api STATIC IMPORTED GLOBAL) 
    SET_PROPERTY(TARGET paddle_api PROPERTY IMPORTED_LOCATION ${LITE_TINY_LIB_FILE})
  endif()
ELSE(WIN32)
  if(USE_FULL_API)
    add_definitions(-DUSE_FULL_API)
    ADD_LIBRARY(paddle_api SHARED IMPORTED GLOBAL) 
    SET_PROPERTY(TARGET paddle_api PROPERTY IMPORTED_LOCATION ${LITE_FULL_LIB_FILE})
  else()
    ADD_LIBRARY(paddle_api SHARED IMPORTED GLOBAL) 
    SET_PROPERTY(TARGET paddle_api PROPERTY IMPORTED_LOCATION ${LITE_TINY_LIB_FILE})
  endif()
ENDIF(WIN32)

ADD_LIBRARY(mklml_iomp5 SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET mklml_iomp5 PROPERTY IMPORTED_LOCATION ${LITE_IOMP5_DLL_FILE})
IF(WIN32)
  SET_PROPERTY(TARGET mklml_iomp5 PROPERTY IMPORTED_IMPLIB ${LITE_IOMP5_LIB_FILE})
ENDIF(WIN32)

# add static/shared lib
if(LINK_STATIC)
  add_library(model_test STATIC model_test.cc)
  target_link_libraries(model_test paddle_api mklml_iomp5)
else()
  add_library(model_test SHARED model_test.cc)
  target_link_libraries(model_test paddle_api mklml_iomp5)
endif()

# 5.add executable output
if(WIN32)
  if (NOT WITH_STATIC_MKL)
    ADD_LIBRARY(mklml_dll SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET mklml_dll PROPERTY IMPORTED_LOCATION ${LITE_MKLML_DLL_FILE})
    SET_PROPERTY(TARGET mklml_dll PROPERTY IMPORTED_IMPLIB ${LITE_MKLML_LIB_FILE})

    target_link_libraries(model_test mklml_dll)
    add_custom_command(TARGET model_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${LITE_MKLML_DLL_FILE} ${CMAKE_BINARY_DIR}/Release
    )
  endif()
  add_custom_command(TARGET model_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${LITE_IOMP5_DLL_FILE} ${CMAKE_BINARY_DIR}/Release
  )
endif()

# add execution
add_executable(main_test main_test.cc)
add_dependencies(main_test model_test)
target_link_libraries(main_test model_test)

if(WIN32)
  target_link_libraries(main_test shlwapi.lib)
endif()