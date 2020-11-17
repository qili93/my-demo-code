cmake_minimum_required(VERSION 3.10)

project(model_test)

set(TARGET_EXE model_test)

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
  set(LITE_FULL_DLL_FILE   ${PADDLE_LIB_DIR}/libpaddle_full_api_shared.so)
  set(LITE_TINY_DLL_FILE   ${PADDLE_LIB_DIR}/libpaddle_light_api_shared.so)
  set(LITE_FULL_LIB_FILE   ${PADDLE_LIB_DIR}/libpaddle_api_full_bundled.a)
  set(LITE_TINY_LIB_FILE   ${PADDLE_LIB_DIR}/libpaddle_api_light_bundled.a)
  set(LITE_IOMP5_DLL_FILE  ${MKLML_LIB_DIR}/libiomp5.so)
endif()

if(NOT DEFINED USE_FULL_API)
  set(USE_FULL_API TRUE)
endif()

IF(USE_STATIC_API)
  if(USE_FULL_API)
    add_definitions(-DUSE_FULL_API)
    ADD_LIBRARY(paddle_api STATIC IMPORTED GLOBAL) 
    SET_PROPERTY(TARGET paddle_api PROPERTY IMPORTED_LOCATION ${LITE_FULL_LIB_FILE})
  else()
    ADD_LIBRARY(paddle_api STATIC IMPORTED GLOBAL) 
    SET_PROPERTY(TARGET paddle_api PROPERTY IMPORTED_LOCATION ${LITE_TINY_LIB_FILE})
  endif()
ELSE(USE_STATIC_API)
  if(USE_FULL_API)
    add_definitions(-DUSE_FULL_API)
    ADD_LIBRARY(paddle_api SHARED IMPORTED GLOBAL) 
    SET_PROPERTY(TARGET paddle_api PROPERTY IMPORTED_LOCATION ${LITE_FULL_DLL_FILE})
  else()
    ADD_LIBRARY(paddle_api SHARED IMPORTED GLOBAL) 
    SET_PROPERTY(TARGET paddle_api PROPERTY IMPORTED_LOCATION ${LITE_FULL_DLL_FILE})
  endif()
ENDIF(USE_STATIC_API)

ADD_LIBRARY(mklml_iomp5 SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET mklml_iomp5 PROPERTY IMPORTED_LOCATION ${LITE_IOMP5_DLL_FILE})
IF(WIN32)
  SET_PROPERTY(TARGET mklml_iomp5 PROPERTY IMPORTED_IMPLIB ${LITE_IOMP5_LIB_FILE})
ENDIF(WIN32)

# 5.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

# 6. link libraries
target_link_libraries(${TARGET_EXE} paddle_api mklml_iomp5)

if(WIN32)
  if (NOT WITH_STATIC_MKL)
    ADD_LIBRARY(mklml_dll SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET mklml_dll PROPERTY IMPORTED_LOCATION ${LITE_MKLML_DLL_FILE})
    SET_PROPERTY(TARGET mklml_dll PROPERTY IMPORTED_IMPLIB ${LITE_MKLML_LIB_FILE})

    target_link_libraries(${TARGET_EXE} mklml_dll)
    add_custom_command(TARGET ${TARGET_EXE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${LITE_MKLML_DLL_FILE} ${CMAKE_BINARY_DIR}/Release
    )
  endif()
  target_link_libraries(${TARGET_EXE} shlwapi.lib)
  add_custom_command(TARGET ${TARGET_EXE} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${LITE_IOMP5_DLL_FILE} ${CMAKE_BINARY_DIR}/Release
  )
endif()
