cmake_minimum_required(VERSION 3.10)

project(model_test)

set(TARGET_EXE model_test)

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
  option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)
  if (MSVC_STATIC_CRT)
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
  endif()
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
  set(LITE_FULL_LIB_NAME "paddle_full_api_shared")
  set(LITE_TINY_LIB_NAME "paddle_light_api_shared")
  # set(LITE_MKLML_LIB_NAME "mklml") # dyload
  set(LITE_IOMP5_LIB_NAME "iomp5")
elseif(WIN32)
  set(LITE_FULL_LIB_NAME "libpaddle_api_full_bundled")
  set(LITE_TINY_LIB_NAME "libpaddle_api_light_bundled")
  # set(LITE_MKLML_LIB_NAME "mklml") # dyload
  set(LITE_IOMP5_LIB_NAME "libiomp5md")
else()
  set(LITE_FULL_LIB_NAME "paddle_full_api_shared")
  set(LITE_TINY_LIB_NAME "paddle_light_api_shared")
  # set(LITE_MKLML_LIB_NAME "mklml") # dyload
  set(LITE_IOMP5_LIB_NAME "iomp5")
endif()

# message("prefix: ${CMAKE_FIND_LIBRARY_PREFIXES}")
# message("suffix: ${CMAKE_FIND_LIBRARY_SUFFIXES}")

# message(STATUS "LITE_FULL_LIB_NAME=${LITE_FULL_LIB_NAME}")
# message(STATUS "LITE_TINY_LIB_NAME=${LITE_TINY_LIB_NAME}")
# message(STATUS "LITE_MKLML_LIB_NAME=${LITE_MKLML_LIB_NAME}")
# message(STATUS "LITE_IOMP5_LIB_NAME=${LITE_IOMP5_LIB_NAME}")

# Create demo_deps_library function
set(dep_libs CACHE INTERNAL "dependent libraries")
function(demo_deps_library TARGET_NAME PATH_DIR)
  find_library(${TARGET_NAME}_FILE NAMES ${TARGET_NAME} PATHS "${PATH_DIR}" NO_DEFAULT_PATH)
  if(NOT ${TARGET_NAME}_FILE)
    message(FATAL_ERROR "Can not find ${TARGET_NAME} Library in ${PATH_DIR}")
  else()
    message(STATUS "Found ${TARGET_NAME}: ${${TARGET_NAME}_FILE}")
    if(WIN32)
      add_library(${TARGET_NAME}_lib_dep STATIC IMPORTED GLOBAL)
    else()
      add_library(${TARGET_NAME}_lib_dep SHARED IMPORTED GLOBAL)
    endif()
    set_property(TARGET ${TARGET_NAME}_lib_dep PROPERTY IMPORTED_LOCATION "${${TARGET_NAME}_FILE}")
    set(dep_libs "${dep_libs};${${TARGET_NAME}_FILE}" CACHE INTERNAL "")
    # list(APPEND DEP_LIBS "${${TARGET_NAME}_FILE}")
  endif()
endfunction()

if(NOT DEFINED USE_FULL_API)
  set(USE_FULL_API TRUE)
endif()
if(USE_FULL_API)
  add_definitions(-DUSE_FULL_API)
  demo_deps_library(${LITE_FULL_LIB_NAME} ${PADDLE_LIB_DIR})
else()
  demo_deps_library(${LITE_TINY_LIB_NAME} ${PADDLE_LIB_DIR})
endif()

if(NOT DEFINED WITH_MKL)
  set(WITH_MKL TRUE)
endif()
if(WITH_MKL)
  demo_deps_library(${LITE_IOMP5_LIB_NAME} ${MKLML_LIB_DIR})
endif()

# 5.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

# 6. link libraries
message(STATUS "dep_libs=${dep_libs}")
target_link_libraries(${TARGET_EXE} ${dep_libs} ${OpenCV_LIBS})

if(WIN32)
  target_link_libraries(${TARGET_EXE} shlwapi.lib)
  add_custom_command(TARGET ${TARGET_EXE} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_LIB_DIR}/mklml.dll ${CMAKE_BINARY_DIR}/Release
      COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_LIB_DIR}/libiomp5md.dll ${CMAKE_BINARY_DIR}/Release
  )
endif()