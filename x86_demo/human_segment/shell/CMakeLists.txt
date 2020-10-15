cmake_minimum_required(VERSION 3.10)

project(human_seg_demo)

set(TARGET_EXE human_seg_demo)

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
if(NOT DEFINED USE_FULL_API)
  set(USE_FULL_API TRUE)
endif()

if(APPLE)
  set(LITE_FULL_LIB_NAME "libpaddle_full_api_shared.dylib")
  set(LITE_TINY_LIB_NAME "libpaddle_light_api_shared.dylib")
  # set(LITE_MKLML_LIB_NAME "libmklml.dylib")
  set(LITE_IOMP5_LIB_NAME "libiomp5.dylib")
elseif(WIN32)
  set(LITE_FULL_LIB_NAME "libpaddle_api_full_bundled.lib")
  set(LITE_TINY_LIB_NAME "libpaddle_api_light_bundled.lib")
  # set(LITE_MKLML_LIB_NAME "mklml.lib")
  set(LITE_IOMP5_LIB_NAME "libiomp5md.lib")
else()
  set(LITE_FULL_LIB_NAME "libpaddle_full_api_shared.so")
  set(LITE_TINY_LIB_NAME "libpaddle_light_api_shared.so")
  # set(LITE_MKLML_LIB_NAME "libmklml.so")
  set(LITE_IOMP5_LIB_NAME "libiomp5.so")
endif()

# message(STATUS "LITE_FULL_LIB_NAME=${LITE_FULL_LIB_NAME}")
# message(STATUS "LITE_TINY_LIB_NAME=${LITE_TINY_LIB_NAME}")
# message(STATUS "LITE_MKLML_LIB_NAME=${LITE_MKLML_LIB_NAME}")
# message(STATUS "LITE_IOMP5_LIB_NAME=${LITE_IOMP5_LIB_NAME}")

# find paddle-lite cxx libs
if(USE_FULL_API)
  add_definitions(-DUSE_FULL_API)
  find_library(PADDLE_LIBRARY_FILE NAMES ${LITE_FULL_LIB_NAME} PATHS "${PADDLE_LIB_DIR}" NO_DEFAULT_PATH)
else()
  find_library(PADDLE_LIBRARY_FILE NAMES ${LITE_TINY_LIB_NAME}  PATHS "${PADDLE_LIB_DIR}" NO_DEFAULT_PATH)
endif()
if(NOT PADDLE_LIBRARY_FILE)
  message(FATAL_ERROR "Can not find Paddle-Lite Library in ${PADDLE_LIB_DIR}")
else()
  message(STATUS "Found Paddle-Lite Library of ${PADDLE_LIBRARY_FILE}")
  if(WIN32)
    add_library(paddle_lite_api STATIC IMPORTED GLOBAL)
  else()
    add_library(paddle_lite_api SHARED IMPORTED GLOBAL)
  endif()
  set_property(TARGET paddle_lite_api PROPERTY IMPORTED_LOCATION ${PADDLE_LIBRARY_FILE})
endif()

# find intel mklml libs
# find_library(MKLML_LIBRARY_FILE NAMES ${LITE_MKLML_LIB_NAME} PATHS "${MKLML_LIB_DIR}" NO_DEFAULT_PATH)
# if(NOT MKLML_LIBRARY_FILE)
#   message(FATAL_ERROR "Can not find ${LITE_MKLML_LIB_NAME} Library in ${MKLML_LIB_DIR}")
# else()
#   message(STATUS "Found Intel MKLML Library of ${MKLML_LIBRARY_FILE}")
#   if(WIN32)
#     add_library(intel_mklml STATIC IMPORTED GLOBAL)
#   else()
#     add_library(intel_mklml SHARED IMPORTED GLOBAL)
#   endif()
#   set_property(TARGET intel_mklml PROPERTY IMPORTED_LOCATION ${MKLML_LIBRARY_FILE})
# endif()

find_library(IOMP5_LIBRARY_FILE NAMES ${LITE_IOMP5_LIB_NAME} PATHS "${MKLML_LIB_DIR}" NO_DEFAULT_PATH)
if(NOT IOMP5_LIBRARY_FILE)
  message(FATAL_ERROR "Can not find ${LITE_IOMP5_LIB_NAME} Library in ${MKLML_LIB_DIR}")
else()
  message(STATUS "Found Intel IOMP5 Library of ${IOMP5_LIBRARY_FILE}")
  if(WIN32)
    add_library(intel_iomp5 STATIC IMPORTED GLOBAL)
  else()
    add_library(intel_iomp5 SHARED IMPORTED GLOBAL)
  endif()
  set_property(TARGET intel_iomp5 PROPERTY IMPORTED_LOCATION ${IOMP5_LIBRARY_FILE})
endif()

# 5.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

# 6. link libraries
target_link_libraries(${TARGET_EXE} paddle_lite_api intel_iomp5)

if(WIN32)
  target_link_libraries(${TARGET_EXE} shlwapi.lib)
  add_custom_command(TARGET ${TARGET_EXE} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_LIB_DIR}/mklml.dll ${CMAKE_BINARY_DIR}/Release
      COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_LIB_DIR}/libiomp5md.dll ${CMAKE_BINARY_DIR}/Release
  )
endif()