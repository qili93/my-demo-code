cmake_minimum_required(VERSION 3.10)
project(model_test)
set(TARGET_EXE model_test)

# 1. path to Paddle-Lite lib
if(${PADDLE_LITE_DIR} STREQUAL "")
  message(FATAL_ERROR "PADDLE_LITE_DIR not set.")
  return()
endif()
message(STATUS "PADDLE_LITE_DIR=${PADDLE_LITE_DIR}")

# 2. link Paddle-Lite directory
#link_directories(${PADDLE_LITE_DIR}/lib)
include_directories(${PADDLE_LITE_DIR}/include)

# 3.add executable output
add_executable(${TARGET_EXE} ${TARGET_EXE}.cc)

# 4. add andriod ndk log
find_library(log-lib log)
message(STATUS "log-lib=${log-lib}")

# 5. link paddle-lite libraries
if(NOT DEFINED USE_FULL_API)
  set(USE_FULL_API TRUE)
endif()

if(USE_FULL_API)
  add_definitions(-DUSE_FULL_API)
  add_library(paddle_api_shared SHARED IMPORTED GLOBAL)
  set_property(TARGET paddle_api_shared PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/lib/libpaddle_full_api_shared.so)
else()
  add_library(paddle_api_shared SHARED IMPORTED GLOBAL)
  set_property(TARGET paddle_api_shared PROPERTY IMPORTED_LOCATION ${PADDLE_LITE_DIR}/lib/libpaddle_light_api_shared.so)
endif()

target_link_libraries(${TARGET_EXE} paddle_api_shared ${log-lib})