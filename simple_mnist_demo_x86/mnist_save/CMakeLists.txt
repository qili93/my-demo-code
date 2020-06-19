cmake_minimum_required(VERSION 3.4.1)

project(mnist_save)

if(${LIB_DIR_PATH} STREQUAL "")
  message(FATAL_ERROR "LIB_DIR_PATH not set.")
  return()
endif()

include_directories(${LIB_DIR_PATH}/include)
add_executable(mnist_save mnist_save.cc)
if(USE_FULL_API)
    add_library(paddle_api_shared SHARED IMPORTED GLOBAL)
    set_property(TARGET paddle_api_shared PROPERTY IMPORTED_LOCATION ${LIB_DIR_PATH}/lib/libpaddle_full_api_shared.so)
else()
    add_library(paddle_api_shared SHARED IMPORTED GLOBAL)
    set_property(TARGET paddle_api_shared PROPERTY IMPORTED_LOCATION ${LIB_DIR_PATH}/lib/libpaddle_light_api_shared.so)
endif()

target_link_libraries(mnist_save paddle_api_shared)