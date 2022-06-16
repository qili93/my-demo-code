cmake_minimum_required(VERSION 3.10)

project(paddle-custom-cpu CXX C)

# Use different compiler of clang++
set(CMAKE_CXX_COMPILER "clang++")

set(CUSTOM_CPU_NAME        "paddle-custom-cpu")
set(CUSTOM_CPU_VERSION     "0.0.1")

# Import paddle from lib/python3.7/site-packages/paddle
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
include(paddle)
include_directories(${PADDLE_INC_DIR})

# add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)  # for out CustomContext

# Add custom cpu target and link to paddlepaddle
add_library(${CUSTOM_CPU_NAME} SHARED runtime.cc)
target_link_libraries(${CUSTOM_CPU_NAME} PRIVATE ${PADDLE_CORE_LIB})

# Install the libpaddle-custom-cpu.so to lib/python3.7/site-packages/paddle-plugins
add_custom_command(TARGET ${CUSTOM_CPU_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${Python_SITEARCH}/paddle-plugins/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/lib${CUSTOM_CPU_NAME}.so ${Python_SITEARCH}/paddle-plugins/
    COMMENT "Copying lib${CUSTOM_CPU_NAME}.so to ${Python_SITEARCH}/paddle-plugins/"
)
