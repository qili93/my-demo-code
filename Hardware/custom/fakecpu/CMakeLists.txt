# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

cmake_minimum_required(VERSION 3.10)

project(paddle-fake-cpu CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

option(WITH_TESTING    "Compile with unit testing"               OFF)
option(ON_INFER        "Compile with Paddle Inference C++ lib"   OFF)
option(WITH_MKL        "Compile with MKL/OpenBlas support, default use MKL."      ON)
option(WITH_STATIC_LIB "Compile with static/shared library, default use shared."  OFF)

set(FAKE_CPU_NAME      "paddle-fake-cpu")
set(FAKE_CPU_VERSION   "0.0.1")

if(ON_INFER)
  include(paddle_inference)
else()
  include(paddle_training)
endif()
link_directories(${PADDLE_LIB_DIR})
include(generic)
# include(third_party)

add_definitions(-std=c++14)
# TODO(qiil93): avoid compile error, to be removed
add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)

file(GLOB_RECURSE FAKE_CPU_SRCS RELATIVE ${CMAKE_SOURCE_DIR} kernels/*.cc)
list(APPEND FAKE_CPU_SRCS runtime/runtime.cc)

# build shared library
add_library(${FAKE_CPU_NAME} SHARED ${FAKE_CPU_SRCS})
target_link_libraries(${FAKE_CPU_NAME} PRIVATE ${PADDLE_CORE_LIB})  # special name

# link third_party
include(third_party)
add_dependencies(${FAKE_CPU_NAME} third_party)

if(NOT ON_INFER)
  # packing wheel package
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
  add_custom_command(TARGET ${FAKE_CPU_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/lib${FAKE_CPU_NAME}.so ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
      COMMENT "Creating plugin dirrectories------>>>"
  )
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
      COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
      DEPENDS ${FAKE_CPU_NAME}
      COMMENT "Packing whl packages------>>>"
  )
  add_custom_target(python_package ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
endif()

if (WITH_TESTING)
  set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Paddle")
  enable_testing()
  add_subdirectory(tests)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
  COMMAND cp -r ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(python_tests ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp)
endif()
