
cmake_minimum_required(VERSION 3.0)

project(mkl-lab-solution)

set(TARGET_LIB mkl-lab-solution)


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
message(STATUS "CMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}")

# 1. path to MKLROOT
if(${MKL_ROOT} STREQUAL "")
  message(FATAL_ERROR "MKL_ROOT not set.")
  return()
endif()
message(STATUS "MKL_ROOT=${MKL_ROOT}")

set(MKLML_INC_DIR ${MKL_ROOT}/include)
set(MKLML_LIB_DIR ${MKL_ROOT}/lib/intel64)
set(MKLML_IOMP_DIR ${IOMP_ROOT}/lib/intel64)

include_directories(${MKLML_INC_DIR})

# 2. Add MKL Libs
set(MKLML_LIBRARIES CACHE INTERNAL "dependent libraries")
function(demo_deps_library TARGET_NAME PATH_DIR)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(demo_deps_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  find_library(${TARGET_NAME}_FILE NAMES ${TARGET_NAME} PATHS "${PATH_DIR}" NO_DEFAULT_PATH)
  if(NOT ${TARGET_NAME}_FILE)
    message(FATAL_ERROR "Can not find ${TARGET_NAME} Library in ${PATH_DIR}")
  else()
    message(STATUS "Found ${TARGET_NAME}: ${${TARGET_NAME}_FILE}")
    if (demo_deps_library_SHARED OR demo_deps_library_shared)
      add_library(${TARGET_NAME}_lib_dep SHARED IMPORTED GLOBAL)
      message(STATUS "Adding Shared Lib: ${${TARGET_NAME}_FILE}")
    else()
      add_library(${TARGET_NAME}_lib_dep STATIC IMPORTED GLOBAL)
      message(STATUS "Adding Static Lib: ${${TARGET_NAME}_FILE}")
    endif()
    set_property(TARGET ${TARGET_NAME}_lib_dep PROPERTY IMPORTED_LOCATION "${${TARGET_NAME}_FILE}")
    set(MKLML_LIBRARIES "${MKLML_LIBRARIES};${${TARGET_NAME}_FILE}" CACHE INTERNAL "")
  endif()
endfunction()

# demo_deps_library(mkl_intel_lp64 ${MKL_LIB_DIR})
# demo_deps_library(mkl_intel_thread ${MKL_LIB_DIR})
# demo_deps_library(mkl_core_dll ${MKL_LIB_DIR})
# demo_deps_library(libiomp5md ${MKL_IOMP_DIR})
# message(STATUS "dep_libs=${dep_libs}")


# add static/share mkl lib
if(WITH_STATIC_MKL)
  SET(MKLML_LP64_LIB            mkl_intel_lp64)
  SET(MKLML_THREAD_LIB          mkl_intel_thread)
  SET(MKLML_CORE_LIB            mkl_core)
  SET(MKLML_IOMP_LIB            libiomp5md)
else(WITH_STATIC_MKL)
  SET(MKLML_LP64_LIB            mkl_intel_lp64_dll)
  SET(MKLML_THREAD_LIB          mkl_intel_thread_dll)
  SET(MKLML_CORE_LIB            mkl_core_dll)
  SET(MKLML_IOMP_LIB            libiomp5md)
endif(WITH_STATIC_MKL)

message(STATUS "MKLML_LP64_LIB=${MKLML_LP64_LIB}")
message(STATUS "MKLML_THREAD_LIB=${MKLML_THREAD_LIB}")
message(STATUS "MKLML_CORE_LIB=${MKLML_CORE_LIB}")
message(STATUS "MKLML_IOMP_LIB=${MKLML_IOMP_LIB}")

if(WITH_STATIC_MKL)
  demo_deps_library(${MKLML_LP64_LIB} ${MKLML_LIB_DIR} STATIC)
  demo_deps_library(${MKLML_THREAD_LIB} ${MKLML_LIB_DIR} STATIC)
  demo_deps_library(${MKLML_CORE_LIB} ${MKLML_LIB_DIR} STATIC)
  demo_deps_library(${MKLML_IOMP_LIB} ${MKLML_IOMP_DIR} SHARED)
  message(STATUS "MKLML_LIBRARIES=${MKLML_LIBRARIES}")
else()
  demo_deps_library(${MKLML_LP64_LIB} ${MKLML_LIB_DIR} SHARED)
  demo_deps_library(${MKLML_THREAD_LIB} ${MKLML_LIB_DIR} SHARED)
  demo_deps_library(${MKLML_CORE_LIB} ${MKLML_LIB_DIR} SHARED)
  demo_deps_library(${MKLML_IOMP_LIB} ${MKLML_IOMP_DIR} SHARED)
  message(STATUS "MKLML_LIBRARIES=${MKLML_LIBRARIES}")
endif()

message(STATUS "WITH_STATIC_MKL=${WITH_STATIC_MKL}")
message(STATUS "MKLML_LIBRARIES=${MKLML_LIBRARIES}")

# add static/shared lib
if(LINK_STATIC)
  add_library(mkl-lab-solution STATIC mkl-lab-solution.cc)
  target_link_libraries(mkl-lab-solution ${MKLML_LIBRARIES})
else(LINK_STATIC)
  add_library(mkl-lab-solution SHARED mkl-lab-solution.cc)
  target_link_libraries(mkl-lab-solution ${MKLML_LIBRARIES})
endif(LINK_STATIC)

# add execution
add_executable(main_test main_test.cc)
add_dependencies(main_test mkl-lab-solution)
target_link_libraries(main_test mkl-lab-solution)

add_custom_command(TARGET ${TARGET_LIB} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${DLL_COPY_DIR}/compiler/libiomp5md.dll ${CMAKE_BINARY_DIR}/Release
)

if(NOT WITH_STATIC_MKL)
  add_custom_command(TARGET ${TARGET_LIB} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${DLL_COPY_DIR}/mkl/mkl_intel_thread.dll ${CMAKE_BINARY_DIR}/Release
      COMMAND ${CMAKE_COMMAND} -E copy ${DLL_COPY_DIR}/mkl/mkl_core.dll ${CMAKE_BINARY_DIR}/Release
      COMMAND ${CMAKE_COMMAND} -E copy ${DLL_COPY_DIR}/mkl/mkl_avx2.dll ${CMAKE_BINARY_DIR}/Release
      COMMAND ${CMAKE_COMMAND} -E copy ${DLL_COPY_DIR}/mkl/mkl_def.dll ${CMAKE_BINARY_DIR}/Release
  )
endif()