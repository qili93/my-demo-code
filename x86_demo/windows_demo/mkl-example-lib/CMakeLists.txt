
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

set(MKL_INC_DIR ${MKL_ROOT}/include)
set(MKL_LIB_DIR ${MKL_ROOT}/lib/intel64)
set(MKL_IOMP_DIR ${IOMP_ROOT}/lib/intel64)

include_directories(${MKL_INC_DIR})

# 2. Add MKL Libs
set(dep_libs CACHE INTERNAL "dependent libraries")
function(demo_deps_library TARGET_NAME PATH_DIR)
  find_library(${TARGET_NAME}_FILE NAMES ${TARGET_NAME} PATHS "${PATH_DIR}" NO_DEFAULT_PATH)
  if(NOT ${TARGET_NAME}_FILE)
    message(FATAL_ERROR "Can not find ${TARGET_NAME} Library in ${PATH_DIR}")
  else()
    message(STATUS "Found ${TARGET_NAME}: ${${TARGET_NAME}_FILE}")
    add_library(${TARGET_NAME}_lib_dep STATIC IMPORTED GLOBAL)
    set_property(TARGET ${TARGET_NAME}_lib_dep PROPERTY IMPORTED_LOCATION "${${TARGET_NAME}_FILE}")
    set(dep_libs "${dep_libs};${${TARGET_NAME}_FILE}" CACHE INTERNAL "")
  endif()
endfunction()

demo_deps_library(mkl_intel_lp64 ${MKL_LIB_DIR})
demo_deps_library(mkl_intel_thread ${MKL_LIB_DIR})
demo_deps_library(mkl_core ${MKL_LIB_DIR})
demo_deps_library(libiomp5md ${MKL_IOMP_DIR})
message(STATUS "dep_libs=${dep_libs}")

# add static/shared lib
if(LINK_STATIC)
  add_library(mkl-lab-solution STATIC mkl-lab-solution.cc)
  target_link_libraries(mkl-lab-solution ${dep_libs})
else(LINK_STATIC)
  set(mkl-lab-solution_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}mkl-lab-solution${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE STRING "output library name for target mkl-lab-solution")
  add_library(mkl-lab-solution SHARED mkl-lab-solution.cc)
  target_link_libraries(mkl-lab-solution ${dep_libs})
endif(LINK_STATIC)

# add execution
add_executable(main_test main_test.cc)
add_dependencies(main_test mkl-lab-solution)
target_link_libraries(main_test mkl-lab-solution)

add_custom_command(TARGET ${TARGET_LIB} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${IOMP_LINK}/libiomp5md.dll ${CMAKE_BINARY_DIR}/Release
)