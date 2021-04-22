cmake_minimum_required(VERSION 3.1)
project(hypre_app CXX C)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(ENABLE_CUDA "Enable build targeting GPU" OFF)
option(ENABLE_UMPIRE "Enable Umpire GPU memory pools" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS FALSE)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

if (ENABLE_CUDA)
enable_language(CUDA)
find_package(CUDA REQUIRED)
find_library(CUDA_NVTX_LIBRARY
  NAMES nvToolsExt
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib)
endif()

find_package(HYPRE REQUIRED)
include_directories(SYSTEM ${HYPRE_INCLUDE_DIRS})

if (ENABLE_UMPIRE)
  set(CMAKE_PREFIX_PATH ${UMPIRE_DIR} ${CMAKE_PREFIX_PATH})
  find_package(Umpire REQUIRED)
  include_directories(SYSTEM ${UMPIRE_INCLUDE_DIRS})
endif()

set(CMAKE_PREFIX_PATH ${YAML_ROOT_DIR} ${CMAKE_PREFIX_PATH})
find_package(YAML-CPP)
if (YAML-CPP_FOUND)
  include_directories(SYSTEM ${YAML_CPP_INCLUDE_DIR})
else()
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Build tests")
  set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Build tools")
  set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Build contrib")

  add_subdirectory(deps/yaml-cpp)
  include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/deps/yaml-cpp/include/")
endif()

if (ENABLE_CUDA)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "--relocatable-device-code=true ")
endif()
add_subdirectory(src)
