
cmake_minimum_required(VERSION 3.1)
project(hypre_app CXX C)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS FALSE)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

find_package(HYPRE REQUIRED)
include_directories(SYSTEM ${HYPRE_INCLUDE_DIRS})

set(YAML_CPP_BUILD_TESTS OFF)
set(YAML_CPP_BUILD_TOOLS OFF)
set(YAML_CPP_BUILD_CONTRIB OFF)

add_subdirectory(deps/yaml-cpp)

include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/deps/yaml-cpp/include/")
add_subdirectory(src)
