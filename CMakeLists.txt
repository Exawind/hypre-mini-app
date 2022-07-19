############################ BASE ######################################
cmake_minimum_required (VERSION 3.17.0 FATAL_ERROR)
project(hypre_app CXX C)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

########################## OPTIONS #####################################
option(ENABLE_CUDA "Enable Nvidia CUDA" OFF)
option(ENABLE_HIP "Enable HIP/ROCm" OFF)
option(ENABLE_UMPIRE "Enable Umpire GPU memory pools" OFF)

set(CMAKE_CXX_STANDARD 14)       # Set nalu-wind C++14 standard
set(CMAKE_CXX_EXTENSIONS OFF)    # Do not enable GNU extensions
set(CMAKE_CXX_STANDARD_REQUIRED) # Force error if C++14 standard is not supported

# Create targets
set(hypre_app_name "hypre_app")
add_library(hypre_app_lib "")
add_executable(${hypre_app_name} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

########################## MPI ####################################
find_package(MPI REQUIRED)
target_link_libraries(hypre_app_lib PUBLIC $<$<BOOL:${MPI_CXX_FOUND}>:MPI::MPI_CXX>)
target_link_libraries(hypre_app_lib PUBLIC $<$<BOOL:${MPI_Fortran_FOUND}>:MPI::MPI_Fortran>)


########################## CUDA ####################################
if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Found CUDAToolkit = ${CUDAToolkit_VERSION} (${CUDAToolkit_LIBRARY_DIR})")
  target_link_libraries(hypre_app_lib PUBLIC
    CUDA::cusparse
    CUDA::curand
    CUDA::cudart
    CUDA::cublas
    CUDA::nvToolsExt)
  # Are these needed??
  #target_compile_options(hypre_app_lib PUBLIC $<$<COMPILE_LANGUAGE:CXX>:--expt-relaxed-constexpr>)
  #set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "--relocatable-device-code=true ")
endif()


########################## HIP ####################################
if (ENABLE_HIP)
  find_package(HIP REQUIRED)
  if (NOT DEFINED AMD_ARCH)
    # Set default AMD architectures (based on Tulip for now)
    set(AMD_ARCH "gfx906,gfx908")
  endif()

  message(STATUS "HIP: ${HIP_VERSION}, platform = ${HIP_PLATFORM}, compiler = ${HIP_COMPILER}")
  target_link_libraries(hypre_app_lib PUBLIC
    hip::host)
endif()


########################## HYPRE ####################################
set(CMAKE_PREFIX_PATH ${HYPRE_DIR} ${CMAKE_PREFIX_PATH})
find_package(HYPRE 2.20.0 REQUIRED)
message(STATUS "Found HYPRE = ${HYPRE_DIR}")
target_link_libraries(hypre_app_lib PUBLIC ${HYPRE_LIBRARIES})
target_include_directories(hypre_app_lib SYSTEM PUBLIC ${HYPRE_INCLUDE_DIRS})
include(CheckCXXSymbolExists)
check_cxx_symbol_exists(
  HYPRE_BIGINT "${HYPRE_INCLUDE_DIRS}/HYPRE_config.h" HYPRE_APP_BIGINT)
if(NOT HYPRE_APP_BIGINT AND NOT ENABLE_CUDA AND NOT ENABLE_HIP)
  message(WARNING
    "HYPRE does not enable 64-bit integer support; will fail on large problems!")
endif()

  
########################## UMPIRE ####################################
if (ENABLE_UMPIRE)
  set(CMAKE_PREFIX_PATH ${UMPIRE_DIR} ${CMAKE_PREFIX_PATH})
  message(STATUS "Looking for Umpire at ${UMPIRE_DIR}")
  find_package(umpire REQUIRED)
  message(STATUS "Found Umpire at ${umpire_DIR}")
  target_include_directories(hypre_app_lib SYSTEM PUBLIC ${umpire_DIR}/../../../include)
endif()


########################## YAML ####################################
set(CMAKE_PREFIX_PATH ${YAML_ROOT_DIR} ${CMAKE_PREFIX_PATH})
find_package(YAML-CPP 0.6.3 QUIET REQUIRED)
message(STATUS "Found YAML-CPP = ${YAML_CPP_CMAKE_DIR}")
target_link_libraries(hypre_app_lib PUBLIC yaml-cpp)
target_include_directories(hypre_app_lib SYSTEM PUBLIC ${YAML_CPP_INCLUDE_DIR})

########################### Hypre Mini APp #####################################
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")


# Logic for handling warnings
if(ENABLE_ALL_WARNINGS)
  # GCC, Clang, and Intel seem to accept these
  list(APPEND HYPRE_APP_CXX_FLAGS "-Wall" "-Wextra" "-pedantic")
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # ifort doesn't like -Wall
    list(APPEND HYPRE_APP_Fortran_FLAGS "-Wall")
  else()
    # Intel always reports some diagnostics we don't necessarily care about
    list(APPEND HYPRE_APP_CXX_FLAGS "-diag-disable:11074,11076")
  endif()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.0)
    # Avoid notes about -faligned-new with GCC > 7
    list(APPEND HYPRE_APP_CXX_FLAGS "-faligned-new")
  endif()
endif()

# Add our extra flags according to language
separate_arguments(HYPRE_APP_CXX_FLAGS)
separate_arguments(HYPRE_APP_Fortran_FLAGS)
target_compile_options(hypre_app_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${HYPRE_APP_CXX_FLAGS}>)
target_compile_options(hypre_app_lib PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${HYPRE_APP_Fortran_FLAGS}>)
target_compile_options(hypre_app_lib PRIVATE $<$<BOOL:${ENABLE_WERROR}>:-Werror>)

target_link_libraries(${hypre_app_name} PRIVATE hypre_app_lib)
add_subdirectory(src)
install(TARGETS ${hypre_app_name})
