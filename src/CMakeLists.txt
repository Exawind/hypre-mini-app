
add_executable(hypre_app
  mmio.c
  HypreSystem.cpp
  main.cpp
  )
if (YAML-CPP_FOUND)
  target_link_libraries(hypre_app ${YAML_CPP_LIBRARIES})
else()
  target_link_libraries(hypre_app yaml-cpp)
endif()

target_link_libraries(hypre_app
  ${HYPRE_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
  ${MPI_C_LIBRARIES}
  ${CUDA_cusparse_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_NVTX_LIBRARY}
  )

set_target_properties(hypre_app PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  )

if(MPI_COMPILE_FLAGS)
  set_target_properties(hypre_app PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif(MPI_COMPILE_FLAGS)
if(MPI_LINK_FLAGS)
  set_target_properties(hypre_app PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif(MPI_LINK_FLAGS)
