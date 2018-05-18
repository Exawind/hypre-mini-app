
add_executable(hypre_app
  mmio.c
  HypreSystem.cpp
  main.cpp
  )
target_link_libraries(hypre_app ${HYPRE_LIBRARIES} ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(hypre_app PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif(MPI_COMPILE_FLAGS)
if(MPI_LINK_FLAGS)
  set_target_properties(hypre_app PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif(MPI_LINK_FLAGS)
