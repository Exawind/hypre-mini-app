
#include "HypreSystem.h"
#include "mpi.h"

#include "yaml-cpp/yaml.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <chrono>
#define useProjection 1

int main(int argc, char* argv[])
{
  int iproc;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
  //HYPRE_Init(argc, argv);
  auto start = std::chrono::system_clock::now();


  if (argc != 2) {
    std::cout << "ERROR!! Incorrect arguments passed to program." << std::endl
      << "Usage: hypre_app INPUT_FILE" << std::endl << std::endl;
    return 1;
  }

  std::string yaml_filename(argv[1]);
  YAML::Node inpfile = YAML::LoadFile(yaml_filename);

  nalu::HypreSystem linsys(MPI_COMM_WORLD, inpfile);

  linsys.loadSetup();
  int num_matrices = linsys.get_num_matrices();
  printf("TOTAL NUM MATRICES %d ", num_matrices);

  for (int ii = 1; ii<= num_matrices; ++ii){
    printf("\n\n\n ============== LOADING MATRIX %d ======================================================\n\n\n", ii);
    linsys.loadMatrix(ii);
   if (useProjection){

      linsys.solve2();
      linsys.projectionSpaceUpdate(ii);
    }
    else{
      linsys.solve();
    }
    linsys.destroyMatrix();
}


  linsys.summarize_timers();
  linsys.check_solution();
  linsys.output_linear_system();

  MPI_Barrier(MPI_COMM_WORLD);
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  if (iproc == 0)
    std::cout << "Total time: " << elapsed.count() << " seconds" << std::endl;

  MPI_Finalize();
  return 0;
}
