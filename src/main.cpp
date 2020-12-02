
#include "HypreSystem.h"
#include "mpi.h"

#include "yaml-cpp/yaml.h"

#include <iostream>
#include <chrono>

int main(int argc, char* argv[])
{
    int iproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);

    HYPRE_Int ret = HYPRE_Init();

#ifdef HAVE_CUDA
    int count, device;
    cudaGetDeviceCount(&count);
    //cudaSetDevice(iproc_ % count);
    cudaGetDevice(&device);
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("\trank=%d : %s %s %d : device=%d of %d : free memory=%1.8g GB, total memory=%1.8g GB\n",
	   iproc,__FUNCTION__,__FILE__,__LINE__,device,count,free/1.e9,total/1.e9);

    hypre_HandleDefaultExecPolicy(hypre_handle()) = HYPRE_EXEC_DEVICE;
    hypre_HandleSpgemmUseCusparse(hypre_handle()) = 0;
#endif

    auto start = std::chrono::system_clock::now();


    if (argc != 2) {
        std::cout << "ERROR!! Incorrect arguments passed to program." << std::endl
                  << "Usage: hypre_app INPUT_FILE" << std::endl << std::endl;
        return 1;
    }

    std::string yaml_filename(argv[1]);
    YAML::Node inpfile = YAML::LoadFile(yaml_filename);

    nalu::HypreSystem linsys(MPI_COMM_WORLD, inpfile);

    linsys.load();
    linsys.solve();

    linsys.check_solution();
    linsys.output_linear_system();
    linsys.summarize_timers();

    MPI_Barrier(MPI_COMM_WORLD);
    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    if (iproc == 0)
        std::cout << "Total time: " << elapsed.count() << " seconds" << std::endl;

#ifdef HAVE_CUDA
    HYPRE_Finalize();
#endif

    MPI_Finalize();
#ifdef HAVE_CUDA
    /* Need this at the end so cuda memcheck leak-check can work properly */
    cudaDeviceReset();
#endif
    return 0;
}
