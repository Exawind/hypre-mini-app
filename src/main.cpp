#include "HypreSystem.h"

#if defined(HYPRE_USING_CUDA)
#include <cuda_runtime.h>
#elif defined(HYPRE_USING_HIP)
#include <hip/hip_runtime.h>
#endif

int getDevice(int nproc, int device_count)
{
  int rank, is_rank0, nodes;
  MPI_Comm shmcomm;

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
		      MPI_INFO_NULL, &shmcomm);
  MPI_Comm_rank(shmcomm, &rank);
  is_rank0 = (rank == 0) ? 1 : 0;
  MPI_Allreduce(&is_rank0, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Comm_free(&shmcomm);

  // Here we make sure that we don't try to mod with more than the actual number of devices.
  int ngpus_to_use = nproc/nodes;
  ngpus_to_use = ngpus_to_use < device_count ? ngpus_to_use : device_count;
  
  // Here we assign device based on the rank in the shmcomm, not the global.
  int device = rank%ngpus_to_use;

  return device;
}

int main(int argc, char* argv[])
{
  int iproc, nproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

#ifdef HYPRE_USING_CUDA
    int count;
    cudaGetDeviceCount(&count);

    // get the device from
    int device = getDevice(nproc, count);

    // set the device before calling HypreInit.
    cudaSetDevice(device);
    cudaGetDevice(&device);
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("\trank=%d : %s %s %d : %s (cc=%d.%d): device=%d of %d : free memory=%1.8g GB, total memory=%1.8g GB\n",
	   iproc,__FUNCTION__,__FILE__,__LINE__,prop.name,prop.major,prop.minor,device,count,free/1.e9,total/1.e9);
#endif

#ifdef HYPRE_USING_HIP
    int count;
    hipGetDeviceCount(&count);

    // get the device from
    int device = getDevice(nproc, count);

    // set the device before calling HypreInit.
    hipSetDevice(device);
    hipGetDevice(&device);
    size_t free, total;
    hipMemGetInfo(&free, &total);

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    printf("rank=%d : %s %s %d : %s arch=%d : device=%d of %d : free memory=%1.8g GB, total memory=%1.8g GB\n",
	   iproc,__FUNCTION__,__FILE__,__LINE__,prop.name,prop.gcnArch,device,count,free/1.e9,total/1.e9);
#endif
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    HYPRE_Int ret = HYPRE_Init();

#ifdef HYPRE_USING_CUB_ALLOCATOR
    /* CUB Allocator */
    hypre_uint mempool_bin_growth   = 8,
      mempool_min_bin      = 3,
      mempool_max_bin      = 9;
    size_t mempool_max_cached_bytes = 2000LL * 1024 * 1024;

   /* To be effective, hypre_SetCubMemPoolSize must immediately follow HYPRE_Init */
   HYPRE_SetGPUMemoryPoolSize( mempool_bin_growth, mempool_min_bin,
                               mempool_max_bin, mempool_max_cached_bytes );
#endif

#if defined(HYPRE_USING_UMPIRE)
   /* Setup Umpire pools */
   //HYPRE_SetUmpireDevicePoolName("HYPRE_DEVICE_POOL");
   HYPRE_SetUmpireUMPoolName("HYPRE_UM_POOL");
   //HYPRE_SetUmpireHostPoolName("HYPRE_HOST_POOL");
   //HYPRE_SetUmpirePinnedPoolName("HYPRE_PINNED_POOL");
   //HYPRE_SetUmpireDevicePoolSize(4LL * 1024 * 1024 * 1024);
   HYPRE_SetUmpireUMPoolSize(1LL * 1024 * 1024 * 1024);
   //HYPRE_SetUmpireHostPoolSize(1LL * 1024 * 1024 * 1024 / 1024);
   //HYPRE_SetUmpirePinnedPoolSize(1LL * 1024 * 1024 * 1024 / 1024);
#endif

#ifdef HYPRE_USING_GPU
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_DEVICE;
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_DEVICE;

   /* default memory location */
   HYPRE_SetMemoryLocation(memory_location);

   /* default execution policy */
   HYPRE_SetExecutionPolicy(default_exec_policy);

   HYPRE_CSRMatrixSetSpGemmUseCusparse(false);
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

#ifdef HYPRE_USING_CUDA
    cudaMemGetInfo(&free, &total);
    printf("\trank=%d : %s %s %d : %s (cc=%d.%d): device=%d of %d : free memory=%1.8g GB, total memory=%1.8g GB\n",
	   iproc,__FUNCTION__,__FILE__,__LINE__,prop.name,prop.major,prop.minor,device,count,free/1.e9,total/1.e9);
#endif

#ifdef HYPRE_USING_HIP
    hipMemGetInfo(&free, &total);
    printf("rank=%d : %s %s %d : %s arch=%d : device=%d of %d : free memory=%1.8g GB, total memory=%1.8g GB\n",
	   iproc,__FUNCTION__,__FILE__,__LINE__,prop.name,prop.gcnArch,device,count,free/1.e9,total/1.e9);
#endif
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    linsys.solve();

    linsys.check_solution();
    linsys.output_linear_system();
    linsys.summarize_timers();

    MPI_Barrier(MPI_COMM_WORLD);
    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    if (iproc == 0)
        std::cout << "Total time: " << elapsed.count() << " seconds" << std::endl;

    linsys.destroy_system();

    HYPRE_Finalize();

    MPI_Finalize();

    /* Need this at the end so cuda memcheck leak-check can work properly */
#if defined(HYPRE_USING_CUDA)
   cudaDeviceReset();
#elif defined(HYPRE_USING_HIP)
   hipDeviceReset();
#endif
    return 0;
}
