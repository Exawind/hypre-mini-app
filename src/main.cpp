#include "HypreSystem.h"

#if defined(HYPRE_USING_CUDA)
#include <cuda_runtime.h>
#elif defined(HYPRE_USING_HIP)
#include <hip/hip_runtime.h>
#endif

int getDevice(int nproc, int device_count) {
  int rank, is_rank0, nodes;
  MPI_Comm shmcomm;

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &shmcomm);
  MPI_Comm_rank(shmcomm, &rank);
  is_rank0 = (rank == 0) ? 1 : 0;
  MPI_Allreduce(&is_rank0, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Comm_free(&shmcomm);

  // Here we make sure that we don't try to mod with more than the actual number
  // of devices.
  int ngpus_to_use = nproc / nodes;
  ngpus_to_use = ngpus_to_use < device_count ? ngpus_to_use : device_count;

  // Here we assign device based on the rank in the shmcomm, not the global.
  int device = rank % ngpus_to_use;

  return device;
}

int main(int argc, char *argv[]) {
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
  if (iproc == 0)
	  printf("\trank=%d : %s %s %d : %s (cc=%d.%d): device=%d of %d : free "
				"memory=%1.8g GB, total memory=%1.8g GB\n",
				iproc, __FUNCTION__, __FILE__, __LINE__, prop.name, prop.major,
				prop.minor, device, count, free / 1.e9, total / 1.e9);
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
  if (iproc == 0)
	  printf("rank=%d : %s %s %d : %s arch=%d : device=%d of %d : free "
				"memory=%1.8g GB, total memory=%1.8g GB\n",
				iproc, __FUNCTION__, __FILE__, __LINE__, prop.name, prop.gcnArch,
				device, count, free / 1.e9, total / 1.e9);
#endif
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);

  HYPRE_Int ret = HYPRE_Init();

  auto start = std::chrono::system_clock::now();

  if (argc != 2) {
    std::cout << "ERROR!! Incorrect arguments passed to program." << std::endl
              << "Usage: hypre_app INPUT_FILE" << std::endl
              << std::endl;
    return 1;
  }

  std::string yaml_filename(argv[1]);
  YAML::Node inpfile = YAML::LoadFile(yaml_filename);
  YAML::Node node = inpfile["solver_settings"];

#ifdef HYPRE_USING_DEVICE_POOL
  /* CUB Allocator */
  hypre_uint mempool_bin_growth = 8, mempool_min_bin = 3, mempool_max_bin = 9;
  size_t mempool_max_cached_bytes = 2000LL * 1024 * 1024;

  /* To be effective, hypre_SetCubMemPoolSize must immediately follow HYPRE_Init
   */
  HYPRE_SetGPUMemoryPoolSize(mempool_bin_growth, mempool_min_bin,
                             mempool_max_bin, mempool_max_cached_bytes);
#endif

#if defined(HYPRE_USING_UMPIRE)
  long long device_pool_size =
      nalu::get_optional(node, "umpire_device_pool_mbs", 4096);
  if (!iproc)
    std::cout << "umpire_device_pool_mbs=" << device_pool_size << std::endl;
  HYPRE_SetUmpireDevicePoolSize(device_pool_size * 1024 * 1024);
#endif

#ifdef HYPRE_USING_GPU
  HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_DEVICE;
  HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_DEVICE;

  /* default memory location */
  HYPRE_SetMemoryLocation(memory_location);

  /* default execution policy */
  HYPRE_SetExecutionPolicy(default_exec_policy);

  if (nalu::get_optional(node, "spgemm_use_vendor", 0) == 1) {
    if (!iproc)
      std::cout << "Using VENDOR SpGemm." << std::endl;
    HYPRE_SetSpGemmUseVendor(true);
  } else {
    if (!iproc)
      std::cout << "NOT Using VENDOR SpGemm." << std::endl;
    HYPRE_SetSpGemmUseVendor(false);
  }

  if (nalu::get_optional(node, "spmv_use_vendor", 0) == 1) {
    if (!iproc)
      std::cout << "Using VENDOR SpMV." << std::endl;
    HYPRE_SetSpMVUseVendor(true);
  } else {
    if (!iproc)
      std::cout << "NOT Using VENDOR SpMV." << std::endl;
    HYPRE_SetSpMVUseVendor(false);
  }

  if (nalu::get_optional(node, "sptrans_use_vendor", 0) == 1) {
    if (!iproc)
      std::cout << "Using VENDOR SpTrans." << std::endl;
    HYPRE_SetSpTransUseVendor(true);
  } else {
    if (!iproc)
      std::cout << "NOT Using VENDOR SpTrans." << std::endl;
    HYPRE_SetSpTransUseVendor(false);
  }
#endif

  /* Timers dumped to a csv file for mutliple tests */
  std::string csv_profile_file = nalu::get_optional<std::string>(node, "csv_profile_file", "");
  bool found_csv_profile_file = csv_profile_file.size()>0;
  std::vector<std::string> names(0);
  std::vector<std::vector<double>> data(0);

  HYPRE_Int num_tests = nalu::get_optional(node, "num_tests", 1);
  for (int i=0; i<num_tests; ++i)
  {
	  // reset the random number generator
	  hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);

	  nalu::HypreSystem linsys(MPI_COMM_WORLD, inpfile);

	  linsys.load();

#ifdef HYPRE_USING_CUDA
	  cudaMemGetInfo(&free, &total);
	  if (iproc == 0)
		  printf("\trank=%d : %s %s %d : %s (cc=%d.%d): device=%d of %d : free "
					"memory=%1.8g GB, total memory=%1.8g GB\n",
					iproc, __FUNCTION__, __FILE__, __LINE__, prop.name, prop.major,
					prop.minor, device, count, free / 1.e9, total / 1.e9);
#endif

#ifdef HYPRE_USING_HIP
	  hipMemGetInfo(&free, &total);
	  if (iproc == 0)
		  printf("rank=%d : %s %s %d : %s arch=%d : device=%d of %d : free "
					"memory=%1.8g GB, total memory=%1.8g GB\n",
					iproc, __FUNCTION__, __FILE__, __LINE__, prop.name, prop.gcnArch,
					device, count, free / 1.e9, total / 1.e9);
#endif
	  fflush(stdout);
	  MPI_Barrier(MPI_COMM_WORLD);

	  linsys.solve();

	  linsys.check_solution();
	  linsys.output_linear_system();
	  linsys.summarize_timers();

	  if (found_csv_profile_file)
		  linsys.retrieve_timers(names,data);

	  MPI_Barrier(MPI_COMM_WORLD);
	  auto stop = std::chrono::system_clock::now();
	  std::chrono::duration<double> elapsed = stop - start;
	  if (iproc == 0)
		  std::cout << "Total time: " << elapsed.count() << " seconds" << std::endl;

	  linsys.destroy_system();
  }

  if (found_csv_profile_file)
  {
	  FILE * fid = fopen(csv_profile_file.c_str(), "wt");
	  size_t N = names.size();
	  size_t M = data[0].size();
	  for (int i = 0; i < N; ++i)
	  {
		  if (i<N-1)
			  fprintf(fid, "%s,", names[i].c_str());
		  else
			  fprintf(fid, "%s\n", names[i].c_str());
	  }
	  for (int j = 0; j < M; ++j)
		  for (int i = 0; i < N; ++i)
		  {
			  if (i<N-1)
				  fprintf(fid, "%1.15g,", data[i][j]);
			  else
			  fprintf(fid, "%1.15g\n", data[i][j]);
		  }
	  fclose(fid);
  }

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
