#if defined (USE_GPU)

#define GPU_HOST __host__
#define GPU_DEVICE __device__
#define GPU_GLOBAL __global__
#define GPU_HOST_DEVICE __host__ __device__

#else

#define GPU_HOST
#define GPU_DEVICE
#define GPU_GLOBAL
#define GPU_HOST_DEVICE

#endif

