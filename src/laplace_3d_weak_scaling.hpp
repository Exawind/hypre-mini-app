#include <cstring>
#include <map>
#include <mpi.h>
#include <set>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

#define HIP_CHECK(err)                          \
{                                               \
    if(err != hipSuccess)                       \
    {                                           \
        fprintf(stderr,                         \
                "HIP ERROR %d in %s line %d\n", \
                err,                            \
                __FILE__,                       \
                __LINE__);                      \
        exit(1);                                \
    }                                           \
}

struct Data
{
    // Number of rows of the global matrix
    int64_t global_nrow;
    // Number of cols of the global matrix
    int64_t global_ncol;

    // Number of rows of the local / diagonal matrix
    int local_nrow;
    // Number of cols of the local / diagonal matrix
    int local_ncol;

    // Interior / diagonal CSR structure - these are device pointers
    int* diagonal_csr_row_ptr;
    int* diagonal_csr_col_ind;
    double* diagonal_csr_val;
    // Number of non-zeros of the local / diagonal matrix
    int64_t diagonal_nnz;

    // Ghost / off-diagonal CSR structure - these are device pointers
    int* offd_csr_row_ptr;
    int* offd_csr_col_ind;
    double* offd_csr_val;
    // Number of non-zeros of the ghost / off-diagonal matrix
    int64_t offd_nnz;

    // RHS data - this is a device pointer
    double* rhs_val;

    ////////////////////////////////////////////////

    // Number of total processes, this rank has to send data to
    int nsend;
    // Number of total processes, this rank receives data from
    int nrecv;

    // Array that stores the process ids, that we need to send data to - host pointer
    int* sends;
    // Array that stores the process ids, that we need to receive data from - host pointer
    int* recvs;

    // Array that stores the number of vertices per neighbor, that we need to send - host pointer
    // e.g. if we have to send 253 vertices to neighbor sends[0], and
    // 300 vertices to neighbor sends[1], then
    // send_index_offset[0] = 0
    // send_index_offset[1] = 253 and
    // send_index_offset[2] = 553 (this is kind of CSR structure)
    int* send_index_offset;
    // Array that stores the number of vertices per neighbor, that we need to receive - host pointer
    int* recv_index_offset;

    // Total elements to be sent
    int boundary_index_size;
    // Vertices that need to be sent
    int* boundary_index;
};

// This function computes all prime factors of a given number n
static void compute_prime_factors(int n, std::vector<int>& p)
{
    int factor = 2;

    // Factorize
    while(n > 1)
    {
        while(n % factor == 0)
        {
            p.push_back(factor);
            n /= factor;
        }

        ++factor;
    }
}

// This function computes the process distribution for each dimension
static void compute_3d_process_distribution(int nprocs, int& nprocx, int& nprocy, int& nprocz)
{
    // Compute prime factors
    std::vector<int> p;
    compute_prime_factors(nprocs, p);

    // Compute number of processes in each dimension
    nprocx = 1;
    nprocy = 1;
    nprocz = 1;

    if(p.size() == 0)
    {
        // No entry, this means we have exactly one process
    }
    else if(p.size() == 1)
    {
        // If we have a single prime number, this is going to be our x dimension
        nprocx = p[0];
    }
    else if(p.size() == 2)
    {
        // For two prime numbers, setup x and y
        nprocx = p[1];
        nprocy = p[0];
    }
    else if(p.size() == 3)
    {
        // Three prime numbers
        nprocx = p[2];
        nprocy = p[1];
        nprocz = p[0];
    }
    else
    {
        // More than three prime numbers

        // #prime numbers
        int    idx    = 0;
        size_t nprime = p.size();

        // cubic root
        double qroot = std::cbrt(nprocs);

        // Determine x dimension
        nprocx = p[nprime-- - 1];

        while(nprocx < qroot && idx < nprime)
        {
            nprocx *= p[idx++];
        }

        // Determine y dimension
        double sqroot = std::sqrt(nprocs / nprocx);

        nprocy = p[nprime-- - 1];

        while(nprocy < sqroot && idx < nprime)
        {
            nprocy *= p[idx++];
        }

        // Determine z dimension
        while(idx < nprime)
        {
            nprocz *= p[idx++];
        }
    }

    // Number of processes must match
    assert(nprocx * nprocy * nprocz == nprocs);
}

template <unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY>
__launch_bounds__(BLOCKSIZEX * BLOCKSIZEY)
__global__ void kernel_assembly(int m,
                                int local_dimx,
                                int local_dimy,
                                int local_dimz,
                                int64_t global_dimx,
                                int64_t global_dimy,
                                int64_t global_dimz,
                                int64_t global_iproc_x,
                                int64_t global_iproc_y,
                                int64_t global_iproc_z,
                                char* __restrict__ global_row_nnz,
                                int64_t* __restrict__ global_csr_col_ind,
                                double* __restrict__ rhs)
{
    // Current local row
    int local_row = blockIdx.x * BLOCKSIZEY + threadIdx.y;

    // Offsets into shared arrays that hold
    // interior vertex marker, to determine if the current vertex is an interior
    // or boundary vertex
    __shared__ bool interior_vertex[16];
    // and column offset, that stores the column index array offset of the
    // current thread index in x direction
    __shared__ int sdata[27 * 16];

    // Offset into current local row
    int* column_offset = sdata + threadIdx.y * BLOCKSIZEX;

    // Initialize interior vertex marker
    if(threadIdx.x == 0)
    {
        interior_vertex[threadIdx.y] = true;
    }

    // Sync interior vertex initialization
    __syncthreads();

    // Do not exceed local number of rows
    if(local_row >= m)
    {
        return;
    }

    // Compute local vertex coordinates
    int iz = local_row / (local_dimx * local_dimy);
    int iy = local_row / local_dimx - local_dimy * iz;
    int ix = local_row - iz * local_dimx * local_dimy - iy * local_dimx;

    // Compute global vertex coordinates
    int64_t global_z = global_iproc_z + iz;
    int64_t global_y = global_iproc_y + iy;
    int64_t global_x = global_iproc_x + ix;

    // Obtain neighboring offsets in x, y and z direction relative to the
    // current vertex and compute the resulting neighboring coordinates
    int64_t nb_global_z = global_z + threadIdx.x / 9 - 1;
    int64_t nb_global_y = global_y + (threadIdx.x % 9) / 3 - 1;
    int64_t nb_global_x = global_x + (threadIdx.x % 3) - 1;

    // Compute current global column for neighboring vertex
    int64_t curcol = nb_global_z * global_dimx * global_dimy + nb_global_y * global_dimx + nb_global_x;

    // Check if current vertex is an interior or boundary vertex
    bool interior = (nb_global_z > -1 && nb_global_z < global_dimz &&
                     nb_global_y > -1 && nb_global_y < global_dimy &&
                     nb_global_x > -1 && nb_global_x < global_dimx);

    // Each thread within the row checks if a neighbor exists for his
    // neighboring offset
    if(interior == false)
    {
        // If no neighbor exists for one of the offsets, we need to re-compute
        // the indexing for the column entry accesses
        interior_vertex[threadIdx.y] = false;
    }

    // Re-compute index into matrix, by marking if current offset is
    // a neighbor or not
    column_offset[threadIdx.x] = interior ? 1 : 0;

    // Wait for threads to finish
    __syncthreads();

    // Do we have an interior vertex?
    bool full_interior = interior_vertex[threadIdx.y];

    // Compute inclusive sum to obtain new matrix index offsets
    int tmp;
    if(threadIdx.x >=  1 && full_interior == false) tmp = column_offset[threadIdx.x -  1]; __syncthreads();
    if(threadIdx.x >=  1 && full_interior == false) column_offset[threadIdx.x] += tmp;     __syncthreads();
    if(threadIdx.x >=  2 && full_interior == false) tmp = column_offset[threadIdx.x -  2]; __syncthreads();
    if(threadIdx.x >=  2 && full_interior == false) column_offset[threadIdx.x] += tmp;     __syncthreads();
    if(threadIdx.x >=  4 && full_interior == false) tmp = column_offset[threadIdx.x -  4]; __syncthreads();
    if(threadIdx.x >=  4 && full_interior == false) column_offset[threadIdx.x] += tmp;     __syncthreads();
    if(threadIdx.x >=  8 && full_interior == false) tmp = column_offset[threadIdx.x -  8]; __syncthreads();
    if(threadIdx.x >=  8 && full_interior == false) column_offset[threadIdx.x] += tmp;     __syncthreads();
    if(threadIdx.x >= 16 && full_interior == false) tmp = column_offset[threadIdx.x - 16]; __syncthreads();
    if(threadIdx.x >= 16 && full_interior == false) column_offset[threadIdx.x] += tmp;     __syncthreads();

    // Number of non-zero entries in the current row
    char row_nnz;

    // Do we have interior or boundary vertex, e.g. do we have a neighbor for each
    // direction?
    if(full_interior == true)
    {
        // Interior vertex
        int idx = local_row * 27 + threadIdx.x;

        // Store current global column
        global_csr_col_ind[idx] = curcol;

        // Interior vertices have 27 neighboring vertices
        row_nnz = 27;
    }
    else
    {
        // Boundary vertex, e.g. at least one neighboring offset is not a neighbor (this
        // happens e.g. on the global domains boundary)
        // We do only process "real" neighbors
        if(interior == true)
        {
            // Obtain current threads index into matrix from above inclusive scan
            // (convert from 1-based to 0-based indexing)
            int offset = column_offset[threadIdx.x] - 1;

            int idx = local_row * 27 + offset;

            // Store current global column
            global_csr_col_ind[idx] = curcol;
        }

        // First thread writes number of neighboring vertices, including the
        // identity vertex
        if(threadIdx.x == 0)
        {
            row_nnz = column_offset[BLOCKSIZEX - 1];
        }
    }

    // For each row, initialize vector arrays and number of vertices
    if(threadIdx.x == 0)
    {
        // Differentiate whether we have interior or boundary vertex, e.g. do we have a
        // neighbor for each direction?
        global_row_nnz[local_row] = row_nnz;

        // Setup rhs vector
        rhs[local_row] = 26.0 - (row_nnz - 1.0);
    }
}

template <unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY>
__launch_bounds__(BLOCKSIZEX * BLOCKSIZEY)
__global__ void kernel_setup_halo_nnz(int m,
                                      int local_dimx,
                                      int local_dimy,
                                      int local_dimz,
                                      int global_iproc_x,
                                      int global_iproc_y,
                                      int global_iproc_z,
                                      int64_t global_dimx,
                                      int64_t global_dimy,
                                      const char* __restrict__ global_row_nnz,
                                      const int64_t* __restrict__ global_csr_col_ind,
                                      int* __restrict__ int_csr_row_ptr,
                                      int* __restrict__ gst_csr_row_ptr)
{
    // Each block processes blockDim.y rows
    int local_row = blockIdx.x * BLOCKSIZEY + threadIdx.y;

    // Do not exceed number of rows
    if(local_row >= m)
    {
        return;
    }

    // Read row nnz
    char row_nnz = global_row_nnz[local_row];

    // Process only non-zeros of current row ; each thread index in x direction processes one column entry
    int neighbor_rank_id = 0;

    if(threadIdx.x < row_nnz)
    {
        // Obtain the corresponding global column index (generated in GenerateProblem.cpp)
        int64_t global_col = global_csr_col_ind[local_row * 27 + threadIdx.x];

        // Determine neighboring process of current global column
        int64_t idx_z = global_col / (global_dimx * global_dimy);
        int64_t idx_y = (global_col - idx_z * global_dimx * global_dimy) / global_dimx;
        int64_t idx_x = global_col % global_dimx;

        int idx_proc_z = idx_z / local_dimz;
        int idx_proc_y = idx_y / local_dimy;
        int idx_proc_x = idx_x / local_dimx;

        // Compute neighboring process id depending on the global column.
        // Each domain has at most 26 neighboring domains.
        // Since the numbering is following a fixed order, we can compute the
        // neighbor process id by the actual x,y,z coordinate of the entry, using
        // the domains offsets into the global numbering.
        neighbor_rank_id = (idx_proc_z - global_iproc_z / local_dimz) * 9
                         + (idx_proc_y - global_iproc_y / local_dimy) * 3
                         + (idx_proc_x - global_iproc_x / local_dimx);



        // This will give us the neighboring process id between [-13, 13] where 0
        // is the local domain. We shift the resulting id by 13 to avoid counting threads
        // that do not participate.
        neighbor_rank_id += 13;
    }

    // Count non-zeros of the interior part
    unsigned long long mask = __ballot(neighbor_rank_id == 13);

    // We are running 32 threads per row (e.g. 2 rows per wavefront)
    // Thus we have to shift the mask accordingly
    if((threadIdx.y & 1) == 0)
    {
        // Shift left to zero out left bits we are not interested in
        mask <<= 32;
    }

    // Shift right to zero out right bits we are not interested in
    mask >>= 32;

    // Get the number of interior nnz
    int nnz = __popcll(mask);

    // Write back to global memory
    if(threadIdx.x == 0)
    {
        int_csr_row_ptr[local_row] = nnz;
        gst_csr_row_ptr[local_row] = row_nnz - nnz;
    }
}

template <unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY>
__launch_bounds__(BLOCKSIZEX * BLOCKSIZEY)
__global__ void kernel_setup_halo(int m,
                                  int max_boundary,
                                  int max_sending,
                                  int local_dimx,
                                  int local_dimy,
                                  int local_dimz,
                                  int global_iproc_x,
                                  int global_iproc_y,
                                  int global_iproc_z,
                                  int npx,
                                  int npy,
                                  int npz,
                                  int64_t global_dimx,
                                  int64_t global_dimy,
                                  const char* __restrict__ global_row_nnz,
                                  const int64_t* __restrict__ global_csr_col_ind,
                                  const int* __restrict__ int_csr_row_ptr,
                                  const int* __restrict__ gst_csr_row_ptr,
                                  int* __restrict__ int_csr_col_ind,
                                  double* __restrict__ int_csr_val,
                                  int* __restrict__ nsend_per_rank,
                                  int* __restrict__ nrecv_per_rank,
                                  int* __restrict__ neighbors,
                                  int* __restrict__ send_indices,
                                  int64_t* __restrict__ recv_indices,
                                  int* __restrict__ halo_indices)
{
    // Each block processes blockDim.y rows
    int local_row = blockIdx.x * BLOCKSIZEY + threadIdx.y;

    // Some shared memory to mark rows that need to be sent to neighboring processes
    __shared__ bool sdata[BLOCKSIZEX * BLOCKSIZEY];
    sdata[threadIdx.x + threadIdx.y * BLOCKSIZEX] = false;

    __syncthreads();

    // Do not exceed number of rows
    if(local_row >= m)
    {
        return;
    }

    // Global ID for 1D grid of 2D blocks
    char row_nnz = global_row_nnz[local_row];

    // Row entry points
    int int_row_begin = int_csr_row_ptr[local_row];
    int gst_row_begin = gst_csr_row_ptr[local_row];

    // Neighbor this vertex belongs to
    int neighbor_rank_id = 0;

    // Process only non-zeros of current row ; each thread index in x direction processes one column entry
    if(threadIdx.x < row_nnz)
    {
        // Compute local vertex coordinates
        int iz = local_row / (local_dimx * local_dimy);
        int iy = local_row / local_dimx - local_dimy * iz;
        int ix = local_row - iz * local_dimx * local_dimy - iy * local_dimx;

        // Compute global vertex coordinates
        int64_t global_z = global_iproc_z + iz;
        int64_t global_y = global_iproc_y + iy;
        int64_t global_x = global_iproc_x + ix;

        // Global row
        int64_t global_row = global_z * global_dimx * global_dimy + global_y * global_dimx + global_x;

        // Obtain the corresponding global column index (generated in GenerateProblem.cpp)
        int64_t global_col = global_csr_col_ind[local_row * 27 + threadIdx.x];

        // Determine neighboring process of current global column
        int64_t idx_z = global_col / (global_dimx * global_dimy);
        int64_t idx_y = (global_col - idx_z * global_dimx * global_dimy) / global_dimx;
        int64_t idx_x = global_col % global_dimx;

        int idx_proc_z = idx_z / local_dimz;
        int idx_proc_y = idx_y / local_dimy;
        int idx_proc_x = idx_x / local_dimx;

        // Compute neighboring process id depending on the global column.
        // Each domain has at most 26 neighboring domains.
        // Since the numbering is following a fixed order, we can compute the
        // neighbor process id by the actual x,y,z coordinate of the entry, using
        // the domains offsets into the global numbering.
        neighbor_rank_id = (idx_proc_z - global_iproc_z / local_dimz) * 9
                         + (idx_proc_y - global_iproc_y / local_dimy) * 3
                         + (idx_proc_x - global_iproc_x / local_dimx);

        // This will give us the neighboring process id between [-13, 13] where 0
        // is the local domain. We shift the resulting id by 13 to avoid negative indices.
        neighbor_rank_id += 13;

        // Count threads that are contributing to the interior part
        unsigned long long mask = __ballot(neighbor_rank_id == 13);

        // We are running 32 threads per row (e.g. 2 rows per wavefront)
        // Thus we have to shift the mask accordingly
        if((threadIdx.y & 1) == 0)
        {
            // Shift left to zero out left bits we are not interested in
            mask <<= 32;
        }

        // Shift right to zero out right bits we are not interested in
        mask >>= 32;

        // Get the lanemask
        unsigned long long lanemask_le = UINT64_MAX >> (sizeof(unsigned long long) * CHAR_BIT - (threadIdx.x + 1));

        // Check whether we are in the local domain or not
        if(neighbor_rank_id != 13)
        {
            // Mark current row for sending, to avoid multiple entries with the same row index
            sdata[neighbor_rank_id + threadIdx.y * BLOCKSIZEX] = true;

            // Also store the "real" process id this global column index belongs to
            neighbors[neighbor_rank_id] = idx_proc_x + idx_proc_y * npx + idx_proc_z * npy * npx;

            // Count up the global column that we have to receive by a neighbor using atomics
            int idx = atomicAdd(&nrecv_per_rank[neighbor_rank_id], 1);

            // Get the offset
            int offset = __popcll(lanemask_le & (~mask)) - 1;

            // Halo indices array stores the global id, so we can easily access the matrix
            // column array at the halo position
            halo_indices[neighbor_rank_id * max_boundary + idx] = gst_row_begin + offset;

            // Store the global column id that we have to receive from a neighbor
            recv_indices[neighbor_rank_id * max_boundary + idx] = global_col;
        }
        else
        {
            // Determine local column index
            int lz = idx_z % local_dimz;
            int ly = global_col / global_dimx % local_dimy;
            int lx = global_col % local_dimx;

            // Store the local column index in the local matrix column array
            int idx = int_row_begin + __popcll(lanemask_le & mask) - 1;

            int_csr_col_ind[idx] = lz * local_dimy * local_dimx + ly * local_dimx + lx;

            int_csr_val[idx] = (global_row == global_col) ? 26.0 : -1.0;
        }
    }

    __syncthreads();

    // Check if current row has been marked for sending its entry
    if(sdata[threadIdx.x + threadIdx.y * BLOCKSIZEX] == true)
    {
        // If current row has been marked for sending, store its index
        int idx = atomicAdd(&nsend_per_rank[threadIdx.x], 1);
        send_indices[threadIdx.x * max_sending + idx] = local_row;
    }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_halo_columns(int size,
                                    int m,
                                    int rank_offset,
                                    const int* __restrict__ halo_indices,
                                    const int64_t* __restrict__ offsets,
                                    int* __restrict__ csr_col_ind,
                                    double* __restrict__ csr_val)
{
    // 1D thread indexing
    int gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    // Do not run out of bounds
    if(gid >= size)
    {
        return;
    }

    // Loop over all halo entries of the current row
    for(int i = offsets[gid]; i < offsets[gid + 1]; ++i)
    {
        // Get the index value to access the halo entry in the local matrix column array
        int idx = halo_indices[i];

        // Numbering of halo entries are consecutive with number of local rows as offset
        csr_col_ind[idx] = gid + rank_offset;
        csr_val[idx]     = -1.0;
    }
}

void generate_3d_laplacian_hip(int local_dimx,
                               int local_dimy,
                               int local_dimz,
                               int nproc_x,
                               int nproc_y,
                               int nproc_z,
                               const MPI_Comm* comm,
                               int rank,
                               int nprocs,
                               Data* data)
{
    assert(nprocs > 1);
    assert(local_dimx > 0);
    assert(local_dimy > 0);
    assert(local_dimz > 0);
    assert(nproc_x > 0);
    assert(nproc_y > 0);
    assert(nproc_z > 0);

    // Determine process index into the unit cube
    int iproc_z = rank / (nproc_x * nproc_y);
    int iproc_y = (rank - iproc_z * nproc_x * nproc_y) / nproc_x;
    int iproc_x = rank % nproc_x;

    // Global sizes
    int64_t global_dimx = static_cast<int64_t>(nproc_x) * local_dimx;
    int64_t global_dimy = static_cast<int64_t>(nproc_y) * local_dimy;
    int64_t global_dimz = static_cast<int64_t>(nproc_z) * local_dimz;

    // Global process entry points
    int64_t global_iproc_x = iproc_x * local_dimx;
    int64_t global_iproc_y = iproc_y * local_dimy;
    int64_t global_iproc_z = iproc_z * local_dimz;

    // Number of rows (global and local)
    int64_t local_nrow  = local_dimx * local_dimy * local_dimz;
    int64_t global_nrow = global_dimx * global_dimy * global_dimz;

    // Assemble CSR matrix row nnz and column indices
    char* d_global_row_nnz;
    int64_t* d_global_csr_col_ind;
    double* d_rhs;

    HIP_CHECK(hipMalloc((void**)&d_global_row_nnz, sizeof(char) * local_nrow));
    HIP_CHECK(hipMalloc((void**)&d_global_csr_col_ind, sizeof(int64_t) * local_nrow * 27));
    HIP_CHECK(hipMalloc((void**)&d_rhs, sizeof(double) * local_nrow));

    // Launch parameters
    dim3 GridSize((local_nrow - 1) / 16 + 1);
    dim3 BlockSize(27, 16);

    kernel_assembly<27, 16><<<GridSize, BlockSize>>>(
        local_nrow,
        local_dimx,
        local_dimy,
        local_dimz,
        global_dimx,
        global_dimy,
        global_dimz,
        global_iproc_x,
        global_iproc_y,
        global_iproc_z,
        d_global_row_nnz,
        d_global_csr_col_ind,
        d_rhs);

    // We can already initialize some sizes
    data->global_nrow = global_nrow;
    data->global_ncol = global_nrow;
    data->local_nrow  = local_nrow;
    data->local_ncol  = local_nrow;

    // Determine two largest dimensions
    int max_dim_1 = std::max(local_dimx, std::max(local_dimy, local_dimz));
    int max_dim_2 = ((local_dimx >= local_dimy && local_dimx <= local_dimz) || (local_dimx >= local_dimz && local_dimx <= local_dimy)) ? local_dimx
                          : ((local_dimy >= local_dimz && local_dimy <= local_dimx) || (local_dimy >= local_dimx && local_dimy <= local_dimz)) ? local_dimy
                          : local_dimz;

    // Maximum of entries that can be sent to a single neighboring rank
    int max_sending = max_dim_1 * max_dim_2;

    // 27 pt stencil has a maximum of 9 boundary entries per boundary plane
    // and thus, the maximum number of boundary elements can be computed to be
    // 9 * max_dim_1 * max_dim_2
    int max_boundary = 9 * max_dim_1 * max_dim_2;

    // A maximum of 27 neighbors, including outselves, is possible for each process
    int max_neighbors = 27;

    // Arrays to hold send and receive element offsets per rank
    int* d_nsend_per_rank;
    int* d_nrecv_per_rank;

    // Number of elements is stored for each neighboring rank
    HIP_CHECK(hipMalloc((void**)&d_nsend_per_rank, sizeof(int) * max_neighbors));
    HIP_CHECK(hipMalloc((void**)&d_nrecv_per_rank, sizeof(int) * max_neighbors));

    // Since we use increments, we have to initialize with 0
    HIP_CHECK(hipMemset(d_nsend_per_rank, 0, sizeof(int) * max_neighbors));
    HIP_CHECK(hipMemset(d_nrecv_per_rank, 0, sizeof(int) * max_neighbors));


    // Array to store the neighboring process ids
    int* d_neighbors;
    HIP_CHECK(hipMalloc((void**)&d_neighbors, sizeof(int) * max_neighbors));

    // Array to hold send indices
    int* d_send_indices;

    // d_send_indices holds max_sending elements per neighboring rank, at max
    HIP_CHECK(hipMalloc((void**)&d_send_indices, sizeof(int) * max_sending * max_neighbors));

    // Array to hold receive and halo indices
    int64_t* d_recv_indices;
    int* d_halo_indices;

    // Both arrays hold max_boundary elements per neighboring rank, at max
    HIP_CHECK(hipMalloc((void**)&d_recv_indices, sizeof(int64_t) * max_boundary * max_neighbors));
    HIP_CHECK(hipMalloc((void**)&d_halo_indices, sizeof(int) * max_boundary * max_neighbors));

    // Interior and ghost csr row pointer arrays
    int* d_int_csr_row_ptr;
    int* d_gst_csr_row_ptr;

    HIP_CHECK(hipMalloc((void**)&d_int_csr_row_ptr, sizeof(int) * (local_nrow + 1)));
    HIP_CHECK(hipMalloc((void**)&d_gst_csr_row_ptr, sizeof(int) * (local_nrow + 1)));

    // SetupHalo nnz kernel
    kernel_setup_halo_nnz<32, 16><<<(local_nrow - 1) / 16 + 1, dim3(32, 16)>>>(
        local_nrow,
        local_dimx,
        local_dimy,
        local_dimz,
        global_iproc_x,
        global_iproc_y,
        global_iproc_z,
        global_dimx,
        global_dimy,
        d_global_row_nnz,
        d_global_csr_col_ind,
        d_int_csr_row_ptr,
        d_gst_csr_row_ptr);

    // rocPRIM
    size_t rocprim_size;
    void*  rocprim_buffer;

    // Prefix sum to obtain row pointer arrays
    rocprim::exclusive_scan(NULL,
                            rocprim_size,
                            d_int_csr_row_ptr,
                            d_int_csr_row_ptr,
                            0,
                            local_nrow + 1,
                            rocprim::plus<int>());
    hipMalloc(&rocprim_buffer, rocprim_size);
    rocprim::exclusive_scan(rocprim_buffer,
                            rocprim_size,
                            d_int_csr_row_ptr,
                            d_int_csr_row_ptr,
                            0,
                            local_nrow + 1,
                            rocprim::plus<int>());
    rocprim::exclusive_scan(rocprim_buffer,
                            rocprim_size,
                            d_gst_csr_row_ptr,
                            d_gst_csr_row_ptr,
                            0,
                            local_nrow + 1,
                            rocprim::plus<int>());
    hipFree(rocprim_buffer);

    // Sizes
    int local_nnz;
    int ghost_nnz;

    HIP_CHECK(hipMemcpy(&local_nnz, d_int_csr_row_ptr + local_nrow, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&ghost_nnz, d_gst_csr_row_ptr + local_nrow, sizeof(int), hipMemcpyDeviceToHost));

    // Allocate local structures
    int* d_int_csr_col_ind;
    int* d_gst_csr_col_ind;
    double* d_int_csr_val;
    double* d_gst_csr_val;

    HIP_CHECK(hipMalloc((void**)&d_int_csr_col_ind, sizeof(int) * local_nnz));
    HIP_CHECK(hipMalloc((void**)&d_gst_csr_col_ind, sizeof(int) * ghost_nnz));
    HIP_CHECK(hipMalloc((void**)&d_int_csr_val, sizeof(double) * local_nnz));
    HIP_CHECK(hipMalloc((void**)&d_gst_csr_val, sizeof(double) * ghost_nnz));

    // SetupHalo kernel
    kernel_setup_halo<32, 16><<<(local_nrow - 1) / 16 + 1, dim3(32, 16)>>>(
        local_nrow,
        max_boundary,
        max_sending,
        local_dimx,
        local_dimy,
        local_dimz,
        global_iproc_x,
        global_iproc_y,
        global_iproc_z,
        nproc_x,
        nproc_y,
        nproc_z,
        global_dimx,
        global_dimy,
        d_global_row_nnz,
        d_global_csr_col_ind,
        d_int_csr_row_ptr,
        d_gst_csr_row_ptr,
        d_int_csr_col_ind,
        d_int_csr_val,
        d_nsend_per_rank,
        d_nrecv_per_rank,
        d_neighbors,
        d_send_indices,
        d_recv_indices,
        d_halo_indices);

    HIP_CHECK(hipFree(d_global_row_nnz));
    HIP_CHECK(hipFree(d_global_csr_col_ind));

    // Prefix sum to obtain send index offsets
    std::vector<int> nsend_per_rank(max_neighbors + 1);
    HIP_CHECK(hipMemcpy(nsend_per_rank.data() + 1,
                        d_nsend_per_rank,
                        sizeof(int) * max_neighbors,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_nsend_per_rank));

    nsend_per_rank[0] = 0;
    for(int i = 0; i < max_neighbors; ++i)
    {
        nsend_per_rank[i + 1] += nsend_per_rank[i];
    }

    // Total elements to be sent
    data->boundary_index_size = nsend_per_rank[max_neighbors];

    // Array to hold number of entries that have to be sent to each process
    data->send_index_offset = new int[nprocs];

    // Sort send indices to obtain elementsToSend array
    // elementsToSend array has to be in increasing order, so other processes know
    // where to place the elements.
    int* d_boundary;
    HIP_CHECK(hipMalloc((void**)&d_boundary, sizeof(int) * data->boundary_index_size));
    data->nsend = 0;

    // Loop over all possible neighboring processes
    for(int i = 0; i < max_neighbors; ++i)
    {
        // Compute number of entries to be sent to i-th rank
        int entriesToSend = nsend_per_rank[i + 1] - nsend_per_rank[i];

        // Check if this is actually a neighbor that receives some data
        if(entriesToSend == 0)
        {
            // Nothing to be sent / sorted, skip
            continue;
        }

        size_t rocprim_size;
        void* rocprim_buffer;

        // Obtain buffer size
        HIP_CHECK(rocprim::radix_sort_keys(NULL,
                                           rocprim_size,
                                           d_send_indices + i * max_sending,
                                           d_boundary + nsend_per_rank[i],
                                           entriesToSend));
        HIP_CHECK(hipMalloc(&rocprim_buffer, rocprim_size));

        // Sort send indices to obtain increasing order
        HIP_CHECK(rocprim::radix_sort_keys(rocprim_buffer,
                                           rocprim_size,
                                           d_send_indices + i * max_sending,
                                           d_boundary + nsend_per_rank[i],
                                           entriesToSend));
        HIP_CHECK(hipFree(rocprim_buffer));
        rocprim_buffer = NULL;

        // Store number of elements that have to be sent to i-th process
        data->send_index_offset[++data->nsend] = entriesToSend;
    }

    // Free up memory
    HIP_CHECK(hipFree(d_send_indices));

    // Exclusive scan to obtain send index offsets
    data->send_index_offset[0] = 0;
    for(int i = 0; i < data->nsend; ++i)
    {
        data->send_index_offset[i + 1] += data->send_index_offset[i];
    }

    // Prefix sum to obtain receive indices offsets (with duplicates)
    std::vector<int> nrecv_per_rank(max_neighbors + 1);
    HIP_CHECK(hipMemcpy(nrecv_per_rank.data() + 1,
                        d_nrecv_per_rank,
                        sizeof(int) * max_neighbors,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_nrecv_per_rank));

    nrecv_per_rank[0] = 0;
    for(int i = 0; i < max_neighbors; ++i)
    {
        // Verify boundary size does not exceed maximum boundary elements
        assert(nrecv_per_rank[i + 1] < max_boundary);

        nrecv_per_rank[i + 1] += nrecv_per_rank[i];
    }

    // Initialize number of external values
    int recv_index_size = 0;

    // Array to hold number of elements that have to be received from each neighboring
    // process
    data->recv_index_offset = new int[nprocs];

    // Counter for number of neighbors we are actually receiving data from
    data->nrecv = 0;

    // Create rank indexing array for send, recv and halo lists
    std::vector<int64_t*> d_recvList(max_neighbors);
    std::vector<int*> d_haloList(max_neighbors);

    for(int i = 0; i < max_neighbors; ++i)
    {
        d_recvList[i] = d_recv_indices + i * max_boundary;
        d_haloList[i] = d_halo_indices + i * max_boundary;
    }

    // Own rank can be buffer, nothing should be sent/received by ourselves
    int64_t* d_recvBuffer = d_recvList[13];
    int* d_haloBuffer = d_haloList[13];

    // Array to hold the process ids of all neighbors that we receive data from
    data->recvs = new int[nprocs - 1];

    // Buffer to process the GPU data
    std::vector<int> neighbors(max_neighbors);
    HIP_CHECK(hipMemcpy(neighbors.data(),
                        d_neighbors,
                        sizeof(int) * max_neighbors,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_neighbors));

    // Loop over all possible neighbors
    for(int i = 0; i < max_neighbors; ++i)
    {
        // Number of entries that have to be received from i-th rank
        int entriesToRecv = nrecv_per_rank[i + 1] - nrecv_per_rank[i];

        // Check if we actually receive data
        if(entriesToRecv == 0)
        {
            // Nothing to receive, skip
            continue;
        }

        size_t rocprim_size;
        void* rocprim_buffer;

        // Obtain buffer size
        HIP_CHECK(rocprim::radix_sort_pairs(NULL,
                                            rocprim_size,
                                            d_recvList[i],
                                            d_recvBuffer,
                                            d_haloList[i],
                                            d_haloBuffer,
                                            entriesToRecv));
        HIP_CHECK(hipMalloc(&rocprim_buffer, rocprim_size));

        // Sort receive index array and halo index array
        HIP_CHECK(rocprim::radix_sort_pairs(rocprim_buffer,
                                            rocprim_size,
                                            d_recvList[i],
                                            d_recvBuffer,
                                            d_haloList[i],
                                            d_haloBuffer,
                                            entriesToRecv));
        HIP_CHECK(hipFree(rocprim_buffer));

        // Swap receive buffer pointers
        int64_t* gptr = d_recvBuffer;
        d_recvBuffer = d_recvList[i];
        d_recvList[i] = gptr;

        // Swap halo buffer pointers
        int* lptr = d_haloBuffer;
        d_haloBuffer = d_haloList[i];
        d_haloList[i] = lptr;

        // No need to allocate new memory, we can use existing buffers
        int64_t* d_num_runs;
        HIP_CHECK(hipMalloc((void**)&d_num_runs, sizeof(int64_t) * data->boundary_index_size));
        int64_t* d_offsets = reinterpret_cast<int64_t*>(d_recvBuffer);
        int64_t* d_unique_out = reinterpret_cast<int64_t*>(d_haloBuffer);;

        // Obtain rocprim buffer size
        HIP_CHECK(rocprim::run_length_encode(NULL,
                                             rocprim_size,
                                             d_recvList[i],
                                             entriesToRecv,
                                             d_unique_out,
                                             d_offsets + 1,
                                             d_num_runs));
        HIP_CHECK(hipMalloc(&rocprim_buffer, rocprim_size));

        // Perform a run length encode over the receive indices to obtain the number
        // of halo entries in each row
        HIP_CHECK(rocprim::run_length_encode(rocprim_buffer,
                                             rocprim_size,
                                             d_recvList[i],
                                             entriesToRecv,
                                             d_unique_out,
                                             d_offsets + 1,
                                             d_num_runs));
        HIP_CHECK(hipFree(rocprim_buffer));

        // Copy the number of halo entries with respect to the i-th neighbor
        int64_t num_runs;
        HIP_CHECK(hipMemcpy(&num_runs, d_num_runs, sizeof(int64_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_num_runs));

        // Store the number of halo entries we need to get from i-th neighbor
        data->recv_index_offset[data->nrecv + 1] = num_runs;

        // d_offsets[0] = 0
        HIP_CHECK(hipMemset(d_offsets, 0, sizeof(int64_t)));

        // Obtain rocprim buffer size
        HIP_CHECK(rocprim::inclusive_scan(NULL, rocprim_size, d_offsets + 1, d_offsets + 1, num_runs, rocprim::plus<int64_t>()));
        HIP_CHECK(hipMalloc(&rocprim_buffer, rocprim_size));

        // Perform inclusive sum to obtain the offsets to the first halo entry of each row
        HIP_CHECK(rocprim::inclusive_scan(rocprim_buffer, rocprim_size, d_offsets + 1, d_offsets + 1, num_runs, rocprim::plus<int64_t>()));
        HIP_CHECK(hipFree(rocprim_buffer));
        rocprim_buffer = NULL;

        // Launch kernel to fill all halo columns in the local matrix column index array for the i-th neighbor
        kernel_halo_columns<128><<<(num_runs - 1) / 128 + 1, 128>>>(
            num_runs,
            local_nrow,
            recv_index_size,
            d_haloList[i],
            d_offsets,
            d_gst_csr_col_ind,
            d_gst_csr_val);

        // Increase the number of external values by i-th neighbors halo entry contributions
        recv_index_size += num_runs;

        // Store the "real" neighbor id for i-th neighbor
        data->recvs[data->nrecv++] = neighbors[i];
    }

    // Free up data
    HIP_CHECK(hipFree(d_recv_indices));
    HIP_CHECK(hipFree(d_halo_indices));

    // Exclusive scan to obtain receive index offsets
    data->recv_index_offset[0] = 0;
    for(int i = 0; i < data->nrecv; ++i)
    {
        data->recv_index_offset[i + 1] += data->recv_index_offset[i];
    }

    // Set up the interior / diagonal matrix pointers
    data->diagonal_csr_row_ptr = d_int_csr_row_ptr;
    data->diagonal_csr_col_ind = d_int_csr_col_ind;
    data->diagonal_csr_val     = d_int_csr_val;
    data->diagonal_nnz         = local_nnz;

    // Set up the ghost / offdiagonal matrix pointers
    data->offd_csr_row_ptr = d_gst_csr_row_ptr;
    data->offd_csr_col_ind = d_gst_csr_col_ind;
    data->offd_csr_val     = d_gst_csr_val;
    data->offd_nnz         = ghost_nnz;

    // Set up the rhs
    data->rhs_val = d_rhs;

    // Set boundary
    data->boundary_index = d_boundary;
}

void free(Data data)
{
	HIP_CHECK(hipFree(data.rhs_val));
   HIP_CHECK(hipFree(data.diagonal_csr_row_ptr));
   HIP_CHECK(hipFree(data.diagonal_csr_col_ind));
	HIP_CHECK(hipFree(data.diagonal_csr_val));
   HIP_CHECK(hipFree(data.offd_csr_row_ptr));
   HIP_CHECK(hipFree(data.offd_csr_col_ind));
	HIP_CHECK(hipFree(data.offd_csr_val));
	HIP_CHECK(hipFree(data.boundary_index));
}
