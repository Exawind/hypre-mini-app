#ifndef HYPRESYSTEM_H
#define HYPRESYSTEM_H

#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/uio.h>
#else
#include <sys/io.h>
#endif

#include <sys/mman.h>

#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "krylov.h"
#include "mpi.h"

#include "yaml-cpp/yaml.h"

extern "C" {
#include "mmio.h"
}

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#if defined(HYPRE_USING_CUDA) || (HYPRE_USING_HIP)
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#endif

#define HIP_CALL(call)                                                         \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (hipSuccess != err) {                                                   \
      printf("HIP ERROR (code = %d, %s) at %s:%d\n", err,                      \
             hipGetErrorString(err), __FILE__, __LINE__);                      \
      assert(0);                                                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace nalu {

namespace {
template <typename T>
T get_optional(YAML::Node &node, std::string key, T default_value) {
  if (node[key])
    return node[key].as<T>();
  else
    return default_value;
}
} // namespace

class HypreSystem {
public:
  HypreSystem(MPI_Comm, YAML::Node &);

  void load();
  void setup_precon_and_solver();
  void solve();

  //! Output the HYPRE matrix, rhs and solution vectors
  void output_linear_system();

  //! Check HYPRE solution against reference solution provided by user
  void check_solution();

  //! Summarize timers
  void summarize_timers();

  //! retrieve timers to dump to csv file
  void retrieve_timers(std::vector<std::string> &names,
                       std::vector<std::vector<double>> &data);

  //! Destroy hypre linear system
  void destroy_system();

  //! check amoutn of device memory in use
  void checkMemory();

private:
  HypreSystem() = delete;

  HypreSystem(const HypreSystem &) = delete;

  //! build 27 pt stencil
  void build_27pt_stencil();
  void validateDiagData(int nnz, int *drows, int *dcols);
  void validateOffdData(int nnz, int *drows, int *dcols);

  //! Load files in matrix market format
  void load_matrix_market();

  //! Load files in HYPRE IJMatrix format
  void load_hypre_format();

  //! Load files using HYPRE_IJ{Matrix,Vector}Read
  void load_hypre_native();

  //! Determine global sizes from IJ files
  void determine_ij_system_sizes(std::string, int);

  //! Initialize data structures when reading IJ files
  void init_row_decomposition();

  //! Read IJ Matrix into memory
  void build_ij_matrix(std::string, int);

  //! Read IJ Vector into memory
  void build_ij_vector(std::vector<std::string> &, int,
                       std::vector<HYPRE_IJVector> &);

  //! Scan and load the Matrix Market file
  void build_mm_matrix(std::string);

  //! Determine the dimensions of the matrix and the largest row ID : matrix
  //! market
  void determine_mm_system_sizes(std::string);

  //! Build the HYPRE_IJMatrix from data loaded from either IJ or matrix market
  void hypre_matrix_set_values();

  //! Build the HYPRE_IJVector from data loaded from either IJ or matrix market
  void hypre_vector_set_values(std::vector<HYPRE_IJVector> &vec, int component);

  //! Load the matrix into HYPRE_IJVector
  void build_mm_vector(std::vector<std::string> &,
                       std::vector<HYPRE_IJVector> &);

  //! Initialize hypre linear system
  void init_system();

  //! assemble system
  void assemble_system();

  //! Setup BoomerAMG
  void setup_boomeramg_precond();

  void setup_boomeramg_solver();

  //! Setup GMRES
  void setup_gmres();
  void setup_cogmres();
  void setup_fgmres();
  void setup_bicg();
  void setup_cg();

  //! Setup ILU
  void setup_ilu_precond();

  void setup_ilu();

  //! MPI Communicator object
  MPI_Comm comm_;

  YAML::Node &inpfile_;

  //! Flag indicating whether a row was filled
  std::vector<int> rowFilled_;

  //! COO data structures read in only once
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
  std::vector<HYPRE_BigInt> rows_;
  std::vector<HYPRE_BigInt> cols_;
  HYPRE_BigInt *d_rows_ = NULL;
  HYPRE_BigInt *d_cols_ = NULL;
  HYPRE_BigInt *d_offd_rows_ = NULL;
  HYPRE_BigInt *d_offd_cols_ = NULL;

  std::vector<HYPRE_BigInt> vector_indices_;
  HYPRE_BigInt *d_vector_indices_ = NULL;

  //! Global number of rows in the linear system
  HYPRE_BigInt totalRows_{0};

  //! Number of rows in this processor
  HYPRE_BigInt numRows_{0};

  //! Global ID of the first row on this processor
  HYPRE_BigInt iLower_{0};

  //! Global ID of the last row on this processor
  HYPRE_BigInt iUpper_{0};
#else
  std::vector<HYPRE_Int> rows_;
  std::vector<HYPRE_Int> cols_;
  HYPRE_Int *d_rows_ = NULL;
  HYPRE_Int *d_cols_ = NULL;
  HYPRE_Int *d_offd_rows_ = NULL;
  HYPRE_Int *d_offd_cols_ = NULL;

  std::vector<HYPRE_Int> vector_indices_;
  HYPRE_Int *d_vector_indices_ = NULL;

  //! Global number of rows in the linear system
  HYPRE_Int totalRows_{0};

  //! Number of rows in this processor
  HYPRE_Int numRows_{0};

  //! Global ID of the first row on this processor
  HYPRE_Int iLower_{0};

  //! Global ID of the last row on this processor
  HYPRE_Int iUpper_{0};

#endif

  std::vector<double> vals_;
  HYPRE_Complex *d_vals_ = NULL;

  std::vector<double> vector_values_;
  HYPRE_Complex *d_vector_vals_ = NULL;

  //! Timers
  std::vector<std::pair<std::string, double>> timers_;

  //! HYPRE IJ Matrix
  HYPRE_IJMatrix mat_ = NULL;

  //! Instance of the Hypre parallel matrix
  HYPRE_ParCSRMatrix parMat_;

  //! The rhs vector
  std::vector<HYPRE_IJVector> rhs_;

  //! Instance of the Hypre parallel RHS vector
  std::vector<HYPRE_ParVector> parRhs_;

  //! The solution vector
  std::vector<HYPRE_IJVector> sln_;

  //! Instance of Hypre parallel solution vector
  std::vector<HYPRE_ParVector> parSln_;

  //! The reference solution vector
  std::vector<HYPRE_IJVector> slnRef_;

  //! Instance of Hypre parallel reference solution vector
  std::vector<HYPRE_ParVector> parSlnRef_;

  HYPRE_Solver solver_ = NULL;

  HYPRE_Solver precond_ = NULL;

  //! The number of rhs/sln components
  HYPRE_Int numComps_{1};
  HYPRE_Int numSolves_{1};
  HYPRE_Int numVectors_{1};

  HYPRE_Int (*solverDestroyPtr_)(HYPRE_Solver);
  HYPRE_Int (*solverSetupPtr_)(HYPRE_Solver, HYPRE_ParCSRMatrix,
                               HYPRE_ParVector, HYPRE_ParVector);
  HYPRE_Int (*solverSolvePtr_)(HYPRE_Solver, HYPRE_ParCSRMatrix,
                               HYPRE_ParVector, HYPRE_ParVector);
  HYPRE_Int (*solverPrecondPtr_)(HYPRE_Solver, HYPRE_PtrToParSolverFcn,
                                 HYPRE_PtrToParSolverFcn, HYPRE_Solver);

  HYPRE_Int (*precondDestroyPtr_)(HYPRE_Solver);
  HYPRE_Int (*precondSetupPtr_)(HYPRE_Solver, HYPRE_ParCSRMatrix,
                                HYPRE_ParVector, HYPRE_ParVector);
  HYPRE_Int (*precondSolvePtr_)(HYPRE_Solver, HYPRE_ParCSRMatrix,
                                HYPRE_ParVector, HYPRE_ParVector);

  int M_{0};
  int N_{0};
  int nx_{0};
  int ny_{0};
  int nz_{0};
  int nnz_{0};
  int iproc_{0};
  int nproc_{0};

  bool segregatedSolve_{true};
  bool solveComplete_{false};
  bool checkSolution_{false};
  bool outputSystem_{false};
  bool outputSolution_{false};
  bool usePrecond_{true};
  bool writeAmgMatrices_{false};
  double atol_{1.e-8};
  double rtol_{1.e-6};
};

} // namespace nalu

#endif /* HYPRESYSTEM_H */
