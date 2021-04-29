#ifndef HYPRESYSTEM_H
#define HYPRESYSTEM_H

#include "mpi.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "krylov.h"
#include "HYPRE.h"

#if defined(HYPRE_USING_CUDA)
#include <cuda_runtime.h>
#elif defined(HYPRE_USING_HIP)
#include <hip/hip_runtime.h>
#endif


#include "yaml-cpp/yaml.h"

extern "C"
{
#include "mmio.h"
}

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

namespace nalu {

class HypreSystem
{
public:
    HypreSystem(MPI_Comm, YAML::Node&);

    void load();

    void solve();

    //! Output the HYPRE matrix, rhs and solution vectors
    void output_linear_system();

    //! Check HYPRE solution against reference solution provided by user
    void check_solution();

    //! Summarize timers
    void summarize_timers();

    //! Destroy hypre linear system
    void destroy_system();

private:

    HypreSystem() = delete;

    HypreSystem(const HypreSystem&) = delete;

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
    void build_ij_vector(std::string, int, HYPRE_IJVector&);

    //! Scan and load the Matrix Market file
    void build_mm_matrix(std::string);

    //! Determine the dimensions of the matrix and the largest row ID : matrix market
    void determine_mm_system_sizes(std::string);

    //! Build the HYPRE_IJMatrix from data loaded from either IJ or matrix market
    void hypre_matrix_set_values();

    //! Build the HYPRE_IJVector from data loaded from either IJ or matrix market
    void hypre_vector_set_values(HYPRE_IJVector& vec);

    //! Load the matrix into HYPRE_IJVector
    void build_mm_vector(std::string, HYPRE_IJVector&);

    //! Initialize hypre linear system
    void init_system();

    //! finalize system
    void finalize_system();

    //! Setup BoomerAMG
    void setup_boomeramg_precond();

    void setup_boomeramg_solver();

    //! Setup GMRES
    void setup_gmres();
    void setup_cogmres();
    void setup_fgmres();
    void setup_bicg();

    
    //! MPI Communicator object
    MPI_Comm comm_;

    YAML::Node& inpfile_;

    //! Flag indicating whether a row was filled
    std::vector<int> rowFilled_;

    //! COO data structures read in only once
    std::vector<HYPRE_Int> rows_;
    std::vector<HYPRE_Int> cols_;
    std::vector<double> vals_;
    std::vector<HYPRE_Int> vector_indices_;
    std::vector<double> vector_values_;

    //! Timers
    std::vector<std::pair<std::string, double>> timers_;

    //! HYPRE IJ Matrix
    HYPRE_IJMatrix mat_=NULL;

    //! The rhs vector
    HYPRE_IJVector rhs_=NULL;

    //! The solution vector
    HYPRE_IJVector sln_=NULL;

    //! The solution vector
    HYPRE_IJVector slnRef_=NULL;

    //! Instance of the Hypre parallel matrix
    HYPRE_ParCSRMatrix parMat_;

    //! Instance of the Hypre parallel RHS vector
    HYPRE_ParVector parRhs_;

    //! Instance of Hypre parallel solution vector
    HYPRE_ParVector parSln_;

    //! Instance of Hypre parallel solution vector
    HYPRE_ParVector parSlnRef_;

    HYPRE_Solver solver_=NULL;

    HYPRE_Solver precond_=NULL;

    //! Global number of rows in the linear system
    HYPRE_Int totalRows_{0};

    //! Number of rows in this processor
    HYPRE_Int numRows_{0};

    //! Global ID of the first row on this processor
    HYPRE_Int iLower_{0};

    //! Global ID of the last row on this processor
    HYPRE_Int iUpper_{0};

    HYPRE_Int (*solverDestroyPtr_)(HYPRE_Solver);
    HYPRE_Int (*solverSetupPtr_)(
        HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
    HYPRE_Int (*solverSolvePtr_)(
        HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
    HYPRE_Int (*solverPrecondPtr_)(
        HYPRE_Solver,
        HYPRE_PtrToParSolverFcn,
        HYPRE_PtrToParSolverFcn,
        HYPRE_Solver);

    HYPRE_Int (*precondDestroyPtr_)(HYPRE_Solver);
    HYPRE_Int (*precondSetupPtr_)(
        HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
    HYPRE_Int (*precondSolvePtr_)(
        HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);

    int M_{0};
    int N_{0};
    int nnz_{0};
    int iproc_{0};
    int nproc_{0};

    bool solveComplete_{false};
    bool checkSolution_{false};
    bool outputSystem_{false};
    bool usePrecond_{true};
};

} // namespace nalu

#endif /* HYPRESYSTEM_H */
