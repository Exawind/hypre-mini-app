#ifndef HYPRESYSTEM_H
#define HYPRESYSTEM_H

#include "mpi.h"
#include "HYPRE_utilities.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "krylov.h"
#include "HYPRE.h"

#include <string>
#include <vector>
#include <utility>

namespace nalu {

class HypreSystem
{
public:
    HypreSystem(MPI_Comm);

    void load_matrix(std::string);

    void load_rhs_vector(std::string rhsfile)
    {
        read_mm_vector(rhsfile, rhs_);
    }

    void load_sln_vector(std::string slnfile)
    {
        read_mm_vector(slnfile, slnRef_);
    }

    void setup_preconditioner();

    void setup_solver();

    void solve();

    //! Output the HYPRE matrix, rhs and solution vectors
    void output_linear_system();

    //! Check HYPRE solution against reference solution provided by user
    void check_solution();

    //! Summarize timers
    void summarize_timers();

private:
    HypreSystem() = delete;
    HypreSystem(const HypreSystem&) = delete;

    //! Determine the dimensions of the matrix and the largest row ID
    void determine_system_sizes(std::string);

    //! Load the matrix into HYPRE_IJMatrix
    void read_mm_matrix(std::string);

    void read_mm_vector(std::string, HYPRE_IJVector&);

    //! Initialize hypre linear system
    void init_system();

    //! finalize system
    void finalize_system();

    //! MPI Communicator object
    MPI_Comm comm_;

    //! Flag indicating whether a row was filled
    std::vector<int> rowFilled_;

    //! Row ordering
    std::vector<HYPRE_Int> rowOrder_;

    //! Timers
    std::vector<std::pair<std::string, double>> timers_;

    //! HYPRE IJ Matrix
    HYPRE_IJMatrix mat_;

    //! The rhs vector
    HYPRE_IJVector rhs_;

    //! The solution vector
    HYPRE_IJVector sln_;

    //! The solution vector
    HYPRE_IJVector slnRef_;

    //! Instance of the Hypre parallel matrix
    HYPRE_ParCSRMatrix parMat_;

    //! Instance of the Hypre parallel RHS vector
    HYPRE_ParVector parRhs_;

    //! Instance of Hypre parallel solution vector
    HYPRE_ParVector parSln_;

    //! Instance of Hypre parallel solution vector
    HYPRE_ParVector parSlnRef_;

    HYPRE_Solver solver_;

    HYPRE_Solver precond_;

    //! Global number of rows in the linear system
    HYPRE_Int totalRows_{0};

    //! Number of rows in this processor
    HYPRE_Int numRows_{0};

    //! Global ID of the first row on this processor
    HYPRE_Int iLower_{0};

    //! Global ID of the last row on this processor
    HYPRE_Int iUpper_{0};

    int M_{0};
    int N_{0};
    int nnz_{0};
    int iproc_{0};
    int nproc_{0};

    bool solveComplete_{false};
};

} // namespace nalu

#endif /* HYPRESYSTEM_H */
