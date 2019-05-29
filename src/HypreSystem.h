#ifndef HYPRESYSTEM_H
#define HYPRESYSTEM_H

#include "mpi.h"
#include "HYPRE_utilities.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "krylov.h"
#include "HYPRE.h"

#include "yaml-cpp/yaml.h"

#include <string>
#include <vector>
#include <utility>

namespace nalu {

class HypreSystem
{
public:
    HypreSystem(MPI_Comm, YAML::Node&);

    void load();

//this replaces load for sequence of matrices as Setup needs to be loaded ONCE
    void loadSetup();
    void loadMatrix(int i);
    void destroyMatrix();
    int get_num_matrices();
    void set_num_matrices(int num);
void projectionSpaceUpdate(int i);
    void solve();
//solve2 takes init guess improvement into account
    void solve2();

    //! Output the HYPRE matrix, rhs and solution vectors
    void output_linear_system();

    //! Check HYPRE solution against reference solution provided by user
    void check_solution();

    //! Summarize timers
    void summarize_timers();

    void createProjectedInitGuess(int i);
private:
    HypreSystem() = delete;
    HypreSystem(const HypreSystem&) = delete;

    //! Load files in matrix market format
    void load_matrix_market();

    void load_matrix_market_one(int i);
    //! Load files in HYPRE IJMatrix format
    void load_hypre_format();
    void load_hypre_format_one(int i);

    //! Load files using HYPRE_IJ{Matrix,Vector}Read
    void load_hypre_native();
    void load_hypre_native_one(int i);

    //! Determine global sizes from IJ files
    void determine_ij_system_sizes(std::string, int);

    //! Initialize data structures when reading IJ files
    void init_ij_system();

    //! Read IJ Matrix into memory
    void read_ij_matrix(std::string, int);

    //! Read IJ Vector into memory
    void read_ij_vector(std::string, int, HYPRE_IJVector&);

    //! Scan and load the Matrix Market file
    void load_mm_matrix(std::string);

    //! Determine the dimensions of the matrix and the largest row ID
    void determine_system_sizes(std::string);

    //! Load the matrix into HYPRE_IJMatrix
    void read_mm_matrix(std::string);

    void read_mm_vector(std::string, HYPRE_IJVector&);

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
    //! MPI Communicator object
    MPI_Comm comm_;

    YAML::Node& inpfile_;

    //! Flag indicating whether a row was filled
    std::vector<int> rowFilled_;

    //! Row ordering
    std::vector<HYPRE_Int> rowOrder_;

    //! Timers
    std::vector<std::pair<std::string, double>> timers_;

    //! HYPRE IJ Matrix
    HYPRE_IJMatrix mat_;

//    HYPRE_IJMatrix matSeq_[30];
  
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



    HYPRE_ParVector parY_;
    HYPRE_ParVector parZ_;
    HYPRE_ParVector parOldRhs_;//for left precon
    HYPRE_ParVector projectionSpace; // Q
    HYPRE_ParVector projectionSpaceRaw; //Raw is U
    double * R, *RGPU, *rv; // rv is a temp vector
    void dropFirstColumn();
    void planeRot(double * x, double * G);
    //! Instance of Hypre parallel solution vector
    HYPRE_ParVector parSlnRef_;
    int spaceSize;
double * CPUtmp, *GPUtmp;    
int currentSpaceSize;
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
//for projection spaces
    int num_matrices;
 //   int projectionSpaceSize;


    bool solveComplete_{false};
    bool checkSolution_{false};
    bool outputSystem_{false};
    bool usePrecond_{true};
    bool needFinalize_{true};
};

} // namespace nalu

#endif /* HYPRESYSTEM_H */
