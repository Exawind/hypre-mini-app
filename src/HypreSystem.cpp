
#include "HypreSystem.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE.h"
#include "HYPRE_config.h"

extern "C"
{
#include "mmio.h"
}

#include <iomanip>
#include <algorithm>
#include <chrono>
#include <limits>
#include <cmath>
#include <cstdio>
#include <iostream>

namespace nalu {

HypreSystem::HypreSystem(
    MPI_Comm comm
) : comm_(comm),
    rowFilled_(0)
{
    MPI_Comm_rank(comm, &iproc_);
    MPI_Comm_size(comm, &nproc_);
}

void HypreSystem::setup_preconditioner()
{
    HYPRE_BoomerAMGCreate(&precond_);
    HYPRE_BoomerAMGSetPrintLevel(precond_, 1);
    HYPRE_BoomerAMGSetCoarsenType(precond_, 8);
    HYPRE_BoomerAMGSetOldDefault(precond_);
    HYPRE_BoomerAMGSetCycleType (precond_, 1);
    HYPRE_BoomerAMGSetRelaxType(precond_, 6);
    HYPRE_BoomerAMGSetNumSweeps(precond_, 1);
    HYPRE_BoomerAMGSetTol(precond_, 0.0);
    HYPRE_BoomerAMGSetMaxIter(precond_, 1);
    HYPRE_BoomerAMGSetRelaxOrder(precond_, 1);
}

void HypreSystem::setup_solver()
{
    HYPRE_ParCSRGMRESCreate(comm_, &solver_);
    HYPRE_ParCSRGMRESSetTol(solver_, 1e-5);
    HYPRE_ParCSRGMRESSetMaxIter(solver_, 1000);
    HYPRE_ParCSRGMRESSetKDim(solver_, 10);
    HYPRE_ParCSRGMRESSetPrintLevel(solver_, 4);
}

void HypreSystem::solve()
{
    finalize_system();

    auto start = std::chrono::system_clock::now();
    HYPRE_ParCSRGMRESSetPrecond(
        solver_, &HYPRE_BoomerAMGSolve, &HYPRE_BoomerAMGSetup, precond_);
    HYPRE_ParCSRGMRESSetup(solver_, parMat_, parRhs_, parSln_);
    MPI_Barrier(comm_);
    auto stop1 = std::chrono::system_clock::now();
    std::chrono::duration<double> setup = stop1 - start;
    HYPRE_ParCSRGMRESSolve(solver_, parMat_, parRhs_, parSln_);
    MPI_Barrier(comm_);
    auto stop2 = std::chrono::system_clock::now();
    std::chrono::duration<double> solve = stop2 - stop1;

    if (iproc_ == 0) {
        timers_.emplace_back("Preconditioner setup", setup.count());
        timers_.emplace_back("Solve", solve.count());
    }

    solveComplete_ = true;
}

void HypreSystem::output_linear_system()
{
    auto start = std::chrono::system_clock::now();

    HYPRE_IJMatrixPrint(mat_, "IJM.mat");
    HYPRE_IJVectorPrint(rhs_, "IJV.rhs");
    HYPRE_IJVectorPrint(sln_, "IJV.sln");

    MPI_Barrier(comm_);
    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = stop - start;

    timers_.emplace_back("Output system", elapsed.count());
}

void HypreSystem::check_solution()
{
    if (!solveComplete_)
        throw std::runtime_error("Solve was not called before check_solution");

    auto start = std::chrono::system_clock::now();
    double sol1, sol2, diff;
    double maxerr = std::numeric_limits<double>::lowest();

    for (HYPRE_Int i=iLower_; i < iUpper_; i++) {
        HYPRE_IJVectorGetValues(sln_, 1, &i, &sol1);
        HYPRE_IJVectorGetValues(slnRef_, 1, &i, &sol2);
        diff = sol1 - sol2;
        maxerr = std::max(maxerr, std::fabs(diff));
    }

    double gmax;
    MPI_Reduce(&maxerr, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
    if (iproc_ == 0)
        std::cout << "Solution error: " << gmax << std::endl;

    MPI_Barrier(comm_);
    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = stop - start;

    timers_.emplace_back("Check solution", elapsed.count());
}

void HypreSystem::summarize_timers()
{
    if (iproc_ != 0) return;

    std::cout << "\nTimer summary: " << std::endl;
    for (auto& timer: timers_) {
        std::cout << "    " << std::setw(25) << std::left << timer.first
                  << timer.second << " seconds" << std::endl;
    }
}

void HypreSystem::load_matrix(std::string matfile)
{
    // Scan the matrix and determine the sizes
    determine_system_sizes(matfile);
    MPI_Barrier(comm_);

    // Communicate matrix information to all processors
    MPI_Bcast(&totalRows_, 1, MPI_INT, 0, comm_);

    HYPRE_Int rowsPerProc = totalRows_ / nproc_;
    HYPRE_Int remainder = totalRows_ % nproc_;

    iLower_ = rowsPerProc * iproc_ + std::min<HYPRE_Int>(iproc_, remainder);
    iUpper_ = rowsPerProc * (iproc_ + 1) + std::min<HYPRE_Int>(iproc_ + 1, remainder) - 1;
    numRows_ = iUpper_ - iLower_ + 1;

    std::cout << "  Rank: " << std::setw(4) << iproc_ << ":: iLower = "
              << std::setw(9) << iLower_ << "; iUpper = "
              << std::setw(9) << iUpper_ << "; numRows = "
              << numRows_ << std::endl;
    MPI_Barrier(comm_);

    // Create HYPRE data structures
    init_system();

    // Initialize data
    rowFilled_.resize(totalRows_);
    std::fill(rowFilled_.begin(), rowFilled_.end(), 0);

    // Populate the matrix (proc 0 only)
    read_mm_matrix(matfile);
    MPI_Barrier(comm_);

    // Broadcast filled row information to all procs; need to handle missing rows
    MPI_Bcast(rowFilled_.data(), totalRows_, MPI_INT, 0, comm_);
}

void HypreSystem::init_system()
{
    auto start = std::chrono::system_clock::now();
    if (iproc_ == 0) {
        std::cout << "Initializing data HYPRE structures... ";
    }

    HYPRE_IJMatrixCreate(comm_, iLower_, iUpper_, iLower_, iUpper_, &mat_);
    HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(mat_);
    HYPRE_IJMatrixGetObject(mat_, (void**)&parMat_);

    HYPRE_IJVectorCreate(comm_, iLower_, iUpper_, &rhs_);
    HYPRE_IJVectorSetObjectType(rhs_, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(rhs_);
    HYPRE_IJVectorGetObject(rhs_, (void**)&parRhs_);

    HYPRE_IJVectorCreate(comm_, iLower_, iUpper_, &sln_);
    HYPRE_IJVectorSetObjectType(sln_, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(sln_);
    HYPRE_IJVectorGetObject(sln_, (void**)&parSln_);

    HYPRE_IJVectorCreate(comm_, iLower_, iUpper_, &slnRef_);
    HYPRE_IJVectorSetObjectType(slnRef_, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(slnRef_);
    HYPRE_IJVectorGetObject(sln_, (void**)&parSlnRef_);

    HYPRE_IJMatrixSetConstantValues(mat_, 0.0);
    HYPRE_ParVectorSetConstantValues(parRhs_, 0.0);
    HYPRE_ParVectorSetConstantValues(parSln_, 0.0);
    HYPRE_ParVectorSetConstantValues(parSlnRef_, 0.0);

    MPI_Barrier(comm_);
    if (iproc_ == 0) {
        auto stop = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = stop - start;
        std::cout << elapsed.count() << " seconds" << std::endl;

        timers_.emplace_back("Initialize system", elapsed.count());
    }
}

void HypreSystem::finalize_system()
{
    auto start = std::chrono::system_clock::now();
    if (iproc_ == 0) {
        std::cout << "Assembling data HYPRE structures... ";
    }

    HYPRE_Int nrows = 1;
    HYPRE_Int ncols = 1;
    double setval = 1.0; // Set diagonal to 1 for missing rows
    for (HYPRE_Int i=iLower_; i < iUpper_; i++) {
        if (rowFilled_[i] > 0) continue;
        HYPRE_IJMatrixSetValues(mat_, nrows, &ncols, &i, &i, &setval);
    }

    HYPRE_IJMatrixAssemble(mat_);
    HYPRE_IJMatrixGetObject(mat_, (void**)&parMat_);

    HYPRE_IJVectorAssemble(rhs_);
    HYPRE_IJVectorGetObject(rhs_, (void**)&(parRhs_));

    HYPRE_IJVectorAssemble(sln_);
    HYPRE_IJVectorGetObject(sln_, (void**)&(parSln_));

    HYPRE_IJVectorAssemble(slnRef_);
    HYPRE_IJVectorGetObject(sln_, (void**)&(parSlnRef_));

    MPI_Barrier(comm_);
    if (iproc_ == 0) {
        auto stop = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = stop - start;
        std::cout << elapsed.count() << " seconds" << std::endl;

        timers_.emplace_back("Finalize system", elapsed.count());
    }
}

void HypreSystem::read_mm_matrix(std::string matfile)
{
    if (iproc_ != 0) return;
    std::cout << "Loading matrix into HYPRE_IJMatrix... ";

    auto start = std::chrono::system_clock::now();

    // Set up row order array that will be used later with RHS and solution files
    rowOrder_.resize(totalRows_);

    FILE* fh;
    int err;
    int msize, nsize, nnz;
    HYPRE_Int irow, icol;
    double value;

    if ((fh = fopen(matfile.c_str(), "r")) == NULL) {
        throw std::runtime_error("Cannot open matrix file: " + matfile);
    }

    // Skip checks this time; already performed during scan step

    err = mm_read_mtx_crd_size(fh, &msize, &nsize, &nnz);
    if (err != 0)
        throw std::runtime_error("Cannot read matrix sizes in file: " + matfile);

    HYPRE_Int lastSeen = totalRows_ + 10;
    HYPRE_Int idx = 0;
    HYPRE_Int ncols = 1;
    for (int i=0; i < nnz; i++) {
#ifdef HYPRE_BIGINT
        fscanf(fh, "%lld %lld %lf\n", &irow, &icol, &value);
#else
        fscanf(fh, "%d %d %lf\n", &irow, &icol, &value);
#endif
        irow--;
        icol--;
        HYPRE_IJMatrixAddToValues(mat_, 1, &ncols, &irow, &icol, &value);
        rowFilled_[irow] = 1;

        if (irow != lastSeen) {
            rowOrder_[idx++] = irow;
            lastSeen = irow;
        }
    }

    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    std::cout << elapsed.count() << " seconds" << std::endl;
    timers_.emplace_back("Read matrix", elapsed.count());

    fclose(fh);
}

void HypreSystem::determine_system_sizes(std::string matfile)
{
    if (iproc_ != 0) return;
    auto start = std::chrono::system_clock::now();

    FILE* fh;
    MM_typecode matcode;
    int err;
    int msize, nsize, nnz;
    HYPRE_Int irow, icol;
    double value;

    if ((fh = fopen(matfile.c_str(), "r")) == NULL) {
        throw std::runtime_error("Cannot open matrix file: " + matfile);
    }

    err = mm_read_banner(fh, &matcode);
    if (err != 0)
        throw std::runtime_error("Cannot read matrix banner");

    if (!mm_is_valid(matcode) || !mm_is_coordinate(matcode))
        throw std::runtime_error("Invalid matrix market file encountered");

    err = mm_read_mtx_crd_size(fh, &msize, &nsize, &nnz);
    if (err != 0)
        throw std::runtime_error("Cannot read matrix sizes in file: " + matfile);

    M_ = msize;
    N_ = nsize;
    nnz_= nnz;

    std::cout << "Scanning matrix structure... ";
    totalRows_ = 0;
    for (int i=0; i < nnz; i++)
    {
#ifdef HYPRE_BIGINT
        fscanf(fh, "%lld %lld %lf\n", &irow, &icol, &value);
#else
        fscanf(fh, "%d %d %lf\n", &irow, &icol, &value);
#endif
        totalRows_ = std::max(totalRows_, irow);
    }

    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    std::cout << elapsed.count() << " seconds" << std::endl;
    std::cout << "Matrix parameters: "
              << "M = " << msize
              << "; N = " << nsize
              << "; nnz = " << nnz
              << "; maxRowID = " << totalRows_ << std::endl;

    timers_.emplace_back("Scan matrix", elapsed.count());
    fclose(fh);
}

void HypreSystem::read_mm_vector(std::string mmfile, HYPRE_IJVector& vec)
{
    if (iproc_ != 0) return;
    auto start = std::chrono::system_clock::now();

    FILE* fh;
    MM_typecode matcode;
    int err;
    int msize, nsize;
    double value;

    std::cout << "Reading array into HYPRE_IJVector.... ";
    if ((fh = fopen(mmfile.c_str(), "r")) == NULL) {
        throw std::runtime_error("Cannot open matrix file: " + mmfile);
    }

    err = mm_read_banner(fh, &matcode);
    if (err != 0)
        throw std::runtime_error("Cannot read matrix banner");

    if (!mm_is_valid(matcode) || !mm_is_array(matcode))
        throw std::runtime_error("Invalid matrix market file encountered");

    err = mm_read_mtx_array_size(fh, &msize, &nsize);
    if (err != 0)
        throw std::runtime_error("Cannot read matrix sizes in file: " + mmfile);

    if ((msize != M_))
        throw std::runtime_error("Inconsistent sizes for Matrix and Vector");

    for (int i=0; i < msize; i++) {
        fscanf(fh, "%lf\n", &value);
        HYPRE_IJVectorAddToValues(vec, 1, &rowOrder_[i], &value);
    }

    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    std::cout << elapsed.count() << " seconds" << std::endl;
    timers_.emplace_back("Read vector", elapsed.count());
    fclose(fh);
}

} // namespace nalu
