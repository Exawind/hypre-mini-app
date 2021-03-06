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
#include <sstream>
#include <string.h>

namespace nalu {

  namespace {

    template<typename T>
      T get_optional(YAML::Node& node, std::string key, T default_value)
      {
        if (node[key])
          return node[key].as<T>();
        else
          return default_value;
      }

  }

  HypreSystem::HypreSystem(
      MPI_Comm comm,
      YAML::Node& inpfile) : comm_(comm),
			     inpfile_(inpfile)
  {
    MPI_Comm_rank(comm, &iproc_);
    MPI_Comm_size(comm, &nproc_);
  }

  void
    HypreSystem::load()
    {
      YAML::Node linsys = inpfile_["linear_system"];
      std::string mat_format = get_optional<std::string>(linsys, "type", "matrix_market") ;

      if (mat_format == "matrix_market") {
        load_matrix_market();
      } else if (mat_format == "hypre_ij") {
        load_hypre_format();
      } else {
        throw std::runtime_error("Invalid linear system format option: " + mat_format);
      }

      if (linsys["write_outputs"])
        outputSystem_ = linsys["write_outputs"].as<bool>();

      YAML::Node solver = inpfile_["solver_settings"];
      std::string method = solver["method"].as<std::string>();
      std::string preconditioner = solver["preconditioner"].as<std::string>();

      if (preconditioner == "boomeramg") {
        setup_boomeramg_precond();
      } else if (preconditioner == "none") {
        usePrecond_ = false;
        if (iproc_ == 0)
          std::cout << "No preconditioner used" << std::endl;
      }
      else {
        throw std::runtime_error("Invalid option for preconditioner provided"
            + preconditioner);
      }
      if (!method.compare( "gmres")){       
        if (iproc_ == 0) std::cout << "using GMRES solver" << std::endl;
        setup_gmres();
      } else if (!method.compare("boomeramg")) {
        if (iproc_ == 0) std::cout << "using BOOMERANG solver" << std::endl;
        setup_boomeramg_solver();
      }
      else if (!method.compare("cogmres")){
        if (iproc_ == 0) std::cout << "using CO-GMRES solver" << std::endl;
        setup_cogmres();
      }
      else {
        throw std::runtime_error("Invalid option for solver method provided: "
            + method);
      }
    }

    void HypreSystem::setup_boomeramg_solver()
    {
      YAML::Node node = inpfile_["solver_settings"];

      HYPRE_BoomerAMGCreate(&solver_);
      HYPRE_BoomerAMGSetTol(solver_, get_optional(node, "tolerance", 1.0e-5));
      HYPRE_BoomerAMGSetMaxIter(solver_, get_optional(node, "max_iterations", 1000));
      HYPRE_BoomerAMGSetPrintLevel(solver_, get_optional(node, "print_level", 4));

      HYPRE_BoomerAMGSetCoarsenType(
          precond_, get_optional(node, "coarsen_type", 8));
      HYPRE_BoomerAMGSetCycleType (
          precond_, get_optional(node, "cycle_type", 1));
      HYPRE_BoomerAMGSetRelaxType(
          precond_, get_optional(node, "relax_type", 6));
      HYPRE_BoomerAMGSetNumSweeps(
          precond_, get_optional(node, "num_sweeps", 1));
      HYPRE_BoomerAMGSetRelaxOrder(
          precond_, get_optional(node, "relax_order", 1));
      HYPRE_BoomerAMGSetMaxLevels(
          precond_, get_optional(node, "max_levels", 20));
      HYPRE_BoomerAMGSetStrongThreshold(
          precond_, get_optional(node, "strong_threshold", 0.57));

      solverDestroyPtr_ = &HYPRE_BoomerAMGDestroy;
      solverSetupPtr_ = &HYPRE_BoomerAMGSetup;
      solverPrecondPtr_ = nullptr;
      solverSolvePtr_ = &HYPRE_BoomerAMGSolve;
      usePrecond_ = false;
    }

    void HypreSystem::setup_boomeramg_precond()
    {
      YAML::Node node = inpfile_["boomeramg_settings"];

      HYPRE_BoomerAMGCreate(&precond_);
      HYPRE_BoomerAMGSetPrintLevel(
          precond_, get_optional(node, "print_level", 1));
      HYPRE_BoomerAMGSetCoarsenType(
          precond_, get_optional(node, "coarsen_type", 8));
      HYPRE_BoomerAMGSetCycleType (
          precond_, get_optional(node, "cycle_type", 1));
      HYPRE_BoomerAMGSetRelaxType(
          precond_, get_optional(node, "relax_type", 6));
      HYPRE_BoomerAMGSetNumSweeps(
          precond_, get_optional(node, "num_sweeps", 1));
      HYPRE_BoomerAMGSetTol(
          precond_, get_optional(node, "tolerance", 0.0));
      HYPRE_BoomerAMGSetMaxIter(
          precond_, get_optional(node, "max_iterations", 1));
      HYPRE_BoomerAMGSetRelaxOrder(
          precond_, get_optional(node, "relax_order", 1));
      HYPRE_BoomerAMGSetMaxLevels(
          precond_, get_optional(node, "max_levels", 20));
      HYPRE_BoomerAMGSetStrongThreshold(
          precond_, get_optional(node, "strong_threshold", 0.57));

      if (node["non_galerkin_tol"]) {
        double non_galerkin_tol = node["non_galerkin_tol"].as<double>();
        HYPRE_BoomerAMGSetNonGalerkinTol(precond_, non_galerkin_tol);

        if (node["non_galerkin_level_tols"]) {
          auto ngnode = node["non_galerkin_level_tols"];
          std::vector<int> levels = ngnode["levels"].as<std::vector<int>>();
          std::vector<double> tol = ngnode["tolerances"].as<std::vector<double>>();

          if (levels.size() != tol.size())
            throw std::runtime_error(
                "Hypre Config:: Invalid non_galerkin_level_tols");

          for (size_t i = 0; i < levels.size(); i++) {
            HYPRE_BoomerAMGSetLevelNonGalerkinTol(precond_, tol[i], levels[i]);
          }
        }
      }

      if (node["variant"]) {
        int int_value = node["variant"].as<int>();
        HYPRE_BoomerAMGSetVariant(precond_, int_value);
      }

      if (node["keep_transpose"]) {
        int int_value = node["keep_transpose"].as<int>();
        HYPRE_BoomerAMGSetKeepTranspose(precond_, int_value);
      }

      if (node["interp_type"]) {
        int int_value = node["interp_type"].as<int>();
        HYPRE_BoomerAMGSetInterpType(precond_, int_value);
      }

      if (node["min_coarse_size"]) {
        int int_value = node["min_coarse_size"].as<int>();
        HYPRE_BoomerAMGSetMinCoarseSize(precond_, int_value);
      }
      if (node["max_coarse_size"]) {
        int int_value = node["max_coarse_size"].as<int>();
        HYPRE_BoomerAMGSetMaxCoarseSize(precond_, int_value);
      }
      if (node["pmax_elmts"]) {
        int int_value = node["pmax_elmts"].as<int>();
        HYPRE_BoomerAMGSetAggPMaxElmts(precond_, int_value);
      }
      if (node["agg_num_levels"]) {
        int int_value = node["agg_num_levels"].as<int>();
        HYPRE_BoomerAMGSetAggNumLevels(precond_, int_value);
      }
      if (node["agg_interp_type"]) {
        int int_value = node["agg_interp_type"].as<int>();
        HYPRE_BoomerAMGSetAggInterpType(precond_, int_value);
      }
      if (node["agg_pmax_elmts"]) {
        int int_value = node["agg_pmax_elmts"].as<int>();
        HYPRE_BoomerAMGSetAggPMaxElmts(precond_, int_value);
      }
      if (node["trunc_factor"]) {
        double float_value = node["trunc_factor"].as<double>();
        HYPRE_BoomerAMGSetTruncFactor(precond_, float_value);
      }

      if (node["smooth_type"]) {
        int smooth_type = node["smooth_type"].as<int>();
        HYPRE_BoomerAMGSetSmoothType(precond_, smooth_type);
      }

      precondSetupPtr_ = &HYPRE_BoomerAMGSetup;
      precondSolvePtr_ = &HYPRE_BoomerAMGSolve;
      precondDestroyPtr_ = &HYPRE_BoomerAMGDestroy;
    }

    void HypreSystem::setup_cogmres()
    {
      YAML::Node node = inpfile_["solver_settings"];

      HYPRE_ParCSRCOGMRESCreate(comm_, &solver_);
      HYPRE_ParCSRCOGMRESSetTol(solver_, get_optional(node, "tolerance", 1.0e-5));
      HYPRE_ParCSRCOGMRESSetMaxIter(solver_, get_optional(node, "max_iterations", 1000));
      HYPRE_ParCSRCOGMRESSetKDim(solver_, get_optional(node, "kspace", 10));
      HYPRE_ParCSRCOGMRESSetPrintLevel(solver_, get_optional(node, "print_level", 4));

      solverDestroyPtr_ = &HYPRE_ParCSRCOGMRESDestroy;
      solverSetupPtr_ = &HYPRE_ParCSRCOGMRESSetup;
      solverPrecondPtr_ = &HYPRE_ParCSRCOGMRESSetPrecond;
      solverSolvePtr_ = &HYPRE_ParCSRCOGMRESSolve;
    }

    void HypreSystem::setup_gmres()
    {
      YAML::Node node = inpfile_["solver_settings"];

      HYPRE_ParCSRGMRESCreate(comm_, &solver_);
      HYPRE_ParCSRGMRESSetTol(solver_, get_optional(node, "tolerance", 1.0e-5));
      HYPRE_ParCSRGMRESSetMaxIter(solver_, get_optional(node, "max_iterations", 1000));
      HYPRE_ParCSRGMRESSetKDim(solver_, get_optional(node, "kspace", 10));
      HYPRE_ParCSRGMRESSetPrintLevel(solver_, get_optional(node, "print_level", 4));

      solverDestroyPtr_ = &HYPRE_ParCSRGMRESDestroy;
      solverSetupPtr_ = &HYPRE_ParCSRGMRESSetup;
      solverPrecondPtr_ = &HYPRE_ParCSRGMRESSetPrecond;
      solverSolvePtr_ = &HYPRE_ParCSRGMRESSolve;
    }

    void HypreSystem::init_row_decomposition()
    {
      HYPRE_Int rowsPerProc = totalRows_ / nproc_;
      HYPRE_Int remainder = totalRows_ % nproc_;

      iLower_ = rowsPerProc * iproc_ + std::min<HYPRE_Int>(iproc_, remainder);
      iUpper_ = rowsPerProc * (iproc_ + 1) + std::min<HYPRE_Int>(iproc_ + 1, remainder) - 1;
      numRows_ = iUpper_ - iLower_ + 1;

      std::cout << "  Rank: " << std::setw(4) << iproc_ << ":: iLower = "
        << std::setw(9) << iLower_ << "; iUpper = "
        << std::setw(9) << iUpper_ << "; numRows = "
        << numRows_ << std::endl;
    }

    void HypreSystem::init_system()
    {
      auto start = std::chrono::system_clock::now();
      if (iproc_ == 0)
        std::cout << "Initializing data HYPRE structures... ";

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

      HYPRE_IJMatrixAssemble(mat_);
      HYPRE_IJVectorAssemble(rhs_);
      HYPRE_IJVectorAssemble(sln_);
      HYPRE_IJMatrixGetObject(mat_, (void**)&parMat_);
      HYPRE_IJVectorGetObject(rhs_, (void**)&(parRhs_));
      HYPRE_IJVectorGetObject(sln_, (void**)&(parSln_));

      if (checkSolution_) {
        HYPRE_IJVectorAssemble(slnRef_);
        HYPRE_IJVectorGetObject(sln_, (void**)&(parSlnRef_));
      }

      MPI_Barrier(comm_);
      if (iproc_ == 0) {
        auto stop = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = stop - start;
        std::cout << elapsed.count() << " seconds" << std::endl;
        timers_.emplace_back("Finalize system", elapsed.count());
      }
    }

    void HypreSystem::solve()
    {
      finalize_system();

      auto start = std::chrono::system_clock::now();
      if (usePrecond_) {
        solverPrecondPtr_(
            solver_, precondSolvePtr_, precondSetupPtr_, precond_);
      }
      solverSetupPtr_(solver_, parMat_, parRhs_, parSln_);
      MPI_Barrier(comm_);
      auto stop1 = std::chrono::system_clock::now();
      std::chrono::duration<double> setup = stop1 - start;
      solverSolvePtr_(solver_, parMat_, parRhs_, parSln_);
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
      if (!outputSystem_) return;

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
      if (!checkSolution_) {
        if (iproc_ == 0) {
          std::cout << "Reference solution not provided; skipping error check."
            << std::endl;
        }
        return;
      }

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

    /********************************************************************************/
    /* Generic methods for building matrices/vectors for CUDA/CPU/... useable from  */
    /* IJ or MM matrix formats                                                      */
    /********************************************************************************/

    void HypreSystem::hypre_matrix_set_values()
    {
      if (iproc_==0)
	std::cout << "Loading matrix into HYPRE_IJMatrix... ";

      auto start = std::chrono::system_clock::now();

      int nnz_this_rank = rows_.size();

#ifdef HAVE_CUDA
      HYPRE_Int * d_cols, * d_rows;
      double * d_values;

      cudaMallocManaged((void**)&d_rows, nnz_this_rank*sizeof(HYPRE_Int));
      cudaMallocManaged((void**)&d_cols, nnz_this_rank*sizeof(HYPRE_Int));
      cudaMallocManaged((void**)&d_values, nnz_this_rank*sizeof(double));

      memcpy(d_rows, rows_.data(), nnz_this_rank*sizeof(HYPRE_Int));
      memcpy(d_cols, cols_.data(), nnz_this_rank*sizeof(HYPRE_Int));
      memcpy(d_values, vals_.data(), nnz_this_rank*sizeof(double));

      /* Call this on UVM data */
      HYPRE_IJMatrixSetValues2(mat_, nnz_this_rank, NULL, d_rows, NULL, d_cols, d_values);
      cudaDeviceSynchronize();

      cudaFree(d_rows);
      cudaFree(d_cols);
      cudaFree(d_values);
#else
      HYPRE_IJMatrixSetValues2(mat_, nnz_this_rank, NULL, rows_.data(), NULL, cols_.data(), vals_.data());
#endif

      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      if (iproc_ == 0)
	std::cout << elapsed.count() << " seconds" << std::endl;
      timers_.emplace_back("Build HYPRE matrix", elapsed.count());
    }

    void HypreSystem::hypre_vector_set_values(HYPRE_IJVector& vec)
    {
      if (iproc_==0)
	std::cout << "Loading matrix into HYPRE_IJVector... ";

      auto start = std::chrono::system_clock::now();

#ifdef HAVE_CUDA
      HYPRE_Int * d_indices;
      double * d_values;
      cudaMallocManaged((void**)&d_values, vector_values_.size()*sizeof(double));
      cudaMallocManaged((void**)&d_indices, vector_indices_.size()*sizeof(HYPRE_Int));

      memcpy(d_values, vector_values_.data(), vector_values_.size()*sizeof(double));
      memcpy(d_indices, vector_indices_.data(), vector_indices_.size()*sizeof(HYPRE_Int));

      /* Call this on UVM data */
      HYPRE_IJVectorSetValues(vec, iUpper_-iLower_+1, d_indices, d_values);
      cudaDeviceSynchronize();

      cudaFree(d_values);
      cudaFree(d_indices);
#else
      HYPRE_IJVectorSetValues(vec, iUpper_-iLower_+1, vector_indices_.data(), vector_values_.data());
#endif

      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      if (iproc_ == 0)
	std::cout << elapsed.count() << " seconds" << std::endl;
      timers_.emplace_back("Build HYPRE vector", elapsed.count());
    }

    /********************************************************************************/
    /*                               HYPRE IJ Format                                */
    /********************************************************************************/

    void HypreSystem::load_hypre_format()
    {
      YAML::Node linsys = inpfile_["linear_system"];
      int nfiles = get_optional(linsys, "num_partitions", nproc_);
      bool useCuda=false;

#ifdef HAVE_CUDA
      // This is hack to prevent the cuda build from using the native load procedure
      useCuda=true;
#endif
      if (nfiles == nproc_ && !useCuda)
        load_hypre_native();
      else {
        std::string matfile = linsys["matrix_file"].as<std::string>();
        std::string rhsfile = linsys["rhs_file"].as<std::string>();

	// Scan the matrix and determine the sizes
	determine_ij_system_sizes(matfile, nfiles);
	
	// generic method for IJ and MM
	init_row_decomposition();
	
	// Create HYPRE data structures
	init_system();

	// build matrices and vectors
        build_ij_matrix(matfile, nfiles);
        build_ij_vector(rhsfile, nfiles, rhs_);
     
        if (linsys["sln_file"]) {
          std::string slnfile = linsys["sln_file"].as<std::string>();
          checkSolution_ = true;
          build_ij_vector(slnfile, nfiles, slnRef_);
        }
      }
    }

    /*******************
     *
     *******************/
    void HypreSystem::load_hypre_native()
    {
      MPI_Barrier(comm_);
      auto start = std::chrono::system_clock::now();
      if (iproc_ == 0) {
        std::cout << "Loading HYPRE IJ files... ";
      }

      YAML::Node linsys = inpfile_["linear_system"];

      std::string matfile = linsys["matrix_file"].as<std::string>();
      std::string rhsfile = linsys["rhs_file"].as<std::string>();

      HYPRE_IJMatrixRead(matfile.c_str(), comm_, HYPRE_PARCSR, &mat_);
      HYPRE_IJVectorRead(rhsfile.c_str(), comm_, HYPRE_PARCSR, &rhs_);

      if (linsys["sln_file"]) {
        std::string slnfile = linsys["sln_file"].as<std::string>();
        checkSolution_ = true;
        HYPRE_IJVectorRead(slnfile.c_str(), comm_, HYPRE_PARCSR, &slnRef_);
      }

      // Figure out local range
      HYPRE_Int jlower, jupper;
      HYPRE_IJMatrixGetLocalRange(mat_, &iLower_, &iUpper_, &jlower, &jupper);
      numRows_ = (iUpper_ - iLower_ + 1);

      // Initialize the solution vector
      HYPRE_IJVectorCreate(comm_, iLower_, iUpper_, &sln_);
      HYPRE_IJVectorSetObjectType(sln_, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(sln_);
      HYPRE_IJVectorGetObject(sln_, (void**)&parSln_);
      HYPRE_ParVectorSetConstantValues(parSln_, 0.0);

      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      if (iproc_ == 0)
        std::cout << elapsed.count() << " seconds" << std::endl;

      timers_.emplace_back("Load system", elapsed.count());

      MPI_Barrier(comm_);
      std::cout << "  Rank: " << std::setw(4) << iproc_ << ":: iLower = "
        << std::setw(9) << iLower_ << "; iUpper = "
        << std::setw(9) << iUpper_ << "; numRows = "
        << numRows_ << std::endl;
      MPI_Barrier(comm_);
    }

    /*******************
     *
     *******************/
    void HypreSystem::determine_ij_system_sizes(std::string matfile, int nfiles)
    {
      MPI_Barrier(comm_);
      auto start = std::chrono::system_clock::now();

      HYPRE_Int ilower, iupper, jlower, jupper;
      HYPRE_Int imin = 0;
      HYPRE_Int imax = 0;
      HYPRE_Int gmin = 0;
      HYPRE_Int gmax = 0;

      for (int ii=iproc_; ii < nfiles; ii+=nproc_) {
        FILE* fh;
        std::ostringstream suffix;
        suffix << matfile << "." << std::setw(5) << std::setfill('0') << ii;

        if ((fh = fopen(suffix.str().c_str(), "r")) == NULL) {
          throw std::runtime_error("Cannot open matrix file: " + suffix.str());
        }

#ifdef HYPRE_BIGINT
        fscanf(fh, "%lld %lld %lld %lld\n", &ilower, &iupper, &jlower, &jupper);
#else
        fscanf(fh, "%d %d %d %d\n", &ilower, &iupper, &jlower, &jupper);
#endif
        imin = std::min(imin, ilower);
        imax = std::max(imax, iupper);
        fclose(fh);
      }

      MPI_Allreduce(&imin, &gmin, 1, HYPRE_MPI_INT, MPI_MIN, comm_);
      MPI_Allreduce(&imax, &gmax, 1, HYPRE_MPI_INT, MPI_MAX, comm_);
      totalRows_ = (gmax - gmin) + 1;

      MPI_Barrier(comm_);
      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      if (iproc_==0) 
	std::cout << elapsed.count() << " seconds" << std::endl;
      timers_.emplace_back("IJ : determine system size", elapsed.count());
    }

    /*******************
     *
     *******************/
    void HypreSystem::build_ij_matrix(std::string matfile, int nfiles)
    {
      MPI_Barrier(comm_);
      auto start = std::chrono::system_clock::now();

      // read the files
      if (iproc_ == 0)
        std::cout << "Reading " << nfiles << " HYPRE IJ Matrix files... " << std::endl;

      HYPRE_Int ilower, iupper, jlower, jupper;
      HYPRE_Int irow, icol;
      double value;

      /* store the loaded matrix into these vectors */
      rows_.resize(0);
      cols_.resize(0);
      vals_.resize(0);

      // Need to loop over all the files
      for (int ii=0; ii < nfiles; ii++) {
        FILE* fh;
        std::ostringstream suffix;
        suffix << matfile << "." << std::setw(5) << std::setfill('0') << ii;

        if ((fh = fopen(suffix.str().c_str(), "r")) == NULL) {
          throw std::runtime_error("Cannot open matrix file: " + suffix.str());
        }

#ifdef HYPRE_BIGINT
        fscanf(fh, "%lld %lld %lld %lld\n", &ilower, &iupper, &jlower, &jupper);
#else
        fscanf(fh, "%d %d %d %d\n", &ilower, &iupper, &jlower, &jupper);
#endif

	// need the + 1 so that the upper boundary are inclusive
	int overlap = std::max(0, std::min(iUpper_+1, iupper+1) - std::max(iLower_, ilower));
        if (overlap) {
#ifdef HYPRE_BIGINT
	  while (fscanf(fh, "%lld %lld%*[ \t]%le\n", &irow, &icol, &value) != EOF) {
#else
	  while (fscanf(fh, "%d %d%*[ \t]%le\n", &irow, &icol, &value) != EOF) {
#endif
            if (irow>=iLower_ && irow<=iUpper_) {
              rows_.push_back(irow);
	      cols_.push_back(icol);
	      vals_.push_back(value);
	    }
	  }
	}
        fclose(fh);
      }

      // Set the values of the matrix
      hypre_matrix_set_values();

      MPI_Barrier(comm_);
      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      timers_.emplace_back("IJ : read and build matrix", elapsed.count());
    }

    /*******************
     *
     *******************/
    void HypreSystem::build_ij_vector(std::string vecfile, int nfiles, HYPRE_IJVector& vec)
    {
      MPI_Barrier(comm_);
      auto start = std::chrono::system_clock::now();
      if (iproc_ == 0)
        std::cout << "Reading " << nfiles << " HYPRE IJ Vector files... " << std::endl;

      HYPRE_Int ilower, iupper;
      HYPRE_Int irow;
      double value;

      /* resize these */
      vector_indices_.resize(0);
      vector_values_.resize(0);

      for (int ii=0; ii < nfiles; ii++) {
        FILE* fh;
        std::ostringstream suffix;
        suffix << vecfile << "." << std::setw(5) << std::setfill('0') << ii;

        if ((fh = fopen(suffix.str().c_str(), "r")) == NULL) {
          throw std::runtime_error("Cannot open vector file: " + suffix.str());
        }

#ifdef HYPRE_BIGINT
        fscanf(fh, "%lld %lld\n", &ilower, &iupper);
#else
        fscanf(fh, "%d %d\n", &ilower, &iupper);
#endif

	// need the + 1 so that the upper boundary are inclusive
	int overlap = std::max(0, std::min(iUpper_+1, iupper+1) - std::max(iLower_, ilower));
        if (overlap) {
#ifdef HYPRE_BIGINT
          while (fscanf(fh, "%lld%*[ \t]%le\n", &irow, &value) != EOF) {
#else
          while (fscanf(fh, "%d%*[ \t]%le\n", &irow, &value) != EOF) {
#endif
	    vector_indices_.push_back(irow);
	    vector_values_.push_back(value);
	  }
	}
        fclose(fh);
      }

      /* Build the vector */
      hypre_vector_set_values(vec);

      MPI_Barrier(comm_);
      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      timers_.emplace_back("IJ : read and build vector", elapsed.count());
    }

    /********************************************************************************/
    /*                          Matrix Market Format                                */
    /********************************************************************************/

    void HypreSystem::load_matrix_market()
    {
      YAML::Node linsys = inpfile_["linear_system"];

      std::string matfile = linsys["matrix_file"].as<std::string>();
      std::string rhsfile = linsys["rhs_file"].as<std::string>();

      // Scan the matrix and determine the sizes
      determine_mm_system_sizes(matfile);

      // generic method for IJ and MM
      init_row_decomposition();

      // Create HYPRE data structures
      init_system();

      // Build matrices and vectors
      build_mm_matrix(matfile);
      build_mm_vector(rhsfile, rhs_);

      if (linsys["sln_file"]) {
        std::string slnfile = linsys["sln_file"].as<std::string>();
        checkSolution_ = true;
        build_mm_vector(slnfile, slnRef_);
      }
    }

    /*******************
     *
     *******************/
    void HypreSystem::determine_mm_system_sizes(std::string matfile)
    {
      MPI_Barrier(comm_);
      auto start = std::chrono::system_clock::now();

      FILE* fh;
      MM_typecode matcode;
      int err;
      int msize, nsize, nnz;

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

      totalRows_ = M_ = msize;
      N_ = nsize;
      nnz_= nnz;

      MPI_Barrier(comm_);
      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      if (iproc_==0)
	std::cout << elapsed.count() << " seconds" << std::endl;
      timers_.emplace_back("Matrix market : determine system size", elapsed.count());
      fclose(fh);
    }
  
    /*******************
     *
     *******************/
    void HypreSystem::build_mm_matrix(std::string matfile)
    {
      MPI_Barrier(comm_);
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

      rows_.resize(0);
      cols_.resize(0);
      vals_.resize(0);
      for (int i=0; i < nnz; i++)
      {
#ifdef HYPRE_BIGINT
        fscanf(fh, "%lld %lld %lf\n", &irow, &icol, &value);
#else
        fscanf(fh, "%d %d %lf\n", &irow, &icol, &value);
#endif

	irow--;
	icol--;

	if (irow>=iLower_ && irow<=iUpper_) {
	  rows_.push_back(irow);
	  cols_.push_back(icol);
	  vals_.push_back(value);
	}
      }

      // Set the values of the matrix
      hypre_matrix_set_values();

      MPI_Barrier(comm_);
      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      if (iproc_==0)
	std::cout << elapsed.count() << " seconds" << std::endl;
      timers_.emplace_back("Matrix market : read and build matrix", elapsed.count());
      fclose(fh);
    }

    /*******************
     *
     *******************/
    void HypreSystem::build_mm_vector(std::string mmfile, HYPRE_IJVector& vec)
    {
      MPI_Barrier(comm_);
      auto start = std::chrono::system_clock::now();

      FILE* fh;
      MM_typecode matcode;
      int err;
      int msize, nsize;
      double value;

      if (iproc_==0)
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

      vector_indices_.resize(0);
      vector_values_.resize(0);
      for (int i=0; i < msize; i++) {
	/* only read in the part owned by this rank */
	if (i>=iLower_ && i<=iUpper_) {
	  fscanf(fh, "%lf\n", &value);
	  vector_values_.push_back(value);
	  vector_indices_.push_back(i);
	}
      }

      /* build the vector */
      hypre_vector_set_values(vec);

      MPI_Barrier(comm_);
      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      if (iproc_==0)
	std::cout << elapsed.count() << " seconds" << std::endl;
      timers_.emplace_back("Matrix market : read and build vector", elapsed.count());
      fclose(fh);
    }
} // namespace nalu
