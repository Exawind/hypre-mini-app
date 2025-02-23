#include "HypreSystem.h"
#include "GpuQualifiers.h"

#if defined (HYPRE_USING_HIP)
#include "laplace_3d_weak_scaling.hpp"
#endif


namespace nalu {
HypreSystem::HypreSystem(MPI_Comm comm, YAML::Node &inpfile)
    : comm_(comm), inpfile_(inpfile) {
  MPI_Comm_rank(comm, &iproc_);
  MPI_Comm_size(comm, &nproc_);
}

void HypreSystem::load() {
  YAML::Node linsys = inpfile_["linear_system"];
  if (linsys["write_amg_matrices"]) {
    if (linsys["write_amg_matrices"].as<bool>())
      writeAmgMatrices_ = true;
  }
  std::string mat_format =
      get_optional<std::string>(linsys, "type", "matrix_market");

  if (iproc_ == 0)
     printf("%s : Using %s mat_format\n", __FUNCTION__, mat_format.c_str());

  if (mat_format == "matrix_market") {
    load_matrix_market();
  } else if (mat_format == "hypre_ij") {
    load_hypre_format();
  } else if (mat_format == "build_27pt_stencil") {
#if defined(HYPRE_USING_HIP)
    build_27pt_stencil();
#else
throw std::runtime_error("Cannot use build_27pt_stencil() without Hypre HIP support");
#endif
  } else {
    throw std::runtime_error("Invalid linear system format option: " +
                             mat_format);
  }

  if (linsys["write_outputs"])
    outputSystem_ = linsys["write_outputs"].as<bool>();
  if (linsys["write_solution"])
    outputSolution_ = linsys["write_solution"].as<bool>();
}

void HypreSystem::setup_precon_and_solver() {
  YAML::Node solver = inpfile_["solver_settings"];
  std::string method = solver["method"].as<std::string>();
  std::string preconditioner = solver["preconditioner"].as<std::string>();

  if (iproc_ == 0)
     printf("%s : Using %s solver with %s preconditioner\n",
            __FUNCTION__, method.c_str(), preconditioner.c_str());

  if (preconditioner == "boomeramg") {
    setup_boomeramg_precond();
  }
  else if (preconditioner == "ilu"){
    setup_ilu_precond(); 
  } else if (preconditioner == "none") {
    usePrecond_ = false;
  } else {
    throw std::runtime_error("Invalid option for preconditioner provided" +
                             preconditioner);
  }
  if (!method.compare("gmres")) {
    setup_gmres();
  } else if (!method.compare("cg")) {
    setup_cg();
  } else if (!method.compare("bicg")) {
    setup_bicg();
  } else if (!method.compare("fgmres")) {
    setup_fgmres();
  } else if (!method.compare("boomeramg")) {
    setup_boomeramg_solver();
  } else if (!method.compare("cogmres")) {
    setup_cogmres();
  } else if (!method.compare("ilu")) {
    setup_ilu();
  } else {
    throw std::runtime_error("Invalid option for solver method provided: " +
                             method);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  fflush(stdout);
}

void HypreSystem::setup_boomeramg_solver() {
  YAML::Node node = inpfile_["solver_settings"];

  HYPRE_BoomerAMGCreate(&solver_);
  HYPRE_BoomerAMGSetTol(solver_, get_optional(node, "tolerance", 1.0e-5));
  HYPRE_BoomerAMGSetMaxIter(solver_,
                            get_optional(node, "max_iterations", 1000));
  HYPRE_BoomerAMGSetPrintLevel(solver_, get_optional(node, "print_level", 4));

  HYPRE_BoomerAMGSetCoarsenType(precond_,
                                get_optional(node, "coarsen_type", 8));
  HYPRE_BoomerAMGSetCycleType(precond_, get_optional(node, "cycle_type", 1));
  HYPRE_BoomerAMGSetRelaxType(precond_, get_optional(node, "relax_type", 6));
  HYPRE_BoomerAMGSetNumSweeps(precond_, get_optional(node, "num_sweeps", 1));
  HYPRE_BoomerAMGSetSmoothNumSweeps(precond_,
                                    get_optional(node, "smooth_num_sweeps", 1));
  HYPRE_BoomerAMGSetRelaxOrder(precond_, get_optional(node, "relax_order", 1));
  HYPRE_BoomerAMGSetMaxLevels(precond_, get_optional(node, "max_levels", 20));
  HYPRE_BoomerAMGSetStrongThreshold(
      precond_, get_optional(node, "strong_threshold", 0.57));

  solverDestroyPtr_ = &HYPRE_BoomerAMGDestroy;
  solverSetupPtr_ = &HYPRE_BoomerAMGSetup;
  solverPrecondPtr_ = nullptr;
  solverSolvePtr_ = &HYPRE_BoomerAMGSolve;
  usePrecond_ = false;
}

void HypreSystem::setup_boomeramg_precond() {
  YAML::Node node = inpfile_["boomeramg_settings"];

  HYPRE_BoomerAMGCreate(&precond_);
  HYPRE_BoomerAMGSetPrintLevel(precond_, get_optional(node, "print_level", 1));
  HYPRE_BoomerAMGSetDebugFlag(precond_, get_optional(node, "debug_flag", 1));
  HYPRE_BoomerAMGSetCoarsenType(precond_,
                                get_optional(node, "coarsen_type", 8));
  HYPRE_BoomerAMGSetCycleType(precond_, get_optional(node, "cycle_type", 1));

  if (node["down_relax_type"] && node["up_relax_type"] &&
      node["coarse_relax_type"]) {
    HYPRE_BoomerAMGSetCycleRelaxType(
        precond_, get_optional(node, "down_relax_type", 8), 1);
    HYPRE_BoomerAMGSetCycleRelaxType(precond_,
                                     get_optional(node, "up_relax_type", 8), 2);
    HYPRE_BoomerAMGSetCycleRelaxType(
        precond_, get_optional(node, "coarse_relax_type", 8), 3);
  } else {
    HYPRE_BoomerAMGSetRelaxType(precond_, get_optional(node, "relax_type", 8));
  }

  if (node["num_down_sweeps"] && node["num_up_sweeps"] &&
      node["num_coarse_sweeps"]) {
    HYPRE_BoomerAMGSetCycleNumSweeps(
        precond_, get_optional(node, "num_down_sweeps", 1), 1);
    HYPRE_BoomerAMGSetCycleNumSweeps(precond_,
                                     get_optional(node, "num_up_sweeps", 1), 2);
    HYPRE_BoomerAMGSetCycleNumSweeps(
        precond_, get_optional(node, "num_coarse_sweeps", 1), 3);
  } else {
    HYPRE_BoomerAMGSetNumSweeps(precond_, get_optional(node, "num_sweeps", 1));
  }
  HYPRE_BoomerAMGSetSmoothNumSweeps(precond_,
                                    get_optional(node, "smooth_num_sweeps", 1));
  HYPRE_BoomerAMGSetTol(precond_, get_optional(node, "tolerance", 0.0));
  HYPRE_BoomerAMGSetMaxIter(precond_, get_optional(node, "max_iterations", 1));
  HYPRE_BoomerAMGSetRelaxOrder(precond_, get_optional(node, "relax_order", 1));
  HYPRE_BoomerAMGSetMaxLevels(precond_, get_optional(node, "max_levels", 20));
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

  if (node["rap2"]) {
    int int_value = node["rap2"].as<int>();
    HYPRE_BoomerAMGSetRAP2(precond_, int_value);
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

  /* Smoothing parameters : used for complex algorithms such as ILU and
   * Chebyshev */
  if (node["smooth_type"]) {
    int smooth_type = node["smooth_type"].as<int>();
    HYPRE_BoomerAMGSetSmoothType(precond_, smooth_type);
  }
  if (node["smooth_num_sweeps"]) {
    int smooth_num_sweeps = node["smooth_num_sweeps"].as<int>();
    HYPRE_BoomerAMGSetSmoothNumSweeps(precond_, smooth_num_sweeps);
  }
  if (node["smooth_num_levels"]) {
    int smooth_num_levels = node["smooth_num_levels"].as<int>();
    HYPRE_BoomerAMGSetSmoothNumLevels(precond_, smooth_num_levels);
  }

  /* ILU parameters */
  if (node["ilu_type"]) {
    int ilu_type = node["ilu_type"].as<int>();
    HYPRE_BoomerAMGSetILUType(precond_, ilu_type);
  }
  if (node["ilu_level"]) {
    int ilu_level = node["ilu_level"].as<int>();
    HYPRE_BoomerAMGSetILULevel(precond_, ilu_level);
  }
  if (node["ilu_reordering_type"]) {
    int ilu_reordering_type = node["ilu_reordering_type"].as<int>();
    HYPRE_BoomerAMGSetILULocalReordering(precond_, ilu_reordering_type);
  }
  if (node["ilu_max_row_nnz"]) {
    int ilu_max_row_nnz = node["ilu_max_row_nnz"].as<int>();
    HYPRE_BoomerAMGSetILUMaxRowNnz(precond_, ilu_max_row_nnz);
  }
  if (node["ilu_max_iter"]) {
    int ilu_max_iter = node["ilu_max_iter"].as<int>();
    HYPRE_BoomerAMGSetILUMaxIter(precond_, ilu_max_iter);
  }
  if (node["ilu_drop_tol"]) {
    double ilu_drop_tol = node["ilu_drop_tol"].as<double>();
    HYPRE_BoomerAMGSetILUDroptol(precond_, ilu_drop_tol);
  }

  if (node["iterative_ilu_algorithm_type"]) {
  // 0 : Non-iterative algorithm (default)
  // 1 : Asynchronous with in-place storage
  // 2 : Asynchronous with explicit storage splitting
  // 3 : Synchronous with explicit storage splitting
  // 4 : Semi-synchronous with explicit storage splitting
  // Iterative ILU is available only for zero fill-in and it depends on rocSparse
  int iterative_ilu_algorithm_type = node["iterative_ilu_algorithm_type"].as<int>();
  HYPRE_BoomerAMGSetILUIterSetupType(precond_, iterative_ilu_algorithm_type);
  }

  if (node["iterative_ilu_setup_option"]) {
  int iterative_ilu_setup_option = node["iterative_ilu_setup_option"].as<int>();
  HYPRE_BoomerAMGSetILUIterSetupOption(precond_, iterative_ilu_setup_option);
  }

  if (node["iterative_ilu_max_iterations"]) {
  int max_iterative_ilu_iterations = node["iterative_ilu_max_iterations"].as<int>();
  HYPRE_BoomerAMGSetILUIterSetupMaxIter(
      precond_, get_optional(node, "iterative_ilu_max_iterations", 1));
  }

  if (node["iterative_ilu_tolerance"]) {
  double iterative_ilu_tolerance = node["iterative_ilu_tolerance"].as<double>();
  HYPRE_BoomerAMGSetILUIterSetupTolerance(
      precond_, iterative_ilu_tolerance);
  }

  // 0: iterative
  // 1: direct (default)
  HYPRE_ILUSetTriSolve(precond_, get_optional(node, "trisolve", 1));


  if (node["ilu_tri_solve"]) {
    int ilu_tri_solve = node["ilu_tri_solve"].as<int>();
    HYPRE_BoomerAMGSetILUTriSolve(precond_, ilu_tri_solve);
    printf("%s %s %d\n", __FILE__, __FUNCTION__, __LINE__);
  }
  if (node["ilu_lower_jacobi_iters"]) {
    int ilu_lower_jacobi_iters = node["ilu_lower_jacobi_iters"].as<int>();
    HYPRE_BoomerAMGSetILULowerJacobiIters(precond_, ilu_lower_jacobi_iters);
  }
  if (node["ilu_upper_jacobi_iters"]) {
    int ilu_upper_jacobi_iters = node["ilu_upper_jacobi_iters"].as<int>();
    HYPRE_BoomerAMGSetILUUpperJacobiIters(precond_, ilu_upper_jacobi_iters);
  }

  precondSetupPtr_ = &HYPRE_BoomerAMGSetup;
  precondSolvePtr_ = &HYPRE_BoomerAMGSolve;
  precondDestroyPtr_ = &HYPRE_BoomerAMGDestroy;
}

void HypreSystem::setup_ilu_precond() {
  YAML::Node node = inpfile_["ilu_preconditioner_settings"];

  HYPRE_ILUCreate(&precond_);
  HYPRE_ILUSetType(precond_, get_optional(node, "ilu_type", 0));
  HYPRE_ILUSetMaxIter(precond_, get_optional(node, "max_iterations", 1));
  HYPRE_ILUSetTol(precond_, get_optional(node, "tolerance", 0.0));
  HYPRE_ILUSetLocalReordering(precond_, get_optional(node, "local_reordering", 0));
  HYPRE_ILUSetPrintLevel(precond_, get_optional(node, "print_level", 1));

  // ILUK parameters
  HYPRE_ILUSetLevelOfFill(precond_, get_optional(node, "fill", 0));

  // ILUT parameters
  HYPRE_ILUSetMaxNnzPerRow(precond_,get_optional(node, "max_nnz_per_row", 1000));
  HYPRE_ILUSetDropThreshold(precond_,get_optional(node, "drop_threshold", 1.0e-2));

  // 0 : Non-iterative algorithm (default)
  // 1 : Asynchronous with in-place storage
  // 2 : Asynchronous with explicit storage splitting
  // 3 : Synchronous with explicit storage splitting
  // 4 : Semi-synchronous with explicit storage splitting
  // Iterative ILU is available only for zero fill-in and it depends on rocSparse
  HYPRE_ILUSetIterativeSetupType(precond_,
                                 get_optional(node, "iterative_algorithm_type", 0));
  HYPRE_ILUSetIterativeSetupOption(precond_, get_optional(node, "iterative_setup_option", 2));
  HYPRE_ILUSetIterativeSetupMaxIter(
      precond_, get_optional(node, "iterative_ilu_max_iterations", 1));
  HYPRE_ILUSetIterativeSetupTolerance(
      precond_, get_optional(node, "iterative_ilu_tolerance", 1e-5));

  // 0: iterative
  // 1: direct (default)
  HYPRE_ILUSetTriSolve(precond_, get_optional(node, "trisolve", 1));

  // Jacobi iterations for lower and upper triangular solves
  HYPRE_ILUSetLowerJacobiIters(precond_,get_optional(node, "lower_jacobi_iters", 5));
  HYPRE_ILUSetUpperJacobiIters(precond_,get_optional(node, "upper_jacobi_iters", 5));

  precondSetupPtr_ = &HYPRE_ILUSetup;
  precondSolvePtr_ = &HYPRE_ILUSolve;
  precondDestroyPtr_ = &HYPRE_ILUDestroy;
}

void HypreSystem::setup_cogmres() {
  YAML::Node node = inpfile_["solver_settings"];

  HYPRE_ParCSRCOGMRESCreate(comm_, &solver_);
  HYPRE_ParCSRCOGMRESSetTol(solver_, get_optional(node, "tolerance", 1.0e-5));
  HYPRE_ParCSRCOGMRESSetMaxIter(solver_,
                                get_optional(node, "max_iterations", 1000));
  HYPRE_ParCSRCOGMRESSetKDim(solver_, get_optional(node, "kspace", 10));
  HYPRE_ParCSRCOGMRESSetPrintLevel(solver_,
                                   get_optional(node, "print_level", 4));
  HYPRE_ParCSRCOGMRESSetCGS(solver_, get_optional(node, "cgs", 0));

  solverDestroyPtr_ = &HYPRE_ParCSRCOGMRESDestroy;
  solverSetupPtr_ = &HYPRE_ParCSRCOGMRESSetup;
  solverPrecondPtr_ = &HYPRE_ParCSRCOGMRESSetPrecond;
  solverSolvePtr_ = &HYPRE_ParCSRCOGMRESSolve;
}

void HypreSystem::setup_gmres() {
  YAML::Node node = inpfile_["solver_settings"];
  HYPRE_ParCSRGMRESCreate(comm_, &solver_);
  HYPRE_ParCSRGMRESSetTol(solver_, get_optional(node, "tolerance", 1.0e-5));
  HYPRE_ParCSRGMRESSetMaxIter(solver_,
                              get_optional(node, "max_iterations", 1000));
  HYPRE_ParCSRGMRESSetKDim(solver_, get_optional(node, "kspace", 10));
  HYPRE_ParCSRGMRESSetPrintLevel(solver_, get_optional(node, "print_level", 4));
  // HYPRE_ParCSRGMRESSetCGS(solver_, get_optional(node, "cgs", 0));

  solverDestroyPtr_ = &HYPRE_ParCSRGMRESDestroy;
  solverSetupPtr_ = &HYPRE_ParCSRGMRESSetup;
  solverPrecondPtr_ = &HYPRE_ParCSRGMRESSetPrecond;
  solverSolvePtr_ = &HYPRE_ParCSRGMRESSolve;
}

void HypreSystem::setup_fgmres() {
  YAML::Node node = inpfile_["solver_settings"];

  HYPRE_ParCSRFlexGMRESCreate(comm_, &solver_);
  HYPRE_ParCSRFlexGMRESSetTol(solver_, get_optional(node, "tolerance", 1.0e-5));
  HYPRE_ParCSRFlexGMRESSetMaxIter(solver_,
                                  get_optional(node, "max_iterations", 1000));
  HYPRE_ParCSRFlexGMRESSetKDim(solver_, get_optional(node, "kspace", 10));
  HYPRE_ParCSRFlexGMRESSetPrintLevel(solver_,
                                     get_optional(node, "print_level", 4));

  solverDestroyPtr_ = &HYPRE_ParCSRFlexGMRESDestroy;
  solverSetupPtr_ = &HYPRE_ParCSRFlexGMRESSetup;
  solverPrecondPtr_ = &HYPRE_ParCSRFlexGMRESSetPrecond;
  solverSolvePtr_ = &HYPRE_ParCSRFlexGMRESSolve;
}

void HypreSystem::setup_bicg() {
  YAML::Node node = inpfile_["solver_settings"];

  HYPRE_ParCSRBiCGSTABCreate(comm_, &solver_);
  HYPRE_ParCSRBiCGSTABSetTol(solver_, get_optional(node, "tolerance", 1.0e-5));
  HYPRE_ParCSRBiCGSTABSetMaxIter(solver_,
                                 get_optional(node, "max_iterations", 1000));
  // HYPRE_ParCSRBiCGSTABSetKDim(solver_, get_optional(node, "kspace", 10));
  HYPRE_ParCSRBiCGSTABSetPrintLevel(solver_,
                                    get_optional(node, "print_level", 4));

  solverDestroyPtr_ = &HYPRE_ParCSRBiCGSTABDestroy;
  solverSetupPtr_ = &HYPRE_ParCSRBiCGSTABSetup;
  solverPrecondPtr_ = &HYPRE_ParCSRBiCGSTABSetPrecond;
  solverSolvePtr_ = &HYPRE_ParCSRBiCGSTABSolve;
}

void HypreSystem::setup_cg() {
  YAML::Node node = inpfile_["solver_settings"];

  HYPRE_ParCSRPCGCreate(comm_, &solver_);
  HYPRE_ParCSRPCGSetTol(solver_, get_optional(node, "tolerance", 1.0e-5));
  HYPRE_ParCSRPCGSetMaxIter(solver_,
                                 get_optional(node, "max_iterations", 1000));
  // HYPRE_ParCSRPCGSetKDim(solver_, get_optional(node, "kspace", 10));
  HYPRE_ParCSRPCGSetPrintLevel(solver_,
                                    get_optional(node, "print_level", 4));

  solverDestroyPtr_ = &HYPRE_ParCSRPCGDestroy;
  solverSetupPtr_ = &HYPRE_ParCSRPCGSetup;
  solverPrecondPtr_ = &HYPRE_ParCSRPCGSetPrecond;
  solverSolvePtr_ = &HYPRE_ParCSRPCGSolve;
}

void HypreSystem::setup_ilu() {
  YAML::Node node = inpfile_["solver_settings"];
  HYPRE_ILUCreate(&solver_);
  HYPRE_ILUSetType(solver_, get_optional(node, "ilu_type", 0));
  HYPRE_ILUSetMaxIter(solver_, get_optional(node, "max_iterations", 20));
  HYPRE_ILUSetTol(solver_, get_optional(node, "tolerance", 0.0));
  HYPRE_ILUSetLocalReordering(solver_, get_optional(node,"local_reordering", 0));
  HYPRE_ILUSetPrintLevel(solver_, get_optional(node, "print_level", 4));

  // ILUK parameters
  HYPRE_ILUSetLevelOfFill(solver_, get_optional(node, "fill", 0));

  // ILUT parameters
  HYPRE_ILUSetMaxNnzPerRow(solver_,get_optional(node, "max_nnz_per_row", 1000));
  HYPRE_ILUSetDropThreshold(solver_,get_optional(node, "drop_threshold", 1.0e-2));

  // 0 : Non-iterative algorithm (default)
  // 1 : Asynchronous with in-place storage
  // 2 : Asynchronous with explicit storage splitting
  // 3 : Synchronous with explicit storage splitting
  // 4 : Semi-synchronous with explicit storage splitting
  // Iterative ILU is available only for zero fill-in and it depends on rocSparse
  HYPRE_ILUSetIterativeSetupType(solver_,
                                 get_optional(node, "iterative_algorithm_type", 0));
  HYPRE_ILUSetIterativeSetupOption(solver_, get_optional(node, "iterative_setup_option", 2));
  HYPRE_ILUSetIterativeSetupMaxIter(
      solver_, get_optional(node, "iterative_ilu_max_iterations", 1));
  HYPRE_ILUSetIterativeSetupTolerance(
      solver_, get_optional(node, "iterative_ilu_tolerance", 1e-5));
  // 0: iterative
  // 1: direct (default)
  HYPRE_ILUSetTriSolve(solver_, get_optional(node, "trisolve", 1));
  // Jacobi iterations for lower and upper triangular solves
  HYPRE_ILUSetLowerJacobiIters(solver_, get_optional(node, "lower_jacobi_iters", 5));
  HYPRE_ILUSetUpperJacobiIters(solver_, get_optional(node, "upper_jacobi_iters", 5));

  solverDestroyPtr_ = &HYPRE_ILUDestroy;
  solverSetupPtr_ = &HYPRE_ILUSetup;
  solverPrecondPtr_ = nullptr;
  solverSolvePtr_ = &HYPRE_ILUSolve;
}

void HypreSystem::destroy_system() {
  if (mat_)
    HYPRE_IJMatrixDestroy(mat_);
  for (int i = 0; i < rhs_.size(); ++i)
    if (rhs_[i])
      HYPRE_IJVectorDestroy(rhs_[i]);
  for (int i = 0; i < sln_.size(); ++i)
    if (sln_[i])
      HYPRE_IJVectorDestroy(sln_[i]);
  for (int i = 0; i < slnRef_.size(); ++i)
    if (slnRef_[i])
      HYPRE_IJVectorDestroy(slnRef_[i]);
  if (solver_)
    solverDestroyPtr_(solver_);
  if (precond_)
    precondDestroyPtr_(precond_);

  if (d_vector_indices_) hypre_TFree(d_vector_indices_, HYPRE_MEMORY_DEVICE);
  if (d_vector_vals_)    hypre_TFree(d_vector_vals_, HYPRE_MEMORY_DEVICE);
  if (d_rows_)           hypre_TFree(d_rows_, HYPRE_MEMORY_DEVICE);
  if (d_cols_)           hypre_TFree(d_cols_, HYPRE_MEMORY_DEVICE);
  if (d_offd_rows_)      hypre_TFree(d_offd_rows_, HYPRE_MEMORY_DEVICE);
  if (d_offd_cols_)      hypre_TFree(d_offd_cols_, HYPRE_MEMORY_DEVICE);
  if (d_vals_)           hypre_TFree(d_vals_, HYPRE_MEMORY_DEVICE);
}

void HypreSystem::init_row_decomposition() {
  if (iproc_ == 0)
    printf("\tComputing row decomposition\n");

  HYPRE_Int rowsPerProc = totalRows_ / nproc_;
  HYPRE_Int remainder = totalRows_ % nproc_;

  iLower_ = rowsPerProc * iproc_ + std::min<HYPRE_Int>(iproc_, remainder);
  iUpper_ = rowsPerProc * (iproc_ + 1) +
            std::min<HYPRE_Int>(iproc_ + 1, remainder) - 1;
  numRows_ = iUpper_ - iLower_ + 1;

  MPI_Barrier(comm_);
  std::cout << "\tRank: " << std::setw(4) << iproc_
            << " :: iLower = " << std::setw(9) << iLower_
            << "; iUpper = " << std::setw(9) << iUpper_
            << "; numRows = " << numRows_ << std::endl;
  MPI_Barrier(comm_);
  fflush(stdout);
}

void HypreSystem::init_system() {
  MPI_Barrier(comm_);
  auto start = std::chrono::system_clock::now();
  if (iproc_ == 0)
    printf("\tInitializing HYPRE data structures\n");

  HYPRE_IJMatrixCreate(comm_, iLower_, iUpper_, iLower_, iUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void **)&parMat_);
  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);

  rhs_.resize(numSolves_);
  parRhs_.resize(numSolves_);
  sln_.resize(numSolves_);
  parSln_.resize(numSolves_);
  if (checkSolution_) {
    slnRef_.resize(numSolves_);
    parSlnRef_.resize(numSolves_);
  }
  for (int i = 0; i < numSolves_; ++i) {
    HYPRE_IJVectorCreate(comm_, iLower_, iUpper_, &rhs_[i]);
    HYPRE_IJVectorSetObjectType(rhs_[i], HYPRE_PARCSR);
    HYPRE_IJVectorSetNumComponents(rhs_[i], numVectors_);
    HYPRE_IJVectorInitialize(rhs_[i]);
    HYPRE_IJVectorGetObject(rhs_[i], (void **)&parRhs_[i]);

    HYPRE_IJVectorCreate(comm_, iLower_, iUpper_, &sln_[i]);
    HYPRE_IJVectorSetObjectType(sln_[i], HYPRE_PARCSR);
    HYPRE_IJVectorSetNumComponents(sln_[i], numVectors_);
    HYPRE_IJVectorInitialize(sln_[i]);
    HYPRE_IJVectorGetObject(sln_[i], (void **)&parSln_[i]);

    HYPRE_ParVectorSetConstantValues(parRhs_[i], 0.0);
    HYPRE_ParVectorSetConstantValues(parSln_[i], 0.0);

    if (checkSolution_) {
      HYPRE_IJVectorCreate(comm_, iLower_, iUpper_, &slnRef_[i]);
      HYPRE_IJVectorSetObjectType(slnRef_[i], HYPRE_PARCSR);
      HYPRE_IJVectorSetNumComponents(slnRef_[i], numVectors_);
      HYPRE_IJVectorInitialize(slnRef_[i]);
      HYPRE_IJVectorGetObject(slnRef_[i], (void **)&parSlnRef_[i]);
      HYPRE_ParVectorSetConstantValues(parSlnRef_[i], 0.0);
    }
  }

  MPI_Barrier(comm_);
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  timers_.emplace_back("Initialize system", elapsed.count());
  MPI_Barrier(comm_);
  fflush(stdout);
}

void HypreSystem::assemble_system() {
  auto start = std::chrono::system_clock::now();
  if (iproc_ == 0)
    printf("Assembling HYPRE data structures\n");

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void **)&parMat_);

  for (int i = 0; i < numSolves_; ++i) {
    HYPRE_IJVectorAssemble(rhs_[i]);
    HYPRE_IJVectorAssemble(sln_[i]);
    HYPRE_IJVectorGetObject(rhs_[i], (void **)&(parRhs_[i]));
    HYPRE_IJVectorGetObject(sln_[i], (void **)&(parSln_[i]));

    if (checkSolution_) {
      HYPRE_IJVectorAssemble(slnRef_[i]);
      HYPRE_IJVectorGetObject(slnRef_[i], (void **)&(parSlnRef_[i]));
    }
  }

  MPI_Barrier(comm_);
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  timers_.emplace_back("Assemble system", elapsed.count());
  MPI_Barrier(comm_);

  /* delete unneeded memory */
  if (d_vector_indices_) hypre_TFree(d_vector_indices_, HYPRE_MEMORY_DEVICE);
  if (d_vector_vals_)    hypre_TFree(d_vector_vals_, HYPRE_MEMORY_DEVICE);
  if (d_rows_)           hypre_TFree(d_rows_, HYPRE_MEMORY_DEVICE);
  if (d_cols_)           hypre_TFree(d_cols_, HYPRE_MEMORY_DEVICE);
  if (d_offd_rows_)      hypre_TFree(d_offd_rows_, HYPRE_MEMORY_DEVICE);
  if (d_offd_cols_)      hypre_TFree(d_offd_cols_, HYPRE_MEMORY_DEVICE);
  if (d_vals_)           hypre_TFree(d_vals_, HYPRE_MEMORY_DEVICE);

  checkMemory();
}

void HypreSystem::checkMemory() {
#ifdef HYPRE_USING_CUDA
  int count;
  cudaGetDeviceCount(&count);
  int device;
  cudaGetDevice(&device);
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  if (iproc_ == 0)
     printf("\trank=%d : %s %s %d : %s (cc=%d.%d): device=%d of %d : free "
            "memory=%1.8g GB, total memory=%1.8g GB\n",
            iproc_, __FUNCTION__, __FILE__, __LINE__, prop.name, prop.major,
            prop.minor, device, count, free / 1.e9, total / 1.e9);
#endif

#ifdef HYPRE_USING_HIP
   int count;
   hipGetDeviceCount(&count);
   int device;
   hipGetDevice(&device);
   size_t free, total;
   hipMemGetInfo(&free, &total);
   hipDeviceProp_t prop;
   hipGetDeviceProperties(&prop, device);
   //if (iproc_ == 0)
   printf("rank=%d : %s %s %d : %s arch=%s : device=%d of %d : free "
          "memory=%1.8g GB, total memory=%1.8g GB\n",
          iproc_, __FUNCTION__, __FILE__, __LINE__, prop.name, prop.gcnArchName,
          device, count, free / 1.e9, total / 1.e9);
   fflush(stdout);
#endif
}

void HypreSystem::solve() {
  assemble_system();
  std::chrono::duration<double> setup(0);
  std::chrono::duration<double> write_operators(0);
  std::chrono::duration<double> solve(0);

  //hypre_CSRMatrixGpuSpMVAnalysis(hypre_ParCSRMatrixDiag(parMat_));

  for (int i = 0; i < numSolves_; ++i) {
    if (iproc_ == 0)
      printf("Setting up preconditioner\n");

    auto start = std::chrono::system_clock::now();
    if (usePrecond_) {
      solverPrecondPtr_(solver_, precondSolvePtr_, precondSetupPtr_, precond_);
    }
    checkMemory();
    if (iproc_ == 0)
      printf("Setting up solver\n");
    solverSetupPtr_(solver_, parMat_, parRhs_[i], parSln_[i]);
    checkMemory();
    MPI_Barrier(comm_);
    auto stop1 = std::chrono::system_clock::now();
    setup += stop1 - start;
    MPI_Barrier(comm_);
    fflush(stdout);

    /* extract AMG matrices */
    if (writeAmgMatrices_) {
      YAML::Node linsys = inpfile_["linear_system"];
      std::string matfile = linsys["matrix_file"].as<std::string>();
      std::size_t pos = matfile.rfind(".");
      hypre_ParAMGData *amg_data = (hypre_ParAMGData *)precond_;
      hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
      int num_levels = hypre_ParAMGDataNumLevels(amg_data);
      for (int i = 0; i < num_levels; ++i) {
        char fname[1024];
        sprintf(fname, "%s_level_%d.IJ", matfile.substr(0, pos).c_str(), i);
        hypre_ParCSRMatrixPrintIJ(A_array[i], 0, 0, fname);
      }
      MPI_Barrier(comm_);
    }
    auto stop2 = std::chrono::system_clock::now();
    write_operators += stop2 - stop1;
    MPI_Barrier(comm_);
    fflush(stdout);

    if (iproc_ == 0)
      printf("Solving the system\n");

    solverSolvePtr_(solver_, parMat_, parRhs_[i], parSln_[i]);
    MPI_Barrier(comm_);
    auto stop3 = std::chrono::system_clock::now();
    solve += stop3 - stop2;
    MPI_Barrier(comm_);
    fflush(stdout);
  }

  timers_.emplace_back("Preconditioner setup", setup.count());
  if (writeAmgMatrices_)
     timers_.emplace_back("Write AMG Matrices", write_operators.count());
  timers_.emplace_back("Solve", solve.count());

  solveComplete_ = true;
}

void HypreSystem::output_linear_system() {
  if (!outputSystem_ and !outputSolution_)
    return;

  auto start = std::chrono::system_clock::now();

  if (outputSystem_) {
    HYPRE_IJMatrixPrint(mat_, "IJM.mat");
    for (int i = 0; i < numSolves_; ++i) {
      std::string r = "IJV" + std::to_string(i) + ".rhs";
      HYPRE_IJVectorPrint(rhs_[i], r.c_str());
      std::string s = "IJV" + std::to_string(i) + ".sln";
      HYPRE_IJVectorPrint(sln_[i], s.c_str());
    }
  }

  if (outputSolution_) {
    for (int i = 0; i < numSolves_; ++i) {
      for (int j = 0; j < numVectors_; ++j) {
        HYPRE_IJVectorSetComponent(sln_[i], j);
        std::string s = "IJV" + std::to_string(std::max(i, j)) + ".sln";
        HYPRE_IJVectorPrint(sln_[i], s.c_str());
      }
    }
  }
  MPI_Barrier(comm_);
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;

  timers_.emplace_back("Output system", elapsed.count());
}

void HypreSystem::check_solution() {
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
  double diff;
  double maxrerr = std::numeric_limits<double>::lowest();
  double maxaerr = std::numeric_limits<double>::lowest();
  double avgrerr = 0;
  double avgaerr = 0;

  HYPRE_Complex *dsln, *dslnRef, *hsln, *hslnRef;
  HYPRE_Int n = iUpper_ - iLower_ + 1;

  dsln = hypre_TAlloc(HYPRE_Complex, n, HYPRE_MEMORY_DEVICE);
  dslnRef = hypre_TAlloc(HYPRE_Complex, n, HYPRE_MEMORY_DEVICE);

  hsln = hypre_TAlloc(HYPRE_Complex, n, HYPRE_MEMORY_HOST);
  hslnRef = hypre_TAlloc(HYPRE_Complex, n, HYPRE_MEMORY_HOST);

  for (int j = 0; j < numSolves_; ++j) {
    for (HYPRE_Int i = 0; i < numVectors_; i++) {
      HYPRE_IJVectorSetComponent(sln_[j], i);
      HYPRE_IJVectorSetComponent(slnRef_[j], i);

      HYPRE_IJVectorGetValues(sln_[j], n, NULL, dsln);
      HYPRE_IJVectorGetValues(slnRef_[j], n, NULL, dslnRef);

      hypre_TMemcpy(hsln, dsln, HYPRE_Complex, n, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(hslnRef, dslnRef, HYPRE_Complex, n, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);

      HYPRE_Int printCount = 0;
      bool isClose = true;
      for (HYPRE_Int k = 0; k < n; k++) {
        diff = std::fabs(hsln[k] - hslnRef[k]);
        double rhs = std::max(
            rtol_ * std::max(std::fabs(hsln[k]), std::fabs(hslnRef[k])), atol_);
        if (diff >= rhs) {
          isClose = false;
          if (printCount < 20 && iproc_ == 1) {
            std::cout << k + iLower_ << ":" << hsln[k] << " " << hslnRef[k]
                      << " " << diff << " " << rhs << std::endl;
            printCount++;
          }
        }
      }
      HYPRE_Int testAll = 1;
      HYPRE_Int test = isClose ? 1 : 0;
      MPI_Reduce(&testAll, &test, 1, MPI_INT, MPI_MIN, 0, comm_);
      if (iproc_ == 0 and !testAll)
        std::cout << "Solve " << j << " comp " << i << " atol=" << atol_
                  << " rtol=" << rtol_ << " allClose=" << testAll << std::endl;
    }
  }
  hypre_TFree(dsln, HYPRE_MEMORY_DEVICE);
  hypre_TFree(dslnRef, HYPRE_MEMORY_DEVICE);
  hypre_TFree(hsln, HYPRE_MEMORY_HOST);
  hypre_TFree(hslnRef, HYPRE_MEMORY_HOST);

  MPI_Barrier(comm_);
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;

  timers_.emplace_back("Check solution", elapsed.count());
}

void HypreSystem::retrieve_timers(std::vector<std::string>& names,
                                  std::vector<std::vector<double>> & data) {
  if (iproc_ != 0)
    return;

  if (names.size()==0)
  {
     for (auto &timer : timers_)
        names.push_back(std::string(timer.first));
     data.resize(names.size());
     int k=0;
     for (auto &timer : timers_)
     {
        data[k].push_back(double(timer.second));
        k++;
     }
  }
  else
  {
     for (auto &timer : timers_)
     {
        auto it = std::find(names.begin(), names.end(), timer.first);

        // If element was found
        if (it != names.end())
        {
           int k = it - names.begin();
           data[k].push_back(double(timer.second));
        }
     }
  }
}

void HypreSystem::summarize_timers() {
  if (iproc_ != 0)
    return;

  std::cout << "\nTimer summary: " << std::endl;
  for (auto &timer : timers_) {
    std::cout << "    " << std::setw(25) << std::left << timer.first
              << timer.second << " seconds" << std::endl;
  }
}

/********************************************************************************/
/* Generic methods for building matrices/vectors for CUDA/CPU/... useable from
 */
/* IJ or MM matrix formats */
/********************************************************************************/

void HypreSystem::hypre_matrix_set_values() {
  if (iproc_ == 0)
     printf("%s : loading matrix into HYPRE_IJMatrix\n", __FUNCTION__);

  auto start = std::chrono::system_clock::now();

  int nnz_this_rank = rows_.size();

#if defined(HYPRE_USING_GPU)
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
  d_rows_ = hypre_TAlloc(HYPRE_BigInt, nnz_this_rank, HYPRE_MEMORY_DEVICE);
  hypre_TMemcpy(d_rows_, rows_.data(), HYPRE_BigInt, nnz_this_rank,
                HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

  d_cols_ = hypre_TAlloc(HYPRE_BigInt, nnz_this_rank, HYPRE_MEMORY_DEVICE);
  hypre_TMemcpy(d_cols_, cols_.data(), HYPRE_BigInt, nnz_this_rank,
                HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
#else
  d_rows_ = hypre_TAlloc(HYPRE_Int, nnz_this_rank, HYPRE_MEMORY_DEVICE);
  hypre_TMemcpy(d_rows_, rows_.data(), HYPRE_Int, nnz_this_rank,
                HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

  d_cols_ = hypre_TAlloc(HYPRE_Int, nnz_this_rank, HYPRE_MEMORY_DEVICE);
  hypre_TMemcpy(d_cols_, cols_.data(), HYPRE_Int, nnz_this_rank,
                HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
#endif

  d_vals_ = hypre_TAlloc(HYPRE_Complex, nnz_this_rank, HYPRE_MEMORY_DEVICE);
  hypre_TMemcpy(d_vals_, vals_.data(), HYPRE_Complex, nnz_this_rank,
                HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

  /* Use the fast path */
  //if (iproc_ == 0)
     printf("rank=%d : %s %s %d : nnz_this_rank=%d\n", iproc_, __FILE__,
            __FUNCTION__, __LINE__, nnz_this_rank);
  YAML::Node node = inpfile_["solver_settings"];
  if (get_optional(node, "fast_matrix_assemble", 0)) {
#if 0
    HYPRE_IJMatrixSetMaxOnProcElmts(mat_, nnz_this_rank);
    HYPRE_IJMatrixSetOffProcSendElmts(mat_, 0);
    HYPRE_IJMatrixSetOffProcRecvElmts(mat_, 0);
#endif
  }

  /* Call this on UVM data */
  HYPRE_IJMatrixSetValues2(mat_, nnz_this_rank, NULL, d_rows_, NULL, d_cols_,
                           d_vals_);
#else
  HYPRE_IJMatrixSetValues2(mat_, nnz_this_rank, NULL, rows_.data(), NULL,
                           cols_.data(), vals_.data());
#endif

  MPI_Barrier(comm_);
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  timers_.emplace_back("Build HYPRE matrix", elapsed.count());
  MPI_Barrier(comm_);
  fflush(stdout);
}

void HypreSystem::hypre_vector_set_values(std::vector<HYPRE_IJVector> &vec,
                                          int component) {
  if (iproc_ == 0)
     printf("%s : loading vector into HYPRE_IJVector\n", __FUNCTION__);

  auto start = std::chrono::system_clock::now();

  HYPRE_IJVector v;
  if (numSolves_ == 1) {
    v = vec[0];
    HYPRE_IJVectorSetComponent(v, component);
  } else {
    v = vec[component];
    HYPRE_IJVectorSetComponent(v, 0);
  }

#if defined(HYPRE_USING_GPU)
  size_t N = vector_values_.size();

#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
  d_vector_indices_ = hypre_TAlloc(HYPRE_BigInt, N, HYPRE_MEMORY_DEVICE);
  hypre_TMemcpy(d_vector_indices_, vector_indices_.data(), HYPRE_BigInt, N,
                HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
#else
  d_vector_indices_ = hypre_TAlloc(HYPRE_Int, N, HYPRE_MEMORY_DEVICE);
  hypre_TMemcpy(d_vector_indices_, vector_indices_.data(), HYPRE_Int, N,
                HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
#endif
  d_vector_vals_ = hypre_TAlloc(HYPRE_Complex, N, HYPRE_MEMORY_DEVICE);
  hypre_TMemcpy(d_vector_vals_, vector_values_.data(), HYPRE_Complex, N,
                HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

  /* Use the fast path. This probably doesn't work with multivectors yet */
  if (iproc_ == 0)
     printf("rank=%d : %s %s %d : N=%d\n", iproc_, __FILE__, __FUNCTION__,
            __LINE__, N);
  YAML::Node node = inpfile_["solver_settings"];
  if (get_optional(node, "fast_vector_assemble", 0)) {
#if 0
    HYPRE_IJVectorSetMaxOnProcElmts(v, N);
    HYPRE_IJVectorSetOffProcSendElmts(v, 0);
    HYPRE_IJVectorSetOffProcRecvElmts(v, 0);
#endif
  }

  HYPRE_IJVectorSetValues(v, iUpper_ - iLower_ + 1, d_vector_indices_,
                          d_vector_vals_);
#else
  HYPRE_IJVectorSetValues(v, iUpper_ - iLower_ + 1, vector_indices_.data(),
                          vector_values_.data());
#endif

  MPI_Barrier(comm_);
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  timers_.emplace_back("Build HYPRE vector", elapsed.count());
  MPI_Barrier(comm_);
  fflush(stdout);
}

/********************************************************************************/
/*                               HYPRE IJ Format */
/********************************************************************************/

void HypreSystem::load_hypre_format() {
  YAML::Node linsys = inpfile_["linear_system"];
  int nfiles = get_optional(linsys, "num_partitions", nproc_);
  bool useGPU = false;

#if defined(HYPRE_USING_GPU)
  // This is hack to prevent the cuda build from using the native load procedure
  useGPU = true;
#endif
  if (nfiles == nproc_ && !useGPU)
    load_hypre_native();
  else {
    numComps_ = get_optional(linsys, "num_components", 1);
    segregatedSolve_ = (bool)get_optional(linsys, "segregated_solve", 1);
    numSolves_ = segregatedSolve_ ? numComps_ : 1;
    numVectors_ = segregatedSolve_ ? 1 : numComps_;
    rtol_ = (double)get_optional(linsys, "rtol", 1.0e-6);
    atol_ = (double)get_optional(linsys, "atol", 1.0e-8);

    std::string matfile = linsys["matrix_file"].as<std::string>();

    std::vector<std::string> rhsfile(numComps_);
    std::vector<std::string> slnfile(numComps_);
    if (numComps_ == 1 && linsys["rhs_file"]) {
      rhsfile[0] = linsys["rhs_file"].as<std::string>();
      if (linsys["sln_file"]) {
        slnfile[0] = linsys["sln_file"].as<std::string>();
        checkSolution_ = true;
      }
    } else {
      int count = 0;
      for (int i = 0; i < numComps_; ++i) {
        rhsfile[i] = linsys["rhs_file" + std::to_string(i)].as<std::string>();
        std::string sfile = "sln_file" + std::to_string(i);
        if (linsys[sfile]) {
          slnfile[i] = linsys[sfile].as<std::string>();
          count++;
        }
      }
      if (count == numComps_)
        checkSolution_ = true;
    }

    // Scan the matrix and determine the sizes
    determine_ij_system_sizes(matfile, nfiles);

    // generic method for IJ and MM
    init_row_decomposition();

    // Create HYPRE data structures
    init_system();

    // build matrices and vectors
    build_ij_matrix(matfile, nfiles);
    build_ij_vector(rhsfile, nfiles, rhs_);

    if (checkSolution_) {
      build_ij_vector(slnfile, nfiles, slnRef_);
    }
  }
}

/*******************
 *
 *******************/
void HypreSystem::load_hypre_native() {
#if 0
      auto start = std::chrono::system_clock::now();
      if (iproc_ == 0)
         std::cout << "Loading HYPRE IJ files" << std::endl;

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
      fflush(stdout);
#endif
}

/*******************
 *
 *******************/
void HypreSystem::determine_ij_system_sizes(std::string matfile, int nfiles) {
  auto start = std::chrono::system_clock::now();

  HYPRE_Int ilower, iupper, jlower, jupper;
  HYPRE_Int imin = 0;
  HYPRE_Int imax = 0;
  HYPRE_Int gmin = 0;
  HYPRE_Int gmax = 0;

  for (int ii = iproc_; ii < nfiles; ii += nproc_) {
    FILE *fh;
    std::ostringstream suffix;
    suffix << matfile << "." << std::setw(5) << std::setfill('0') << ii;

    if ((fh = fopen(suffix.str().c_str(), "r")) == NULL) {
      throw std::runtime_error("Cannot open matrix file: " + suffix.str());
    }

#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
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
  timers_.emplace_back("IJ : determine system size", elapsed.count());
  MPI_Barrier(comm_);
  fflush(stdout);
}

/*******************
 *
 *******************/
void HypreSystem::build_ij_matrix(std::string matfile, int nfiles) {
  MPI_Barrier(comm_);
  auto start = std::chrono::system_clock::now();

  // read the files
  if (iproc_ == 0)
     printf("%s : Reading %d HYPRE IJ Matrix files\n", __FUNCTION__, nfiles);

  HYPRE_Int ilower, iupper, jlower, jupper;
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
  HYPRE_BigInt irow, icol;
#else
  HYPRE_Int irow, icol;
#endif
  double value;

  /* store the loaded matrix into these vectors */
  rows_.resize(0);
  cols_.resize(0);
  vals_.resize(0);

  // Need to loop over all the files
  for (int ii = 0; ii < nfiles; ii++) {
    FILE *fh;
    std::ostringstream suffix;
    suffix << matfile << "." << std::setw(5) << std::setfill('0') << ii;

    if ((fh = fopen(suffix.str().c_str(), "r")) == NULL) {
      throw std::runtime_error("Cannot open matrix file: " + suffix.str());
    }

#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
    fscanf(fh, "%lld %lld %lld %lld\n", &ilower, &iupper, &jlower, &jupper);
#else
    fscanf(fh, "%d %d %d %d\n", &ilower, &iupper, &jlower, &jupper);
#endif

    // need the + 1 so that the upper boundary are inclusive
    int overlap = std::max(0, std::min(iUpper_ + 1, iupper + 1) -
                                  std::max(iLower_, ilower));
    if (overlap) {
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
      while (fscanf(fh, "%lld %lld%*[ \t]%le\n", &irow, &icol, &value) != EOF)
#else
      while (fscanf(fh, "%d %d%*[ \t]%le\n", &irow, &icol, &value) != EOF)
#endif
      {
        if (irow >= iLower_ && irow <= iUpper_) {
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
  MPI_Barrier(comm_);
  fflush(stdout);
}

/*******************
 *
 *******************/
void HypreSystem::build_ij_vector(std::vector<std::string> &vecfiles,
                                  int nfiles,
                                  std::vector<HYPRE_IJVector> &vec) {
  MPI_Barrier(comm_);
  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < numComps_; ++i) {
    std::string vecfile = vecfiles[i];

    if (iproc_ == 0)
       printf("%s : Reading %d HYPRE IJ Vector files %s\n", __FUNCTION__, nfiles, vecfile.c_str());

    HYPRE_Int ilower, iupper;
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
    HYPRE_BigInt irow;
#else
    HYPRE_Int irow;
#endif
    double value;

    /* resize these */
    vector_indices_.resize(0);
    vector_values_.resize(0);

    for (int ii = 0; ii < nfiles; ii++) {
      FILE *fh;
      std::ostringstream suffix;
      suffix << vecfile << "." << std::setw(5) << std::setfill('0') << ii;

      if ((fh = fopen(suffix.str().c_str(), "r")) == NULL) {
        throw std::runtime_error("Cannot open vector file: " + suffix.str());
      }
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
      fscanf(fh, "%lld %lld\n", &ilower, &iupper);
#else
      fscanf(fh, "%d %d\n", &ilower, &iupper);
#endif

      // need the + 1 so that the upper boundary are inclusive
      int overlap = std::max(0, std::min(iUpper_ + 1, iupper + 1) -
                                    std::max(iLower_, ilower));
      if (overlap) {
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
        while (fscanf(fh, "%lld%*[ \t]%le\n", &irow, &value) != EOF)
#else
        while (fscanf(fh, "%d%*[ \t]%le\n", &irow, &value) != EOF)
#endif
        {
          if (irow >= iLower_ && irow <= iUpper_) {
            vector_indices_.push_back(irow);
            vector_values_.push_back(value);
          }
        }
      }
      fclose(fh);
    }
    /* Build the vector */
    hypre_vector_set_values(vec, i);
  }

  MPI_Barrier(comm_);
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  timers_.emplace_back("IJ : read and build vector", elapsed.count());
  MPI_Barrier(comm_);
  fflush(stdout);
}

/********************************************************************************/
/*                         Build 27 Pt Stencil                                  */
/********************************************************************************/
#if defined(HYPRE_USING_HIP)
GPU_GLOBAL void
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
fillGlobalRowIndices(HYPRE_BigInt n, HYPRE_BigInt iLower, int * row_ptr, HYPRE_BigInt * global_row_inds)
#else
fillGlobalRowIndices(HYPRE_Int n, HYPRE_Int iLower, int * row_ptr, HYPRE_Int * global_row_inds)
#endif
{
   if (blockIdx.x<n)
   {
      int row_start = row_ptr[blockIdx.x];
      int row_end   = row_ptr[blockIdx.x+1];
      for (int i=threadIdx.x; i<row_end-row_start; i+=blockDim.x) {
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
         global_row_inds[row_start+i] = (HYPRE_BigInt)(blockIdx.x+iLower);
#else
         global_row_inds[row_start+i] = (HYPRE_Int)(blockIdx.x+iLower);
#endif
      }
   }
   return;
}
#endif

#if defined(HYPRE_USING_HIP)
GPU_GLOBAL void
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
fillGlobalColIndices(HYPRE_BigInt nnz, HYPRE_BigInt shift, int * col_inds, HYPRE_BigInt * global_col_inds)
#else
fillGlobalColIndices(HYPRE_Int nnz, HYPRE_Int shift, int * col_inds, HYPRE_Int * global_col_inds)
#endif
{
   for (int tid = blockIdx.x*blockDim.x + threadIdx.x; tid<nnz; tid+=blockDim.x*gridDim.x)
   {
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
      global_col_inds[tid] = (HYPRE_BigInt) (col_inds[tid]+shift);
#else
      global_col_inds[tid] = (HYPRE_Int) (col_inds[tid]+shift);
#endif
   }
   return;
}
#endif

#if (HYPRE_USING_HIP)
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
void HypreSystem::validateDiagData(HYPRE_BigInt nnz, HYPRE_BigInt *drows, HYPRE_BigInt *dcols)
{
   std::vector<HYPRE_BigInt> hrows(nnz);
   std::vector<HYPRE_BigInt> hcols(nnz);
   HIP_CALL(hipMemcpy(hrows.data(), drows, nnz*sizeof(HYPRE_BigInt), hipMemcpyDeviceToHost));
   HIP_CALL(hipMemcpy(hcols.data(), dcols, nnz*sizeof(HYPRE_BigInt), hipMemcpyDeviceToHost));
   HYPRE_BigInt countr=0;
   HYPRE_BigInt countc=0;
   for (int i=0; i<nnz; ++i)
   {
      if (hrows[i]<iLower_ || hrows[i]>iUpper_)
      {
         countr++;
      }
      if (hcols[i]<iLower_ || hcols[i]>iUpper_)
      {
         countc++;
      }
   }
   if (countr) printf("rank %d : Found %ld of %ld bad diag row indices\n", iproc_, countr, nnz);
   if (countc) printf("rank %d : Found %ld of %ld bad diag col indices\n", iproc_, countc, nnz);
   fflush(stdout);
   return;
}
#else
void HypreSystem::validateDiagData(HYPRE_Int nnz, HYPRE_Int *drows, HYPRE_Int *dcols)
{
   std::vector<HYPRE_Int> hrows(nnz);
   std::vector<HYPRE_Int> hcols(nnz);
   HIP_CALL(hipMemcpy(hrows.data(), drows, nnz*sizeof(HYPRE_Int), hipMemcpyDeviceToHost));
   HIP_CALL(hipMemcpy(hcols.data(), dcols, nnz*sizeof(HYPRE_Int), hipMemcpyDeviceToHost));
   HYPRE_Int countr=0;
   HYPRE_Int countc=0;
   for (int i=0; i<nnz; ++i)
   {
      if (hrows[i]<iLower_ || hrows[i]>iUpper_)
      {
         countr++;
      }
      if (hcols[i]<iLower_ || hcols[i]>iUpper_)
      {
         countc++;
      }
   }
   if (countr) printf("rank %d : Found %d of %d bad diag row indices\n", iproc_, countr, nnz);
   if (countc) printf("rank %d : Found %d of %d bad diag col indices\n", iproc_, countc, nnz);
   fflush(stdout);
   return;
}
#endif
#endif

#if defined(HYPRE_USING_HIP)
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
void HypreSystem::validateOffdData(HYPRE_BigInt nnz, HYPRE_BigInt *drows, HYPRE_BigInt *dcols)
{
   std::vector<HYPRE_BigInt> hrows(nnz);
   std::vector<HYPRE_BigInt> hcols(nnz);
   HIP_CALL(hipMemcpy(hrows.data(), drows, nnz*sizeof(HYPRE_BigInt), hipMemcpyDeviceToHost));
   HIP_CALL(hipMemcpy(hcols.data(), dcols, nnz*sizeof(HYPRE_BigInt), hipMemcpyDeviceToHost));
   HYPRE_BigInt countr=0;
   HYPRE_BigInt countc=0;
   for (int i=0; i<nnz; ++i)
   {
      if (hrows[i]<iLower_ || hrows[i]>iUpper_)
      {
         countr++;
      }
      if (hcols[i]>=iLower_ && hcols[i]<=iUpper_)
      {
         countc++;
      }
   }
   if (countr) printf("rank %d : Found %ld of %ld bad offd row indices\n", iproc_, countr, nnz);
   if (countc) printf("rank %d : Found %ld of %ld bad offd col indices\n", iproc_, countc, nnz);
   fflush(stdout);
   return;
}
#else
void HypreSystem::validateOffdData(HYPRE_Int nnz, HYPRE_Int *drows, HYPRE_Int *dcols)
{
   std::vector<HYPRE_Int> hrows(nnz);
   std::vector<HYPRE_Int> hcols(nnz);
   HIP_CALL(hipMemcpy(hrows.data(), drows, nnz*sizeof(HYPRE_Int), hipMemcpyDeviceToHost));
   HIP_CALL(hipMemcpy(hcols.data(), dcols, nnz*sizeof(HYPRE_Int), hipMemcpyDeviceToHost));
   HYPRE_Int countr=0;
   HYPRE_Int countc=0;
   for (int i=0; i<nnz; ++i)
   {
      if (hrows[i]<iLower_ || hrows[i]>iUpper_)
      {
         countr++;
      }
      if (hcols[i]>=iLower_ && hcols[i]<=iUpper_)
      {
         countc++;
      }
   }
   if (countr) printf("rank %d : Found %d of %d bad offd row indices\n", iproc_, countr, nnz);
   if (countc) printf("rank %d : Found %d of %d bad offd col indices\n", iproc_, countc, nnz);
   fflush(stdout);
   return;
}
#endif
#endif

#if defined(HYPRE_USING_HIP)
void HypreSystem::build_27pt_stencil() {
  auto start = std::chrono::system_clock::now();

  YAML::Node linsys = inpfile_["linear_system"];
  numComps_ = get_optional(linsys, "num_components", 1);
  segregatedSolve_ = (bool)get_optional(linsys, "segregated_solve", 1);
  numSolves_ = segregatedSolve_ ? numComps_ : 1;
  numVectors_ = segregatedSolve_ ? 1 : numComps_;
  rtol_ = (double)get_optional(linsys, "rtol", 1.0e-6);
  atol_ = (double)get_optional(linsys, "atol", 1.0e-8);

  nx_ = (int)get_optional(linsys, "nx", 128);
  ny_ = (int)get_optional(linsys, "ny", 128);
  nz_ = (int)get_optional(linsys, "nz", 128);

  // Determine process pattern for the unit cube
  int nproc_x;
  int nproc_y;
  int nproc_z;

  compute_3d_process_distribution(nproc_, nproc_x, nproc_y, nproc_z);
  if(iproc_ == 0)
  {
     printf("\tProcess distribution: %d x %d x %d\n", nproc_x, nproc_y, nproc_z);
  }

  // Generate problem
  Data data;

  generate_3d_laplacian_hip(nx_,
                            ny_,
                            nz_,
                            nproc_x,
                            nproc_y,
                            nproc_z,
                            &comm_,
                            iproc_,
                            nproc_,
                            &data);

  totalRows_ = M_ = N_ = nx_*ny_*nz_*nproc_;

  // generic method for IJ and MM
  init_row_decomposition();

  // Create HYPRE data structures
  init_system();

  /********************************************/
  /* Create the COO matrix in global indexing */
  /********************************************/

#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
  d_rows_ = hypre_TAlloc(HYPRE_BigInt, data.diagonal_nnz, HYPRE_MEMORY_DEVICE);
  d_cols_ = hypre_TAlloc(HYPRE_BigInt, data.diagonal_nnz, HYPRE_MEMORY_DEVICE);
  d_offd_rows_ = hypre_TAlloc(HYPRE_BigInt, data.offd_nnz, HYPRE_MEMORY_DEVICE);
  d_offd_cols_ = hypre_TAlloc(HYPRE_BigInt, data.offd_nnz, HYPRE_MEMORY_DEVICE);
#else
  d_rows_ = hypre_TAlloc(HYPRE_Int, data.diagonal_nnz, HYPRE_MEMORY_DEVICE);
  d_cols_ = hypre_TAlloc(HYPRE_Int, data.diagonal_nnz, HYPRE_MEMORY_DEVICE);
  d_offd_rows_ = hypre_TAlloc(HYPRE_Int, data.offd_nnz, HYPRE_MEMORY_DEVICE);
  d_offd_cols_ = hypre_TAlloc(HYPRE_Int, data.offd_nnz, HYPRE_MEMORY_DEVICE);
#endif

  fillGlobalRowIndices<<<numRows_,128>>>(numRows_, iLower_, data.diagonal_csr_row_ptr, d_rows_);
  HIP_CALL(hipGetLastError());

  fillGlobalRowIndices<<<numRows_,128>>>(numRows_, iLower_, data.offd_csr_row_ptr, d_offd_rows_);
  HIP_CALL(hipGetLastError());

  /* transform column indices to global */
  hipDeviceProp_t prop;
  int device;
  hipGetDevice(&device);
  hipGetDeviceProperties(&prop, device);
  int CUcount = prop.multiProcessorCount;

  fillGlobalColIndices<<<CUcount*4,256>>>(data.diagonal_nnz, iLower_, data.diagonal_csr_col_ind, d_cols_);
  HIP_CALL(hipGetLastError());
  fillGlobalColIndices<<<CUcount*4,256>>>(data.offd_nnz, ((iproc_==0) ? iUpper_+1 : 0), data.offd_csr_col_ind, d_offd_cols_);
  HIP_CALL(hipGetLastError());

  /* Validate data */
  //validateDiagData(data.diagonal_nnz, d_rows_, d_cols_);
  //validateOffdData(data.offd_nnz, d_offd_rows_, d_offd_cols_);

  /********************************************/
  /* Call Hypre Matrix Assembly Routines      */
  /********************************************/

  /* Set matrix diagonal values */
  HYPRE_IJMatrixSetValues2(mat_, data.diagonal_nnz,
                           NULL, d_rows_, NULL, d_cols_, data.diagonal_csr_val);
  HIP_CALL(hipGetLastError());

  /* Set matrix off diagonal values */
  HYPRE_IJMatrixAddToValues2(mat_, data.offd_nnz,
                             NULL, d_offd_rows_, NULL, d_offd_cols_, data.offd_csr_val);
  HIP_CALL(hipGetLastError());

  /********************************************/
  /* Call Hypre Vector Assembly Routines      */
  /********************************************/

  /* Set the rhs vector */
  HYPRE_IJVector v;
  if (numSolves_ == 1 && numVectors_ == 1) {
    v = rhs_[0];
    HYPRE_IJVectorSetComponent(v, 0);
  } else {
     /* Throw exception */
  }
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
  d_vector_indices_ = hypre_TAlloc(HYPRE_BigInt, numRows_, HYPRE_MEMORY_DEVICE);
#else
  d_vector_indices_ = hypre_TAlloc(HYPRE_Int, numRows_, HYPRE_MEMORY_DEVICE);
#endif
  thrust::sequence(thrust::device, d_vector_indices_, d_vector_indices_ + numRows_, iLower_);
  HIP_CALL(hipGetLastError());

  HYPRE_IJVectorSetValues(v, numRows_, d_vector_indices_, data.rhs_val);
  HIP_CALL(hipGetLastError());

  free(data);

  MPI_Barrier(comm_);
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  timers_.emplace_back("Build 27Pt Stencil HYPRE matrix", elapsed.count());
  MPI_Barrier(comm_);
  fflush(stdout);
}
#endif
/********************************************************************************/
/*                          Matrix Market Format                                */
/********************************************************************************/

void HypreSystem::load_matrix_market() {
  YAML::Node linsys = inpfile_["linear_system"];
  numComps_ = get_optional(linsys, "num_components", 1);
  segregatedSolve_ = (bool)get_optional(linsys, "segregated_solve", 1);
  numSolves_ = segregatedSolve_ ? numComps_ : 1;
  numVectors_ = segregatedSolve_ ? 1 : numComps_;
  rtol_ = (double)get_optional(linsys, "rtol", 1.0e-6);
  atol_ = (double)get_optional(linsys, "atol", 1.0e-8);

  std::string matfile = linsys["matrix_file"].as<std::string>();

  std::vector<std::string> rhsfile(numComps_);
  std::vector<std::string> slnfile(numComps_);
  if (numComps_ == 1 && linsys["rhs_file"]) {
    rhsfile[0] = linsys["rhs_file"].as<std::string>();
    if (linsys["sln_file"]) {
      slnfile[0] = linsys["sln_file"].as<std::string>();
      checkSolution_ = true;
    }
  } else {
    int count = 0;
    for (int i = 0; i < numComps_; ++i) {
      rhsfile[i] = linsys["rhs_file" + std::to_string(i)].as<std::string>();
      std::string sfile = "sln_file" + std::to_string(i);
      if (linsys[sfile]) {
        slnfile[i] = linsys[sfile].as<std::string>();
        count++;
      }
    }
    if (count == numComps_)
      checkSolution_ = true;
  }

  // Scan the matrix and determine the sizes
  determine_mm_system_sizes(matfile);

  // generic method for IJ and MM
  init_row_decomposition();

  // Create HYPRE data structures
  init_system();

  // Build matrices and vectors
  build_mm_matrix(matfile);
  build_mm_vector(rhsfile, rhs_);

  if (checkSolution_) {
    build_mm_vector(slnfile, slnRef_);
  }
}

/*******************
 *
 *******************/
void HypreSystem::determine_mm_system_sizes(std::string matfile) {
  MPI_Barrier(comm_);
  auto start = std::chrono::system_clock::now();

  FILE *fh;
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
  nnz_ = nnz;

  MPI_Barrier(comm_);
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  timers_.emplace_back("Matrix market : determine system size",
                       elapsed.count());
  MPI_Barrier(comm_);
  fflush(stdout);
  fclose(fh);
}

/*******************
 *
 *******************/
void HypreSystem::build_mm_matrix(std::string matfile) {
   MPI_Barrier(comm_);
   if (iproc_ == 0)
       printf("%s : Reading from %s into HYPRE_IJMatrix\n", __FUNCTION__, matfile.c_str());

  auto start = std::chrono::system_clock::now();

  FILE *fh;
  MM_typecode matcode;
  int err;
  int msize, nsize, nnz;
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
  HYPRE_BigInt irow, icol;
#else
  HYPRE_Int irow, icol;
#endif
  double value;

  if ((fh = fopen(matfile.c_str(), "rt")) == NULL) {
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


  fclose(fh);
  int fd = open(matfile.c_str(), O_RDONLY);
  struct stat s;
  int status = fstat(fd, &s);
  int64_t size = s.st_size;
  char * f = (char *) mmap (0, size, PROT_READ, MAP_FILE|MAP_PRIVATE, fd, 0);
  std::string all_lines(f);
  std::string line;
  int64_t found=0, pos=-1, rsize=0, len=0;
  bool foundHeader=false;
  rows_.resize(0);
  cols_.resize(0);
  vals_.resize(0);
  while (rsize<size)
  {
    found = all_lines.find('\n', pos+1);
    int64_t len = found-pos;
    rsize+=len;
    line = all_lines.substr(pos+1, len);
    pos=found;
    if (line.find("%",0)==0)
    {
      foundHeader=true;
      continue;
    }
    if (foundHeader)
    {
      foundHeader=false;
      continue;
    }
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
    sscanf(line.c_str(), "%lld %lld %lf", &irow, &icol, &value);
#else
    sscanf(line.c_str(), "%d %d %lf", &irow, &icol, &value);
#endif

    irow--;
    icol--;
    
    if (irow >= iLower_ && irow <= iUpper_) {
      rows_.push_back(irow);
      cols_.push_back(icol);
      vals_.push_back(value);
    }
  }

  int unmap_result = munmap(f, size);
  close(fd);

  // Set the values of the matrix
  hypre_matrix_set_values();

  MPI_Barrier(comm_);
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  timers_.emplace_back("Matrix market : read and build matrix",
                       elapsed.count());
  MPI_Barrier(comm_);
  fflush(stdout);
}

/*******************
 *
 *******************/
void HypreSystem::build_mm_vector(std::vector<std::string> &mmfiles,
                                  std::vector<HYPRE_IJVector> &vec) {
  MPI_Barrier(comm_);
  auto start = std::chrono::system_clock::now();

  for (int j = 0; j < numComps_; ++j) {
    std::string mmfile = mmfiles[j];
    if (iproc_ == 0)
       printf("%s : Reading from %s into HYPRE_IJVector\n", __FUNCTION__, mmfile.c_str());

    FILE *fh;
    MM_typecode matcode;
    int err;
    int msize, nsize;
    double value;

    if ((fh = fopen(mmfile.c_str(), "r")) == NULL) {
      throw std::runtime_error("Cannot open vector file: " + mmfile);
    }

    err = mm_read_banner(fh, &matcode);
    if (err != 0)
      throw std::runtime_error("Cannot read array banner");

    if (!mm_is_valid(matcode) || !mm_is_array(matcode))
      throw std::runtime_error("Invalid matrix market file encountered");

    err = mm_read_mtx_array_size(fh, &msize, &nsize);
    if (err != 0)
      throw std::runtime_error("Cannot read array sizes in file: " + mmfile);

    if ((msize != M_))
      throw std::runtime_error("Inconsistent sizes for Matrix and Vector");

    /* Read in the file through memory mapping */
    fclose(fh);
    int fd = open(mmfile.c_str(), O_RDONLY);
    struct stat s;
    int status = fstat(fd, &s);
    int64_t size = s.st_size;
    char * f = (char *) mmap (0, size, PROT_READ, MAP_FILE|MAP_PRIVATE, fd, 0);
    std::string all_lines(f);
    std::string line;
    int64_t found=0, pos=-1, rsize=0, len=0, i=0;
    bool foundHeader=false;
    vector_indices_.resize(0);
    vector_values_.resize(0);

    while (rsize<size)
    {
      found = all_lines.find('\n', pos+1);
      int64_t len = found-pos;
      rsize+=len;
      line = all_lines.substr(pos+1, len);
      pos=found;
      if (line.find("%",0)==0)
      {
   foundHeader=true;
   continue;
      }
      if (foundHeader)
      {
   foundHeader=false;
   continue;
      }
      if (i >= iLower_ && i <= iUpper_) {
   sscanf(line.c_str(), "%lf", &value);
   vector_values_.push_back(value);
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
   vector_indices_.push_back((HYPRE_BigInt)i);
#else
   vector_indices_.push_back(i);
#endif
      }
      i++;
    }
    
    int unmap_result = munmap(f, size);
    close(fd);
    
    /* build the vector */
    hypre_vector_set_values(vec, j);
  }

  MPI_Barrier(comm_);
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  timers_.emplace_back("Matrix market : read and build vector",
                       elapsed.count());
  MPI_Barrier(comm_);
  fflush(stdout);
}
} // namespace nalu
