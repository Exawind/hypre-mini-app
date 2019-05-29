
#include "HypreSystem.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE.h"
#include "HYPRE_config.h"

#define debugMode 0
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

#include <cuda_runtime_api.h>

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
      YAML::Node& inpfile
      ) : comm_(comm),
  inpfile_(inpfile),
  rowFilled_(0)
  {
    MPI_Comm_rank(comm, &iproc_);
    MPI_Comm_size(comm, &nproc_);
  }

  void HypreSystem::set_num_matrices(int num){
    num_matrices = num;
  }
  int HypreSystem::get_num_matrices(){
    return num_matrices;
  }
  void
    HypreSystem::loadSetup()
    {
      //for multiple matrices -- multiple solver calls invoked

      cudaError_t ierr;
      int numGPUs;
      int modeK;
      ierr = cudaGetDeviceCount(&numGPUs);
      printf("%d GPUs available!\n", numGPUs);      
      cudaDeviceGetAttribute (&modeK, cudaDevAttrConcurrentManagedAccess,iproc_%numGPUs);
      printf("managed access on device %d? %d\n", iproc_ %numGPUs, modeK);
      HYPRE_DEVICE = iproc_%numGPUs;
      HYPRE_DEVICE_COUNT = numGPUs;
      if (ierr != cudaSuccess)
	throw std::runtime_error("Error getting GPU count");
      ierr = cudaSetDevice(iproc_ % numGPUs);
      printf("Hi I am rank %d Setting my GPUs to %d !\n",iproc_, iproc_%numGPUs);      
      if (ierr != cudaSuccess) 
	throw std::runtime_error("Error setting GPU device for " + std::to_string(iproc_));

      YAML::Node linsys = inpfile_["linear_system"];


      num_matrices = get_optional(linsys, "num_matrices", 1); 
      spaceSize = get_optional(linsys, "space_size", 5);

      cudaMalloc ( &GPUtmp, sizeof(HYPRE_Real)*spaceSize);
      CPUtmp = (HYPRE_Real*) calloc(spaceSize, sizeof(HYPRE_Real)); 


      YAML::Node solver = inpfile_["solver_settings"];
      std::string method = solver["method"].as<std::string>();
      /*std::string preconditioner = solver["preconditioner"].as<std::string>();
      //this is done once, and matrix independent
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
      }*/
      //end of precond setup
      //read GMRES setting

      if (!method.compare( "gmres")){       
	printf("using GMRES solver \n");       
	setup_gmres();
      } else if (!method.compare("boomeramg")) {
	printf("using BOOMERANG solver\n");       
	setup_boomeramg_solver();
      }
      else if (!method.compare("cogmres")){
	printf("using CO-GMRES solver\n");       
	setup_cogmres();
	//printf("done with setup \n");      
      }
      else {
	throw std::runtime_error("Invalid option for solver method provided: "
	    + method);
      }
      currentSpaceSize = 0;   
    }
  void HypreSystem::loadMatrix(int i){

    YAML::Node linsys = inpfile_["linear_system"];

    std::string mat_format = get_optional<std::string>(linsys, "type", "matrix_market") ;

    if (mat_format == "matrix_market") {
      load_matrix_market_one(i);
    } else if (mat_format == "hypre_ij") {
      load_hypre_format_one(i);
    } else {
      throw std::runtime_error("Invalid linear system format option: " + mat_format);
    }

    //we destroyed it last time to need to reset

    YAML::Node solver = inpfile_["solver_settings"];
    std::string preconditioner = solver["preconditioner"].as<std::string>();
    //this is done once, and matrix independent
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
  }  

  void HypreSystem::destroyMatrix(){

    HYPRE_IJMatrixDestroy(mat_);
    HYPRE_IJVectorDestroy(sln_);
    HYPRE_IJVectorDestroy(rhs_);
    precondDestroyPtr_(precond_);
  }

  void HypreSystem::planeRot(double * x, double * G){
    //Givens rot on a 2x1 vector, returns 2x3 matrix ORDERED COLUMNWISE, third column is a vectori y such that G*x = y
    if (x[1] != 0){

      double r = sqrt(x[0]*x[0]+x[1]*x[1]);
      G[0] = x[0]/r;
      G[2] = x[1]/r;
      G[1] = (-1.0/r)*x[1];
      G[3] = x[0]/r;

      G[4] = r;
      G[5] = 0.0; 
    }
    else {
      G[0] = 1.0f;
      G[3] = 1.0f;
      G[1] = 0.0f;
      G[2] = 0.0f;


      G[4] = x[0];
      G[5] = x[1];
    }

  }
  void HypreSystem::dropFirstColumn(){
    // the hardest
    /* for (i = 1:1:(dim - 1))
     *  G = planerot(R(i:(i + 1), i));
     keyboard
     R(i:(i + 1), :) = G*R(i:(i + 1), :);

     Q(:, i:(i + 1)) = Q(:, i:(i + 1))*G';

     U(:, i:(i + 1)) = U(:, i:(i + 1))*G';
     *
     * */

    double * G = (double*)  calloc(6, sizeof(double));
    double * x = (double*)  calloc(2, sizeof(double));
    for (int i=0; i<spaceSize-1; ++i){
      x[0] = R[i*(spaceSize+1)+spaceSize];
      x[1] = R[i*(spaceSize+1)+spaceSize+1];
#if debugMode
      printf("BEFORE ROT x[0] = %16.15f x[1] = %16.15f\n", x[0], x[1]);
#endif 
      planeRot(x, G);
#if debugMode
      printf("G matrix %f %f %f %f x[0] = %16.15f x[1] = %16.15f\n", G[0], G[1], G[2], G[3], x[0], x[1]);
#endif
      //update R

      for (int j=1; j<spaceSize; ++j){ 
	double aux = R[j*spaceSize+i ];
	double aux2 = R[j*spaceSize+i+1 ];
#if debugMode
	printf("Will be changing %f %f \n", R[j*spaceSize+i ], R[j*spaceSize+i+1 ]);      
#endif
	R[j*spaceSize+i ] = G[0]*aux+ G[2]*aux2;
	R[j*spaceSize+i+1] = G[1]*aux+G[3]*aux2;
      }
      //Update projectionSpace (Q)
      hypre_ParKrylovGivensRotRight(i,
	  i+1,
	  projectionSpace,
	  projectionSpace,
	  G[0], G[2],G[1],G[3]);

      hypre_ParKrylovGivensRotRight(
	  i,
	  i+1,
	  projectionSpaceRaw,
	  projectionSpaceRaw,
	  G[0], G[2],G[1],G[3]);
    }
    //now we copy
    for (int i=1; i<spaceSize; ++i){
      for (int j=0; j<i; ++j){
	R[(i-1)*spaceSize+j] = R[i*spaceSize+j]; 
      }
      for (int j=i; j<spaceSize; ++j){
	R[(i-1)*spaceSize+j] = 0.0f;
      }
    }
#if debugMode
    for (int i=0; i<spaceSize; ++i){
      for (int j=0; j<spaceSize; ++j){
	printf(" %16.16f ", R[j*spaceSize+i]);
      }
      printf("\n");

    }
#endif


  }

  void HypreSystem::createProjectedInitGuess(int i){
    /*
     * */
#if debugMode
    printf("SPACE SIZE %d \n", i);
#endif
    if (i<1){
      // we don't have a space yet
      // FOR LEFT PRECON ONLY!!!
      // rhs =  AMGVcycle(L,bn,1);
      //take 0s as init guess (std)
printf("I IS SMALLER THAN 1 \n");
      HYPRE_ParVectorSetConstantValues(parSln_, 0.0);

      //parY = parY -R()*Q;

      parY_ =(hypre_ParVector*)  hypre_ParKrylovCreateVector((hypre_ParVector *)parSln_);//tmp
      parZ_ =(hypre_ParVector*)  hypre_ParKrylovCreateVector((hypre_ParVector *)parSln_);//another tmp
      parOldRhs_ =(hypre_ParVector*)  hypre_ParKrylovCreateVector((hypre_ParVector *)parSln_);//another tmp
    }
    else {

      /*
	 alpha = R(1:d,d);
	 bnt = bn - Q(:, 1:d)*alpha;
	 unpara = U(:, 1:d)*alpha;
	 y = AMGVcycle(L,bnt,1);  
	 unperp = y;
	 un = unpara + unperp;
	 un = un + AMGVcycle(L,bn - A*un,1);
	 un = un + AMGVcycle(L,bn - A*un,1);

      //LEFT precon
      //done in "solve"
      rhs = AMGVcycle(L,bn,1);      

*/

      //parSln = 0s
      HYPRE_ParVectorSetConstantValues(parSln_, 0.0);
      //parY = bn
#if 1
      hypre_ParKrylovCopyVectorOneOfMult(parRhs_, 0,
	  parY_, 0 );
      //bnt: = parY = parY -R()*Q;
#if debugMode
      printf("START: IP of  parY (=rhs) with itself %16.16f \n", hypre_ParKrylovInnerProdOneOfMult(parY_, 0,
	    parY_, 0 ));
      printf("START: IP of  sln with itself (should be ZERO) %16.16f \n", hypre_ParKrylovInnerProdOneOfMult(parSln_, 0,
	    parSln_, 0 ));
      printf("CURRENT SPACE SIZE %d \n", currentSpaceSize);
      //works
      printf("trying to multiply by (current space size %d) and putting the result in Q(%d) \n", currentSpaceSize, currentSpaceSize);
      for (int kk=0; kk<currentSpaceSize; ++kk){
	printf("%16.16f \n", R[spaceSize*(currentSpaceSize-1)+kk]);
      }
#endif
      //GPUtmp = alpha (alpha = R(1:d, d))
      cudaMemcpy ( GPUtmp, &R[spaceSize*(currentSpaceSize-1)],
	  currentSpaceSize*sizeof(HYPRE_Real),
	  cudaMemcpyHostToDevice );

      //parY = bnt = bn - Q*alpha = parY-Q*alpha
      hypre_ParKrylovMassAxpyMult(GPUtmp,
	  projectionSpace,currentSpaceSize,
	  parY_,
	  0);
#if debugMode
      printf("IP of parY with itself %16.16f \n", hypre_ParKrylovInnerProdOneOfMult(parY_, 0,
	    parY_, 0 ));
#endif


      //parSln = 0 - U*alpha
      hypre_ParKrylovMassAxpyMult(GPUtmp,
	  projectionSpaceRaw,currentSpaceSize,
	  parSln_,
	  0);
      //scale by -1.0 (by default mass AXPY performs y - alpha_1 x_1 - alpha_2 x_2 etc.

      hypre_ParKrylovScaleVectorOneOfMult(-1.0f,parSln_, 0 );

#if debugMode
      printf("AGAIN: IP of parY with itself %16.16fi and parSln with itself %16.16f \n", hypre_ParKrylovInnerProdOneOfMult(parY_, 0,
	    parY_, 0 ), hypre_ParKrylovInnerProdOneOfMult(parSln_, 0,parSln_, 0 ));
#endif
      //apply AMG V cycle to bnt

      //important
      hypre_ParKrylovUpdateVectorCPU(parY_);
      hypre_ParKrylovClearVector(parZ_);
      //parZ = AMGVcycle(L,parY,1);  

      precondSolvePtr_(precond_, parMat_, parY_, parZ_);

      //un = parZ = parZ + parSln_
      hypre_ParKrylovAxpyOneOfMult(1.0f,
	  parSln_,0,
	  parZ_,
	  0);


      //parY = parRhs = bn
      hypre_ParKrylovCopyVectorOneOfMult(parRhs_, 0,
	  parY_, 0 );

      //parY = parY - A*parZ = bn - A*un
      hypre_ParKrylovMatvecMult(NULL,
	  -1.0f,
	  parMat_,
	  parZ_,
	  0,
	  1.0f,
	  parY_, 0);
      //
      //parSln = AMGVcycle(parY)

      hypre_ParKrylovUpdateVectorCPU(parY_);
      hypre_ParKrylovClearVector(parSln_);
      precondSolvePtr_(precond_, parMat_, parY_, parSln_);

      //un = un + AMGVcycle(L,bn - A*un,1);ie parSln = parZ + parSln
      hypre_ParKrylovAxpyOneOfMult(1.0f,
	  parZ_,0,
	  parSln_,
	  0);

#if 0
      // one more Richardson
      //parY = bn
      hypre_ParKrylovCopyVectorOneOfMult(parRhs_, 0,
	  parY_, 0 );
      //parY = parY-A*Sln 
      hypre_ParKrylovMatvecMult(NULL,
	  -1.0f,
	  parMat_,
	  parSln_,
	  0,
	  1.0f,
	  parY_, 0);
      //
      //parZ = AMGVcycle(parY)
      hypre_ParKrylovUpdateVectorCPU(parY_);
      hypre_ParKrylovClearVector(parZ_);
      precondSolvePtr_(precond_, parMat_, parY_, parZ_);

      //un = un + AMGVcycle(L,bn - A*un,1);ie parSln = parZ + parSln
      hypre_ParKrylovAxpyOneOfMult(1.0f,
	  parZ_,0,
	  parSln_,
	  0);

#endif

      hypre_ParKrylovUpdateVectorCPU(parSln_);

#if debugMode
      printf("END IP of parSln with itself %16.16f \n", hypre_ParKrylovInnerProdOneOfMult(parSln_, 0,
	    parSln_, 0 ));
#endif
#endif
    }

  }


  void HypreSystem::projectionSpaceUpdate(int i){
    auto start = std::chrono::system_clock::now();

    // need to create matvec

    if (i==1){

      projectionSpace =(hypre_ParVector*)  hypre_ParKrylovCreateMultiVector((hypre_ParVector *)parRhs_, spaceSize);//Q
      projectionSpaceRaw = (hypre_ParVector*)hypre_ParKrylovCreateMultiVector((hypre_ParVector *)parRhs_, spaceSize);//U
      //and allocate R
      R = (double *) calloc((spaceSize)*(spaceSize), sizeof(double));

      hypre_ParKrylovMatvecMult(NULL,
	  1.0,
	  parMat_,
	  parSln_,
	  0,
	  0.0f,
	  projectionSpace, 0);
      //copy parSln_ to 
      hypre_ParKrylovCopyVectorOneOfMult(parSln_, 0,
	  projectionSpaceRaw, 0 );
      R[0] = hypre_ParKrylovInnerProdOneOfMult(projectionSpace, 0,
	  projectionSpace, 0 );
      R[0] = sqrt(R[0]);

      double tt = 1.0f/R[0];
#if debugMode
      printf("R of zero: R[0] = %f \n ", R[0]);

      printf("VERY BEFORE Norm of the first vector %f \n", hypre_ParKrylovInnerProdOneOfMult(projectionSpace, 0,projectionSpace, 0));
      //scale both

      printf("BEFORE NORM of the solution %f\n", hypre_ParKrylovInnerProdOneOfMult(parSln_, 0,parSln_, 0));
      printf("BEFORE NORM of the first vector Q(0) %f NORM of U(0) %f\n", hypre_ParKrylovInnerProdOneOfMult(projectionSpace, 0,projectionSpace, 0),  hypre_ParKrylovInnerProdOneOfMult(projectionSpaceRaw, 0,projectionSpaceRaw, 0));
#endif     
      hypre_ParKrylovScaleVectorOneOfMult(tt,projectionSpace, 0 ); 
      hypre_ParKrylovScaleVectorOneOfMult(tt,projectionSpaceRaw, 0 ); 
#if debugMode    
      printf("NORM of the first vector Q(0) %f NORM of U(0) %f\n", hypre_ParKrylovInnerProdOneOfMult(projectionSpace, 0,projectionSpace, 0),  hypre_ParKrylovInnerProdOneOfMult(projectionSpaceRaw, 0,projectionSpaceRaw, 0));
#endif    
      currentSpaceSize = 1;
    }
    else{
      // the space exists 
      //check if we need to drop
      if (i>spaceSize){
#if debugMode    
	printf("dropping a column\n");
#endif	
	dropFirstColumn();
	currentSpaceSize = spaceSize-1;
      }

#if debugMode    
      printf("ADDING a vector, spaceSize %d currentSpaceSize %d i %d \n", spaceSize, currentSpaceSize, i);
#endif      
      //projectionSpace(currentSpaceSize) = A*parSln_    
      hypre_ParKrylovMatvecMult(NULL,
	  1.0,
	  parMat_,
	  parSln_,
	  0,
	  0.0f,
	  projectionSpace, currentSpaceSize);

#if debugMode    
      printf("initial Norm of the vector %d:  %f \n",currentSpaceSize, hypre_ParKrylovInnerProdOneOfMult(projectionSpace, currentSpaceSize,projectionSpace, currentSpaceSize));
#endif
      //U(currenSpaceSize) = sln
      hypre_ParKrylovCopyVectorOneOfMult(parSln_, 0,
	  projectionSpaceRaw, currentSpaceSize);


      //orthogonalize new vector using CGS-2

      hypre_ParKrylovMassInnerProdMult(projectionSpace,currentSpaceSize,projectionSpace, currentSpaceSize, GPUtmp);


      cudaMemcpy ( &R[spaceSize*currentSpaceSize],GPUtmp,
	  currentSpaceSize*sizeof(HYPRE_Real),
	  cudaMemcpyDeviceToHost );
      hypre_ParKrylovMassAxpyMult(GPUtmp,
	  projectionSpace,currentSpaceSize,
	  projectionSpace,
	  currentSpaceSize);
#if debugMode
      for (int jj = 0; jj<currentSpaceSize; ++jj){
	printf("BEFORE REORTH R[%d, %d] = %f \n", jj, currentSpaceSize, R[spaceSize*currentSpaceSize+jj]);
      }
#endif
      //next, reorth

      hypre_ParKrylovMassInnerProdMult(projectionSpace,currentSpaceSize,projectionSpace, currentSpaceSize, GPUtmp);


      cudaMemcpy (CPUtmp,GPUtmp,
	  currentSpaceSize*sizeof(HYPRE_Real),
	  cudaMemcpyDeviceToHost );

      for (int j=0; j<currentSpaceSize; j++){
	HYPRE_Int id = currentSpaceSize*spaceSize+j;
	R[id]+=CPUtmp[j];
#if debugMode    
	printf("adding %16.16f to R[ %d * %d + %d] \n", CPUtmp[j], currentSpaceSize, spaceSize, j);       
#endif
      }
#if debigMode
      for (int jj = 0; jj<currentSpaceSize; ++jj){
	printf("AFTER R[%d, %d] = %f \n", jj, currentSpaceSize, R[spaceSize*currentSpaceSize+jj]);
      }
#endif
      hypre_ParKrylovMassAxpyMult(GPUtmp,
	  projectionSpace,currentSpaceSize,
	  projectionSpace,
	  currentSpaceSize);


      double t = hypre_ParKrylovInnerProdOneOfMult(projectionSpace, currentSpaceSize,projectionSpace, currentSpaceSize);
      t = sqrt(t);
      R[spaceSize*currentSpaceSize+currentSpaceSize] = t;
#if debugMode    
      printf("t = %f R of not zero i. e R[%d, %d] = %f \n",t, currentSpaceSize, currentSpaceSize, R[spaceSize*currentSpaceSize+ currentSpaceSize]);
#endif
      if (t!=0){

	t = 1.0/t;

#if debugMode    
	printf("NORM sq OF Q(%d) BEFORR %16.16f t = %16.16f \n",currentSpaceSize, hypre_ParKrylovInnerProdOneOfMult(projectionSpace, currentSpaceSize,
	      projectionSpace, currentSpaceSize ), t);
#endif
	hypre_ParKrylovScaleVectorOneOfMult(t,
	    projectionSpace, currentSpaceSize);

#if debugMode    
	printf("NORM of the solution %f\n", hypre_ParKrylovInnerProdOneOfMult(parSln_, 0,parSln_, 0));
	printf("NORM sq OF Q(%d) %16.16f \n",currentSpaceSize, hypre_ParKrylovInnerProdOneOfMult(projectionSpace, currentSpaceSize,
	      projectionSpace, currentSpaceSize ));
#endif
      }
      if (currentSpaceSize>1){
	for (int kk=1; kk<currentSpaceSize; kk++){
#if debugMode    
	  printf("IP Q(%d)^TQ(%d) = %16.16f \n",currentSpaceSize,kk, hypre_ParKrylovInnerProdOneOfMult(projectionSpace, currentSpaceSize,
		projectionSpace, kk));
#endif
	}
      }
      //update U
      /*
	 U(:, d + 1) = un - U(:, 1:d)*R(1:d, d + 1);
	 U(:, d + 1) = U(:, d + 1)/R(d + 1, d + 1);
	 */

      cudaMemcpy (GPUtmp, &R[spaceSize*currentSpaceSize],
	  currentSpaceSize*sizeof(HYPRE_Real),
	  cudaMemcpyHostToDevice );
      hypre_ParKrylovMassAxpyMult(GPUtmp,
	  projectionSpaceRaw,currentSpaceSize,
	  projectionSpaceRaw,
	  currentSpaceSize);
      if(t!=0)
      {
#if debugMode    
	printf("NORM sq OF U(%d) BEFORE %16.16f t = %16.16f \n",currentSpaceSize, hypre_ParKrylovInnerProdOneOfMult(projectionSpaceRaw, currentSpaceSize,    projectionSpaceRaw, currentSpaceSize ), t);
#endif     
	hypre_ParKrylovScaleVectorOneOfMult(t,
	    projectionSpaceRaw, currentSpaceSize);
      }

#if debugMode    
      printf("NORM sq OF U(%d) %16.16f \n",currentSpaceSize, hypre_ParKrylovInnerProdOneOfMult(projectionSpaceRaw, currentSpaceSize,
	    projectionSpaceRaw, currentSpaceSize ));
#endif
      currentSpaceSize ++;
    }

    MPI_Barrier(comm_);
    auto stop1 = std::chrono::system_clock::now(); 
    std::chrono::duration<double> solve = stop1 - start;

    if (iproc_ == 0) {
      timers_.emplace_back("Space Update", solve.count());
    }
  }

  void
    HypreSystem::load()
    {

      cudaError_t ierr;
      int numGPUs;
      int modeK;
      ierr = cudaGetDeviceCount(&numGPUs);
      printf("%d GPUs available!\n", numGPUs);      
      cudaDeviceGetAttribute (&modeK, cudaDevAttrConcurrentManagedAccess,iproc_%numGPUs);
      printf("managed access on device %d? %d\n", iproc_ %numGPUs, modeK);
      HYPRE_DEVICE = iproc_%numGPUs;
      HYPRE_DEVICE_COUNT = numGPUs;
      if (ierr != cudaSuccess)
	throw std::runtime_error("Error getting GPU count");
      ierr = cudaSetDevice(iproc_ % numGPUs);
      printf("Hi I am rank %d Setting my GPUs to %d !\n",iproc_, iproc_%numGPUs);      
      if (ierr != cudaSuccess) 
	throw std::runtime_error("Error setting GPU device for " + std::to_string(iproc_));

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
      std::cout<<"METHOD IS "<<method<<'\n';
      //      if (method == "gmres") {
      if (!method.compare( "gmres")){       
	printf("using GMRES solver \n");       
	setup_gmres();
      } else if (!method.compare("boomeramg")) {
	printf("using BOOMERANG solver\n");       
	setup_boomeramg_solver();
      }
      else if (!method.compare("cogmres")){
	printf("using CO-GMRES solver\n");       
	setup_cogmres();
	//printf("done with setup \n");      
      }
      else {
	throw std::runtime_error("Invalid option for solver method provided: "
	    + method);
      }
    }

    void HypreSystem::load_matrix_market_one(int i)
    {
      YAML::Node linsys = inpfile_["linear_system"];

      std::string matfile = linsys["matrix_file"].as<std::string>();
      std::string rhsfile = linsys["rhs_file"].as<std::string>();
      std::string matfile_one = matfile + std::to_string(i) + ".mm";
      std::string rhsfile_one = rhsfile + std::to_string(i) + ".mm";

      load_mm_matrix(matfile_one);
      read_mm_vector(rhsfile_one, rhs_);


      // Indicate that we need a check on missing rows and a final assemble call
      needFinalize_ = true;
    }
    void HypreSystem::load_matrix_market()
    {
      YAML::Node linsys = inpfile_["linear_system"];

      std::string matfile = linsys["matrix_file"].as<std::string>();
      std::string rhsfile = linsys["rhs_file"].as<std::string>();

      load_mm_matrix(matfile);
      read_mm_vector(rhsfile, rhs_);

      if (linsys["sln_file"]) {
	std::string slnfile = linsys["sln_file"].as<std::string>();
	checkSolution_ = true;
	read_mm_vector(slnfile, slnRef_);
      }

      // Indicate that we need a check on missing rows and a final assemble call
      needFinalize_ = true;
    }

    void HypreSystem::load_hypre_format()
    {
      YAML::Node linsys = inpfile_["linear_system"];
      int nfiles = get_optional(linsys, "num_partitions", nproc_);

      if (nfiles == nproc_)
	load_hypre_native();
      else {
	std::string matfile = linsys["matrix_file"].as<std::string>();
	std::string rhsfile = linsys["rhs_file"].as<std::string>();


	determine_ij_system_sizes(matfile, nfiles);
	init_ij_system();
	read_ij_matrix(matfile, nfiles);
	read_ij_vector(rhsfile, nfiles, rhs_);

	if (linsys["sln_file"]) {
	  std::string slnfile = linsys["sln_file"].as<std::string>();
	  checkSolution_ = true;
	  read_ij_vector(slnfile, nfiles, slnRef_);
	}
	needFinalize_ = false;
      }
    }

    void HypreSystem::load_hypre_format_one(int i)
    {
      YAML::Node linsys = inpfile_["linear_system"];
      int nfiles = get_optional(linsys, "num_partitions", nproc_);

      if (nfiles == nproc_)
	load_hypre_native_one(i);
      else {
	std::string matfile = linsys["matrix_file"].as<std::string>();
	std::string rhsfile = linsys["rhs_file"].as<std::string>();
	std::string matfile_one = matfile + std::to_string(i) + ".mm";
	std::string rhsfile_one = rhsfile + std::to_string(i) + ".mm";

	//	std::cout<<"Trying to open "<<matfile_one<<"  and  "<<rhsfile_one;      
	determine_ij_system_sizes(matfile_one, nfiles);
	init_ij_system();
	read_ij_matrix(matfile_one, nfiles);
	read_ij_vector(rhsfile_one, nfiles, rhs_);

	needFinalize_ = false;
      }
    }
    void HypreSystem::load_hypre_native()
    {
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

#if debugMode    
      printf("init solution \n");      
#endif
      HYPRE_IJVectorInitialize(sln_);
#if debugMode    
      printf("initializing sln_ 2\n");
#endif
      HYPRE_IJVectorGetObject(sln_, (void**)&parSln_);
      HYPRE_ParVectorSetConstantValues(parSln_, 0.0);

      // Indicate that the assemble has already been called by HYPRE API
      needFinalize_ = false;

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

    void HypreSystem::load_hypre_native_one(int i)
    {
      auto start = std::chrono::system_clock::now();
      if (iproc_ == 0) {
	std::cout << "Loading HYPRE IJ files... ";
      }

      YAML::Node linsys = inpfile_["linear_system"];

      std::string matfile = linsys["matrix_file"].as<std::string>();
      std::string rhsfile = linsys["rhs_file"].as<std::string>();

      std::string matfile_one = matfile + std::to_string(i) + ".mm";
      std::string rhsfile_one = rhsfile + std::to_string(i) + ".mm";

      //    std::cout<<"Trying to open "<<matfile_one<<"  and  "<<rhsfile_one;      
      HYPRE_IJMatrixRead(matfile_one.c_str(), comm_, HYPRE_PARCSR, &mat_);
      HYPRE_IJVectorRead(rhsfile_one.c_str(), comm_, HYPRE_PARCSR, &rhs_);


      // Figure out local range
      HYPRE_Int jlower, jupper;
      HYPRE_IJMatrixGetLocalRange(mat_, &iLower_, &iUpper_, &jlower, &jupper);
      numRows_ = (iUpper_ - iLower_ + 1);


      // Initialize the solution vector
      HYPRE_IJVectorCreate(comm_, iLower_, iUpper_, &sln_);
      HYPRE_IJVectorSetObjectType(sln_, HYPRE_PARCSR);

#if debugMode    
      printf("init solution \n");      
#endif      
      HYPRE_IJVectorInitialize(sln_);
#if debugMode          
      printf("initializing sln_ 3\n");
#endif
      HYPRE_IJVectorGetObject(sln_, (void**)&parSln_);
      HYPRE_ParVectorSetConstantValues(parSln_, 0.0);

      // Indicate that the assemble has already been called by HYPRE API
      needFinalize_ = false;

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
	  precond_, get_optional(node, "relax_type", 18));
      HYPRE_BoomerAMGSetNumSweeps(
	  precond_, get_optional(node, "num_sweeps", 1));
      HYPRE_BoomerAMGSetRelaxOrder(
	  precond_, get_optional(node, "relax_order", 0));
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
	  precond_, get_optional(node, "relax_type", 18));
      HYPRE_BoomerAMGSetNumSweeps(
	  precond_, get_optional(node, "num_sweeps", 1));
      HYPRE_BoomerAMGSetTol(
	  precond_, get_optional(node, "tolerance", 0.0));
      HYPRE_BoomerAMGSetMaxIter(
	  precond_, get_optional(node, "max_iterations", 1));
      HYPRE_BoomerAMGSetRelaxOrder(
	  precond_, get_optional(node, "relax_order", 0));
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
      HYPRE_ParCSRCOGMRESSetGSoption(solver_, get_optional(node, "GSoption", 0));
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


    void HypreSystem::solve2()
    {
      finalize_system();

#if debugMode    
      printf("system has been finalized \n");
#endif
      auto start = std::chrono::system_clock::now();
      if (usePrecond_) {
	solverPrecondPtr_(
	    solver_, precondSolvePtr_, precondSetupPtr_, precond_);
      }
#if debugMode    
      printf("starting solver setup \n");
#endif      
      //works ok if this command is never called      
      solverSetupPtr_(solver_, parMat_, parRhs_, parSln_);
#if debugMode    
      printf("solver setup done \n");
#endif      
      MPI_Barrier(comm_);
      auto stop1 = std::chrono::system_clock::now();
      std::chrono::duration<double> setup = stop1 - start;

      createProjectedInitGuess(currentSpaceSize);
      //printf("solver setup done \n");      

#if 1      
      auto stop3 = std::chrono::system_clock::now();
      std::chrono::duration<double> initGuessUpdate = stop3 - stop1;

      if (iproc_ == 0) {
	timers_.emplace_back("Init Guess Update", initGuessUpdate.count());
      }
#endif

      //For left precon
      /* DONT DO ANYTHING HERE, GMRES WOULD DO IT FOR YOu
	 HYPRE_Solver solver,
	 HYPRE_ParCSRMatrix A,
	 HYPRE_ParVector b,
	 HYPRE_ParVector x
       * */

#if debugMode    
      printf("BEFORE applying precon to RHS %16.16f \n", hypre_ParKrylovInnerProdOneOfMult(parRhs_, 0,
	    parRhs_, 0 ));
#endif     
      //   hypre_ParKrylovClearVector(parY_);
      //   precondSolvePtr_(precond_, parMat_, parRhs_, parY_);

#if debugMode    
      printf("AFTER applying precon to RHS (parRhs)%16.16f \n", hypre_ParKrylovInnerProdOneOfMult(parRhs_, 0,
	    parRhs_, 0 ));

      printf("AFTER applying precon to RHS (parY) %16.16f \n", hypre_ParKrylovInnerProdOneOfMult(parY_, 0,
	    parY_, 0 ));

      hypre_ParKrylovCopyVectorOneOfMult(parRhs_, 0,
	  parOldRhs_, 0 );
#endif
#if 0
      hypre_ParKrylovCopyVectorOneOfMult(parY_, 0,
	  parRhs_, 0 );
      hypre_ParKrylovMatvecMult(NULL,
	  1.0f,
	  parMat_,
	  parSln_,
	  0,
	  0.0f,
	  parY_, 0);

      hypre_ParKrylovUpdateVectorCPU(parY_);
      hypre_ParKrylovClearVector(parZ_);
      precondSolvePtr_(precond_, parMat_, parY_, parZ_);

      hypre_ParKrylovCopyVectorOneOfMult(parRhs_, 0,
	  parY_, 0 );
      hypre_ParKrylovAxpyOneOfMult(-1.0f,
	  parZ_,0,
	  parY_,
	  0);
      printf("BEFORE SOLVER CALL (parY) %16.16f \n", sqrt(hypre_ParKrylovInnerProdOneOfMult(parY_, 0,
	      parY_, 0 )));

      printf("BEFORE SOLVER CALL (parRhs) %16.16f \n", sqrt(hypre_ParKrylovInnerProdOneOfMult(parRhs_, 0,
	      parRhs_, 0 )));

#endif
      //      hypre_ParKrylovCopyVectorOneOfMult(parY_, 0,
      //	  parRhs_, 0 );
      solverSolvePtr_(solver_, parMat_, parRhs_, parSln_);
      //compute TRUE residual
#if debugMode    
      double NormB =  sqrt(hypre_ParKrylovInnerProdOneOfMult(parOldRhs_, 0,
	    parOldRhs_, 0 ));
      hypre_ParKrylovMatvecMult(NULL,
	  -1.0f,
	  parMat_,
	  parSln_,
	  0,
	  1.0f,
	  parOldRhs_, 0);

      //hypre_ParKrylovUpdateVectorCPU(parY_);
      //hypre_ParKrylovClearVector(parZ_);
      //precondSolvePtr_(precond_, parMat_, parY_, parZ_);

      //hypre_ParKrylovCopyVectorOneOfMult(parRhs_, 0,
      //parY_, 0 );
      //hypre_ParKrylovAxpyOneOfMult(-1.0f,
      //  parZ_,0,
      //  parY_,
      // 0);

      printf("\n AFTER SOLVER CALL (parY) %16.16f \n", sqrt(hypre_ParKrylovInnerProdOneOfMult(parOldRhs_, 0,
	      parOldRhs_, 0 )));
      printf("AFTER SOLVER CALL RELATIVE RES %16.16f \n", sqrt(hypre_ParKrylovInnerProdOneOfMult(parOldRhs_, 0,
	      parOldRhs_, 0 ))/NormB);
#endif      
      MPI_Barrier(comm_);

#if 1      
      auto stop2 = std::chrono::system_clock::now();
      std::chrono::duration<double> solve = stop2 - stop3;

      if (iproc_ == 0) {
	timers_.emplace_back("Preconditioner setup", setup.count());
	timers_.emplace_back("Solve", solve.count());
      }
#endif
      solveComplete_ = true;
    }

    void HypreSystem::solve()
    {
      finalize_system();

      auto start = std::chrono::system_clock::now();
      if (usePrecond_) {
	solverPrecondPtr_(
	    solver_, precondSolvePtr_, precondSetupPtr_, precond_);
      }
      //left precon is applied to b INSIDE SOLVER!!! DONT DO ANYTHING TO RHS HERE


      solverSetupPtr_(solver_, parMat_, parRhs_, parSln_);
      //printf("solver setup done \n");      
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
      HYPRE_IJVectorCopyDataGPUtoCPU(sln_);
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

      double totalSolve =0.0;
      double totalIGU =0.0;
      double totalSU =0.0;
      std::string slv ("Solve");
      std::string igu ("Init Guess Update");
      std::string su ("Space Update");
      std::cout << "\nTimer summary: " << std::endl;
      for (auto& timer: timers_) {
	std::cout << "    " << std::setw(25) << std::left << timer.first
	  << timer.second << " seconds" << std::endl;
	if (slv.compare(timer.first) == 0){
	  //printf("adding %f to %f (total solve time) \n", timer.second, totalSolve);
	  totalSolve+=timer.second;
	}		
	if (igu.compare(timer.first) == 0) totalIGU+=timer.second;		
	if (su.compare(timer.first) == 0) totalSU+=timer.second;		

      }
      std::cout << "    " << std::setw(25) << std::left << "TOTAL SOLVE TIME"
	<< totalSolve << " seconds" << std::endl;
      std::cout << "    " << std::setw(25) << std::left << "TOTAL INIT GUESS TIME"
	<< totalIGU << " seconds" << std::endl;
      std::cout << "    " << std::setw(25) << std::left << "TOTAL SPACE UPDATE TIME"
	<< totalSU << " seconds" << std::endl;
    }

    void HypreSystem::load_mm_matrix(std::string matfile)
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

    void HypreSystem::init_ij_system()
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
      MPI_Barrier(comm_);

      // Create HYPRE data structures
      init_system();
    }

    void HypreSystem::read_ij_vector(std::string vecfile, int nfiles, HYPRE_IJVector& vec)
    {
      auto start = std::chrono::system_clock::now();

      HYPRE_Int ilower, iupper;
      HYPRE_Int irow;
      double value;
      //int ret;

      for (int ii=iproc_; ii < nfiles; ii+=nproc_) {
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
	HYPRE_Int numrows = (iupper - ilower) + 1;
	for (HYPRE_Int j=0; j < numrows; j++) {
#ifdef HYPRE_BIGINT
	  fscanf(fh, "%lld%*[ \t]%le\n", &irow, &value);
#else
	  fscanf(fh, "%d%*[ \t]%le\n", &irow, &value);
#endif
	  HYPRE_IJVectorAddToValues(vec, 1, &irow, &value);
	}
	fclose(fh);
      }

      MPI_Barrier(comm_);
      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      timers_.emplace_back("Read vector", elapsed.count());
    }

    void HypreSystem::read_ij_matrix(std::string matfile, int nfiles)
    {
      auto start = std::chrono::system_clock::now();

      HYPRE_Int ilower, iupper, jlower, jupper;
      HYPRE_Int irow, icol;
      HYPRE_Int ncols = 1;
      double value;
      int ret;

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

#ifdef HYPRE_BIGINT
	ret = fscanf(fh, "%lld %lld%*[ \t]%le\n", &irow, &icol, &value);
#else
	ret = fscanf(fh, "%d %d%*[ \t]%le\n", &irow, &icol, &value);
#endif
	while (ret != EOF) {
	  HYPRE_IJMatrixAddToValues(mat_, 1, &ncols, &irow, &icol, &value);
#ifdef HYPRE_BIGINT
	  ret = fscanf(fh, "%lld %lld%*[ \t]%le\n", &irow, &icol, &value);
#else
	  ret = fscanf(fh, "%d %d%*[ \t]%le\n", &irow, &icol, &value);
#endif
	}
	fclose(fh);
      }

      MPI_Barrier(comm_);
      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      timers_.emplace_back("Read matrix", elapsed.count());
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

      //printf("init rhs\n");      
      HYPRE_IJVectorInitialize(rhs_);

#if debugMode    
      printf("initializing rhs_\n");
#endif
      HYPRE_IJVectorGetObject(rhs_, (void**)&parRhs_);

      HYPRE_IJVectorCreate(comm_, iLower_, iUpper_, &sln_);
      HYPRE_IJVectorSetObjectType(sln_, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(sln_);
#if debugMode    
      printf("initializing sln_\n");
#endif
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

      if (needFinalize_) {
	HYPRE_Int nrows = 1;
	HYPRE_Int ncols = 1;
	double setval = 1.0; // Set diagonal to 1 for missing rows
	for (HYPRE_Int i=iLower_; i < iUpper_; i++) {
	  if (rowFilled_[i] > 0) continue;
	  HYPRE_IJMatrixSetValues(mat_, nrows, &ncols, &i, &i, &setval);
	}

      }

      HYPRE_IJMatrixAssemble(mat_);
      HYPRE_IJVectorAssemble(rhs_);
      MPI_Barrier(comm_);
      //printf("BEFORE COPY!\n");
      HYPRE_IJMatrixCopyCPUtoGPU(mat_);
      MPI_Barrier(comm_);



      HYPRE_IJVectorAssemble(sln_);
      HYPRE_IJVectorCopyDataCPUtoGPU(rhs_);
      HYPRE_IJVectorCopyDataCPUtoGPU(sln_);


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

    void HypreSystem::read_mm_matrix(std::string matfile)
    {
      if (iproc_ != 0) return;
      std::cout << "Loading matrix into HYPRE_IJMatrix... ";

      auto start = std::chrono::system_clock::now();

      // Set up row order array that will be used later with RHS and solution files
      rowOrder_.resize(totalRows_);

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

      bool isSymmetric = mm_is_symmetric(matcode);
      HYPRE_Int seenRow = totalRows_ + 10;
      HYPRE_Int seenCol = totalRows_ + 10;
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
	if (isSymmetric && (irow != icol)) {
	  HYPRE_IJMatrixAddToValues(mat_, 1, &ncols, &icol, &irow, &value);
	  rowFilled_[icol] = 1;
	}

	if ((irow != seenRow) && (icol != seenCol)) {
	  rowOrder_[idx++] = irow;
	  seenRow = irow;
	  seenCol = icol;
	}
      }

      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      std::cout << elapsed.count() << " seconds" << std::endl;
      timers_.emplace_back("Read matrix", elapsed.count());

      fclose(fh);
    }

    void HypreSystem::determine_ij_system_sizes(std::string matfile, int nfiles)
    {
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

      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = stop - start;
      std::cout << elapsed.count() << " seconds" << std::endl;
      timers_.emplace_back("Scan matrix", elapsed.count());
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
