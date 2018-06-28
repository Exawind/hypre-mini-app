#!/bin/bash

#PBS -l walltime=01:00:00  # WALLTIME limit
#PBS -q batch-h  # queue
#PBS -l nodes=43:ppn=24  # 172 Number of nodes, put x processes on each
#PBS -N V27_41Jun27thMPIpure1032BadPrecon50gmresOMP1 # Name of job
#PBS -A windsim  # Project handle
#PBS -j oe  # Combine output and error file
#PBS -l qos=high
#PBS -o out.$PBS_JOBNAME
echo $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

#  Put your job commands after this line
{
   module purge
   module use /nopt/nrel/ecom/ecp/base/modules/gcc-6.2.0
   module load gcc/6.2.0
   module load binutils openmpi/1.10.4 netlib-lapack cmake
} &> /dev/null

#mpiexec -np 2 ~sanantha/code/nalu/Nalu/build_master/naluX -i oversetSphereTIOGA.i
#export OMP_NUM_THREADS=12
#mpiexec -np 384 /home/sanantha/exawind/source/Nalu/build_gcc/naluX -i fullV27_41.i.hypre -o fullV27_41.out
export OMP_NUM_THREADS=1
export OMP_NESTED=false
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
#export OMP_SCHEDULE=dynamic
#mpirun -np 1 -x OMP_NUM_THREADS hypre_app inputFileV27.yaml
#mpirun -np 384 -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND hypre_app inputFileV27.yaml
#mpirun -np 688 --map-by ppr:4:node:pe=6 -bind-to core -x OMP_NUM_THREADS -x OMP_SCHEDULE hypre_app inputFileV27.yaml
#mpirun -np 688 --map-by ppr:4:node:pe=6 -bind-to core -x OMP_NUM_THREADS /home/sthomas1/Nalu/build/naluX -i fullV27_41.i.hypre -o fullV27_41-688-6.out
mpirun -np 1032 --map-by ppr:24:node:pe=1 -bind-to core -x OMP_NUM_THREADS ./hypre_app inputFileV27.yaml
