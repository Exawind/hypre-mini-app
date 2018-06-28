#!/bin/bash

#PBS -N NaluCOGMRESJun2018
#PBS -l nodes=16:ppn=24
#PBS -l walltime=00:20:00
#PBS -A windsim
#PBS -q batch-h
#PBS -o out.$PBS_JOBNAME
#PBS -j oe

module purge
module use /nopt/nrel/ecom/ecp/base/modules/gcc-6.2.0/
module load gcc/6.2.0 binutils openmpi/1.10.4


export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=threads

cd $PBS_O_WORKDIR
mpirun -np 384 ./hypre_app inputFileV27.yaml 


