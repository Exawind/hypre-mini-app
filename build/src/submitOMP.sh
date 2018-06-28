#!/bin/bash

#PBS -l walltime=01:00:00  # WALLTIME limit
#PBS -q batch-h  # queue
#PBS -l nodes=172:ppn=24  # 86 Number of nodes, put x processes on each
#PBS -N ompComboMultiRun # Name of job
#PBS -A windsim  # Project handle
#PBS -j oe  # Combine output and error file
#PBS -l qos=high
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
#export OMP_NUM_THREADS=1
export OMP_NESTED=false
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

for value in {1..100}
do
mpirun -np 1032 --map-by ppr:24:node:pe=1 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27gmres.yaml | grep gram
done
echo gmres with one thread done

for value in {1..100}
do
mpirun -np 1032 --map-by ppr:24:node:pe=1 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27cogmres.yaml| grep gram 
done
echo cogmres with one thread done

export OMP_NUM_THREADS=2
for value in {1..100}
do
mpirun -np 1032 --map-by ppr:12:node:pe=2 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27gmres.yaml | grep gram
done
echo gmres with two thread done

for value in {1..100}
do
mpirun -np 1032 --map-by ppr:12:node:pe=2 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27cogmres.yaml | grep gram
done
echo cogmres with two thread done

export OMP_NUM_THREADS=3
for value in {1..100}
do
mpirun -np 1032 --map-by ppr:8:node:pe=3 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27gmres.yaml | grep gram
done
echo gmres with three thread done
for value in {1..100}
do
mpirun -np 1032 --map-by ppr:8:node:pe=3 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27cogmres.yaml | grep gram
done
echo cogmres with three thread done

export OMP_NUM_THREADS=4
for value in {1..100}
do
mpirun -np 1032 --map-by ppr:6:node:pe=4 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27gmres.yaml| grep gram
done
echo gmres with four thread done
for value in {1..100}
do
mpirun -np 1032 --map-by ppr:6:node:pe=4 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27cogmres.yaml | grep gram
done
echo cogmres with four thread done

#export OMP_NUM_THREADS=6
#mpirun -np 1032 --map-by ppr:4:node:pe=6 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27gmres.yaml
#mpirun -np 1032 --map-by ppr:4:node:pe=6 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27cogmres.yaml

#export OMP_NUM_THREADS=8
#mpirun -np 1032 --map-by ppr:3:node:pe=8 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27gmres.yaml
#mpirun -np 1032 --map-by ppr:3:node:pe=8 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27cogmres.yaml

#export OMP_NUM_THREADS=12
#mpirun -np 1032 --map-by ppr:2:node:pe=12 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27gmres.yaml
#mpirun -np 1032 --map-by ppr:2:node:pe=12 -bind-to core -x OMP_NUM_THREADS hypre_app inputFileV27cogmres.yaml

