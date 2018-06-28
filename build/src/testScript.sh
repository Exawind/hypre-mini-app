#!/bin/bash
# Basic range in for loop
for value in {1..10}
do
mpirun -np 1 ./hypre_app inputFile.yaml | grep gram
done
echo All done

