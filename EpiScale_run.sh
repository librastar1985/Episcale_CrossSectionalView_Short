#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --output=N01G00_CenterLineCalcTest# This affects the print out of the "std::cout" in the script, make sure this is changed for different jobs.
#SBATCH --mail-user=useremail@address.com
#SBATCH --mail-type=ALL
#SBATCH --job-name="N01_0"
#SBATCH -p gpu # This is the default partition, you can use any of the following; intel, batch, highmem, gpu

module load extra
module load GCC
module load cuda

srun -p gpu --gres=gpu:1 ./bin/runDiscSimulation_M -slurm N01_0

