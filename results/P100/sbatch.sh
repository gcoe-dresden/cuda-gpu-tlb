#!/bin/bash
#SBATCH -J cuda-gpu-tlb-P100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --mem=12000M # gpu2
#SBATCH --partition=test
#SBATCH --exclusive

TLBBENCH=$HOME/cuda-workspace/cuda-gpu-tlb/release_t1/tlb-bench
TLBSHARE=$HOME/cuda-workspace/cuda-gpu-tlb/release_t1/tlb-sharing
RESULTS=$HOME/cuda-workspace/cuda-gpu-tlb/results/P100/
SR=srun --gpufreq=715:1189
#SR=

cd $RESULTS
module purge
module load gcc/5.3.0 cuda/8.0.61_t1

$SR $TLBBENCH 12 80 1024 4096 0 0
$SR $TLBBENCH 1472 4992 16384 65536 0 0

$SR $TLBSHARE 2048 16 0 0 >& ./TLBsharing-2048-16.txt
$SR $TLBSHARE 32768 65 0 0 >& ./TLBsharing-32768-65.txt
