#!/bin/bash
#SBATCH -J cuda-gpu-tlb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --mem=12000M # gpu2
#SBATCH --partition=gpu2
#SBATCH --exclusive

TLBBENCH=$HOME/cuda-workspace/cuda-gpu-tlb/release/tlb-bench
TLBSHARE=$HOME/cuda-workspace/cuda-gpu-tlb/release/tlb-sharing
RESULTS=$HOME/cuda-workspace/cuda-gpu-tlb/results/K80/

cd $RESULTS
module load gcc/5.3.0 cuda/8.0.61

srun --gpufreq=2505:823 $TLBBENCH 1 5 64 256 0 0
srun --gpufreq=2505:823 $TLBBENCH 48 300 1024 4096 0 0
srun --gpufreq=2505:823 $TLBBENCH 1500 5000 1024 4096 0 0

srun --gpufreq=2505:823 $TLBSHARE 128 16 >& ./TLBsharing-128-16.txt
srun --gpufreq=2505:823 $TLBSHARE 2048 65 >& ./TLBsharing-2048-65.txt
srun --gpufreq=2505:823 $TLBSHARE 2048 1032 >& ./TLBsharing-2048-1032.txt
