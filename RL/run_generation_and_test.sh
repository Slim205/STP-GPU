#!/bin/bash
#SBATCH --job-name=stprepo
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --mem=128GB
#SBATCH --partition=serial_requeue
#SBATCH --constraint=a100
#SBATCH --export=ALL 

module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01 
conda activate /n/netscratch/amin_lab/Lab/slim/env 
cd /n/netscratch/amin_lab/Lab/slim/STP/RL
python generate_and_test.py  --model kfdong/STP_model_Lean_0320 --exp_dir /n/netscratch/amin_lab/Lab/slim/STP/benchmark_results --temperature 1.0 --save_file_name "tests" --raw_dataset_config dataset_configs/miniF2F_ProofNet.json --seed 1 --cpu 18 --gpu 1
