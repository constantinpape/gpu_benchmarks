#! /bin/bash
#SBATCH -A kreshuk
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -t 0-0:30
#SBATCH -o train_p100.out
#SBAtCH -e train_p100.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=constantin.pape@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=P100
#SBATCH --gres=gpu:1

module load cuDNN
/g/kreshuk/pape/Work/my_projects/gpu_benchmarks/benchmarks/training/training_benchmark.py /g/kreshuk/data/benchmark/project 1 --input_path /g/kreshuk/data/benchmark/sample_A_20160501.hdf 
