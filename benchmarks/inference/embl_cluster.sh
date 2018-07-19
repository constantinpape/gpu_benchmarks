#! /bin/bash
#SBATCH -A kreshuk
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 0-0:30
#SBATCH -o inference_1080Ti.out
#SBATCH -e inference_1080Ti.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=constantin.pape@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=1080Ti
#SBATCH --gres=gpu:1

module load cuDNN
/g/kreshuk/pape/Work/my_projects/gpu_benchmarks/benchmarks/inference/inference_benchmark.py out.n5 1  --checkpoint /g/kreshuk/data/benchmark/model.pytorch --input_file /g/kreshuk/data/benchmark/sample_A_padded_20160501.hdf
