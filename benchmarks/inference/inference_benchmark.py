#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/inferno/bin/python

import sys

import argparse
import os
from concurrent import futures
from subprocess import call
import numpy as np

import h5py
import z5py
from simpleference.inference.util import get_offset_lists


def single_inference(in_file, in_key,
                     out_file, out_key,
                     checkpoint, gpu_id):
    call(['./inference_single_gpu.sh', str(gpu_id),
          in_file, in_key, out_file, out_key, checkpoint])


# iterate over all chunk inference times
# and accumulated gpu inference times and evaluate
def evaluate_bench():
    all_files = os.listdir('.')
    t_gpus = {}
    t_blocks = []
    for ff in all_files:
        if ff.startswith('t_inference'):
            gpu_id = int(ff.split('.')[0].split('_')[-1])
            with open(ff) as f:
                t_gpus[gpu_id] = float(f.readline())
            os.remove(ff)

        elif ff.startswith('t_offset'):
            with open(ff) as f:
                t_blocks.append(float(f.readline()))
            os.remove(ff)
        else:
            continue
    print("Mean block-inference:", np.mean(t_blocks), "+-", np.std(t_blocks))
    print("Min block-inference", np.min(t_blocks))
    print("Max block-inference", np.max(t_blocks))
    print("Sum block inference", np.sum(t_blocks))
    for gpu, t in t_gpus.items():
        print("Inference for gpu", gpu, ":", t, "s")


def run_inference(input_file, input_key,
                  output_file, output_key,
                  checkpoint, n_gpus):

    out_blocks = (65, 675, 675)
    chunks = (1,) + out_blocks

    with h5py.File(input_file) as f:
        shape = f[input_key].shape
    aff_shape = (3,) + shape

    f = z5py.N5File(output_file)
    f.require_dataset(output_key, shape=aff_shape, chunks=chunks,
                      compression='gzip', dtype='uint8')

    get_offset_lists(shape, list(range(n_gpus)), './offsets',
                     output_shape=out_blocks)

    with futures.ProcessPoolExecutor(n_gpus) as pp:
        tasks = [pp.submit(single_inference,
                           input_file, input_key,
                           output_file, output_key,
                           checkpoint, gpu_id)
                 for gpu_id in range(n_gpus)]
        [t.result() for t in tasks]

    evaluate_bench()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', type=str)
    parser.add_argument('n_gpus', type=int)
    parser.add_argument('--checkpoint', type=str,
                        default='../data/')  # TODO path to ckpt
    parser.add_argument('--input_file', type=str,
                        default='../data/sample_A_padded_20160501.hdf')
    parser.add_argument('--input_key', type=str, default='volumes/raw')
    parser.add_argument('--output_key', type=str, default='volumes/affinities')
    args = parser.parse_args()

    in_file = args.input_file
    in_key = args.input_key
    assert os.path.exists(in_file), in_file

    out_file = args.output_file
    out_key = args.output_key

    assert os.path.exists(args.checkpoint), args.checkpoint

    run_inference(in_file, in_key, out_file, out_key,
                  args.checkpoint, args.n_gpus)
