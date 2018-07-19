#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/inferno/bin/python

import json
import argparse
import time

from simpleference.inference.inference import run_inference
from simpleference.inference.io import IoN5, IoHDF5
from simpleference.backends.pytorch import PyTorchPredict
from simpleference.backends.pytorch.preprocess import preprocess
from simpleference.postprocessing import clip_float_to_uint8

from inferno.trainers.callbacks import Callback


# Need this to unpickle ...
class TimeTrainingIters(Callback):
    """Log the runtime for each training iteration"""
    def __init__(self, log_file):
        super(TimeTrainingIters, self).__init__()
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            f.write('Iteration | Time [s] \n')

    def begin_of_training_iteration(self, **_):
        # self.t_start = datetime.datetime.now()
        self.t_start = time.time()

    def end_of_training_iteration(self, iteration_num, **_):
        # t_stop = datetime.datetime.now()
        # t_diff = self.t_start - t_stop
        # t_diff = t_diff.microseconds / 1e6
        t_diff = time.time() - self.t_start
        with open(self.log_file, 'a') as f:
            f.write('%i %f \n' % (iteration_num, t_diff))


def inference(gpu_id, input_file, input_key,
              output_file, output_key, checkpoint):

    input_blocks = (65, 675, 675)
    output_blocks = (55, 575, 575)

    with open('offsets/list_gpu_%i.json' % gpu_id, 'r') as f:
        offset_list = json.load(f)

    prediction = PyTorchPredict(checkpoint, crop=output_blocks, gpu=gpu_id)
    io_in = IoHDF5(input_file, [input_key])
    io_out = IoN5(output_file, [output_key],
                  channel_order=[list(range(3))])

    t0 = time.time()
    run_inference(prediction, preprocess, clip_float_to_uint8,
                  io_in, io_out, offset_list, input_blocks, output_blocks)
    t1 = time.time()
    t_diff = t1 - t0

    with open('t_inference_%i.txt' % gpu_id, 'w') as f:
        f.write(str(t_diff))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_id', type=int)
    parser.add_argument('input_file', type=str)
    parser.add_argument('input_key', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('output_key', type=str)
    parser.add_argument('checkpoint', type=str)
    args = parser.parse_args()
    inference(args.gpu_id, args.input_file, args.input_key,
              args.output_file, args.output_key, args.checkpoint)
