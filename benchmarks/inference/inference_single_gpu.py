import json
import argparse
import datetime

from simpleference.inference.inference import run_inference
from simpleference.inference.io import IoN5, IoHDF5
from simpleference.backends.pytorch import InfernoPredict
from simpleference.backends.pytorch.preprocess import preprocess
from simpleference.postprocess import clip_float_to_uint8


def inference(gpu_id, input_file, input_key,
              output_file, output_key, checkpoint):

    input_block = ()
    output_block = ()

    with open('offsets/list_gpu_%i.json' % gpu_id, 'r') as f:
        offset_list = json.load(f)

    prediction = InfernoPredict(checkpoint, crop=output_block, gpu=0)
    io_in = IoHDF5(input_file, input_key)
    io_out = IoN5(output_file, output_key)

    # TODO log inference of individual blocks
    t0 = datetime.now()
    run_inference(prediction, preprocess, clip_float_to_uint8,
                  io_in, io_out, offset_list, input_block)
    t1 = datetime.now()
    t_diff = t1 - t0
    t_diff = t_diff.microseconds / 1e6

    with open('t_inference_%i.txt' % gpu_id) as f:
        f.write(t_diff)


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
