import os
import sys
import logging
import argparse
import time
import yaml
import numpy as np

from inferno.trainers.basic import Trainer
from inferno.utils.io_utils import yaml2dict

from inferno.extensions.criteria import SorensenDiceLoss
from inferno.trainers.callbacks import Callback
import neurofire.models as models
from skunkworks.datasets.cremi.loaders import get_cremi_loaders


logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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


def set_up_training(project_directory,
                    config,
                    data_config,
                    n_iters):
    model_name = config.get('model_name')
    model = getattr(models, model_name)(**config.get('model_kwargs'))

    loss = SorensenDiceLoss()
    # Build trainer and validation metric
    logger.info("Building trainer.")

    trainer = Trainer(model)\
        .save_every((n_iters, 'iterations'), to_directory=os.path.join(project_directory, 'Weights'))\
        .build_criterion(loss)\
        .build_optimizer(**config.get('training_optimizer_kwargs'))\
        .evaluate_metric_every('never')\
        .register_callback(TimeTrainingIters('./tmp_log.txt'))
    return trainer


def training(project_directory,
             train_configuration_file,
             data_configuration_file,
             max_training_iters=int(1001)):

    logger.info("Loading config from {}.".format(train_configuration_file))
    config = yaml2dict(train_configuration_file)

    logger.info("Loading training data loader from %s." % data_configuration_file)
    train_loader = get_cremi_loaders(data_configuration_file)
    data_config = yaml2dict(data_configuration_file)

    trainer = set_up_training(project_directory,
                              config,
                              data_config,
                              max_training_iters)
    trainer.set_max_num_iterations(max_training_iters)

    # Bind loader
    logger.info("Binding loaders to trainer.")
    trainer.bind_loader('train', train_loader)

    # Set devices
    if config.get('devices'):
        logger.info("Using devices {}".format(config.get('devices')))
        trainer.cuda(config.get('devices'))

    # Go!
    t0 = time.time()
    trainer.fit()
    t1 = time.time()
    t_diff = t1 - t0
    return t_diff


def make_train_config(train_config_file, gpus):
    template = yaml2dict('./template_config/train_config.yml')
    template['devices'] = gpus
    with open(train_config_file, 'w') as f:
        yaml.dump(template, f)


def make_data_config(data_config_file, in_path, raw_key, gt_key, n_batches, workers_per_gpu=10):
    template = yaml2dict('./template_config/data_config.yml')
    template['volume_config']['raw']['path'] = in_path
    template['volume_config']['raw']['path_in_h5_dataset'] = raw_key
    template['volume_config']['segmentation']['path'] = in_path
    template['volume_config']['segmentation']['path_in_h5_dataset'] = gt_key
    template['loader_config']['batch_size'] = n_batches
    template['loader_config']['num_workers'] = workers_per_gpu * n_batches
    with open(data_config_file, 'w') as f:
        yaml.dump(template, f)


def evaluate_benchmark(t_tot):
    print("Total train time:", t_tot)
    times = []
    with open('./tmp_log.txt') as f:
        for i, l in enumerate(f):
            if i == 0:
                continue
            t = float(l.split()[1])
            times.append(t)
    print("Mean iteration time:", np.mean(times), "+-", np.std(times))
    print("Max iteration time:", np.max(times))
    print("Min iteration time:", np.min(times))
    print("Sum iteration times:", np.sum(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('ngpus', type=int)
    parser.add_argument('--n_iters', type=int, default=int(1001))
    parser.add_argument('--input_path', type=str, default='../data/sample_A_20160501.hdf')
    parser.add_argument('--raw_key', type=str, default='volumes/raw')
    parser.add_argument('--gt_key', type=str, default='volumes/labels/neuron_ids')

    args = parser.parse_args()

    project_directory = args.project_directory
    if not os.path.exists(project_directory):
        os.mkdir(project_directory)

    # gpus = list(args.ngpus)
    # set the proper CUDA_VISIBLE_DEVICES env variables
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    # gpus = list(range(len(gpus)))
    gpus = list(range(args.ngpus))

    train_config = os.path.join(project_directory, 'train_config.yml')
    make_train_config(train_config, gpus)

    in_path = args.input_path
    assert os.path.exists(in_path)
    in_path = os.path.abspath(in_path)
    raw_key, gt_key = args.raw_key, args.gt_key
    data_config = os.path.join(project_directory, 'data_config.yml')
    make_data_config(data_config, in_path, raw_key, gt_key, len(gpus))

    t_tot = training(project_directory,
                     train_config,
                     data_config,
                     max_training_iters=args.n_iters)
    evaluate_benchmark(t_tot)


if __name__ == '__main__':
    main()
