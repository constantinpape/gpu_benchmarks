from __future__ import print_function
import os
import stat
from argparse import ArgumentParser
from subprocess import call


def gen_script(script_file, gpu_type, n_gpus):
    # allocate 4 cores per gpu
    n_cores = 4 * n_gpus
	# allocate 6 GB per gpu
    mem = 6 * n_gpus
    id_ = "%s_%i" % (gpu_type, n_gpus)
    # define relevant paths
    py_path = "/g/kreshuk/pape/Work/my_projects/gpu_benchmarks/benchmarks/training/training_benchmark.py"
    project_path = "/g/kreshuk/data/benchmark/project_%s" % id_
    input_path = '/g/kreshuk/data/benchmark/sample_A_20160501.hdf'
	# need bigger time-limits for more gpus
    limits_1080Ti = {1: '0-0:30', 2: '0-1:00', 3: '0-1:30', 4: '0-2:00'}
    limits_P100 = {1: '0-0:45', 2: '0-1:30', 3: '0-2:00', 4: '0-2:30'}
    limits = limits_1080Ti if gpu_type == '1080Ti' else limits_P100

    with open(script_file, 'w') as f:
        f.write("#! /bin/bash\n")
        f.write("#SBATCH -A kreshuk\n")
        f.write("#SBATCH -N 1\n")
        f.write("#SBATCH -n %i\n" % n_cores)
        f.write("#SBATCH --mem %iG\n" % mem)
        f.write("#SBATCH -t %s\n" % limits[n_gpus])
        f.write("#SBATCH -o log/train_%s.out\n" % id_)
        f.write("#SBAtCH -e err/train_%s.err\n" % id_)
        f.write("#SBATCH --mail-type=END,FAIL\n")
        f.write("#SBATCH --mail-user=constantin.pape@embl.de\n")
        f.write("#SBATCH -p gpu\n")
        f.write("#SBATCH -C gpu=%s\n" % gpu_type)
        f.write("#SBATCH --gres=gpu:%i\n" % n_gpus)
        f.write("\n")
        f.write("module load cuDNN\n")
        f.write("echo $CUDA_VISIBLE_DEVICES\n")
        f.write("%s %s %i --input_path %s\n" % (py_path, project_path, n_gpus, input_path))
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('gpu_type', type=str)
    parser.add_argument('n_gpus', type=int)
    args = parser.parse_args()
    gpu_type = args.gpu_type
    assert gpu_type in ('P100', '1080Ti')
    n_gpus = args.n_gpus
    assert n_gpus <= 4
    script_file = 'scripts/embl_cluster_%s_%i.sh' % (gpu_type, n_gpus)
    gen_script(script_file, gpu_type, n_gpus)
    call(['sbatch', script_file])
