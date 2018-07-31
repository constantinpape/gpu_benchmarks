# GPU Benchmark Summary

## Method

I have benchmarked slightly idealized versions of our main applications,
training and prediction with a 3D Convolutional Neural Network.
For training, I have run 1000 iterations of a 3D U-Net on one of the CREMI Datasets,
see https://cremi.org/.
For prediction, I have predicted a padded CREMI Dataset (1.9 Gigavoxel).
In both applications, CPU bound pre- and post-processing steps are 
parallelized, s.t. the GPU should (ideally) be fully utilized.
Note that this is easier to realize for prediction, that requires
significantly less pre- or post-processing.

Also, prediction is embarassingly parallel, i.e. it will
speed-up linearly with the number of GPUs.
In contrast, training is not as easily parallelizable:
multi-batch training parallelized over GPUs speeds up convergence, but
increases the time per iteration due to synchronization overhead.

Hence, I have performed prediction benchmarks only for a single GPU (because 
extrapolation to multiple GPUs is trivial) and training benchmarks for a single and two GPUs.


## Benchmark Results

Benchmarks for 4 different set-ups:

- Cluster:1080Ti: EMBL cluster nodes with Geforce GTX1080Ti GPUs (PCIe)
- Cluster:P100: EMBL cluster nodes with P100 GPUs (PCIe)
- SMC:P100: Supermicro test system with P100 GPUs (NVLink)
- SMC:V100: Supermicro test system with V100 GPUs (NVLink)

Note that I have not changed the input volume sizes between the 1080Tis, which have
11 GB of RAM and the P100 / V100, which have 16 GB of RAM.


| System         | CUDA| pytorch | Training [s] | Prediction [s]|
| -----------    | ---:| ------: | -----------: | ------------: |
| Cluster:1080Ti | 9.1 | 0.4.0   | 1754.2		| 215.3		    |
| 1080Ti (x2)    | 9.1 | 0.4.0   | 2053.8       | -             |
| Cluster:P100   | 9.1 | 0.4.0   | 2238.6		| 219.2         |
| P100 (x2)      | 9.1 | 0.4.0   | 2556.7		| -   	   	    |
| SMC:P100       | 9.1 | 0.4.0   | 2071.7       | 201.5         |
| P100 (x2)      | 9.1 | 0.4.0   | 2239.1       | -             |
| SMC:V100       | 9.1 | 0.4.0   | 1866.3       | 148.4         |
| V100 (x2)      | 9.1 | 0.4.0   | 2017.1       | -             |


## Extrapolation

Based on the training benchmarks, we can calculate the number
of iterations per minute and
the time it would take for training to reach 100k iterations,
which in my experience amounts very roughly to the number of iterations after
which most networks have converged.


| System        | Iters / Min | 100k Iters [d] |
| -----------   | ----------: | -------------: |
| Cluster:1080Ti| 34.2        | 2.0		       |
| 1080Ti (x2)   | 29.4        | 2.4            |       
| Cluster:P100  | 26.8	      | 2.6            |
| P100 (x2)     | 23.5	      | 3.0            |
| SMC:P100      | 29.0        | 2.4            |
| P100 (x2)     | 26.8        | 2.6            |
| SMC:V100      | 32.2        | 2.2            |
| V100 (x2)     | 29.8        | 2.3            |


For prediction, we can calculate the number of Megavoxel that are 
predicted per second and extrapolate how long it would take to 
run prediction for the Platyneris dataset (8.1 Teravoxel) with a single GPU.

| System        | Megavox / s| Platyneris [h] |
| -----------   | ---------: | -------------: |
| Cluster:1080Ti| 8.8 	     | 25.7           |
| Cluster:P100  | 8.6	     | 26.3           |
| SMC:P100      | 9.4        | 24.0           |
| SMC:V100      | 12.7       | 17.8           |


## Comments

The 1080Ti is surprisingly fast compared to P100 and V100.

The advantages compared to the P100 can be explained by the
fact that its single precision performance (which is the relevant factor for our applications)
is higher than the P100's according to NVidias specifications.

The V100 might not be as fast as expected because
the CUDA / pytorch combination we have used might not fully leverage the V100's 
tensor cores yet, which are NVidias main innovation for deep learning introduced with these cards.
