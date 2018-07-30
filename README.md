# GPU Benchmarks

Pytorch / Inferno GPU Benchmarks for training and prediction.
Results for the benchmarks:
- Training: we report the time for 1001 iterations of training a 3D U-Net.
- Inference: we report the time for inference of a ~ 1.9 gigavoxel volume.

If not specified otherwise, the results are given for a single gpu.
Data pre-processing is done multi-threaded, s.t. the gpu usage should be ~ 100 %.


| System         | CUDA| pytorch | Training [s] | Inference [s] |
| -----------    | ---:| ------: | -----------: | ------------: |
| VM:P40-Grid    | 9.1 | 0.4.0   | 1816.7       | 204.6         |
| Cluster:1080Ti | 9.1 | 0.4.0   | 1754.2		| 215.3		    |
| 1080Ti (x2)    | 9.1 | 0.4.0   | 2053.8       | -             |
| 1080Ti (x4)    | 9.1 | 0.4.0   | 2371.4       | -             |
| Cluster:P100   | 9.1 | 0.4.0   | 2238.6		| 219.2         |
| P100 (x2)      | 9.1 | 0.4.0   | 2556.7		| -   	   	    |
| SMC: P100      | 9.1 | 0.4.0   | 2071.7       | 201.5         |
| P100 (x2)      | 9.1 | 0.4.0   | 2239.1       | -             |
| P100 (x4)      | 9.1 | 0.4.0   | 2509.9       | -             |
| SMC: V100      | 9.1 | 0.4.0   | 1866.3       | 148.4         |
| V100 (x2)      | 9.1 | 0.4.0   | 2017.1       | -             |
| V100 (x4)      | 9.1 | 0.4.0   | 2308.8       | -             |

Benchmark results translated to quantities we actually care about.
TODO give estimates for volumes for inference (e.g. Dros. brain, Platyneris)

| System        | Train: Iters / Min | Infer: Megavox / s |
| -----------   | -----------------: | -----------------: |
| VM:P40-Grid   | 33.1               | 9.2                |
| Cluster:1080Ti| 34.2  		     | 8.8 		          |
| 1080Ti (x2)   | 29.4               | -                  |       
| 1080Ti (x4)   | 25.3               | -                  |       
| Cluster:P100  | 26.8	             | 8.6	              |
| P100 (x2)     | 23.5	             | -	              |
| SMC: P100     | 29.0               | 9.4                |
| P100 (x2)     | 26.8               | -                  |
| P100 (x4)     | 23.9               | -                  |
| SMC: V100     | 32.2               | 12.7               |
| V100 (x2)     | 29.8               | -                  |
| V100 (x4)     | 26.0               | -                  |
