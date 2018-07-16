# GPU Benchmarks

Pytorch / Inferno GPU Benchmarks for training and prediction.
Results for the benchmarks:
- Training: we report the time for 1001 iterations of training a 3D U-Net.
- Inference: we report the time for inference of a ~ 1.9 gigavoxel volume.

If not specified otherwise, the results are given for a single gpu.
Data pre-processing is done multi-threaded, s.t. the gpu usage should be ~ 100 %.


| System      | CUDA | pytorch | Training [s] | Inference [s] |
| ----------- | ---: | ------: | -----------: | ------------: |
| VM:P40-Grid | 9.1  | 0.4.0   | 1816.7       | 718.8         |
| Cluster:1080Ti | 9.1 | 0.4.0 | 1754.2		  | 708.6		  |
| Cluster:P100 | 9.1 | 0.4.0 | 		  | 		  |

Benchmark results translated to quantities we actually care about.
TODO give estimates for volumes for inference (e.g. Dros. brain, Platyneris)

| System        | Train: Iters / Min | Infer: Megavox / s |
| -----------   | -----------------: | ------------:      |
| VM:P40-Grid   | 33.1               | 2.63               |
| Cluster:1080Ti| 34.2  		     | 2.66 		      |
| Cluster:P100  |		             | 		              |
