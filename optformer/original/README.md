# Official NeurIPS 2022 Implementation
This folder contains code implementing the original OptFormer, based on token quantization. Please see `algorithms_test.py` for an example on how the algorithms are used.

Numerical results used in the paper (for e.g. plotting and comparisons) can be found in the [results folder](https://github.com/google-research/optformer/tree/main/optformer/original/results/hpob).

## NOTE: Legacy Code
Legacy code for the paper [Towards Learning Universal Hyperparameter Optimizers with Transformers (NeurIPS 2022)](https://arxiv.org/abs/2205.13320) can be found in the [`neurips22` branch](https://github.com/google-research/optformer/tree/neurips22).

# Citation
If you found this codebase useful, please consider citing our paper. Thanks!

```bibtex
@inproceedings{optformer,
  author    = {Yutian Chen and
               Xingyou Song and
               Chansoo Lee and
               Zi Wang and
               Qiuyi Zhang and
               David Dohan and
               Kazuya Kawakami and
               Greg Kochanski and
               Arnaud Doucet and
               Marc'Aurelio Ranzato and
               Sagi Perel and
               Nando de Freitas},
  title     = {Towards Learning Universal Hyperparameter Optimizers with Transformers},
  booktitle = {Neural Information Processing Systems (NeurIPS) 2022},
  year      = {2022}
}
```