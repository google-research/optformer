# OptFormer: Transformer-based framework for Hyperparameter Optimization
This is the code used for the paper [Towards Learning Universal Hyperparameter Optimizers with Transformers (NeurIPS 2022)](https://arxiv.org/abs/2205.13320).

# Installation
All base dependencies can be installed from `requirements.txt`. Afterwards, [T5X](https://github.com/google-research/t5x) must be manually installed.

# Usage

## Pre-trained OptFormer as a Policy ([Example Notebook](https://github.com/google-research/optformer/blob/main/optformer/notebooks/OptFormer_inference.ipynb))
To use our OptFormer pre-trained individually on public BBOB and HPO-B benchmarks, follow the steps:

1. (Optional) Download the model checkpoint from `gs://gresearch/optformer/model_checkpoints` for faster loading.
2. Load the model checkpoint into the `InferenceModel`, as shown in [policies_test.py](https://github.com/google-research/optformer/blob/main/optformer/t5x/policies.py).

The `InferenceModel` will then be wrapped into the `OptFormerDesigner`, which follows the same API as a OSS Vizier standard [`Designer`](https://oss-vizier.readthedocs.io/en/latest/guides/developer/writing_algorithms.html).

## Training the OptFormer
To train an OptFormer model, the data will need to consist of a large collection of studies. This data may come from two sources:

1. Our generated dataset, in the form of a [Tensorflow Dataset (TFDS)](https://www.tensorflow.org/datasets), which can be found in [TODO]. Training using this data converged to the model checkpoint found above.
2. Custom user-generated studies, which can be done using [OSS Vizier's benchmarking pipeline](https://oss-vizier.readthedocs.io/en/latest/guides/index.html#for-benchmarking).

## Paper Results
Numerical results used in the paper (for e.g. plotting and comparisons) can be found in the [results folder](https://github.com/google-research/optformer/tree/main/optformer/results).

# Citation
If you found this codebase useful, please consider citing our paper. Thanks!

```
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

**Disclaimer:** This is not an officially supported Google product.
