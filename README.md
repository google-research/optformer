# OptFormer: Transformer-based framework for Hyperparameter Optimization
This is the code used for the paper [Towards Learning Universal Hyperparameter Optimizers with Transformers (NeurIPS 2022)](https://arxiv.org/abs/2205.13320).

# Installation
To get started, manually install [T5X](https://github.com/google-research/t5x). Afterwards, install OptFormer with ``pip install -e .``.

# Usage

## Pre-trained OptFormer as a Policy ([Example Notebook](https://github.com/google-research/optformer/blob/main/optformer/notebooks/OptFormer_inference.ipynb))
To use our OptFormer pre-trained individually on public BBOB and HPO-B benchmarks, follow the steps:

1. (Optional) Download the model checkpoint from `gs://gresearch/optformer/model_checkpoints` for faster loading.
2. Load the model checkpoint into the `InferenceModel`, as shown in [policies_test.py](https://github.com/google-research/optformer/blob/main/optformer/t5x/policies.py).

The `InferenceModel` will then be wrapped into the `OptFormerDesigner`, which follows the same API as a OSS Vizier standard [`Designer`](https://oss-vizier.readthedocs.io/en/latest/guides/developer/writing_algorithms.html).

## Training the OptFormer
To train an OptFormer model, the data will need to consist of a large collection of studies. This data may come from two sources:

1. Our training datasets generated from BBOB and HPO-B benchmarks. We provide cached T5X datasets in `gs://gresearch/optformer/datasets`. Those include one epoch of the training and evaluation datasets. BBOB datasets are generated with multiple hyperparameter optimization algorithms and the HPO-B dataset is generated with Google Vizier's GP-UCB algorithm. Additionally we provide 4 epochs and 30 epochs of cached BBOB and HPO-B training datasets respectively, each epoch using different data augmentations. Those are sufficient for training OptFormer for 100K steps with a batch size of 256. Training using this data converged to the model checkpoint found above.

2. Custom user-generated studies, which can be done using [OSS Vizier's benchmarking pipeline](https://oss-vizier.readthedocs.io/en/latest/guides/index.html#for-benchmarking).

To train an OptFormer with the cached BBOB dataset, download datasets and run the following command to train a model locally (set your Jax platform properly).

```sh
PATH_TO_T5X_ROOT_DIR=...
PATH_TO_OPTFORMER_ROOT_DIR=...
PATH_TO_CACHED_DATASETS_ROOT_DIR=...
MODEL_DIR="/tmp/optformer/$(date +'%y%m%d%H%M%S')"
JAX_PLATFORMS='cpu' \
python3 ${PATH_TO_T5X_ROOT_DIR}/t5x/train.py \
  --seqio_additional_cache_dirs="$PATH_TO_CACHED_DATASETS_ROOT_DIR" \
  --gin_search_paths="${PATH_TO_OPTFORMER_ROOT_DIR}/optformer/optformer/t5x/configs" \
  --gin_file=runs/train.gin \
  --gin_file=tasks/bbob.gin \
  --gin.MODEL_DIR="'$MODEL_DIR'"
```

To use a smaller model size and batch size for debugging, change `t5x/examples/t5/t5_1_1/base.gin` to `t5x/examples/t5/t5_1_1/small.gin` in the task gin file `tasks/bbob.gin` and override the `BATCH_SIZE` as follows

```sh
JAX_PLATFORMS='cpu' \
python3 ${PATH_TO_T5X_ROOT_DIR}/t5x/train.py \
  --seqio_additional_cache_dirs="$PATH_TO_CACHED_DATASETS_ROOT_DIR" \
  --gin_search_paths="${PATH_TO_OPTFORMER_ROOT_DIR}/optformer/optformer/t5x/configs" \
  --gin_file=runs/train.gin \
  --gin_file=tasks/bbob.gin \
  --gin.MODEL_DIR="'$MODEL_DIR'" \
  --gin.BATCH_SIZE=8 \
```

You can also launch a training job with xmanager:

```sh
GOOGLE_CLOUD_BUCKET_NAME=...
PATH_TO_T5X_ROOT_DIR=...
PATH_TO_OPTFORMER_ROOT_DIR=...
MODEL_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/optformer/$(date +'%y%m%d%H%M%S')
python3 PATH_TO_T5X_ROOT_DIR/t5x/scripts/xm_launch.py \
  --extra_args="seqio_additional_cache_dirs=gs://gresearch/optformer/datasets" \
  --gin_search_paths=${PATH_TO_OPTFORMER_ROOT_DIR}/optformer/optformer/t5x/configs \
  --gin_file=runs/train.gin \
  --gin_file=tasks/bbob.gin \
  --model_dir=$MODEL_DIR
```

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
