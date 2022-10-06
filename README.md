# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# OptFormer: Transformer-based framework for Hyperparameter Optimization
This is the code used for the paper [Towards Learning Universal Hyperparameter Optimizers with Transformers (NeurIPS 2022)](https://arxiv.org/abs/2205.13320).

# Installation
All dependencies can be installed in `requirements.txt`. The two main components are [T5X](https://github.com/google-research/t5x) and [OSS Vizier](https://github.com/google/vizier).

# Usage

## Pre-trained OptFormer as a Policy
To use our pre-trained OptFormer (exactly as-is from the paper), follow the steps:

1. Download the model checkpoint from [TODO].
2. Load the model checkpoint into the `InferenceModel`, as shown in [policies_test.py](TODO).

The `InferenceModel` will then be wrapped into the `OptFormerDesigner`, which follows the same API as a OSS Vizier standard [`Designer`](https://oss-vizier.readthedocs.io/en/latest/guides/developer/writing_algorithms.html).

## Training the OptFormer (Coming Soon!)
TODO


**Disclaimer:** This is not an officially supported Google product.
