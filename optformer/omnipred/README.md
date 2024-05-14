# OmniPred: Language Models as Universal Regressors
This folder contains code implementing the OmniPred model.

## General Usage
See `omnipred_test.py` for a setup example.

For regression on your custom object, the idea is to define a custom featurizer which contains the prompt serialization logic:

```python
import tensorflow as tf
from optformer.common.data import featurizers
from optformer.common.serialization import numeric
from optformer.omnipred import vocabs

_T = ... # Type of your object
_DEFAULT_Y_SERIALIZER = numeric.DigitByDigitFloatTokenSerializer()

class MyObjectFeaturizer(featurizers.Featurizer[_T]):

  def to_features(self, obj: _T) -> dict[str, tf.Tensor]:
    inputs: str = ... # Serialize `obj`
    targets: str = _DEFAULT_Y_SERIALIZER.to_str(obj.y_value)

    return {
        'inputs': tf.constant(inputs, dtype=tf.string),
        'targets': tf.constant(targets, dtype=tf.string),
    }
```

The rest of the code follows our regular inference API.

# Citation
If you found this codebase useful, please consider citing our paper. Thanks!

```bibtex
@article{omnipred,
  author       = {Xingyou Song and
                  Oscar Li and
                  Chansoo Lee and
                  Bangding Yang and
                  Daiyi Peng and
                  Sagi Perel and
                  Yutian Chen},
  title        = {OmniPred: Language Models as Universal Regressors},
  journal      = {CoRR},
  volume       = {abs/2402.14547},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2402.14547},
  doi          = {10.48550/ARXIV.2402.14547},
  eprinttype    = {arXiv},
  eprint       = {2402.14547},
}
```