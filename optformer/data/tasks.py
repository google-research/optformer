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

"""Language modeling on vizier studies."""
import functools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from absl import logging
from optformer.data import converters
from optformer.t5x import preprocessors
from optformer.t5x import vocabularies
import seqio
import t5.data
import tensorflow as tf

Study = converters.Study

VOCAB_CC_ALL_100EXTRA_MODEL_FILE = 'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model'
VOCAB_CC_ALL_100EXTRA_SIZE_WITHOUT_INT = 32100
MAX_INTEGER_TOKENS = 1000


def get_vocabulary(
    sentencepiece_model_file: str = VOCAB_CC_ALL_100EXTRA_MODEL_FILE,
    max_integer_tokens: int = MAX_INTEGER_TOKENS,
    expected_vocab_size: Optional[int] = None) -> seqio.Vocabulary:
  """Create a vocabulary with extra integer tokens."""
  extra_tokens = ['<' + str(n) + '>' for n in range(max_integer_tokens)]
  vocabulary = vocabularies.SentencePieceVocabularyWithCustomToken(
      sentencepiece_model_file, extra_tokens=extra_tokens)
  if (expected_vocab_size is not None and
      expected_vocab_size != vocabulary.vocab_size):
    raise ValueError(f'Vocabulary size ({vocabulary.vocab_size}) does not '
                     f'match the expected value ({expected_vocab_size}).')
  return vocabulary


VOCABULARY_CC_ALL_100EXTRA = get_vocabulary()


def vizier_dataset_from_generator(
    study_generator: Callable[[], Iterator[Study]],
    study_converter: converters.Converter,
    converted_study_callback: Optional[Callable[[converters.ConvertedStudy],
                                                None]] = None,
    suppress_error: bool = True) -> tf.data.Dataset:
  """Build a tf.data Dataset of Vizier studies from a generator.

  Generate study with a generator.

  Args:
    study_generator: a study generator that supports the `iter()` protocol.
    study_converter: study converter.
    converted_study_callback: a callback function with the converted study.
    suppress_error: suppress errors and log a warning message instead.

  Returns:
    a tf.data.Dataset containing converted Vizier Study examples.
  """
  def _make_gen():
    # Make generated studies.
    for study in study_generator():
      # Some Study protos have deprecated field values and will cause a
      # ValueError in the converter.
      try:
        texts = study_converter.study_to_texts(study)
      except ValueError as e:
        if suppress_error:
          logging.warning(
              'Skipping a Study %s because of '
              'ValueError in the converter: %s.', study, e)
          continue
        else:
          raise e
      if not (texts.inputs and texts.target_inputs and texts.targets):
        if suppress_error:
          logging.warning(
              'Skipping a Study %s because '
              'inputs, target_inputs or targets are empty.', study)
          continue
        else:
          raise ValueError('A Study cannot be converted.')

      if converted_study_callback is not None:
        converted_study_callback(texts)

      yield dict(
          inputs=texts.inputs,
          target_inputs=texts.target_inputs,
          targets=texts.targets,
          num_parameters=texts.num_parameters)

  output_types = dict(
      inputs=tf.string,
      target_inputs=tf.string,
      targets=tf.string,
      num_parameters=tf.int32)
  output_shapes = dict(
      inputs=(), target_inputs=(), targets=(), num_parameters=())
  return tf.data.Dataset.from_generator(
      generator=_make_gen,
      output_types=output_types,
      output_shapes=output_shapes)


def build_vizier_dataset_from_generator(
    split: str,
    shuffle_files: bool = False,
    seed: Optional[int] = None,
    study_generator: Optional[Callable[[], Study]] = None,
    study_converter: Optional[converters.Converter] = None) -> tf.data.Dataset:
  """A wrapper of vizier_dataset_from_generator with additional inputs."""
  assert split in ['train', 'validation', 'test']
  del shuffle_files
  del seed
  if study_generator is None:
    raise ValueError('study_generator must not be None.')
  if study_converter is None:
    raise ValueError('study_converter must not be None.')

  def _generator():
    while True:
      yield study_generator()
  return vizier_dataset_from_generator(_generator, study_converter)


class DatasetFromStudy(object):
  """Build a dataset to generate data from a study list.

  It adds a task that generates converted studies from `self._study_list` as a
  placehold. Every call to `dataset` will update `self._study_list` and return
  a dataset with the updated content.

  This class is useful for running inference with a list of given studies.
  """

  def __init__(self,
               add_tasks_fn: Optional[Callable[..., Any]],
               study_converter: converters.Converter,
               feature_converter_cls: Callable[..., seqio.FeatureConverter],
               task_feature_lengths: Mapping[str, int],
               batch_size: int,
               task_name: str = 'task_from_study'):
    """Construct a generator.

    Args:
      add_tasks_fn: callable to add a seqio task to the registry, with keyword
        arguments name, and dataset_fn. It is intended to be a partially
        configured method of `add_tasks`.
      study_converter: study converter.
      feature_converter_cls: feature converter class.
      task_feature_lengths: a mapping from feature name to length.
      batch_size: batch size.
      task_name: seqio task name.
    """
    self._study_converter = study_converter
    self._study_list: List[Study] = []
    self._aux_list = []
    self._max_aux_size = 0
    self._batch_size = batch_size

    # Add a task and get dataset.
    add_tasks_fn(name=task_name, dataset_fn=self._dataset_fn)
    feature_converter = feature_converter_cls(pack=False)
    ds = seqio.get_dataset(
        mixture_or_task_name=task_name,
        task_feature_lengths=task_feature_lengths,
        feature_converter=feature_converter)
    ds = ds.batch(batch_size, drop_remainder=False)
    self._dataset: tf.data.Dataset = ds

  def dataset(self,
              study_list: Optional[Sequence[Study]] = None) -> tf.data.Dataset:
    """Returns the TF dataset with the study list updated if specified.

    Note that a new call to `dataset` will change the content of the study list
    from previous calls.

    Args:
      study_list: list of study protocol buffers to update if not None.

    Returns:
      A TF dataset containing the list of studies.
    """
    if study_list is not None:
      self._study_list.clear()
      self._study_list.extend(list(study_list))
      self._aux_list.clear()
      self._max_aux_size = len(study_list)
    return self._dataset

  @property
  def study_converter(self) -> converters.Converter:
    return self._study_converter

  @property
  def study_aux_list(self) -> List[converters.Aux]:
    return self._aux_list

  @property
  def batch_size(self) -> int:
    return self._batch_size

  def _dataset_fn(
      self,
      split: str,
      shuffle_files: bool = False,
      seed: Optional[int] = None,
      ) -> tf.data.Dataset:
    """Build a tf.data Dataset of Vizier studies given split.

    Read data from self._study_list and convert with a python generator.
    Arguments split, shuffle_files and seed are ignored.

    Args:
      split: 'train', 'test', or 'validation'. The split to load. Ignored.
      shuffle_files: whether to shuffle. This should be False. Ignored.
      seed: random seed. Ignored.

    Returns:
      a tf.data.Dataset containing converted Vizier Study examples.
    """
    del split, shuffle_files, seed

    def _generator():
      for study in self._study_list:
        yield study

    def _callback(texts: converters.ConvertedStudy):
      """Called after a study is converted."""
      if len(self._aux_list) <= self._max_aux_size:
        self._aux_list.append(texts.aux)

    return vizier_dataset_from_generator(
        study_generator=_generator,
        study_converter=self._study_converter,
        converted_study_callback=_callback,
        suppress_error=False)


class FakeDataSource(seqio.DataSource):
  """A fake `DataSource` to load cached dataset."""

  def __init__(
      self,
      split_to_num_shards: Dict[str, int],
      splits: Iterable[str],
      num_input_examples: Optional[Mapping[str, int]] = None,
  ):
    """FakeDataSource constructor."""
    self._split_to_num_shards = split_to_num_shards
    super().__init__(
        splits=splits,
        num_input_examples=num_input_examples,
        caching_permitted=True,
    )

  @property
  def supports_arbitrary_sharding(self) -> bool:
    return False

  def get_dataset(
      self,
      split: str = 'train',
      shuffle: bool = True,
      seed: Optional[int] = None,
      shard_info: Optional[seqio.ShardInfo] = None,
      *,
      sequence_length: Optional[Mapping[str, int]] = None,  # Unused
      use_cached: bool = False,  # Unused
      num_epochs: Optional[int] = 1,  # Unused
  ) -> tf.data.Dataset:
    raise NotImplementedError

  def list_shards(self, split: str) -> Sequence[str]:
    return [str(i) for i in range(self._split_to_num_shards[split])]


# Wrap a dataset_fn to match the required positional arguments.
# Gin configured functions have no positional arguments but the dataset_fn
# argument of t5.data.TaskRegistry.add requires positional arguments of split,
# shuffle_files, seed.
def _wrap_dataset_fn(dataset_fn: Callable[..., tf.data.Dataset]
                     ) -> t5.data.DatasetFnCallable:
  def wrapped_fn(
      split: str,
      shuffle_files: bool = False,
      seed: Optional[int] = None,
  ) -> tf.data.Dataset:
    return dataset_fn(split=split, shuffle_files=shuffle_files, seed=seed)
  return wrapped_fn


def add_tasks(
    name: str,
    vocabulary: t5.data.Vocabulary = VOCABULARY_CC_ALL_100EXTRA,
    masked_types: Optional[Sequence[str]] = None,
    num_initial_tokens: int = 1,
    add_eos_in_targets: bool = False,
    dataset_fn: Optional[t5.data.DatasetFnCallable] = None,
    splits: Sequence[str] = ('train', 'validation', 'test'),
    source: Optional[seqio.DataSource] = None,
    supports_caching: bool = False,
):
  """Creates a task."""
  masked_types = masked_types or []

  if (dataset_fn, source).count(None) != 1:
    raise ValueError(
        'Exactly one of either `dataset_fn` or `source` must be provided.')
  splits = splits if dataset_fn else None

  output_features = {
      'inputs':
          t5.data.Feature(vocabulary=vocabulary, add_eos=False),
      'targets':
          t5.data.Feature(vocabulary=vocabulary, add_eos=add_eos_in_targets),
      'target_inputs':
          t5.data.Feature(vocabulary=vocabulary, add_eos=add_eos_in_targets),
      'targets_types':
          t5.data.Feature(
              vocabulary=t5.data.PassThroughVocabulary(size=4), add_eos=False),
      'targets_masks':
          t5.data.Feature(
              vocabulary=t5.data.PassThroughVocabulary(size=2), add_eos=False),
  }

  task_name = name
  dataset_fn_ = _wrap_dataset_fn(dataset_fn) if dataset_fn else None

  t5.data.TaskRegistry.add(
      name=task_name,
      task_cls=t5.data.FunctionTask,
      dataset_fn=dataset_fn_,
      splits=splits,
      source=source,
      text_preprocessor=[],
      output_features=output_features,
      token_preprocessor=[
          functools.partial(
              preprocessors.add_targets_types,
              num_initial_tokens=num_initial_tokens),
          functools.partial(
              preprocessors.add_targets_masks, masked_types=masked_types),
      ],
      metric_fns=[],
      supports_caching=supports_caching,
      shuffle_buffer_size=100000,
  )


################################################################################
# Add cached tasks with a fake data source. These tasks only load data from the
# cached TFRecord files.

NUM_INITIAL_TOKENS = 1
DATASET_SPLIT_TO_NUM_SHARDS = {
    'cached_bbob_gp_train': {
        'train': 697,
    },
    'cached_bbob_gp_eval': {
        'validation': 64,
        'test': 64,
    },
    'cached_bbob_train': {
        'train': 2695,
    },
    'cached_bbob_train_repeat_4': {
        'train': 10780,
    },
    'cached_bbob_eval': {
        'validation': 64,
        'test': 64,
    },
    'cached_hpob_gp_train': {
        'train': 1000,
    },
    'cached_hpob_gp_train_repeat_30': {
        'train': 30000,
    },
    'cached_hpob_gp_eval': {
        'validation': 1000,
        'test': 1000,
    },
    'cached_hpob_train': {
        'train': 3000,
    },
    'cached_hpob_eval': {
        'validation': 2000,
        'test': 2000,
    },
}
# Sweep dataset.
for dataset in ['bbob', 'bbob_gp', 'hpob', 'hpob_gp']:
  # Sweep train/eval mode.
  for mode in ['train', 'eval']:
    splits_ = ('train',) if mode == 'train' else ('validation', 'test')

    # We provide cached training datasets for bbob and hpob_gp that have
    # multiple epochs with random data augmentation to support training for 100K
    # steps using a batch size of 256 without duplication.
    if dataset == 'bbob' and mode == 'train':
      repeats = [1, 4]
    elif dataset == 'hpob_gp' and mode == 'train':
      repeats = [1, 30]
    else:
      repeats = [1]

    for repeat_ in repeats:
      task_name_ = f'cached_{dataset}_{mode}'
      if repeat_ > 1:
        task_name_ += f'_repeat_{repeat_}'
      add_tasks(
          name=task_name_,
          vocabulary=VOCABULARY_CC_ALL_100EXTRA,
          source=FakeDataSource(
              split_to_num_shards=DATASET_SPLIT_TO_NUM_SHARDS[task_name_],
              splits=splits_,
          ),
          masked_types=['separator'],  # No separators prediction in loss.
          num_initial_tokens=NUM_INITIAL_TOKENS,
          add_eos_in_targets=False,
          supports_caching=True,
      )
