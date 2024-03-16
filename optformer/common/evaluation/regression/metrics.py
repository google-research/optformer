# Copyright 2024 Google LLC.
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

"""Metrics for evaluating a surrogate. Follows t5.evaluation.metrics API."""

import numpy as np
import scipy


def _top_k(targets: np.ndarray, preds: np.ndarray, k: int) -> float:
  """See Section 5.1 of https://arxiv.org/pdf/2308.13490.pdf."""
  # TODO: Flip targets and predictions if using maximization.
  sorted_indices = np.argsort(preds, axis=-1)
  k_indices = sorted_indices[..., :k]
  best_pred_k_targets = targets[k_indices]
  return np.min(best_pred_k_targets) / np.min(targets) - 1.0


def _top_k_errors(targets: np.ndarray, preds: np.ndarray) -> dict[str, float]:
  return {
      "top_k=1": _top_k(targets, preds, k=1),
      "top_k=5": _top_k(targets, preds, k=5),
      "top_k=10": _top_k(targets, preds, k=10),
      "top_k=100": _top_k(targets, preds, k=100),
  }


def _relative_errors(
    targets: np.ndarray, preds: np.ndarray
) -> dict[str, float]:
  maxmin_gap = np.max(targets) - np.min(targets)

  normalized_targets = targets / maxmin_gap
  normalized_preds = preds / maxmin_gap

  l1_diff = np.abs(normalized_targets - normalized_preds)
  sq_diff = np.square(normalized_targets - normalized_preds)

  return {
      "relative_mean_abs_error": np.mean(l1_diff),
      "relative_median_abs_error": np.median(l1_diff),
      "relative_max_abs_error": np.max(l1_diff),
      "relative_min_abs_error": np.min(l1_diff),
      "relative_root_mean_sq_error": np.sqrt(np.mean(sq_diff)),
  }


def _ranking_metrics(
    targets: np.ndarray, preds: np.ndarray
) -> dict[str, float]:
  return {
      "kendall": scipy.stats.kendalltau(targets, preds).correlation,
      "spearman_corrcoef": 100 * scipy.stats.spearmanr(targets, preds)[0],
      # "pearson_corrcoef": 100 * scipy.stats.pearsonr(targets, preds)[0],
  }


def evaluate_metrics(
    ys_test: np.ndarray, ys_pred: np.ndarray
) -> dict[str, float | np.ndarray]:
  out = {}
  out.update(_relative_errors(ys_test, ys_pred))
  out.update(_ranking_metrics(ys_test, ys_pred))
  out.update(_top_k_errors(ys_test, ys_pred))
  out.update({"ys_test": ys_test, "ys_pred": ys_pred})
  return out
