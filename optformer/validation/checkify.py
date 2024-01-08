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

"""Checkify module."""

from typing import Callable, TypeVar
import jax
from jax.experimental import checkify


_ENABLED: bool = False  # global variable


def enable_checks(enable: bool) -> None:
  global _ENABLED
  _ENABLED = enable


def enabled() -> bool:
  return _ENABLED


_R = TypeVar('_R')


def _jittable_inner(
    fn: Callable[..., _R], *args, **kwargs
) -> tuple[checkify.Error, _R]:
  """This function is used to avoid retracing."""
  return checkify.checkify(fn)(*args, **kwargs)


def check_and_jit(fn: Callable[..., _R]) -> Callable[..., _R]:
  """Throws checkify errors while preserving the function signature."""

  def inner(*args, **kwargs) -> _R:
    err, result = jax.jit(_jittable_inner, static_argnums=[0])(
        fn, *args, **kwargs
    )
    err.throw()
    return result

  return inner
