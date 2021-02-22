# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for explaining text classifiers."""

from typing import Any, List, Optional
import attr
import numpy as np


@attr.s(auto_attribs=True)
class PosthocExplanation:
  """Represents a post-hoc explanation with feature importance scores.

  Attributes:
    feature_importance: Feature importance scores for each input feature. These
      are the coefficients of the linear model that was fitted to mimic the
      behavior of a (black-box) prediction function.
    intercept: The intercept of the fitted linear model. This is the independent
      term that is added to make a prediction.
    model: The fitted linear model. An explanation only contains this if it was
      explicitly requested from the explanation method.
    score: The R^2 score of the fitted linear model on the perturbations and
      their labels. This reflects how well the linear model was able to fit to
      the perturbation set.
    prediction: The prediction of the linear model on the full input sentence,
      i.e., an all-true boolean mask.
  """
  feature_importance: np.ndarray
  intercept: Optional[float] = None
  model: Optional[Any] = None
  score: Optional[float] = None
  prediction: Optional[float] = None


def normalize_scores(scores: np.ndarray, make_positive: bool = False):
  """Makes the absolute values sum to 1, optionally making them all positive."""
  scores = scores + np.finfo(np.float32).eps
  if make_positive:
    scores = np.abs(scores)
  return scores / np.abs(scores).sum(-1)
