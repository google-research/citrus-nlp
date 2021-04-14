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

"""Local Interpretable Model-agnostic Substring-based Explanations (LIMSSE).

LIMSSE was proposed in the following paper:

> Evaluating neural network explanation methods using hybrid documents and
> morphosyntactic agreement
> Nina Poerner, Hinrich Schütze, Benjamin Roth
> https://www.aclweb.org/anthology/P18-1032/

LIMSSE explains text classifiers by returning a feature attribution score
for each input token. It works as follows:

1) Sample n-grams from the input text.
2) Get predictions from the model for those n-grams. Use these as labels.
3) Fit a linear model to associate the input positions covered by the n-grams
   with the resulting predicted label.

The resulting feature importance scores are the linear model coefficients for
the class that the original model predicted on the input text.
"""
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Tuple
from citrus_nlp import utils
import numpy as np
from sklearn import linear_model

DEFAULT_MIN_NGRAM_LENGTH = 1
DEFAULT_MAX_NGRAM_LENGTH = 6
DEFAULT_NUM_SAMPLES = 3000
DEFAULT_SOLVER = 'cholesky'


def get_masks(start_positions: np.ndarray, lengths: np.ndarray,
              sequence_length: int) -> np.ndarray:
  """Returns a binary mask over input sequence positions.

  The mask is based on the starting positions and lengths of the n-grams.
  The positions where the n-gram is located have 1s, other positions have 0s.

  Args:
    start_positions: The (sampled) starting positions of the n-grams.
      <int>[num_samples]
    lengths: The (sampled) lengths of the n-grams. <int>[num_samples]
    sequence_length: The length of the original sequence.

  Returns:
    The masks <int>[start_position.size, sequence_length].
  """
  assert start_positions.shape == lengths.shape, \
      'start_positions and lengths should have the same shape.'
  if start_positions.ndim == 1:
    start_positions = start_positions[:, np.newaxis]

  if lengths.ndim == 1:
    lengths = lengths[:, np.newaxis]

  end_positions = start_positions + lengths
  positions = np.arange(sequence_length)[np.newaxis, :]
  return np.where((positions >= start_positions) & (positions < end_positions),
                  1, 0)


def sample(sequence_length: int,
           ngram_min_length: int = DEFAULT_MIN_NGRAM_LENGTH,
           ngram_max_length: int = DEFAULT_MAX_NGRAM_LENGTH,
           num_samples: int = DEFAULT_NUM_SAMPLES,
           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
  """Samples n-gram positions and lengths.

  Args:
    sequence_length: The length of the original sequence for which we sample.
    ngram_min_length: Minimum length of the sampled n-grams.
    ngram_max_length: Maximum length of the sampled n-grams.
    num_samples: The number of samples.
    seed: An optional random seed to make sampling deterministic.

  Returns:
    The starting positions <int>[num_samples], the lengths <int>[num_samples].
    The result of starting positions + lengths may exceed the sequence length,
    as `get_mask` will clip the requested length to the sequence boundary.
  """
  assert ngram_min_length > 0, 'ngram minimum length must be at least 1.'
  assert ngram_max_length >= ngram_min_length, \
      'ngram maximum length must be at least as great as ngram minimum length'
  rng = np.random.RandomState(seed)
  start_positions = rng.randint(0, sequence_length, size=num_samples)
  lengths = rng.randint(
      ngram_min_length, ngram_max_length + 1, size=num_samples)
  return start_positions, lengths


def extract_ngrams(tokens: Sequence[str], start_positions: np.ndarray,
                   lengths: np.ndarray) -> Iterator[str]:
  """Extracts n-grams from a sequence given n-gram positions and lengths."""
  for start, length in zip(start_positions, lengths):
    yield ' '.join(tokens[start:start + length])


def explain(
    sentence: str,
    predict_fn: Callable[[Iterable[str]], np.ndarray],
    class_to_explain: int,
    ngram_min_length: int = DEFAULT_MIN_NGRAM_LENGTH,
    ngram_max_length: int = DEFAULT_MAX_NGRAM_LENGTH,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    tokenizer: Any = str.split,
    alpha: float = 1.0,
    solver: str = DEFAULT_SOLVER,
    return_model: bool = False,
    return_score: bool = False,
    return_prediction: bool = False,
    seed: Optional[int] = None,
) -> utils.PosthocExplanation:
  """Returns the LIMSSE explanation for a given sentence.

  By default, this function returns an Explanation object containing feature
  importance scores. Optionally, more information can be returned, such as the
  linear model itself, the intercept, the score of the model on the perturbation
  set, and the prediction that the linear model makes on the original sentence.

  Args:
    sentence: An input to be explained.
    predict_fn: A prediction function that returns an array of probabilities
      given a list of inputs. The output shape is [len(inputs)] for binary
      classification with scalar output, and [len(inputs), num_classes] for
      multi-class classification.
    class_to_explain: The class ID to explain. E.g., 0 for binary classification
      with scalar output, and 1 for the positive class for two-class
      classification.
    ngram_min_length: The minimum length of sampled n-grams.
    ngram_max_length: The maximum length of sampled n-grams.
    num_samples: The number of n-grams to sample.
    tokenizer: A function that splits the input sentence into tokens.
    alpha: Regularization strength of the linear approximation model. See
      `sklearn.linear_model.Ridge` for details.
    solver: Solver to use in the linear approximation model. See
      `sklearn.linear_model.Ridge` for details.
    return_model: Returns the fitted linear model.
    return_score: Returns the score of the linear model on the perturbations.
      This is the R^2 of the linear model predictions w.r.t. their targets.
    return_prediction: Returns the prediction of the linear model on the full
      original sentence.
    seed: Optional random seed to make the explanation deterministic.

  Returns:
    The explanation for the requested label.
  """
  tokens = tokenizer(sentence)
  sequence_length = len(tokens)

  # Sample the starting positions and lengths of the n-grams.
  start_positions, lengths = sample(
      sequence_length,
      ngram_min_length=ngram_min_length,
      ngram_max_length=ngram_max_length,
      num_samples=num_samples + 1,  # Add one because masks[0] is not a sample.
      seed=seed)

  # Compute a binary mask based on the starting positions and lengths.
  # The mask contains 1s for positions covered by the n-gram, and 0s otherwise.
  masks = get_masks(start_positions, lengths, sequence_length)
  masks[0] = np.full_like(masks[0], True)  # First mask is the full sentence.

  # Get the n-grams as a list of strings.
  ngrams = extract_ngrams(tokens, start_positions, lengths)
  ngram_probs = predict_fn(ngrams)
  if len(ngram_probs.shape) > 1:
    assert class_to_explain is not None, \
        'class_to_explain needs to be set when `predict_fn` returns a 2D tensor'
    ngram_probs = ngram_probs[:, class_to_explain]  # Only care about 1 class.

  # Fit a linear model for the requested output class.
  model = linear_model.Ridge(
      alpha=alpha, solver=solver, random_state=seed).fit(masks, ngram_probs)

  explanation = utils.PosthocExplanation(
      feature_importance=model.coef_, intercept=model.intercept_)

  if return_model:
    explanation.model = model

  if return_score:
    explanation.score = model.score(masks, ngram_probs)

  if return_prediction:
    # masks[0] contains the full sentence (all positive mask) by convention.
    explanation.prediction = model.predict(np.expand_dims(
        masks[0], axis=0)).reshape([1, -1])

  return explanation
