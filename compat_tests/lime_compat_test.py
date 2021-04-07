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

import collections
import functools
from absl.testing import absltest
from absl.testing import parameterized
from citrus_nlp import lime
from citrus_nlp import utils
from lime import lime_text  # This is the original third party LIME.
import numpy as np
from scipy import special
from scipy import stats


class LimeCompatTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'for_2d_input',
          'sentence': ' '.join(list('abcdefghijklmnopqrstuvwxyz')),
          'num_samples': 5000,
          'num_classes': 2,
          'class_to_explain': 1,
      }, {
          'testcase_name': 'for_3d_input',
          'sentence': ' '.join(list('abcdefghijklmnopqrstuvwxyz')),
          'num_samples': 5000,
          'num_classes': 3,
          'class_to_explain': 2,
      })
  def test_explain_matches_original_lime(self, sentence, num_samples,
                                         num_classes, class_to_explain):
    """Tests if Citrus LIME matches the original implementation."""

    # Assign some weight to each token a-z.
    # Each token contributes positively/negatively to the prediction.
    rs = np.random.RandomState(seed=0)
    token_weights = {token: rs.normal() for token in sentence.split()}
    token_weights[lime.DEFAULT_MASK_TOKEN] = 0.

    def _predict_fn(sentences):
      """Mock prediction function."""
      rs = np.random.RandomState(seed=0)
      predictions = []
      for sentence in sentences:
        probs = rs.normal(0., 0.1, size=num_classes)
        # To check if LIME finds the right positive/negative correlations.
        for token in sentence.split():
          probs[class_to_explain] += token_weights[token]
        predictions.append(probs)
      return np.stack(predictions, axis=0)

    # Explain the prediction using Citrus LIME.
    explanation = lime.explain(
        sentence,
        _predict_fn,
        class_to_explain=class_to_explain,
        num_samples=num_samples,
        tokenizer=str.split,
        mask_token=lime.DEFAULT_MASK_TOKEN,
        kernel=functools.partial(
            lime.exponential_kernel, kernel_width=lime.DEFAULT_KERNEL_WIDTH))
    scores = explanation.feature_importance  # <float32>[seq_len]
    scores = utils.normalize_scores(scores, make_positive=False)

    # Explain the prediction using original LIME.
    original_lime_explainer = lime_text.LimeTextExplainer(
        class_names=map(str, np.arange(num_classes)),
        mask_string=lime.DEFAULT_MASK_TOKEN,
        kernel_width=lime.DEFAULT_KERNEL_WIDTH,
        split_expression=str.split,
        bow=False)
    num_features = len(sentence.split())
    original_explanation = original_lime_explainer.explain_instance(
        sentence,
        _predict_fn,
        labels=(class_to_explain,),
        num_features=num_features,
        num_samples=num_samples)

    # original_explanation.local_exp is a dict that has a key class_to_explain,
    # which gives a sequence of (index, score) pairs.
    # We convert it to an array <float32>[seq_len] with a score per position.
    original_scores = np.zeros(num_features)
    for index, score in original_explanation.local_exp[class_to_explain]:
      original_scores[index] = score
    original_scores = utils.normalize_scores(
        original_scores, make_positive=False)

    # Test that Citrus LIME and original LIME match.
    np.testing.assert_allclose(scores, original_scores, atol=0.01)


if __name__ == '__main__':
  absltest.main()
