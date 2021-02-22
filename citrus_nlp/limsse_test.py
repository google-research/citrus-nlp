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

"""Tests for language.google.xnlp.citrus.limsse."""
from absl.testing import absltest
from absl.testing import parameterized
from citrus_nlp import limsse
import numpy as np
from scipy import special


class LimsseTest(parameterized.TestCase):

  @parameterized.named_parameters({
      'testcase_name':
          'returns_correct_masks',
      'start_positions':
          np.array([8, 3, 0]),
      'lengths':
          np.array([5, 1, 10]),
      'sequence_length':
          10,
      'expected':
          np.array([
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # n-gram out of bounds.
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          ])
  })
  def test_get_masks(self, start_positions, lengths, sequence_length, expected):
    """Tests constructing a binary mask from start positions and lengths."""
    masks = limsse.get_masks(start_positions, lengths, sequence_length)
    np.testing.assert_array_equal(expected, masks)

  @parameterized.named_parameters(
      {
          'testcase_name': 'works_with_min_length_same_as_max_length',
          'sequence_length': 10,
          'ngram_min_length': 1,
          'ngram_max_length': 1,
          'num_samples': 100
      },
      {
          'testcase_name': 'works_with_defaults_from_the_limsse_paper',
          'sequence_length': 10,
          'ngram_min_length': 1,
          'ngram_max_length': 6,
          'num_samples': 100
      },
      {
          'testcase_name': 'works_with_ngrams_that_go_out_of_bounds',
          'sequence_length': 1,
          'ngram_min_length': 2,
          'ngram_max_length': 3,
          'num_samples': 1
      },
  )
  def test_sample(self, sequence_length, ngram_min_length, ngram_max_length,
                  num_samples):
    """Tests sampling starting positions and lengths."""
    start_positions, lengths = limsse.sample(sequence_length, ngram_min_length,
                                             ngram_max_length, num_samples)
    self.assertEqual((num_samples,), start_positions.shape)
    self.assertEqual((num_samples,), lengths.shape)
    self.assertGreaterEqual(np.min(start_positions), 0)
    self.assertLess(np.max(start_positions), sequence_length)
    self.assertGreaterEqual(np.min(lengths), ngram_min_length)
    self.assertLessEqual(np.max(lengths), ngram_max_length)

  @parameterized.named_parameters({
      'testcase_name':
          'returns_correct_substrings',
      'sentence':
          'It is a great movie but also somewhat bad .',
      'start_positions':
          np.array([8, 3, 0]),
      'lengths':
          np.array([5, 1, 10]),
      'expected': [
          'bad .', 'great', 'It is a great movie but also somewhat bad .'
      ]
  })
  def test_extract_ngrams(self, sentence, start_positions, lengths, expected):
    """Tests extracting n-grams from a token sequence."""
    tokens = sentence.split()
    ngrams = list(limsse.extract_ngrams(tokens, start_positions, lengths))
    self.assertEqual(expected, ngrams)

  @parameterized.named_parameters(
      {
          'testcase_name': 'correctly_identifies_important_tokens_for_1d_input',
          'sentence': 'It is a great movie but also somewhat bad .',
          'num_samples': 1000,
          'positive_token': 'great',
          'negative_token': 'bad',
          'ngram_min_length': 1,
          'ngram_max_length': 6,
          'num_classes': 1,
          'class_to_explain': 0,
      }, {
          'testcase_name': 'correctly_identifies_important_tokens_for_2d_input',
          'sentence': 'It is a great movie but also somewhat bad .',
          'num_samples': 1000,
          'positive_token': 'great',
          'negative_token': 'bad',
          'ngram_min_length': 1,
          'ngram_max_length': 6,
          'num_classes': 2,
          'class_to_explain': 1,
      }, {
          'testcase_name': 'correctly_identifies_important_tokens_for_3d_input',
          'sentence': 'It is a great movie but also somewhat bad .',
          'num_samples': 1000,
          'positive_token': 'great',
          'negative_token': 'bad',
          'ngram_min_length': 1,
          'ngram_max_length': 6,
          'num_classes': 3,
          'class_to_explain': 2,
      })
  def test_explain(self, sentence, num_samples, positive_token, negative_token,
                   ngram_min_length, ngram_max_length, num_classes,
                   class_to_explain):
    """Tests explaining a binary classifier with scalar output."""

    def _predict_fn(sentences):
      """Mock prediction function."""
      predictions = []
      for sentence in sentences:
        probs = np.random.uniform(0., 1., num_classes)
        # To check if LIMSSE finds the right positive/negative correlations.
        if negative_token in sentence:
          probs[class_to_explain] = probs[class_to_explain] - 1.
        if positive_token in sentence:
          probs[class_to_explain] = probs[class_to_explain] + 1.
        predictions.append(probs)

      predictions = np.stack(predictions, axis=0)
      if num_classes == 1:
        predictions = special.expit(predictions)
      else:
        predictions = special.softmax(predictions, axis=-1)
      return predictions

    explanation = limsse.explain(
        sentence,
        _predict_fn,
        class_to_explain,
        ngram_min_length=ngram_min_length,
        ngram_max_length=ngram_max_length,
        num_samples=num_samples,
        tokenizer=str.split)

    self.assertLen(explanation.feature_importance, len(sentence.split()))

    # The positive word should have the highest attribution score.
    positive_token_idx = sentence.split().index(positive_token)
    self.assertEqual(positive_token_idx,
                     np.argmax(explanation.feature_importance))

    # The negative word should have the lowest attribution score.
    negative_token_idx = sentence.split().index(negative_token)
    self.assertEqual(negative_token_idx,
                     np.argmin(explanation.feature_importance))

  def test_explain_returns_explanation_with_intercept(self):
    """Tests if the explanation contains an intercept value."""

    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    explanation = limsse.explain('Test sentence', _predict_fn, 1, num_samples=5)
    self.assertNotEqual(explanation.intercept, 0.)

  def test_explain_returns_explanation_with_model(self):
    """Tests if the explanation contains the model."""

    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    explanation = limsse.explain(
        'Test sentence',
        _predict_fn,
        class_to_explain=1,
        num_samples=5,
        return_model=True)
    self.assertIsNotNone(explanation.model)

  def test_explain_returns_explanation_with_score(self):
    """Tests if the explanation contains a linear model score."""

    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    explanation = limsse.explain(
        'Test sentence',
        _predict_fn,
        class_to_explain=1,
        num_samples=5,
        return_score=True)
    self.assertIsNotNone(explanation.score)

  def test_explain_returns_explanation_with_prediction(self):
    """Tests if the explanation contains a prediction."""

    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    explanation = limsse.explain(
        'Test sentence',
        _predict_fn,
        class_to_explain=1,
        num_samples=5,
        return_prediction=True)
    self.assertIsNotNone(explanation.prediction)


if __name__ == '__main__':
  absltest.main()
