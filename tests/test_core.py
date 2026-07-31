"""Regression tests for the experiment and statistical helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.compute_clinical_metrics import _mean_std, _metrics_from_confusion
from src.util.config import load_configs
from src.util.dataset import split_datasets
from src.util.export import build_results_matrix
from src.util.metrics import compute_alpha, compute_gamma, compute_weighted_alpha
from src.util.runner import CombinationResult, _generate_all_combinations
from src.util.statistics import holm_bonferroni, paired_sign_flip_test, permutation_test

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


class ExperimentDesignTests(unittest.TestCase):
	"""Check invariants that define the 65-pipeline experiment."""

	def test_generates_all_65_ordered_pipelines(self) -> None:
		transforms = tuple(nn.Identity() for _ in range(4))
		combinations = _generate_all_combinations(transforms)
		self.assertEqual(len(combinations), 65)
		self.assertEqual(
			[sum(len(combo) == length for combo in combinations) for length in range(5)], [1, 4, 12, 24, 24]
		)

	def test_split_is_seed_deterministic_and_uses_requested_sizes(self) -> None:
		dataset = TensorDataset(torch.arange(20))
		first = split_datasets(dataset, 0.8, 0.1, 0.1, seed=42)
		second = split_datasets(dataset, 0.8, 0.1, 0.1, seed=42)

		self.assertEqual([len(split) for split in first], [16, 2, 2])
		self.assertEqual([split.indices for split in first], [split.indices for split in second])

	def test_split_rejects_invalid_ratios(self) -> None:
		dataset = TensorDataset(torch.arange(10))
		with self.assertRaises(ValueError):
			split_datasets(dataset, 0.8, 0.2, 0.2)

	def test_primary_config_matches_reported_parameters(self) -> None:
		training, preprocess = load_configs(REPOSITORY_ROOT / "src" / "parameters.toml")

		self.assertEqual(training.num_epochs, 3)
		self.assertEqual(training.resize_dim, 128)
		self.assertEqual((preprocess.normalize.mean, preprocess.normalize.std), (0.4, 0.2))
		self.assertEqual(
			(
				preprocess.denoise.kernel_size,
				preprocess.denoise.sigma_space,
				preprocess.denoise.sigma_color,
			),
			(5, 19, 10),
		)
		self.assertEqual(
			(preprocess.colorspace.source_space, preprocess.colorspace.target_space),
			("RGB", "HSV"),
		)

	def test_export_writes_only_the_compact_results_matrix(self) -> None:
		confusion = {"TP": 6, "TN": 3, "FP": 1, "FN": 0}
		baseline = CombinationResult("Baseline", (), 0.9, confusion, 10.0, "base")
		equalized = CombinationResult(
			"EqualizationTransform",
			("EqualizationTransform",),
			1.0,
			{"TP": 6, "TN": 4, "FP": 0, "FN": 0},
			12.0,
			"base",
		)

		with tempfile.TemporaryDirectory() as temporary_directory:
			output_directory = Path(temporary_directory)
			build_results_matrix(
				{"base": {"Baseline": baseline, "EqualizationTransform": equalized}},
				output_directory,
			)
			matrix = json.loads((output_directory / "results_matrix.json").read_text(encoding="utf-8"))

			self.assertEqual(len(matrix), 2)
			self.assertEqual([path.name for path in output_directory.iterdir()], ["results_matrix.json"])


class MetricTests(unittest.TestCase):
	"""Check the paper's alpha and cost-weighting definitions."""

	def test_alpha_gamma_and_weighted_alpha(self) -> None:
		self.assertAlmostEqual(compute_alpha(0.9, 0.8), 0.1)
		self.assertAlmostEqual(compute_gamma(12.0, 10.0), 1.2)
		self.assertAlmostEqual(compute_weighted_alpha(0.1, 1.2), 1 / 12)
		self.assertAlmostEqual(compute_weighted_alpha(-0.1, 1.2), -0.12)

	def test_clinical_metrics_use_diseased_as_the_positive_class(self) -> None:
		metrics = _metrics_from_confusion(tp=8, tn=9, fp=1, fn=2)

		self.assertAlmostEqual(metrics["accuracy"], 0.85)
		self.assertAlmostEqual(metrics["sensitivity"], 0.8)
		self.assertAlmostEqual(metrics["specificity"], 0.9)
		self.assertAlmostEqual(metrics["balanced_accuracy"], 0.85)

	def test_clinical_metric_dispersion_uses_sample_standard_deviation(self) -> None:
		mean, standard_deviation = _mean_std([1.0, 2.0, 3.0])

		self.assertEqual(mean, 2.0)
		self.assertEqual(standard_deviation, 1.0)


class StatisticsTests(unittest.TestCase):
	"""Protect corrected p-value and resampling behavior."""

	def test_holm_adjusted_values_are_monotone(self) -> None:
		results = holm_bonferroni([("a", 0.01), ("b", 0.03), ("c", 0.04)])
		self.assertEqual([result.test_name for result in results], ["a", "b", "c"])
		self.assertEqual([round(result.corrected_p, 8) for result in results], [0.03, 0.06, 0.06])
		self.assertEqual([result.significant for result in results], [True, False, False])

	def test_monte_carlo_permutation_p_value_is_never_zero(self) -> None:
		result = permutation_test(
			np.array([10.0, 11.0, 12.0]),
			np.array([-10.0, -11.0, -12.0]),
			n_permutations=100,
		)
		self.assertGreaterEqual(result.p_value, 1 / 101)

	def test_paired_sign_flip_uses_seed_pairs(self) -> None:
		result = paired_sign_flip_test(
			np.ones(5),
			np.zeros(5),
		)
		self.assertEqual(result.n_permutations, 32)
		self.assertAlmostEqual(result.p_value, 0.0625)


if __name__ == "__main__":
	unittest.main()
