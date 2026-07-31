"""Structured export for the compact master results matrix."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from .metrics import compute_alpha, compute_gamma, compute_weighted_alpha
from .runner import BASELINE_KEY, CombinationResult

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)


def export_combination_result(
	result: CombinationResult,
	base_accuracy: float,
	base_training_time: float,
) -> dict:
	"""Convert one pipeline result to an exportable mapping.

	Args:
		result: A single CombinationResult from the pipeline.
		base_accuracy: Baseline accuracy for α computation.
		base_training_time: Baseline training time for γ computation.

	Returns:
		Dict entry suitable for inclusion in the master results matrix.
	"""
	alpha = compute_alpha(result.accuracy, base_accuracy)
	gamma = compute_gamma(result.training_time, base_training_time)
	weighted_alpha = compute_weighted_alpha(alpha, gamma)

	return {
		"combo_key": result.combo_key,
		"regime": result.regime,
		"transforms": list(result.transforms),
		"accuracy": result.accuracy,
		"alpha": alpha,
		"gamma": gamma,
		"weighted_alpha": weighted_alpha,
		"confusion_matrix": dict(result.confusion_matrix),
		"training_time_seconds": result.training_time,
	}


def build_results_matrix(
	all_results: dict[str, dict[str, CombinationResult]],
	output_dir: Path,
) -> None:
	"""Export all results and write the master results_matrix.json.

	For each preprocessing regime, the Baseline entry is used to derive
	base_accuracy and base_training_time for α/γ/wα computations.

	Args:
		all_results: Nested dict {regime: {combo_key: CombinationResult}}.
		output_dir: Root output directory.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)
	matrix_entries: list[dict] = []

	for regime, combo_results in all_results.items():
		baseline = combo_results.get(BASELINE_KEY)
		if baseline is None:
			logger.warning(
				"No baseline found for preprocessing regime %s; skipping α/γ/wα",
				regime,
			)
			base_accuracy = 0.0
			base_training_time = 1.0
		else:
			base_accuracy = baseline.accuracy
			base_training_time = baseline.training_time

		for result in combo_results.values():
			entry = export_combination_result(
				result=result,
				base_accuracy=base_accuracy,
				base_training_time=base_training_time,
			)
			matrix_entries.append(entry)

	matrix_path = output_dir / "results_matrix.json"
	matrix_path.write_text(json.dumps(matrix_entries, indent=2), encoding="utf-8")

	logger.info(
		"Results matrix written: %s (%d entries)",
		matrix_path,
		len(matrix_entries),
	)
