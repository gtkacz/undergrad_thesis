"""Hyperparameter grid search for individual preprocessing transforms."""

from __future__ import annotations

import itertools
import json
import logging
import operator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
	from collections.abc import Callable

from util.cnn import configure_cuda
from util.config import TrainingConfig, load_configs
from util.dataset import get_data_loaders
from util.preprocessing import ColorSpaceTransform, DenoiseTransform, NormalizeTransform
from util.runner import evaluate_model

logger = logging.getLogger(__name__)

PARAMS_DIR = Path("params")


def _grid_search(
	name: str,
	param_grid: list[dict],
	build_transforms: Callable[[dict], list[nn.Module]],
	training_config: TrainingConfig,
	device: torch.device,
) -> list[dict]:
	"""Run a grid search over a parameter space for a single transform type.

	Args:
		name: Human-readable name for logging (e.g. "Colorspace").
		param_grid: List of parameter dicts.  Each dict's keys become
			columns in the output alongside accuracy/confusion_matrix/training_time.
		build_transforms: Callable that takes one param dict and returns
			the GPU transform list to evaluate.
		training_config: Training hyperparameters.
		device: Target device.

	Returns:
		Sorted list of result dicts (highest accuracy first).
	"""
	results: list[dict] = []
	train_loader, test_loader, val_loader = get_data_loaders(training_config)

	for params in param_grid:
		logger.info("%s: %s", name, params)
		gpu_transforms = build_transforms(params)

		accuracy, confusion_matrix, training_time, *_ = evaluate_model(
			device=device,
			train_loader=train_loader,
			test_loader=test_loader,
			validation_loader=val_loader,
			training_config=training_config,
			verbose=False,
			gpu_transforms=gpu_transforms,
		)

		results.append({
			**params,
			"accuracy": accuracy,
			"confusion_matrix": dict(confusion_matrix),
			"training_time": training_time,
		})

		torch.cuda.empty_cache()

	results.sort(key=operator.itemgetter("accuracy"), reverse=True)
	return results


def main() -> None:
	"""Run all hyperparameter grid searches and save results."""
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)

	configure_cuda()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.info("Using device: %s", device)

	training_config, _ = load_configs()

	PARAMS_DIR.mkdir(exist_ok=True)

	logger.info("=== Colorspace grid search ===")
	cs_results = _grid_search(
		name="Colorspace",
		param_grid=[{"target_space": s} for s in ["BGR", "HSV", "LAB", "YUV"]],
		build_transforms=lambda p: [ColorSpaceTransform(source_space="RGB", target_space=p["target_space"])],
		training_config=training_config,
		device=device,
	)
	(PARAMS_DIR / "colorspace.json").write_text(json.dumps(cs_results, indent=2), encoding="utf-8")

	logger.info("=== Denoise grid search ===")
	d_results = _grid_search(
		name="Denoise",
		param_grid=[
			{"kernel_size": kernel_size, "sigma_space": sigma_space, "sigma_color": 10}
			for kernel_size, sigma_space in itertools.product(
				[3, 5, 7, 9, 11],
				range(15, 26),
			)
		],
		build_transforms=lambda p: [
			DenoiseTransform(
				kernel_size=p["kernel_size"],
				sigma_space=p["sigma_space"],
				sigma_color=p["sigma_color"],
			),
		],
		training_config=training_config,
		device=device,
	)
	(PARAMS_DIR / "denoise.json").write_text(json.dumps(d_results, indent=2), encoding="utf-8")

	logger.info("=== Normalize grid search ===")
	n_results = _grid_search(
		name="Normalize",
		param_grid=[
			{"mean": round(float(m), 2), "std": round(float(s), 2)}
			for m, s in itertools.product(np.arange(0.15, 1, 0.15), np.arange(0.15, 1, 0.15))
		],
		build_transforms=lambda p: [NormalizeTransform(mean=[p["mean"]], std=[p["std"]])],
		training_config=training_config,
		device=device,
	)
	(PARAMS_DIR / "normalize.json").write_text(json.dumps(n_results, indent=2), encoding="utf-8")

	logger.info("Grid search complete. Results written to %s/", PARAMS_DIR)


if __name__ == "__main__":
	main()
