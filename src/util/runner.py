"""Pipeline orchestrator: evaluate models by composing training + metrics."""

import itertools
import logging
from dataclasses import dataclass
from time import time as timer

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .cnn import BinaryCNN
from .config import PreprocessConfig, TrainingConfig
from .dataset import get_data_loaders
from .metrics import compute_test_metrics
from .preprocessing import (
	ColorSpaceTransform,
	DenoiseTransform,
	EqualizationTransform,
	NormalizeTransform,
)
from .training import train_epoch, validate_epoch
from .types import ConfusionMatrix

logger = logging.getLogger(__name__)

COMBO_SEPARATOR = " -> "
BASELINE_KEY = "Baseline"

_DEFAULT_CRITERION = nn.BCEWithLogitsLoss
_DEFAULT_DEVICE = torch.device("cpu")


def evaluate(
	model: nn.Module,
	optimizer: optim.Optimizer,
	train_loader: DataLoader,
	test_loader: DataLoader,
	validation_loader: DataLoader,
	criterion: nn.Module | None = None,
	device: torch.device | None = None,
	num_epochs: int = 10,
	verbose: bool = True,
	gpu_transforms: list[nn.Module] | None = None,
) -> tuple[float, ConfusionMatrix, float, nn.Module]:
	"""Train a model and evaluate on the test set.

	Orchestrates the full training loop (train_epoch + validate_epoch per epoch)
	then computes test accuracy and the confusion matrix.

	Args:
		model: The neural network model.
		optimizer: Optimizer instance.
		train_loader: DataLoader for training data.
		test_loader: DataLoader for test data.
		validation_loader: DataLoader for validation data.
		criterion: Loss function. Defaults to BCEWithLogitsLoss.
		device: Target device. Defaults to CPU.
		num_epochs: Number of training epochs.
		verbose: Whether to log training progress.
		gpu_transforms: Optional GPU-side preprocessing transforms.

	Returns:
		Tuple of test accuracy, confusion matrix, training duration, and model.
	"""
	if criterion is None:
		criterion = _DEFAULT_CRITERION()
	if device is None:
		device = _DEFAULT_DEVICE

	use_amp = device.type == "cuda"
	scaler = torch.amp.GradScaler("cuda") if use_amp else None

	start_time = timer()

	for epoch in range(num_epochs):
		train_loss, train_acc = train_epoch(
			model,
			train_loader,
			criterion,
			optimizer,
			device,
			use_amp,
			scaler,
			gpu_transforms,
		)
		val_loss, val_acc = validate_epoch(
			model,
			validation_loader,
			criterion,
			device,
			use_amp,
			gpu_transforms,
		)

		if verbose:
			logger.info(
				"Epoch %d/%d, Train Loss: %.4f, Train Accuracy: %.1f%%, "
				"Validation Loss: %.4f, Validation Accuracy: %.1f%%",
				epoch + 1,
				num_epochs,
				train_loss,
				train_acc * 100,
				val_loss,
				val_acc * 100,
			)

	training_duration = timer() - start_time

	if verbose:
		logger.info("Total training duration: %.1f minutes", training_duration / 60)

	test_accuracy, confusion_matrix = compute_test_metrics(
		model,
		test_loader,
		device,
		use_amp,
		gpu_transforms,
	)

	if verbose:
		logger.info("Test Accuracy: %.1f%%", test_accuracy * 100)

	return test_accuracy, confusion_matrix, training_duration, model


def evaluate_model(
	device: torch.device,
	train_loader: DataLoader,
	test_loader: DataLoader,
	validation_loader: DataLoader,
	training_config: TrainingConfig,
	criterion: nn.Module | None = None,
	optimizer_class: type[optim.Optimizer] = optim.Adam,
	verbose: bool = True,
	gpu_transforms: list[nn.Module] | None = None,
) -> tuple[float, ConfusionMatrix, float, BinaryCNN]:
	"""Create a fresh model and run full training + evaluation.

	Args:
		device: Target device.
		train_loader: DataLoader for training data.
		test_loader: DataLoader for test data.
		validation_loader: DataLoader for validation data.
		training_config: Training hyperparameters.
		criterion: Loss function. Defaults to BCEWithLogitsLoss.
		optimizer_class: Optimizer class. Defaults to Adam.
		verbose: Whether to log training progress.
		gpu_transforms: Optional GPU-side preprocessing transforms.

	Returns:
		Tuple of test accuracy, confusion matrix, training duration, and model.
	"""
	if criterion is None:
		criterion = _DEFAULT_CRITERION()
	criterion = criterion.to(device)

	model = BinaryCNN(device=device).to(device, memory_format=torch.channels_last)

	if training_config.num_epochs >= 10 and device.type == "cuda":
		model = torch.compile(model, mode="reduce-overhead")

	optimizer = optimizer_class(model.parameters(), lr=training_config.learning_rate)

	test_accuracy, confusion_matrix, training_time, model = evaluate(
		model=model,
		criterion=criterion,
		device=device,
		verbose=verbose,
		optimizer=optimizer,
		train_loader=train_loader,
		test_loader=test_loader,
		validation_loader=validation_loader,
		num_epochs=training_config.num_epochs,
		gpu_transforms=gpu_transforms,
	)

	return test_accuracy, confusion_matrix, training_time, model


def combo_key(transforms: tuple[nn.Module, ...]) -> str:
	"""Generate a canonical key for a combination of transforms.

	Args:
		transforms: Tuple of transform instances.

	Returns:
		Human-readable key like "DenoiseTransform -> NormalizeTransform".
	"""
	return COMBO_SEPARATOR.join(t.__class__.__name__ for t in transforms)


@dataclass
class CombinationResult:
	"""Result of training and evaluating one preprocessing combination."""

	combo_key: str
	transforms: tuple[str, ...]
	accuracy: float
	confusion_matrix: ConfusionMatrix
	training_time: float
	regime: str = ""


def _build_transforms(
	preprocess_config: PreprocessConfig,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
	"""Build the four GPU transform instances from a PreprocessConfig.

	Args:
		preprocess_config: Configuration for preprocessing parameters.

	Returns:
		Tuple of (normalize, denoise, colorspace, equalization) transform instances.
	"""
	normalize = NormalizeTransform(
		mean=[preprocess_config.normalize.mean],
		std=[preprocess_config.normalize.std],
	)
	denoise = DenoiseTransform(
		kernel_size=preprocess_config.denoise.kernel_size,
		sigma_space=preprocess_config.denoise.sigma_space,
		sigma_color=preprocess_config.denoise.sigma_color,
	)
	colorspace = ColorSpaceTransform(
		source_space=preprocess_config.colorspace.source_space,
		target_space=preprocess_config.colorspace.target_space,
	)
	equalization = EqualizationTransform()
	return normalize, denoise, colorspace, equalization


def _generate_all_combinations(
	transforms: tuple[nn.Module, ...],
) -> list[tuple[nn.Module, ...]]:
	"""Generate all permutation combinations (classes 0-4) from the base transforms.

	Class 0: empty (baseline) — 1
	Class 1: single transforms — 4
	Class 2: permutations of 2 — 12
	Class 3: permutations of 3 — 24
	Class 4: permutations of 4 — 24
	Total: 65

	Args:
		transforms: Tuple of the 4 base transforms.

	Returns:
		List of 65 tuples, starting with the empty tuple (baseline).
	"""
	combinations: list[tuple[nn.Module, ...]] = [()]
	for r in range(1, len(transforms) + 1):
		combinations.extend(itertools.permutations(transforms, r))
	return combinations


def run_combinations(
	preprocess_config: PreprocessConfig,
	training_config: TrainingConfig,
	device: torch.device,
	verbose: bool = False,
	seed: int = 42,
) -> dict[str, CombinationResult]:
	"""Run all 65 preprocessing combinations for one parameter regime.

	Args:
		preprocess_config: Preprocessing parameters for this regime.
		training_config: Training hyperparameters.
		device: Target device.
		verbose: Whether to log per-model training progress.
		seed: Random seed for data splitting.

	Returns:
		Dict mapping combo_key to CombinationResult (65 entries).
	"""
	base_transforms = _build_transforms(preprocess_config)
	all_combos = _generate_all_combinations(base_transforms)
	results: dict[str, CombinationResult] = {}

	train_loader, test_loader, val_loader = get_data_loaders(training_config, seed=seed)

	for i, combo in enumerate(all_combos):
		key = BASELINE_KEY if len(combo) == 0 else combo_key(combo)
		gpu_transforms = list(combo) if combo else None

		logger.info(
			"[%s] Running %d/%d: %s",
			preprocess_config.regime,
			i + 1,
			len(all_combos),
			key,
		)

		accuracy, confusion_matrix, training_time, model = evaluate_model(
			device=device,
			train_loader=train_loader,
			test_loader=test_loader,
			validation_loader=val_loader,
			training_config=training_config,
			verbose=verbose,
			gpu_transforms=gpu_transforms,
		)

		results[key] = CombinationResult(
			combo_key=key,
			transforms=tuple(t.__class__.__name__ for t in combo),
			accuracy=accuracy,
			confusion_matrix=confusion_matrix,
			training_time=training_time,
			regime=preprocess_config.regime,
		)

		del model
		torch.cuda.empty_cache()

	return results


def run_pipeline(
	preprocess_config: PreprocessConfig,
	training_config: TrainingConfig,
	device: torch.device,
	verbose: bool = False,
	seed: int = 42,
) -> dict[str, CombinationResult]:
	"""Run the full 65-combination pipeline for one parameter regime.

	Args:
		preprocess_config: Preprocessing parameters.
		training_config: Training hyperparameters.
		device: Target device.
		verbose: Whether to log per-model training progress.
		seed: Random seed for data splitting.

	Returns:
		Dict mapping combo_key to CombinationResult.
	"""
	logger.info("Starting pipeline for regime: %s (seed=%d)", preprocess_config.regime, seed)
	results = run_combinations(preprocess_config, training_config, device, verbose, seed)
	logger.info(
		"Pipeline complete for %s: %d combinations evaluated",
		preprocess_config.regime,
		len(results),
	)
	return results


def run_full_experiment(
	training_config: TrainingConfig,
	preprocess_configs: list[PreprocessConfig],
	device: torch.device,
	verbose: bool = False,
	seed: int = 42,
) -> dict[str, dict[str, CombinationResult]]:
	"""Run all combinations for all preprocessing regimes.

	Args:
		training_config: Training hyperparameters.
		preprocess_configs: Preprocessing parameter regimes to evaluate.
		device: Target device.
		verbose: Whether to log per-model training progress.
		seed: Random seed for data splitting.

	Returns:
		Nested dict: {regime: {combo_key: CombinationResult}}.
	"""
	results: dict[str, dict[str, CombinationResult]] = {}

	for config in preprocess_configs:
		logger.info("=== Preprocessing regime: %s (seed=%d) ===", config.regime, seed)
		results[config.regime] = run_pipeline(
			config,
			training_config,
			device,
			verbose,
			seed,
		)

	total = sum(len(v) for v in results.values())
	logger.info(
		"Full experiment complete: %d total evaluations across %d preprocessing regimes (seed=%d)",
		total,
		len(results),
		seed,
	)

	return results
