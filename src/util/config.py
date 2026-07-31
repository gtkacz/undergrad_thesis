"""Configuration dataclasses and loader for the thesis pipeline."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NormalizeConfig:
	"""Parameters for NormalizeTransform."""

	mean: float
	std: float


@dataclass(frozen=True)
class DenoiseConfig:
	"""Parameters for DenoiseTransform."""

	kernel_size: int
	sigma_space: float
	sigma_color: float


@dataclass(frozen=True)
class ColorSpaceConfig:
	"""Parameters for ColorSpaceTransform."""

	source_space: str
	target_space: str


@dataclass(frozen=True)
class PreprocessConfig:
	"""Full preprocessing parameter set for one experimental regime."""

	normalize: NormalizeConfig
	denoise: DenoiseConfig
	colorspace: ColorSpaceConfig
	regime: str


@dataclass(frozen=True)
class TrainingConfig:
	"""Training hyperparameters from parameters.toml."""

	num_epochs: int
	num_workers: int
	batch_size: int
	learning_rate: float
	shuffle: bool
	pin_memory: bool
	resize_dim: int


def load_configs(
	path: str | Path = "parameters.toml",
) -> tuple[TrainingConfig, PreprocessConfig]:
	"""Load the training and primary preprocessing configurations.

	Args:
		path: Path to the TOML configuration file.

	Returns:
		The training and preprocessing configurations.
	"""
	raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))

	training = TrainingConfig(
		num_epochs=raw["TRAINING"]["num_epochs"],
		num_workers=raw["TRAINING"]["num_workers"],
		batch_size=raw["TRAINING"]["batch_size"],
		learning_rate=raw["TRAINING"]["learning_rate"],
		shuffle=raw["TRAINING"]["shuffle"],
		pin_memory=raw["TRAINING"]["pin_memory"],
		resize_dim=raw["TRAINING"]["resize_dim"],
	)

	preprocess = raw["PREPROCESS"]

	base = PreprocessConfig(
		normalize=NormalizeConfig(
			mean=preprocess["normalize"]["mean"],
			std=preprocess["normalize"]["std"],
		),
		denoise=DenoiseConfig(
			kernel_size=preprocess["denoise"]["kernel_size"],
			sigma_space=preprocess["denoise"]["sigma_space"],
			sigma_color=preprocess["denoise"]["sigma_color"],
		),
		colorspace=ColorSpaceConfig(
			source_space=preprocess["colorspace"]["source_space"],
			target_space=preprocess["colorspace"]["target_space"],
		),
		regime="base",
	)

	return training, base
