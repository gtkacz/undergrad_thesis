"""Train, evaluate, and export the paper's preprocessing experiment.

Runs the 65 primary pipeline configurations across five random seeds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from util.cnn import configure_cuda
from util.config import load_configs
from util.export import build_results_matrix
from util.runner import run_full_experiment

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
SEEDS = [42, 123, 256, 512, 1024]
PRIMARY_SEED = SEEDS[0]


def main() -> None:
	"""Run the full thesis experiment pipeline across multiple seeds."""
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)

	configure_cuda()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.info("Using device: %s", device)

	training_config, base_config = load_configs()
	preprocess_configs = [base_config]

	all_seed_results: dict[int, dict] = {}

	for seed in SEEDS:
		logger.info("=" * 60)
		logger.info("SEED %d (%d/%d)", seed, SEEDS.index(seed) + 1, len(SEEDS))
		logger.info("=" * 60)

		seed_results = run_full_experiment(
			training_config=training_config,
			preprocess_configs=preprocess_configs,
			device=device,
			seed=seed,
		)

		seed_dir = OUTPUT_DIR / f"seed_{seed}"
		build_results_matrix(seed_results, output_dir=seed_dir)
		all_seed_results[seed] = seed_results

		logger.info("Seed %d complete. Results written to %s/", seed, seed_dir)

	_write_seed_manifest(all_seed_results)
	logger.info(
		"Full multi-seed experiment complete: %d seeds × %d configs = %d total runs",
		len(SEEDS),
		65 * len(preprocess_configs),
		len(SEEDS) * 65 * len(preprocess_configs),
	)


def _write_seed_manifest(
	all_seed_results: dict[int, dict],
) -> None:
	"""Write a manifest listing all seeds and their output directories."""
	manifest = {
		"seeds": list(all_seed_results.keys()),
		"primary_seed": PRIMARY_SEED,
		"seed_dirs": {seed: str(OUTPUT_DIR / f"seed_{seed}") for seed in all_seed_results},
	}
	manifest_path = OUTPUT_DIR / "seed_manifest.json"
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
	logger.info("Seed manifest written to %s", manifest_path)


if __name__ == "__main__":
	main()
