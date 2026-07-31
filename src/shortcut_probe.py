"""Measure label separability from low-level color descriptors.

The negative and positive classes come from different acquisition sources, so
source and class are perfectly aligned. Every image is resized to a common square
to remove native-resolution and aspect-ratio cues, then reduced to spatially
agnostic descriptors that omit lesion shape, border, and spatial texture:

  * colour moments (6-D): per-channel mean and standard deviation;
  * colour histogram (48-D): per-channel 16-bin intensity distribution.

A linear classifier is fit on each descriptor. High test accuracy shows that the
label is recoverable from global color alone and is consistent with a low-level
shortcut. Because acquisition source and pathology are not independently varied,
the probe cannot attribute that separability uniquely to either one.

positive class = diseased (label 1); negative class = healthy (label 0), matching
the ``enumerate(["healthy", "diseased"])`` ordering in :mod:`util.dataset`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_ROOT = Path(__file__).resolve().parent / "dataset"
CLASS_LABELS = {"healthy": 0, "diseased": 1}

# A common square removes native-resolution and aspect-ratio shortcuts.
RESIZE = 64
HIST_BINS = 16
SAMPLE_PER_CLASS = 3000
TEST_SIZE = 0.3
SEED = 42


def _descriptors(path: Path) -> tuple[np.ndarray, np.ndarray]:
	"""Return (colour-moment, colour-histogram) descriptors for one image.

	The image is decoded to RGB and resized to a fixed square so that neither the
	source resolution nor the aspect ratio can leak into the features.
	"""
	with Image.open(path) as handle:
		rgb = handle.convert("RGB").resize((RESIZE, RESIZE))
	pixels = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.0

	moments = np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])

	histogram = np.concatenate([
		np.histogram(pixels[:, channel], bins=HIST_BINS, range=(0.0, 1.0), density=True)[0] for channel in range(3)
	])
	return moments, histogram


def _sampled_paths(class_name: str, rng: np.random.Generator) -> list[Path]:
	"""Deterministically sample image paths for one class.

	Returns:
		The selected paths in stable order.
	"""
	paths = sorted((DATASET_ROOT / class_name).iterdir())
	if len(paths) > SAMPLE_PER_CLASS:
		chosen = rng.choice(len(paths), size=SAMPLE_PER_CLASS, replace=False)
		return [paths[index] for index in sorted(chosen)]
	return paths


def _evaluate(features: np.ndarray, labels: np.ndarray, name: str) -> dict[str, float | int]:
	"""Fit a standardised linear classifier and report held-out performance.

	Returns:
		The evaluation configuration and held-out metrics.
	"""
	x_train, x_test, y_train, y_test = train_test_split(
		features,
		labels,
		test_size=TEST_SIZE,
		stratify=labels,
		random_state=SEED,
	)
	scaler = StandardScaler().fit(x_train)
	model = LogisticRegression(max_iter=1000, random_state=SEED)
	model.fit(scaler.transform(x_train), y_train)

	scores = model.predict_proba(scaler.transform(x_test))[:, 1]
	predictions = (scores >= 0.5).astype(int)

	return {
		"feature_set": name,
		"n_features": features.shape[1],
		"n_train": int(x_train.shape[0]),
		"n_test": int(x_test.shape[0]),
		"accuracy": float((predictions == y_test).mean()),
		"balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
		"auroc": float(roc_auc_score(y_test, scores)),
	}


def main() -> None:
	"""Extract low-level descriptors for both classes and report separability."""
	rng = np.random.default_rng(SEED)

	moment_rows: list[np.ndarray] = []
	histogram_rows: list[np.ndarray] = []
	labels: list[int] = []

	for class_name, label in CLASS_LABELS.items():
		paths = _sampled_paths(class_name, rng)
		for path in paths:
			moments, histogram = _descriptors(path)
			moment_rows.append(moments)
			histogram_rows.append(histogram)
			labels.append(label)
		print(f"  {class_name:8s} (label {label}): {len(paths)} images")

	label_array = np.asarray(labels)
	results = [
		_evaluate(np.vstack(moment_rows), label_array, "colour_moments_6d"),
		_evaluate(np.vstack(histogram_rows), label_array, "colour_histogram_48d"),
	]

	print("\nLow-level shortcut probe (linear classifier, no morphology)")
	print(f"Trivial majority-class baseline: {max(np.bincount(label_array)) / label_array.size:.4f}")
	print("feature_set\t\tn_feat\taccuracy\tbal_acc\t\tauroc")
	for row in results:
		print(
			f"{row['feature_set']:24s}{row['n_features']}\t"
			f"{row['accuracy']:.4f}\t\t{row['balanced_accuracy']:.4f}\t\t{row['auroc']:.4f}",
		)

	Path("output/shortcut_probe.json").write_text(
		json.dumps(
			{
				"config": {
					"resize": RESIZE,
					"hist_bins": HIST_BINS,
					"sample_per_class": SAMPLE_PER_CLASS,
					"test_size": TEST_SIZE,
					"seed": SEED,
				},
				"results": results,
			},
			indent=2,
		),
		encoding="utf-8",
	)
	print("\nWrote output/shortcut_probe.json")


if __name__ == "__main__":
	main()
