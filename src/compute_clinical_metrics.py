"""Derive threshold-dependent clinical metrics from the released confusion matrices.

The main experiment persisted per-pipeline, per-seed confusion matrices but reported
only accuracy. The fixed-threshold, per-class metrics below are recoverable from
the stored confusion matrices with no retraining; this script recomputes them per
seed and aggregates across the five seeds, mirroring the seed-averaging used for the
accuracy-gain analysis.

Positive class = diseased (label 1); negative class = healthy (label 0), matching the
``enumerate(["healthy", "diseased"])`` ordering in :mod:`util.dataset`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SEED_MANIFEST = Path("output/seed_manifest.json")
REGIME = "base"


def _metrics_from_confusion(tp: float, tn: float, fp: float, fn: float) -> dict[str, float]:
	"""Compute clinical metrics from a single confusion matrix.

	Args:
		tp: True positives (diseased classified diseased).
		tn: True negatives (healthy classified healthy).
		fp: False positives (healthy classified diseased).
		fn: False negatives (diseased classified healthy).

	Returns:
		Mapping of metric name to value; undefined ratios collapse to 0.0.
	"""
	total = tp + tn + fp + fn
	sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
	specificity = tn / (tn + fp) if (tn + fp) else 0.0
	precision = tp / (tp + fp) if (tp + fp) else 0.0
	npv = tn / (tn + fn) if (tn + fn) else 0.0
	f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
	balanced_accuracy = (sensitivity + specificity) / 2
	mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
	mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator else 0.0

	return {
		"accuracy": (tp + tn) / total if total else 0.0,
		"sensitivity": sensitivity,
		"specificity": specificity,
		"precision": precision,
		"npv": npv,
		"f1": f1,
		"balanced_accuracy": balanced_accuracy,
		"mcc": mcc,
	}


def _mean_std(values: list[float]) -> tuple[float, float]:
	"""Return the mean and sample standard deviation of ``values``."""
	n = len(values)
	mean = sum(values) / n
	variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
	return mean, math.sqrt(variance)


def main() -> None:
	"""Aggregate per-seed clinical metrics for every pipeline and print a summary."""
	manifest = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
	seed_dirs = manifest["seed_dirs"]

	per_pipeline: dict[str, dict[str, list[float]]] = {}

	for seed_dir in seed_dirs.values():
		matrix = json.loads((Path(seed_dir) / "results_matrix.json").read_text(encoding="utf-8"))
		for entry in matrix:
			if entry["regime"] != REGIME:
				continue
			cm = entry["confusion_matrix"]
			metrics = _metrics_from_confusion(cm["TP"], cm["TN"], cm["FP"], cm["FN"])
			bucket = per_pipeline.setdefault(entry["combo_key"], {})
			for name, value in metrics.items():
				bucket.setdefault(name, []).append(value)

	aggregated = {
		combo: {name: _mean_std(values) for name, values in metric_lists.items()}
		for combo, metric_lists in per_pipeline.items()
	}

	metric_order = [
		"accuracy",
		"sensitivity",
		"specificity",
		"precision",
		"npv",
		"f1",
		"balanced_accuracy",
		"mcc",
	]

	def _fmt(combo: str) -> str:
		stats = aggregated[combo]
		cells = [f"{stats[m][0] * (100 if m != 'mcc' else 1):.2f}" for m in metric_order]
		return combo + "\t" + "\t".join(cells)

	print("Per-seed-averaged clinical metrics (base regime, tau=0.5)")
	print("Percentages except MCC; positive class = diseased")
	print("pipeline\t" + "\t".join(metric_order))
	for named in ("Baseline", "EqualizationTransform"):
		if named in aggregated:
			print(_fmt(named))

	print("\nRange across 64 non-baseline pipelines (mean of per-seed means):")
	for m in metric_order:
		scale = 100 if m != "mcc" else 1
		vals = [stats[m][0] * scale for combo, stats in aggregated.items() if combo != "Baseline"]
		print(f"  {m:18s} min={min(vals):.2f}  max={max(vals):.2f}  median={sorted(vals)[len(vals) // 2]:.2f}")

	print("\nBaseline detail (mean +/- SD across 5 seeds):")
	for m in metric_order:
		scale = 100 if m != "mcc" else 1
		mean, std = aggregated["Baseline"][m]
		print(f"  {m:18s} {mean * scale:.3f} +/- {std * scale:.3f}")

	Path("output/clinical_metrics.json").write_text(
		json.dumps(
			{combo: {m: {"mean": v[0], "std": v[1]} for m, v in stats.items()} for combo, stats in aggregated.items()},
			indent=2,
		),
		encoding="utf-8",
	)
	print("\nWrote output/clinical_metrics.json")


if __name__ == "__main__":
	main()
