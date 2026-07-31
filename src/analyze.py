"""Post-experiment analysis: aggregate multi-seed results and run all statistical tests.

Reads per-seed results_matrix.json files, computes aggregated statistics
(mean ± std per pipeline), runs the full statistical analysis suite
(ANOVA, permutation tests, CIs, multiple testing correction), and exports
a comprehensive analysis JSON for paper generation.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import starmap
from pathlib import Path

import numpy as np
from numpyencoder import NumpyEncoder
from scipy import stats

from util.statistics import (
	bootstrap_ci,
	holm_bonferroni,
	one_way_anova,
	paired_sign_flip_test,
	permutation_correlation_test,
	permutation_test,
)

logger = logging.getLogger(__name__)

BASELINE_KEY = "Baseline"
PRIMARY_REGIME = "base"
OUTPUT_DIR = Path("output")


@dataclass(frozen=True)
class PipelineAggregate:
	"""Aggregated results for one pipeline across all seeds."""

	combo_key: str
	regime: str
	transforms: tuple[str, ...]
	pipeline_length: int
	mean_accuracy: float
	std_accuracy: float
	mean_alpha: float
	std_alpha: float
	ci_alpha_lower: float
	ci_alpha_upper: float
	mean_gamma: float
	std_gamma: float
	mean_weighted_alpha: float
	std_weighted_alpha: float
	mean_fp: float
	mean_fn: float
	mean_tp: float
	mean_tn: float
	n_seeds: int


def _load_seed_results(seed_dir: Path) -> list[dict]:
	"""Load results_matrix.json for one seed.

	Returns:
		List of result dicts from the matrix file.
	"""
	matrix_path = seed_dir / "results_matrix.json"
	return json.loads(matrix_path.read_text(encoding="utf-8"))


def _entry_regime(entry: dict) -> str:
	"""Read the preprocessing regime.

	Returns:
		The stored preprocessing-regime label.
	"""
	return str(entry["regime"])


def _group_by_pipeline(
	all_seeds: dict[int, list[dict]],
	regime: str,
) -> dict[str, list[dict]]:
	"""Group entries across seeds by combo key for one preprocessing regime.

	Returns:
		Dict mapping combo_key to list of per-seed entries.
	"""
	grouped: dict[str, list[dict]] = defaultdict(list)
	for entries in all_seeds.values():
		for entry in entries:
			if _entry_regime(entry) == regime:
				grouped[entry["combo_key"]].append(entry)
	return dict(grouped)


def aggregate_pipeline(
	combo_key: str,
	entries: list[dict],
) -> PipelineAggregate:
	"""Compute aggregated statistics for one pipeline across seeds.

	Returns:
		PipelineAggregate with mean, std, and CIs for all metrics.
	"""
	alphas = np.array([e["alpha"] for e in entries])
	gammas = np.array([e["gamma"] for e in entries])
	w_alphas = np.array([e["weighted_alpha"] for e in entries])
	accuracies = np.array([e["accuracy"] for e in entries])

	fps = np.array([e["confusion_matrix"]["FP"] for e in entries], dtype=float)
	fns = np.array([e["confusion_matrix"]["FN"] for e in entries], dtype=float)
	tps = np.array([e["confusion_matrix"]["TP"] for e in entries], dtype=float)
	tns = np.array([e["confusion_matrix"]["TN"] for e in entries], dtype=float)

	ci = bootstrap_ci(alphas)

	transforms = tuple(entries[0]["transforms"])
	pipeline_length = len(transforms)

	return PipelineAggregate(
		combo_key=combo_key,
		regime=_entry_regime(entries[0]),
		transforms=transforms,
		pipeline_length=pipeline_length,
		mean_accuracy=float(np.mean(accuracies)),
		std_accuracy=float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
		mean_alpha=ci.mean,
		std_alpha=ci.std,
		ci_alpha_lower=ci.ci_lower,
		ci_alpha_upper=ci.ci_upper,
		mean_gamma=float(np.mean(gammas)),
		std_gamma=float(np.std(gammas, ddof=1)) if len(gammas) > 1 else 0.0,
		mean_weighted_alpha=float(np.mean(w_alphas)),
		std_weighted_alpha=float(np.std(w_alphas, ddof=1)) if len(w_alphas) > 1 else 0.0,
		mean_fp=float(np.mean(fps)),
		mean_fn=float(np.mean(fns)),
		mean_tp=float(np.mean(tps)),
		mean_tn=float(np.mean(tns)),
		n_seeds=len(entries),
	)


def _pipeline_set_key(transforms: tuple[str, ...]) -> frozenset[str]:
	"""Unordered set key for grouping pipelines by transform selection.

	Returns:
		Frozenset of transform names.
	"""
	return frozenset(transforms)


def _eta_squared(groups: list[np.ndarray]) -> float:
	"""Return one-way ANOVA eta squared for pre-grouped observations."""
	all_values = np.concatenate(groups)
	grand_mean = float(np.mean(all_values))
	ss_between = sum(len(group) * (float(np.mean(group)) - grand_mean) ** 2 for group in groups)
	ss_within = sum(float(np.sum((group - np.mean(group)) ** 2)) for group in groups)
	ss_total = ss_between + ss_within
	return float(ss_between / ss_total) if ss_total > 0 else 0.0


def run_variance_decomposition(
	aggregates: list[PipelineAggregate],
	grouped_entries: dict[str, list[dict]] | None = None,
) -> dict[int, dict]:
	"""Run ANOVA variance decomposition at each pipeline length.

	Groups pipelines by their unordered transform set and decomposes
	α variance into between-set (selection) and within-set (ordering).

	Returns:
		Dict mapping pipeline length to ANOVA results.
	"""
	by_length: dict[int, list[PipelineAggregate]] = defaultdict(list)
	for agg in aggregates:
		if agg.pipeline_length > 0:
			by_length[agg.pipeline_length].append(agg)

	results: dict[int, dict] = {}

	for length, pipelines in sorted(by_length.items()):
		groups_by_set: dict[frozenset[str], list[float]] = defaultdict(list)
		for p in pipelines:
			groups_by_set[_pipeline_set_key(p.transforms)].append(p.mean_alpha)

		group_arrays = [np.array(v) for v in groups_by_set.values() if len(v) > 1]

		if len(group_arrays) >= 2:
			anova = one_way_anova(group_arrays)
			length_result = {
				"anova": asdict(anova),
				"n_groups": len(group_arrays),
				"n_per_group": [len(g) for g in group_arrays],
				"total_pipelines": len(pipelines),
			}
			if grouped_entries is not None:
				keys_by_set: dict[frozenset[str], list[str]] = defaultdict(list)
				for pipeline in pipelines:
					keys_by_set[_pipeline_set_key(pipeline.transforms)].append(pipeline.combo_key)

				n_seeds = len(grouped_entries[pipelines[0].combo_key])
				rng = np.random.default_rng(42)
				bootstrap_eta = np.empty(10_000, dtype=float)
				for bootstrap_index in range(len(bootstrap_eta)):
					seed_indices = rng.choice(n_seeds, size=n_seeds, replace=True)
					bootstrap_groups = []
					for pipeline_keys in keys_by_set.values():
						means = [
							float(
								np.mean([grouped_entries[key][seed_index]["alpha"] for seed_index in seed_indices]),
							)
							for key in pipeline_keys
						]
						bootstrap_groups.append(np.asarray(means))
					bootstrap_eta[bootstrap_index] = _eta_squared(bootstrap_groups)

				pooled_groups = [
					np.asarray([entry["alpha"] for key in pipeline_keys for entry in grouped_entries[key]])
					for pipeline_keys in keys_by_set.values()
				]
				length_result["eta_squared_bootstrap_ci"] = {
					"lower": float(np.percentile(bootstrap_eta, 2.5)),
					"upper": float(np.percentile(bootstrap_eta, 97.5)),
					"n_resamples": len(bootstrap_eta),
				}
				length_result["pooled_seed_eta_squared"] = _eta_squared(pooled_groups)

			results[length] = length_result
		elif length == 4:
			all_alphas = np.array([p.mean_alpha for p in pipelines])
			results[length] = {
				"note": "Single transform set; ordering is sole variance source",
				"n_pipelines": len(pipelines),
				"alpha_range": float(np.ptp(all_alphas)),
				"alpha_std": float(np.std(all_alphas, ddof=1)),
			}

	return results


def run_length_degradation_test(
	aggregates: list[PipelineAggregate],
) -> dict:
	"""Permutation test on correlation between pipeline length and mean α.

	Returns:
		Dict with correlation test results and positive-proportion breakdown.
	"""
	non_baseline = [a for a in aggregates if a.pipeline_length > 0]

	lengths = np.array([a.pipeline_length for a in non_baseline], dtype=float)
	alphas = np.array([a.mean_alpha for a in non_baseline])

	result = permutation_correlation_test(lengths, alphas)

	proportion_positive = {}
	for length in sorted({int(l) for l in lengths}):
		length_alphas = [a.mean_alpha for a in non_baseline if a.pipeline_length == length]
		n_positive = sum(1 for a in length_alphas if a > 0)
		proportion_positive[length] = {
			"n_positive": n_positive,
			"n_total": len(length_alphas),
			"fraction": n_positive / len(length_alphas),
		}

	return {
		"correlation": asdict(result),
		"proportion_positive_by_length": proportion_positive,
	}


def run_positional_analysis(
	aggregates: list[PipelineAggregate],
) -> dict:
	"""Compute mean α by position for each transform and test E-first significance.

	Returns:
		Dict with positional means, E-first test, and bookend test results.
	"""
	non_baseline = [a for a in aggregates if a.pipeline_length > 0]

	transform_position_alphas: dict[str, dict[int, list[float]]] = defaultdict(
		lambda: defaultdict(list),
	)

	for agg in non_baseline:
		for pos, t_name in enumerate(agg.transforms):
			transform_position_alphas[t_name][pos].append(agg.mean_alpha)

	positional_means: dict[str, dict[int, float]] = {}
	for t_name, positions in transform_position_alphas.items():
		positional_means[t_name] = {pos: float(np.mean(vals)) for pos, vals in sorted(positions.items())}

	e_first_alphas = np.array([
		a.mean_alpha for a in non_baseline if len(a.transforms) > 0 and a.transforms[0] == "EqualizationTransform"
	])
	e_not_first_alphas = np.array([
		a.mean_alpha for a in non_baseline if len(a.transforms) > 0 and a.transforms[0] != "EqualizationTransform"
	])

	e_first_test = permutation_test(e_first_alphas, e_not_first_alphas)
	e_present_not_first_alphas = np.array([
		a.mean_alpha
		for a in non_baseline
		if "EqualizationTransform" in a.transforms and a.transforms[0] != "EqualizationTransform"
	])
	e_within_inclusion_test = permutation_test(e_first_alphas, e_present_not_first_alphas)

	e_first_n_last = np.array([
		a.mean_alpha
		for a in non_baseline
		if (
			len(a.transforms) >= 2
			and a.transforms[0] == "EqualizationTransform"
			and a.transforms[-1] == "NormalizeTransform"
		)
	])
	n_first_e_last = np.array([
		a.mean_alpha
		for a in non_baseline
		if (
			len(a.transforms) >= 2
			and a.transforms[0] == "NormalizeTransform"
			and a.transforms[-1] == "EqualizationTransform"
		)
	])

	bookend_test = None
	if len(e_first_n_last) > 0 and len(n_first_e_last) > 0:
		bookend_test = permutation_test(e_first_n_last, n_first_e_last)

	return {
		"positional_means": positional_means,
		"e_first_test": asdict(e_first_test),
		"e_within_inclusion_test": asdict(e_within_inclusion_test),
		"bookend_test": asdict(bookend_test) if bookend_test else None,
		"e_first_n_last_mean": float(np.mean(e_first_n_last)) if len(e_first_n_last) > 0 else None,
		"n_first_e_last_mean": float(np.mean(n_first_e_last)) if len(n_first_e_last) > 0 else None,
	}


def run_error_decomposition(
	aggregates: list[PipelineAggregate],
) -> dict:
	"""Classify pipelines by error shift pattern using aggregated confusion matrices.

	Returns:
		Dict with FP/FN correlation, category counts, and category fractions.
	"""
	baseline = next((a for a in aggregates if a.pipeline_length == 0), None)
	if baseline is None:
		return {"error": "No baseline found"}

	base_fp = baseline.mean_fp
	base_fn = baseline.mean_fn

	categories: dict[str, list[str]] = {
		"fp_up_fn_down": [],
		"fp_down_fn_up": [],
		"both_up": [],
		"both_down": [],
		"one_unchanged": [],
	}

	delta_fps = []
	delta_fns = []

	for agg in aggregates:
		if agg.pipeline_length == 0:
			continue

		delta_fp = agg.mean_fp - base_fp
		delta_fn = agg.mean_fn - base_fn
		delta_fps.append(delta_fp)
		delta_fns.append(delta_fn)

		if delta_fp > 0 and delta_fn < 0:
			categories["fp_up_fn_down"].append(agg.combo_key)
		elif delta_fp < 0 and delta_fn > 0:
			categories["fp_down_fn_up"].append(agg.combo_key)
		elif delta_fp > 0 and delta_fn > 0:
			categories["both_up"].append(agg.combo_key)
		elif delta_fp <= 0 and delta_fn <= 0 and (delta_fp < 0 or delta_fn < 0):
			categories["both_down"].append(agg.combo_key)
		else:
			categories["one_unchanged"].append(agg.combo_key)

	n_total = len(delta_fps)
	n_opposite_sign = len(categories["fp_up_fn_down"]) + len(categories["fp_down_fn_up"])

	correlation = float(np.corrcoef(delta_fps, delta_fns)[0, 1]) if len(delta_fps) > 1 else 0.0

	return {
		"fp_fn_correlation": correlation,
		"opposite_sign_fraction": n_opposite_sign / n_total if n_total > 0 else 0.0,
		"category_counts": {k: len(v) for k, v in categories.items()},
		"category_fractions": {k: len(v) / n_total for k, v in categories.items()} if n_total > 0 else {},
	}


def run_stratified_bookend_test(
	grouped_entries: dict[str, list[dict]],
) -> dict:
	"""Bookend test (E-first-N-last vs N-first-E-last) within each length stratum.

	Within each length, pipelines in an arm are averaged for each seed. The
	resulting five seed-level arm means are compared with an exact paired
	sign-flip test. This preserves the seed as the independent unit and avoids
	treating multiple pipelines from one seed as independent observations.

	Returns:
		Dict keyed by pipeline length, each with permutation-test results
		and descriptive statistics.
	"""
	results_by_length: dict[int, dict] = {}

	for length in (2, 3, 4):
		e_first_n_last_pipelines: list[list[float]] = []
		n_first_e_last_pipelines: list[list[float]] = []

		for entries in grouped_entries.values():
			transforms = tuple(entries[0]["transforms"])
			if len(transforms) != length:
				continue

			is_e_first_n_last = transforms[0] == "EqualizationTransform" and transforms[-1] == "NormalizeTransform"
			is_n_first_e_last = transforms[0] == "NormalizeTransform" and transforms[-1] == "EqualizationTransform"

			if is_e_first_n_last:
				e_first_n_last_pipelines.append([float(e["alpha"]) for e in entries])
			elif is_n_first_e_last:
				n_first_e_last_pipelines.append([float(e["alpha"]) for e in entries])

		if e_first_n_last_pipelines and n_first_e_last_pipelines:
			e_first_n_last = np.mean(np.asarray(e_first_n_last_pipelines), axis=0)
			n_first_e_last = np.mean(np.asarray(n_first_e_last_pipelines), axis=0)
			test = paired_sign_flip_test(e_first_n_last, n_first_e_last)
			results_by_length[length] = {
				"test": asdict(test),
				"n_seed_pairs": len(e_first_n_last),
				"n_pipelines_per_arm": len(e_first_n_last_pipelines),
				"mean_e_first_n_last": float(np.mean(e_first_n_last)),
				"mean_n_first_e_last": float(np.mean(n_first_e_last)),
				"std_e_first_n_last": float(np.std(e_first_n_last, ddof=1)),
				"std_n_first_e_last": float(np.std(n_first_e_last, ddof=1)),
			}
		else:
			results_by_length[length] = {
				"note": (
					f"Insufficient data: {len(e_first_n_last_pipelines)} vs {len(n_first_e_last_pipelines)} pipelines"
				),
			}

	return results_by_length


def run_weighted_alpha_rank_analysis(
	aggregates: list[PipelineAggregate],
) -> dict:
	"""Spearman ρ between αw (piecewise γ-weighted) and α/γ (uniform) rankings.

	Assesses whether the piecewise weighting scheme used to compute αw
	materially alters pipeline rankings relative to a simpler uniform α/γ
	ratio. High ρ indicates the choice of weighting is not driving rankings.

	Returns:
		Dict with Spearman ρ, p-value, and top-10 overlap count.
	"""
	non_baseline = [a for a in aggregates if a.pipeline_length > 0]

	w_alphas = np.array([a.mean_weighted_alpha for a in non_baseline])
	simple_alpha_gamma = np.array([a.mean_alpha / a.mean_gamma if a.mean_gamma > 0 else 0.0 for a in non_baseline])

	rho, p_value = stats.spearmanr(w_alphas, simple_alpha_gamma)

	order_w = np.argsort(-w_alphas)[:10]
	order_simple = np.argsort(-simple_alpha_gamma)[:10]
	top10_overlap = len(set(order_w.tolist()) & set(order_simple.tolist()))

	return {
		"spearman_rho": float(rho),
		"spearman_p": float(p_value),
		"top10_overlap": int(top10_overlap),
		"n_pipelines": len(non_baseline),
	}


def run_loso_sensitivity(
	all_seeds: dict[int, list[dict]],
	regime: str,
) -> dict:
	"""Leave-one-seed-out sensitivity analysis on primary tests.

	Re-runs variance decomposition, length-degradation, and positional
	analysis with each seed excluded in turn, then applies Holm correction.
	Reports effect size ranges and min/max corrected p-values across the
	five LOSO runs. This analysis checks whether the primary findings depend
	on any single evaluation seed.

	Returns:
		Dict with per-excluded-seed results, effect size ranges, and summary
		of corrected significance across all LOSO iterations.
	"""
	seeds = sorted(all_seeds.keys())
	loso_results: dict[int, dict] = {}

	for excluded_seed in seeds:
		subset_seeds = {s: all_seeds[s] for s in seeds if s != excluded_seed}

		grouped = _group_by_pipeline(subset_seeds, regime)
		aggregates = list(starmap(aggregate_pipeline, grouped.items()))
		aggregates.sort(key=lambda a: (a.pipeline_length, a.combo_key))

		test_results = {
			"variance_decomposition": run_variance_decomposition(aggregates, grouped),
			"length_degradation": run_length_degradation_test(aggregates),
			"positional_analysis": run_positional_analysis(aggregates),
		}
		corrected = apply_multiple_testing_correction(test_results)

		bookend = test_results["positional_analysis"]["bookend_test"]
		loso_results[excluded_seed] = {
			"corrected_p_values": corrected,
			"e_first_cohens_d": test_results["positional_analysis"]["e_first_test"]["cohens_d"],
			"bookend_cohens_d": bookend["cohens_d"] if bookend else None,
			"length_correlation_r": test_results["length_degradation"]["correlation"]["observed_statistic"],
		}

	all_names = {c["test_name"] for r in loso_results.values() for c in r["corrected_p_values"]}
	summary: dict[str, dict] = {}
	for name in all_names:
		entries = [c for r in loso_results.values() for c in r["corrected_p_values"] if c["test_name"] == name]
		if entries:
			ps = [c["corrected_p"] for c in entries]
			summary[name] = {
				"min_corrected_p": min(ps),
				"max_corrected_p": max(ps),
				"significant_in_all_runs": all(c["significant"] for c in entries),
				"n_runs": len(entries),
			}

	e_first_ds = [r["e_first_cohens_d"] for r in loso_results.values()]
	bookend_ds = [r["bookend_cohens_d"] for r in loso_results.values() if r["bookend_cohens_d"] is not None]
	length_rs = [r["length_correlation_r"] for r in loso_results.values()]

	return {
		"per_excluded_seed": {str(k): v for k, v in loso_results.items()},
		"summary": summary,
		"effect_size_ranges": {
			"e_first_cohens_d": {"min": float(min(e_first_ds)), "max": float(max(e_first_ds))},
			"bookend_cohens_d": (
				{"min": float(min(bookend_ds)), "max": float(max(bookend_ds))} if bookend_ds else None
			),
			"length_correlation_r": {"min": float(min(length_rs)), "max": float(max(length_rs))},
		},
	}


def apply_multiple_testing_correction(
	test_results: dict,
) -> list[dict]:
	"""Collect all p-values from the analysis and apply Holm-Bonferroni.

	Returns:
		List of corrected p-value dicts with significance flags.
	"""
	p_values: list[tuple[str, float]] = []

	if "correlation" in test_results.get("length_degradation", {}):
		p_values.append((
			"length_correlation",
			test_results["length_degradation"]["correlation"]["p_value"],
		))

	positional = test_results.get("positional_analysis", {})
	if "e_first_test" in positional:
		p_values.append(("e_first", positional["e_first_test"]["p_value"]))
	if positional.get("bookend_test"):
		p_values.append(("bookend", positional["bookend_test"]["p_value"]))

	for length, anova_data in test_results.get("variance_decomposition", {}).items():
		if "anova" in anova_data:
			p_values.append((f"anova_L{length}", anova_data["anova"]["p_value"]))

	corrected = holm_bonferroni(p_values)
	return [asdict(c) for c in corrected]


def main() -> None:
	"""Run full analysis pipeline on multi-seed experiment results.

	Raises:
		ValueError: If the saved primary matrix is incomplete.
	"""
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)

	manifest_path = OUTPUT_DIR / "seed_manifest.json"

	if not manifest_path.exists():
		logger.error("No seed_manifest.json found. Run main.py first.")
		return

	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	seeds = manifest["seeds"]

	logger.info("Loading results for %d seeds: %s", len(seeds), seeds)

	all_seeds: dict[int, list[dict]] = {}

	for seed in seeds:
		seed_dir = Path(manifest["seed_dirs"][str(seed)])
		all_seeds[seed] = _load_seed_results(seed_dir)

	grouped = _group_by_pipeline(all_seeds, PRIMARY_REGIME)
	aggregates = list(starmap(aggregate_pipeline, grouped.items()))
	aggregates.sort(key=lambda a: (a.pipeline_length, a.combo_key))

	if len(aggregates) != 65:
		raise ValueError(f"Expected 65 primary pipelines, found {len(aggregates)}")
	if any(aggregate.n_seeds != len(seeds) for aggregate in aggregates):
		raise ValueError("At least one primary pipeline is missing a seed result")

	test_results: dict = {}

	logger.info("Running variance decomposition...")
	test_results["variance_decomposition"] = run_variance_decomposition(aggregates, grouped)

	logger.info("Running length-degradation test...")
	test_results["length_degradation"] = run_length_degradation_test(aggregates)

	logger.info("Running positional analysis...")
	test_results["positional_analysis"] = run_positional_analysis(aggregates)

	logger.info("Running error decomposition...")
	test_results["error_decomposition"] = run_error_decomposition(aggregates)

	logger.info("Running stratified bookend test (within-length)...")
	test_results["stratified_bookend"] = run_stratified_bookend_test(grouped)

	logger.info("Running αw rank-correlation analysis...")
	test_results["weighted_alpha_rank"] = run_weighted_alpha_rank_analysis(aggregates)

	logger.info("Applying multiple testing correction...")
	test_results["corrected_p_values"] = apply_multiple_testing_correction(test_results)

	logger.info("Running LOSO sensitivity analysis...")
	test_results["loso_sensitivity"] = run_loso_sensitivity(all_seeds, PRIMARY_REGIME)

	full_analysis = {
		PRIMARY_REGIME: {
			"aggregated_pipelines": [asdict(a) for a in aggregates],
			"statistical_tests": test_results,
			"metadata": {
				"n_seeds": len(seeds),
				"seeds": seeds,
				"n_pipelines": len(aggregates),
			},
		},
	}

	analysis_path = OUTPUT_DIR / "analysis.json"
	analysis_path.write_text(
		json.dumps(full_analysis, indent=2, cls=NumpyEncoder),
		encoding="utf-8",
	)
	logger.info("Full analysis written to %s", analysis_path)


if __name__ == "__main__":
	main()
