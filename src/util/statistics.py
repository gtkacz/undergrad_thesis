"""Statistical analysis for multi-seed experiment results.

Provides: bootstrap confidence intervals, Shapiro-Wilk normality tests,
Holm-Bonferroni multiple testing correction, Cohen's d effect sizes,
one-way ANOVA with full reporting, and permutation tests with effect sizes.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
	from numpy.typing import NDArray


@dataclass(frozen=True)
class BootstrapCI:
	"""Bootstrap confidence interval."""

	mean: float
	ci_lower: float
	ci_upper: float
	std: float


@dataclass(frozen=True)
class ANOVAResult:
	"""One-way ANOVA result with effect size and normality check."""

	f_statistic: float
	p_value: float
	eta_squared: float
	between_var_fraction: float
	within_var_fraction: float
	df_between: int
	df_within: int
	normality_p: float
	normality_ok: bool


@dataclass(frozen=True)
class PermutationTestResult:
	"""Permutation test result with effect size."""

	observed_statistic: float
	p_value: float
	cohens_d: float
	n_permutations: int


@dataclass(frozen=True)
class CorrectedPValue:
	"""A p-value with Holm-Bonferroni correction applied."""

	test_name: str
	raw_p: float
	corrected_p: float
	significant: bool


def bootstrap_ci(
	values: NDArray[np.floating],
	n_bootstrap: int = 10_000,
	confidence: float = 0.95,
	rng: np.random.Generator | None = None,
) -> BootstrapCI:
	"""Compute bootstrap confidence interval for the mean.

	Args:
		values: 1-D array of observations.
		n_bootstrap: Number of bootstrap resamples.
		confidence: Confidence level (e.g. 0.95 for 95% CI).
		rng: Optional random generator for reproducibility.

	Returns:
		BootstrapCI with mean, lower, upper, and std.
	"""
	if rng is None:
		rng = np.random.default_rng(42)

	means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)])

	alpha = 1 - confidence
	lower = float(np.percentile(means, 100 * alpha / 2))
	upper = float(np.percentile(means, 100 * (1 - alpha / 2)))

	return BootstrapCI(
		mean=float(np.mean(values)),
		ci_lower=lower,
		ci_upper=upper,
		std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
	)


def one_way_anova(
	groups: list[NDArray[np.floating]],
) -> ANOVAResult:
	"""One-way ANOVA with eta-squared and Shapiro-Wilk normality test on residuals.

	Args:
		groups: List of arrays, one per group.

	Returns:
		ANOVAResult with F-statistic, p-value, eta-squared, and normality check.
	"""
	f_stat, p_val = stats.f_oneway(*groups)

	all_values = np.concatenate(groups)
	grand_mean = np.mean(all_values)

	ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
	ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
	ss_total = ss_between + ss_within

	eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

	df_between = len(groups) - 1
	df_within = len(all_values) - len(groups)

	residuals = np.concatenate([g - np.mean(g) for g in groups])

	if len(residuals) >= 3:
		_, normality_p = stats.shapiro(residuals)
	else:
		normality_p = float("nan")

	return ANOVAResult(
		f_statistic=float(f_stat),
		p_value=float(p_val),
		eta_squared=float(eta_sq),
		between_var_fraction=float(eta_sq),
		within_var_fraction=float(1 - eta_sq),
		df_between=df_between,
		df_within=df_within,
		normality_p=float(normality_p),
		normality_ok=normality_p > 0.05 if not np.isnan(normality_p) else False,
	)


def permutation_test(
	group_a: NDArray[np.floating],
	group_b: NDArray[np.floating],
	n_permutations: int = 100_000,
	rng: np.random.Generator | None = None,
) -> PermutationTestResult:
	"""Two-sample permutation test on difference of means, with Cohen's d.

	Args:
		group_a: First group observations.
		group_b: Second group observations.
		n_permutations: Number of random permutations.
		rng: Optional random generator for reproducibility.

	Returns:
		PermutationTestResult with observed statistic, p-value, and Cohen's d.
	"""
	if rng is None:
		rng = np.random.default_rng(42)

	observed_diff = float(np.mean(group_a) - np.mean(group_b))

	pooled = np.concatenate([group_a, group_b])
	n_a = len(group_a)
	extreme_count = 0

	for _ in range(n_permutations):
		rng.shuffle(pooled)
		perm_diff = np.mean(pooled[:n_a]) - np.mean(pooled[n_a:])
		if abs(perm_diff) >= abs(observed_diff):
			extreme_count += 1

	p_value = (extreme_count + 1) / (n_permutations + 1)

	pooled_std = float(
		np.sqrt(
			((len(group_a) - 1) * np.var(group_a, ddof=1) + (len(group_b) - 1) * np.var(group_b, ddof=1))
			/ (len(group_a) + len(group_b) - 2),
		),
	)
	cohens_d = observed_diff / pooled_std if pooled_std > 0 else float("inf")

	return PermutationTestResult(
		observed_statistic=observed_diff,
		p_value=float(p_value),
		cohens_d=float(cohens_d),
		n_permutations=n_permutations,
	)


def permutation_correlation_test(
	x: NDArray[np.floating],
	y: NDArray[np.floating],
	n_permutations: int = 100_000,
	rng: np.random.Generator | None = None,
) -> PermutationTestResult:
	"""Permutation test on Pearson correlation coefficient.

	Args:
		x: Independent variable.
		y: Dependent variable.
		n_permutations: Number of random permutations.
		rng: Optional random generator for reproducibility.

	Returns:
		PermutationTestResult with observed r, p-value, and effect size (r itself).
	"""
	if rng is None:
		rng = np.random.default_rng(42)

	observed_r = float(np.corrcoef(x, y)[0, 1])
	extreme_count = 0

	y_shuffled = y.copy()
	for _ in range(n_permutations):
		rng.shuffle(y_shuffled)
		perm_r = np.corrcoef(x, y_shuffled)[0, 1]
		if abs(perm_r) >= abs(observed_r):
			extreme_count += 1

	return PermutationTestResult(
		observed_statistic=observed_r,
		p_value=(extreme_count + 1) / (n_permutations + 1),
		cohens_d=observed_r,
		n_permutations=n_permutations,
	)


def holm_bonferroni(
	tests: list[tuple[str, float]],
	alpha: float = 0.05,
) -> list[CorrectedPValue]:
	"""Apply Holm-Bonferroni correction to a family of p-values.

	Args:
		tests: List of (test_name, raw_p_value) tuples.
		alpha: Family-wise significance level.

	Returns:
		List of CorrectedPValue, sorted by raw p-value.
	"""
	sorted_tests = sorted(tests, key=operator.itemgetter(1))
	n = len(sorted_tests)
	results: list[CorrectedPValue] = []

	raw_adjusted = [min(raw_p * (n - i), 1.0) for i, (_, raw_p) in enumerate(sorted_tests)]
	adjusted_p_values: list[float] = []
	running_max = 0.0
	for value in raw_adjusted:
		running_max = max(running_max, value)
		adjusted_p_values.append(running_max)

	for (name, raw_p), corrected_p in zip(sorted_tests, adjusted_p_values, strict=True):
		results.append(
			CorrectedPValue(
				test_name=name,
				raw_p=raw_p,
				corrected_p=corrected_p,
				significant=corrected_p <= alpha,
			),
		)

	return results


def paired_sign_flip_test(
	group_a: NDArray[np.floating],
	group_b: NDArray[np.floating],
) -> PermutationTestResult:
	"""Run an exact paired sign-flip test on the mean difference.

	Args:
		group_a: First observation from every pair.
		group_b: Second observation from every pair.

	Returns:
		An exact two-sided p-value and paired-sample Cohen's dz.

	Raises:
		ValueError: If the groups are not one-dimensional matched pairs.
	"""
	if group_a.shape != group_b.shape:
		raise ValueError("paired groups must have identical shapes")
	if group_a.ndim != 1 or len(group_a) < 2:
		raise ValueError("paired groups must be one-dimensional with at least two pairs")

	differences = group_a - group_b
	observed_diff = float(np.mean(differences))
	null_means = np.array([
		np.mean(differences * np.asarray(signs)) for signs in product((-1.0, 1.0), repeat=len(differences))
	])
	extreme_count = int(np.sum(np.abs(null_means) >= abs(observed_diff)))
	p_value = extreme_count / len(null_means)

	difference_std = float(np.std(differences, ddof=1))
	cohens_dz = observed_diff / difference_std if difference_std > 0 else float("inf")

	return PermutationTestResult(
		observed_statistic=observed_diff,
		p_value=float(p_value),
		cohens_d=float(cohens_dz),
		n_permutations=len(null_means),
	)
