"""Generate the quantitative figures for the Elsevier manuscript."""

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ELSEVIER_COL_WIDTH = 3.5
ELSEVIER_FONT_SIZE = 9
ELSEVIER_TICK_SIZE = 8
logger = logging.getLogger(__name__)

TRANSFORM_ABBREV = {
	"EqualizationTransform": "E",
	"NormalizeTransform": "N",
	"DenoiseTransform": "D",
	"ColorSpaceTransform": "CS",
}

TRANSFORM_FULL = {
	"EqualizationTransform": "Equalization",
	"NormalizeTransform": "Normalization",
	"DenoiseTransform": "Denoising",
	"ColorSpaceTransform": "Color Space",
}


def setup_elsevier_style() -> None:
	"""Configure matplotlib for Elsevier-quality output."""
	plt.rcParams.update({
		"font.family": "serif",
		"font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
		"font.size": ELSEVIER_FONT_SIZE,
		"axes.titlesize": ELSEVIER_FONT_SIZE,
		"axes.labelsize": ELSEVIER_FONT_SIZE,
		"xtick.labelsize": ELSEVIER_TICK_SIZE,
		"ytick.labelsize": ELSEVIER_TICK_SIZE,
		"legend.fontsize": ELSEVIER_TICK_SIZE,
		"figure.dpi": 300,
		"savefig.dpi": 300,
		"savefig.bbox": "tight",
		"savefig.pad_inches": 0.02,
		"axes.linewidth": 0.5,
		"xtick.major.width": 0.5,
		"ytick.major.width": 0.5,
		"xtick.major.size": 3,
		"ytick.major.size": 3,
		"lines.linewidth": 1.0,
		"lines.markersize": 4,
		"patch.linewidth": 0.5,
		"axes.grid": False,
		"text.usetex": False,
		# Matplotlib defaults to Type 3 fonts, which Elsevier production rejects and
		# which leave figure text unsearchable; 42 selects TrueType instead.
		"pdf.fonttype": 42,
		"ps.fonttype": 42,
	})


def load_analysis(analysis_path: Path) -> dict:
	"""Load the aggregated multi-seed analysis JSON.

	Returns:
		The decoded analysis mapping.
	"""
	with analysis_path.open(encoding="utf-8") as f:
		return json.load(f)


def _pipeline_label(pipeline: dict) -> str:
	"""Return the abbreviated LaTeX label for one pipeline."""
	if pipeline["pipeline_length"] == 0:
		return "Baseline"
	return r"$\to$".join(TRANSFORM_ABBREV[name] for name in pipeline["transforms"])


def _signed(value: float, digits: int) -> str:
	"""Format a signed decimal for a LaTeX math cell.

	Returns:
		The value with an explicit sign and the requested precision.
	"""
	return f"{value:+.{digits}f}"


def write_results_table(analysis: dict, outdir: Path) -> None:
	"""Write the page-breaking 65-pipeline results table."""
	pipelines = analysis["base"]["aggregated_pipelines"]
	lines = [
		r"\begin{longtable}{@{}p{0.34\textwidth} r r c@{}}",
		(
			r"\caption{Pipeline performance at $\tau=0.5$, averaged across five seeds. "
			r"The split-matched baseline averages $\varepsilon_0=98.09\%\pm0.19\%$. "
			r"$\alpha$ is mean accuracy gain in percentage points, $\alpha_w$ is the "
			r"secondary cost-adjusted gain in accuracy-fraction units, and CI$_{95}$ is "
			r"the descriptive percentile-bootstrap interval for $\alpha$.}"
			r"\label{tab:results}\\"
		),
		r"\toprule",
		(
			r"\textbf{Pipeline} & \boldmath$\alpha$\textbf{\,(pp)} & "
			r"\boldmath$\alpha_w$ & \textbf{CI$_{95}$ (pp)} \\"
		),
		r"\midrule",
		r"\endfirsthead",
		r"\multicolumn{4}{c}{\tablename\ \thetable\ -- continued} \\",
		r"\toprule",
		(
			r"\textbf{Pipeline} & \boldmath$\alpha$\textbf{\,(pp)} & "
			r"\boldmath$\alpha_w$ & \textbf{CI$_{95}$ (pp)} \\"
		),
		r"\midrule",
		r"\endhead",
		r"\midrule",
		r"\multicolumn{4}{r}{Continued on next page} \\",
		r"\endfoot",
		r"\bottomrule",
		r"\endlastfoot",
	]

	for length in range(5):
		label = "Baseline" if length == 0 else f"Pipeline length {length}"
		lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{label}}}}} \\")
		for pipeline in pipelines:
			if pipeline["pipeline_length"] != length:
				continue
			alpha = pipeline["mean_alpha"] * 100
			weighted_alpha = pipeline["mean_weighted_alpha"]
			ci_lower = pipeline["ci_alpha_lower"] * 100
			ci_upper = pipeline["ci_alpha_upper"] * 100
			lines.append(
				f"{_pipeline_label(pipeline)} & "
				f"${_signed(alpha, 2)}$ & "
				f"${_signed(weighted_alpha, 3)}$ & "
				f"$[{_signed(ci_lower, 2)},\\,{_signed(ci_upper, 2)}]$ \\\\",
			)
		if length < 4:
			lines.append(r"\addlinespace")

	lines.append(r"\end{longtable}")
	(outdir / "results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
	logger.info("Generated results_table.tex")


def fig1_length_effect(analysis: dict, outdir: Path) -> None:
	"""Box plot of alpha by pipeline length with multi-seed mean alpha values."""
	base = analysis["base"]
	pipes = [p for p in base["aggregated_pipelines"] if p["pipeline_length"] > 0]

	by_length: dict[int, list[float]] = defaultdict(list)
	for p in pipes:
		by_length[p["pipeline_length"]].append(p["mean_alpha"] * 100)

	lengths = [1, 2, 3, 4]
	box_data = [by_length[length] for length in lengths]

	fig, ax = plt.subplots(figsize=(ELSEVIER_COL_WIDTH, 2.4))

	bp = ax.boxplot(
		box_data,
		positions=lengths,
		widths=0.5,
		patch_artist=True,
		showfliers=True,
		flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "gray", "markeredgecolor": "gray", "alpha": 0.7},
		medianprops={"color": "black", "linewidth": 1.0},
		whiskerprops={"linewidth": 0.7},
		capprops={"linewidth": 0.7},
	)

	grays = ["#D9D9D9", "#BFBFBF", "#A6A6A6", "#808080"]
	for patch, color in zip(bp["boxes"], grays, strict=True):
		patch.set_facecolor(color)
		patch.set_edgecolor("black")

	for i, length in enumerate(lengths):
		vals = box_data[i]
		jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
		ax.scatter(
			np.full(len(vals), length) + jitter,
			vals,
			s=8,
			c="black",
			alpha=0.5,
			zorder=5,
			edgecolors="none",
		)

	ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5, alpha=0.6)

	for i, length in enumerate(lengths):
		vals = box_data[i]
		n_pos = sum(1 for v in vals if v > 0)
		pct = n_pos / len(vals) * 100
		y_top = max(vals) + 0.3
		ax.text(
			length,
			y_top,
			f"{pct:.0f}%+",
			ha="center",
			va="bottom",
			fontsize=6,
			fontstyle="italic",
		)

	r = base["statistical_tests"]["length_degradation"]["correlation"]["observed_statistic"]
	corrected = next(
		c for c in base["statistical_tests"]["corrected_p_values"] if c["test_name"] == "length_correlation"
	)
	p_corr = corrected["corrected_p"]
	ax.text(
		0.97,
		0.03,
		f"$r = {r:+.2f}$, $p_{{\\mathrm{{Holm}}}} = {p_corr:.4f}$",
		transform=ax.transAxes,
		ha="right",
		va="bottom",
		fontsize=6,
		bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "gray", "linewidth": 0.5},
	)

	ax.set_xlabel("Pipeline length (number of transforms)")
	ax.set_ylabel("Accuracy gain $\\alpha$ (pp)")
	ax.set_xticks(lengths)
	ax.set_xticklabels([f"{length}\n($n$={len(by_length[length])})" for length in lengths])

	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	fig.tight_layout()
	fig.savefig(outdir / "fig_length_effect.pdf")
	plt.close(fig)
	logger.info("Generated fig_length_effect.pdf")


def fig2_variance_decomposition(analysis: dict, outdir: Path) -> None:
	"""Stacked bar chart of selection vs ordering variance at each pipeline length."""
	vd = analysis["base"]["statistical_tests"]["variance_decomposition"]

	results = {}
	for length_key in ("2", "3"):
		eta_sq = vd[length_key]["anova"]["eta_squared"]
		results[int(length_key)] = {"selection": eta_sq, "ordering": 1 - eta_sq}
	results[4] = {"selection": 0.0, "ordering": 1.0}

	fig, ax = plt.subplots(figsize=(ELSEVIER_COL_WIDTH, 2.2))

	bar_lengths = [2, 3, 4]
	x = np.arange(len(bar_lengths))
	width = 0.55

	sel_vals = [results[length]["selection"] * 100 for length in bar_lengths]
	ord_vals = [results[length]["ordering"] * 100 for length in bar_lengths]

	ax.bar(
		x,
		sel_vals,
		width,
		label="Selection (between-set)",
		color="#D9D9D9",
		edgecolor="black",
		linewidth=0.5,
	)
	ax.bar(
		x,
		ord_vals,
		width,
		bottom=sel_vals,
		label="Ordering (within-set)",
		color="#606060",
		edgecolor="black",
		linewidth=0.5,
	)

	for i, length in enumerate(bar_lengths):
		s = results[length]["selection"] * 100
		o = results[length]["ordering"] * 100
		if s > 8:
			ax.text(x[i], s / 2, f"{s:.1f}%", ha="center", va="center", fontsize=6, color="black")
		if o > 8:
			ax.text(x[i], s + o / 2, f"{o:.1f}%", ha="center", va="center", fontsize=6, color="white")

	n_per_length = {2: 12, 3: 24, 4: 24}
	ax.set_xticks(x)
	ax.set_xticklabels([f"{length}\n($n$={n_per_length[length]})" for length in bar_lengths])
	ax.set_xlabel("Pipeline length")
	ax.set_ylabel("Proportion of variance (%)")
	ax.set_ylim(0, 108)
	ax.set_yticks([0, 25, 50, 75, 100])

	ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=6.5)

	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	fig.tight_layout()
	fig.savefig(outdir / "fig_variance_decomp.pdf")
	plt.close(fig)
	logger.info("Generated fig_variance_decomp.pdf")


def fig3_positional_preferences(analysis: dict, outdir: Path) -> None:
	"""Line plot of mean alpha by ordinal position for each transform with SEMs."""
	pipes = [p for p in analysis["base"]["aggregated_pipelines"] if p["pipeline_length"] > 0]

	transform_order = [
		"EqualizationTransform",
		"NormalizeTransform",
		"DenoiseTransform",
		"ColorSpaceTransform",
	]

	markers = ["s", "^", "o", "D"]
	linestyles = ["-", "-", "--", "--"]
	colors = ["black", "#606060", "#909090", "#B0B0B0"]

	fig, ax = plt.subplots(figsize=(ELSEVIER_COL_WIDTH, 2.4))

	gradients = {}
	for idx, tname in enumerate(transform_order):
		pos_alphas: dict[int, list[float]] = defaultdict(list)
		for p in pipes:
			if tname in p["transforms"]:
				pos = p["transforms"].index(tname) + 1
				pos_alphas[pos].append(p["mean_alpha"] * 100)

		positions = sorted(pos_alphas.keys())
		means = [float(np.mean(pos_alphas[pos])) for pos in positions]
		sems = [float(np.std(pos_alphas[pos], ddof=1) / np.sqrt(len(pos_alphas[pos]))) for pos in positions]

		slope = float(np.polyfit(positions, means, 1)[0])
		gradients[tname] = slope

		label = f"{TRANSFORM_ABBREV[tname]} ({TRANSFORM_FULL[tname]})"
		ax.errorbar(
			positions,
			means,
			yerr=sems,
			marker=markers[idx],
			linestyle=linestyles[idx],
			color=colors[idx],
			label=label,
			capsize=2,
			capthick=0.5,
			markerfacecolor=colors[idx] if idx < 2 else "white",
			markeredgecolor=colors[idx],
			markeredgewidth=0.7,
		)

	ax.axhline(y=0, color="black", linestyle=":", linewidth=0.4, alpha=0.5)

	ax.set_xlabel("Ordinal position in pipeline")
	ax.set_ylabel("Mean accuracy gain $\\alpha$ (pp)")
	ax.set_xticks([1, 2, 3, 4])
	ax.set_xticklabels(["1\n(first)", "2", "3", "4\n(last)"])

	ax.legend(loc="lower left", fontsize=6, frameon=True, fancybox=False, edgecolor="gray", framealpha=0.9)

	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	e_slope = gradients["EqualizationTransform"]
	n_slope = gradients["NormalizeTransform"]
	ax.annotate(
		f"${e_slope:+.2f}$ pp/pos",
		xy=(1.2, -1.0),
		fontsize=5.5,
		fontstyle="italic",
		color="black",
	)
	ax.annotate(
		f"${n_slope:+.2f}$ pp/pos",
		xy=(3.05, -1.3),
		fontsize=5.5,
		fontstyle="italic",
		color="#606060",
	)

	fig.tight_layout()
	fig.savefig(outdir / "fig_positional.pdf")
	plt.close(fig)
	logger.info("Generated fig_positional.pdf")


def main() -> None:
	"""Generate all figures used by the manuscript.

	Raises:
		FileNotFoundError: If the aggregated analysis has not been generated.
	"""
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
	analysis_path = Path(__file__).resolve().parent.parent.parent / "src" / "output" / "analysis.json"
	outdir = Path(__file__).resolve().parent

	if not analysis_path.exists():
		raise FileNotFoundError(f"Analysis file not found: {analysis_path}")

	logger.info("Loading multi-seed analysis from %s", analysis_path)
	analysis = load_analysis(analysis_path)
	n_pipes = len(analysis["base"]["aggregated_pipelines"])
	n_seeds = analysis["base"]["metadata"]["n_seeds"]
	logger.info("Loaded %d aggregated pipelines over %d seeds", n_pipes, n_seeds)

	setup_elsevier_style()
	write_results_table(analysis, outdir)
	fig1_length_effect(analysis, outdir)
	fig2_variance_decomposition(analysis, outdir)
	fig3_positional_preferences(analysis, outdir)


if __name__ == "__main__":
	main()
