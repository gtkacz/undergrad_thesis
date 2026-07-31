# Transform ordering and selection in CNN preprocessing

Code, saved results, and manuscript sources for an exhaustive evaluation of
the 65 ordered pipelines formed from four image transforms:

- global luminance histogram equalization;
- bilateral filtering;
- fixed affine normalization; and
- RGB-to-HSV conversion.

Each pipeline was trained from scratch with five seeds on a balanced
healthy-versus-diseased image task containing 12,078 images. The mean
unprocessed baseline accuracy is 98.09%. No multi-transform pipeline improves
mean accuracy, while ordering accounts for 58% of the observed accuracy-gain
variance at pipeline length three. These results are specific to the exact
implementation and data construction: class is strongly confounded with image
source, and several transform orders cross incompatible value-range or
color-space contracts. The work is not a clinical validation study.

## Repository layout

- `src/main.py`: runs 65 pipelines for each of five seeds (325 training runs).
- `src/analyze.py`: regenerates aggregate statistics from saved seed results.
- `src/compute_clinical_metrics.py`: derives fixed-threshold per-class metrics.
- `src/verify_dataset.py`: verifies the local dataset against the checksum manifest.
- `src/output/`: saved per-seed metrics and generated analysis artifacts.
- `paper/final/`: Elsevier manuscript, references, highlights, and figures.
- `tests/`: unit tests for enumeration, splitting, metrics, and statistics.

## Environment

The project requires Python 3.13 and uses [uv](https://docs.astral.sh/uv/):

```console
uv sync --locked
```

PyTorch is pinned to the CUDA 12.8 package index. A compatible NVIDIA setup is
needed to reproduce all 325 training runs in a practical amount of time.

## Dataset

Place the two classes below `src/dataset/`:

```text
src/dataset/
├── diseased/   # 6,039 HAM10000 images
└── healthy/    # 6,039 Healthy Skin Dataset images
```

The dataset itself is not redistributed. After acquisition, verify the exact
files used in the paper:

```console
cd src
uv run python verify_dataset.py
```

The code performs a deterministic image-level 80/10/remainder split for each
seed. It does not group HAM10000 images by lesion identifier; this is a known
limitation documented in the manuscript.

## Reproduce the saved analysis and paper

From the repository root:

```console
make verify
```

This runs the unit tests, rebuilds the analysis and clinical-metric JSON files,
regenerates the three quantitative figures, and compiles the manuscript.
Compilation uses a local LaTeX installation when available and otherwise can
use Podman or Docker.

To rerun the full training matrix:

```console
cd src
uv run python main.py
```

Training overwrites the saved seed-level result matrices. Per-image logits were
not retained in the original experiment, so AUROC, calibration, and
decision-threshold sweeps cannot be reconstructed without retraining.

## Main result

For this exact CNN, dataset pairing, transform implementation, and parameter
setting, equalization-first/normalization-last is less harmful than the reverse
ordering by 2.00 percentage points on average. Because normalization produces
values outside `[0, 1]` while later transforms clamp or reinterpret those
values, this pattern should be understood partly as an implementation-contract
effect, not as a universal preprocessing prescription.
