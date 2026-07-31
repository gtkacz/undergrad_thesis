"""Verify the local dataset against the file manifest used in the paper."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

CLASS_NAMES = ("healthy", "diseased")
EXPECTED_PER_CLASS = 6_039
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "dataset"
DEFAULT_MANIFEST = SCRIPT_DIR / "dataset_manifest.sha256"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _image_paths(dataset_root: Path) -> list[Path]:
	"""Return all class images in the ordering used by the training code.

	Raises:
		FileNotFoundError: If either class directory is missing.
		ValueError: If a class does not contain the expected number of images.
	"""
	paths: list[Path] = []
	for class_name in CLASS_NAMES:
		class_dir = dataset_root / class_name
		if not class_dir.is_dir():
			raise FileNotFoundError(f"Missing dataset class directory: {class_dir}")
		class_paths = [path for path in sorted(class_dir.iterdir()) if path.suffix.lower() in VALID_EXTENSIONS]
		if len(class_paths) != EXPECTED_PER_CLASS:
			raise ValueError(
				f"Expected {EXPECTED_PER_CLASS} {class_name} images, found {len(class_paths)}",
			)
		paths.extend(class_paths)
	return paths


def _sha256(path: Path) -> str:
	"""Hash one file without loading it all into memory.

	Returns:
		The lowercase SHA-256 hexadecimal digest.
	"""
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def build_manifest(dataset_root: Path) -> str:
	"""Build the deterministic SHA-256 manifest text.

	Returns:
		One SHA-256 and relative path per line.
	"""
	lines = [f"{_sha256(path)}  {path.relative_to(dataset_root).as_posix()}" for path in _image_paths(dataset_root)]
	return "\n".join(lines) + "\n"


def verify_manifest(dataset_root: Path, manifest_path: Path) -> None:
	"""Raise an error when the local files differ from the released manifest.

	Raises:
		ValueError: If file names, counts, or hashes do not match.
	"""
	expected = manifest_path.read_text(encoding="utf-8")
	actual = build_manifest(dataset_root)
	if actual != expected:
		raise ValueError(
			f"Dataset verification failed. File names or SHA-256 hashes differ from {manifest_path}.",
		)


def main() -> None:
	"""Write or verify the dataset manifest."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
	parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
	parser.add_argument(
		"--write-manifest",
		action="store_true",
		help="Create the manifest from the current local dataset.",
	)
	args = parser.parse_args()

	if args.write_manifest:
		args.manifest.write_text(build_manifest(args.dataset_root), encoding="utf-8")
		print(f"Wrote {args.manifest}")
		return

	verify_manifest(args.dataset_root, args.manifest)
	print(f"Verified {2 * EXPECTED_PER_CLASS:,} images against {args.manifest}")


if __name__ == "__main__":
	main()
