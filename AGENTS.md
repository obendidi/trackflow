# Trackflow

Real-time people & object tracking toolkit (Python package `trackflow-python` + static WebGPU browser playground).

## Layout

- `src/trackflow/`: Python library and Typer CLI (`detector.py`, `models.py`, `tracker.py`, `ocsort.py`, `boosttrack.py`, `reid.py`, `pipeline.py`, `privacy.py`, `homography.py`, `lines.py`, `inspector.py`, `inspector_ui.html`, `sinks.py`, `benchmark.py`, `cli.py`).
- `web/`: Static WebGPU playground (plain HTML/JS, zero build step), deployed to GitHub Pages on pushes to `main` via `.github/workflows/pages.yml`.
- `tests/`: Pytest suite including Node-backed JS parity tests (`tests/test_web_parity.py`) and release checks (`tests/test_release.py`).
- `scripts/release_notes.py`: Validates version/tag/changelog consistency during releases.

## Commands

- **Setup**: `uv sync --extra dev` (add `--extra transformers` when working on `rfdetr-large` or `hf:` detectors)
- **Lint**: `uv run ruff check src tests scripts`
- **Test**: `uv run pytest` (requires Node.js on `PATH` for `tests/test_web_parity.py`)
- **Weight smoke tests** (downloads HF weights): `TRACKFLOW_TEST_WEIGHTS=1 uv run pytest tests/test_model_weights.py`
- **Build**: `uv build`
- **Web playground preview**: `python3 -m http.server 4173 --bind 127.0.0.1 --directory web`

## Invariants & Gotchas

- **Web playground parity**: `web/tracker.js` mirrors `ByteTracker`. Keep `tests/test_web_parity.py` passing whenever ByteTrack matching, thresholds, or track lifecycle rules change.
- **Pinned weights & licenses**: Pin every preset in `src/trackflow/models.py` to an exact Hugging Face revision and list its weight license in `README.md` (the package is Apache-2.0, while some detector weights are AGPL-3.0 or non-commercial).
- **Text search vs. identity**: Keep CLIP/SigLIP (`--vlm`) strictly for text-to-image queries (`--query` / `search`); use `--reid osnet` for appearance matching and `--reid-merge`.
- **Privacy & anonymization**: `src/trackflow/privacy.py` must mask every detected person box (including low-confidence detections and coasting tracks) and suppress writing person crops or thumbnails on anonymized runs.
- **Releasing**: Bump `__version__` in both `pyproject.toml` and `src/trackflow/__init__.py`, move `Unreleased` entries in `CHANGELOG.md` under `## [X.Y.Z] - YYYY-MM-DD`, merge to `main`, then push annotated tag `vX.Y.Z`. `.github/workflows/publish.yml` validates via `scripts/release_notes.py` and publishes to PyPI + GitHub Releases.
- **Legacy branch**: Keep `legacy-2017` and tag `v1.0-legacy-2017` (the original GPL-3.0 `Tracking-with-darkflow` codebase) untouched.
