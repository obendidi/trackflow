"""The package version and CHANGELOG.md must be ready for the publish workflow."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import trackflow

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "release_notes.py"


def _notes(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def test_current_version_has_release_notes() -> None:
    tag = f"v{trackflow.__version__}"
    res = _notes(tag)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().startswith("### ")


def test_link_references_stay_out_of_the_notes() -> None:
    spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = (SCRIPT.parent.parent / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = module.changelog_sections(text)
    assert sections[-1][0] == "0.1.0" and "]: https://" not in sections[-1][1]


def test_previous_release_and_mismatched_tags() -> None:
    res = _notes("--previous", f"v{trackflow.__version__}")
    assert res.returncode == 0 and res.stdout.startswith("v0.")
    res = _notes("v99.0.0")
    assert res.returncode == 1 and "doesn't match the package version" in res.stderr


def test_variant_manifests_keep_cpu_and_gpu_onnxruntime_mutually_exclusive() -> None:
    variants_script = SCRIPT.parent / "build_variants.py"
    spec = importlib.util.spec_from_file_location("build_variants", variants_script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    base = (SCRIPT.parent.parent / "pyproject.toml").read_text(encoding="utf-8")

    cpu = module.render_variant_pyproject(base, "cpu")
    assert 'name = "trackflow-python"' in cpu
    assert '"onnxruntime>=' in cpu
    assert "onnxruntime-gpu" not in cpu

    cuda = module.render_variant_pyproject(base, "cuda")
    assert 'name = "trackflow-python-cuda"' in cuda
    assert '"onnxruntime-gpu[cuda,cudnn]>=1.22"' in cuda
    assert '"onnxruntime>=' not in cuda
    assert 'trackflow-python-cuda = "trackflow.cli:app"' in cuda

    trt = module.render_variant_pyproject(base, "tensorrt")
    assert 'name = "trackflow-python-tensorrt"' in trt
    assert 'requires-python = ">=3.11"' in trt
    assert '"onnxruntime-gpu[cuda,cudnn]>=1.28"' in trt
    assert '"tensorrt-cu13-libs>=10,<11"' in trt
    assert '"onnxruntime>=' not in trt
    assert 'trackflow-python-tensorrt = "trackflow.cli:app_tensorrt"' in trt
