"""Build CPU, NVIDIA CUDA, and NVIDIA TensorRT wheel variants from the single source tree.

ONNX Runtime ships ``onnxruntime`` (CPU) and ``onnxruntime-gpu`` (CUDA/TensorRT) under the same
``onnxruntime`` import namespace, so installing both in one environment corrupts GPU providers.
This script stages clean ``pyproject.toml`` manifests for each variant so every user gets a
single-command install with the exact matching runtime and zero library-path setup:

- ``trackflow-python``: CPU / macOS (``onnxruntime``)
- ``trackflow-python-cuda``: NVIDIA CUDA (``onnxruntime-gpu[cuda,cudnn]``)
- ``trackflow-python-tensorrt``: NVIDIA TensorRT FP16 (``onnxruntime-gpu[cuda,cudnn]>=1.28`` +
  ``tensorrt-cu13-libs>=10,<11``; defaults ``--device`` to ``tensorrt``)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VARIANTS: dict[str, dict[str, object]] = {
    "cpu": {
        "name": "trackflow-python",
        "description_suffix": "",
        "requires_python": None,
        "ort_deps": ["onnxruntime>=1.20"],
        "extra_scripts": {},
    },
    "cuda": {
        "name": "trackflow-python-cuda",
        "description_suffix": " (NVIDIA CUDA build with bundled CUDA/cuDNN runtimes.)",
        "requires_python": None,
        "ort_deps": ["onnxruntime-gpu[cuda,cudnn]>=1.22"],
        "extra_scripts": {"trackflow-python-cuda": "trackflow.cli:app"},
    },
    "tensorrt": {
        "name": "trackflow-python-tensorrt",
        "description_suffix": " (NVIDIA TensorRT FP16 build with bundled CUDA 13/cuDNN/TensorRT 10 runtimes.)",
        "requires_python": ">=3.11",
        "ort_deps": [
            "onnxruntime-gpu[cuda,cudnn]>=1.28",
            "tensorrt-cu13-libs>=10,<11",
        ],
        "extra_scripts": {"trackflow-python-tensorrt": "trackflow.cli:app_tensorrt"},
    },
}


def render_variant_pyproject(base_text: str, variant: str) -> str:
    """Rewrite root ``pyproject.toml`` text for ``variant`` without losing formatting."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from: {', '.join(VARIANTS)}")
    spec = VARIANTS[variant]
    if variant == "cpu":
        return base_text

    text, count = re.subn(
        r'^name = "trackflow-python"$',
        f'name = "{spec["name"]}"',
        base_text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Could not find project name line in pyproject.toml")
    if spec["requires_python"]:
        text = re.sub(
            r'^requires-python = "[^"]+"$',
            f'requires-python = "{spec["requires_python"]}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )
    if spec["description_suffix"]:
        text, count = re.subn(
            r'^(description = "[^"]+)"$',
            rf'\1{spec["description_suffix"]}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError("Could not find project description line in pyproject.toml")

    ort_lines = "\n".join(f'  "{dep}",' for dep in spec["ort_deps"])  # type: ignore[attr-defined]
    text, count = re.subn(
        r'^  "onnxruntime>=[^"]+",$',
        ort_lines,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Could not find onnxruntime dependency line in pyproject.toml")

    extra_scripts: dict[str, str] = spec["extra_scripts"]  # type: ignore[assignment]
    if extra_scripts:
        extra_lines = "\n".join(f'{k} = "{v}"' for k, v in extra_scripts.items())
        text, count = re.subn(
            r'^trackflow-python = "trackflow\.cli:app"$',
            extra_lines,
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError("Could not find trackflow-python script entry in pyproject.toml")
    return text


def build_variant(variant: str, out_dir: Path, *, sdist: bool = False) -> list[Path]:
    """Build a single variant wheel (and optionally sdist) into ``out_dir``."""
    base_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    variant_toml = render_variant_pyproject(base_text, variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"trackflow-{variant}-") as tmp:
        stage = Path(tmp)
        (stage / "pyproject.toml").write_text(variant_toml, encoding="utf-8")
        shutil.copy2(ROOT / "README.md", stage / "README.md")
        shutil.copy2(ROOT / "LICENSE", stage / "LICENSE")
        shutil.copytree(ROOT / "src", stage / "src")
        cmd = ["uv", "build", str(stage), "--out-dir", str(out_dir)]
        if not sdist:
            cmd.append("--wheel")
        before = set(out_dir.iterdir())
        subprocess.run(cmd, check=True)
        return sorted(
            p for p in (set(out_dir.iterdir()) - before) if p.suffix in (".whl", ".gz")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default="cpu,cuda,tensorrt",
        help="Comma-separated variants to build: cpu, cuda, tensorrt.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "dist",
        help="Destination directory for built wheels/sdists.",
    )
    parser.add_argument("--sdist", action="store_true", help="Also build source distributions.")
    args = parser.parse_args()

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        built = build_variant(variant, args.out_dir, sdist=args.sdist)
        for path in built:
            print(f"[{variant}] {path}")


if __name__ == "__main__":
    main()
