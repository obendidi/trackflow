"""Model registry and automatic Hugging Face ONNX weight downloading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download

Precision = Literal["fp32", "fp16", "int8"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Descriptor for a built-in Hugging Face ONNX detector model."""

    name: str
    repo_id: str
    architecture: str
    license: str
    approx_size_mb: float
    default_input_size: tuple[int, int]
    description: str


@dataclass(frozen=True, slots=True)
class VlmModelSpec:
    """Descriptor for a built-in vision-language model (``--query`` scoring & ``trackflow search``)."""

    name: str
    repo_id: str
    license: str
    approx_size_mb: float
    vision_files: dict[str, str]
    text_files: dict[str, str]
    description: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "rfdetr-nano": ModelSpec(
        name="rfdetr-nano",
        repo_id="onnx-community/rfdetr_nano-ONNX",
        architecture="RF-DETR Nano",
        license="Apache-2.0",
        approx_size_mb=108.1,
        default_input_size=(384, 384),
        description="384x384 DINOv2 detection transformer; fast on Apple Silicon and WebGPU.",
    ),
    "dfine-n": ModelSpec(
        name="dfine-n",
        repo_id="onnx-community/dfine_n_coco-ONNX",
        architecture="D-FINE Nano",
        license="Apache-2.0",
        approx_size_mb=15.3,
        default_input_size=(640, 640),
        description="Compact 15 MB end-to-end transformer detector.",
    ),
    "rtdetr-v2-r18": ModelSpec(
        name="rtdetr-v2-r18",
        repo_id="onnx-community/rtdetr_v2_r18vd-ONNX",
        architecture="RT-DETRv2 ResNet-18",
        license="Apache-2.0",
        approx_size_mb=81.1,
        default_input_size=(640, 640),
        description="Default detector; balanced accuracy and speed for crowded scenes.",
    ),
    "rfdetr-base": ModelSpec(
        name="rfdetr-base",
        repo_id="onnx-community/rfdetr_base-ONNX",
        architecture="RF-DETR Base",
        license="Apache-2.0",
        approx_size_mb=107.8,
        default_input_size=(560, 560),
        description="Higher-accuracy 560x560 detection transformer.",
    ),
}


@dataclass(frozen=True, slots=True)
class ReidModelSpec:
    """Descriptor for a built-in person Re-ID model (identity embeddings for BoT-SORT & merges)."""

    name: str
    repo_id: str
    filename: str
    revision: str
    license: str
    approx_size_mb: float
    input_hw: tuple[int, int]
    description: str


# Person Re-ID: embeddings trained to tell *people* apart. Used for BoT-SORT appearance matching
# and opt-in cross-exit merges. (CLIP/SigLIP can't do this: on vtest.avi different people scored
# 0.83 cosine on average vs 0.86 for the same person.)
REID_REGISTRY: dict[str, ReidModelSpec] = {
    "osnet": ReidModelSpec(
        name="osnet",
        repo_id="mprib/osnet-ain-x1-0-msmt17-onnx",
        filename="osnet_ain_x1_0_msmt17.onnx",
        revision="306c3dca7875c91a7cbffe48dadc673de73fbb1b",
        license="MIT (weights trained on MSMT17; check dataset terms for commercial use)",
        approx_size_mb=8.7,
        input_hw=(256, 128),
        description="OSNet-AIN x1.0 person Re-ID (Torchreid, MSMT17), 512-d embeddings.",
    ),
}

# Vision-language models: align images with text for --query and `trackflow search`.
VLM_REGISTRY: dict[str, VlmModelSpec] = {
    "clip": VlmModelSpec(
        name="clip",
        repo_id="Xenova/clip-vit-base-patch32",
        license="MIT",
        approx_size_mb=153.0,
        vision_files={
            "int8": "onnx/vision_model_int8.onnx",
            "fp16": "onnx/vision_model_fp16.onnx",
            "fp32": "onnx/vision_model.onnx",
        },
        text_files={
            "int8": "onnx/text_model_int8.onnx",
            "fp16": "onnx/text_model_fp16.onnx",
            "fp32": "onnx/text_model.onnx",
        },
        description="CLIP ViT-B/32 vision+text ONNX for --query and `trackflow search`.",
    ),
    "siglip": VlmModelSpec(
        name="siglip",
        repo_id="Xenova/siglip-base-patch16-224",
        license="Apache-2.0",
        approx_size_mb=205.0,
        vision_files={
            "int8": "onnx/vision_model_int8.onnx",
            "fp16": "onnx/vision_model_fp16.onnx",
            "fp32": "onnx/vision_model.onnx",
        },
        text_files={
            "int8": "onnx/text_model_int8.onnx",
            "fp16": "onnx/text_model_fp16.onnx",
            "fp32": "onnx/text_model.onnx",
        },
        description="SigLIP Base vision+text ONNX; stronger on clothing and attribute queries.",
    ),
}

PRECISION_TO_FILENAME: dict[str, str] = {
    "fp32": "onnx/model.onnx",
    "fp16": "onnx/model_fp16.onnx",
    "int8": "onnx/model_int8.onnx",
}


def resolve_detector_artifacts(
    model: str,
    precision: Precision = "fp32",
    cache_dir: str | Path | None = None,
) -> tuple[Path, Path | None, Path | None]:
    """Resolve a model preset, HF repo id, or local ONNX path into local file paths.

    Returns ``(onnx_path, config_path, preprocessor_config_path)``.
    """
    local_path = Path(model)
    if local_path.is_file():
        parent = local_path.parent
        cfg = parent / "config.json"
        if not cfg.is_file():
            cfg = parent.parent / "config.json"
        prep = parent / "preprocessor_config.json"
        if not prep.is_file():
            prep = parent.parent / "preprocessor_config.json"
        return (
            local_path,
            cfg if cfg.is_file() else None,
            prep if prep.is_file() else None,
        )

    repo_id = MODEL_REGISTRY[model].repo_id if model in MODEL_REGISTRY else model
    onnx_filename = PRECISION_TO_FILENAME.get(precision, "onnx/model.onnx")

    kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    onnx_path = Path(hf_hub_download(repo_id=repo_id, filename=onnx_filename, **kwargs))

    config_path: Path | None = None
    prep_path: Path | None = None
    try:
        config_path = Path(hf_hub_download(repo_id=repo_id, filename="config.json", **kwargs))
    except Exception:
        config_path = None
    try:
        prep_path = Path(
            hf_hub_download(repo_id=repo_id, filename="preprocessor_config.json", **kwargs)
        )
    except Exception:
        prep_path = None

    return onnx_path, config_path, prep_path


def resolve_reid_model(backend: str = "osnet", cache_dir: str | Path | None = None) -> Path:
    """Download and cache a person Re-ID ONNX model (pinned revision)."""
    if backend not in REID_REGISTRY:
        raise ValueError(f"Unknown Re-ID model '{backend}'. Available: {list(REID_REGISTRY)}")
    spec = REID_REGISTRY[backend]
    kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    return Path(
        hf_hub_download(
            repo_id=spec.repo_id, filename=spec.filename, revision=spec.revision, **kwargs
        )
    )


def resolve_vlm_artifacts(
    backend: str = "clip",
    precision: Precision = "int8",
    cache_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Download and cache ONNX vision+text encoder and tokenizer files for NL query & search."""
    if backend not in VLM_REGISTRY:
        raise ValueError(
            f"Unknown vision-language model '{backend}'. Available: {list(VLM_REGISTRY)}"
        )

    spec = VLM_REGISTRY[backend]
    vision_file = spec.vision_files.get(precision, spec.vision_files["int8"])
    text_file = spec.text_files.get(precision, spec.text_files["int8"])
    kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}

    return {
        "vision_onnx": Path(hf_hub_download(repo_id=spec.repo_id, filename=vision_file, **kwargs)),
        "text_onnx": Path(hf_hub_download(repo_id=spec.repo_id, filename=text_file, **kwargs)),
        "tokenizer": Path(
            hf_hub_download(repo_id=spec.repo_id, filename="tokenizer.json", **kwargs)
        ),
        "preprocessor": Path(
            hf_hub_download(repo_id=spec.repo_id, filename="preprocessor_config.json", **kwargs)
        ),
        "config": Path(hf_hub_download(repo_id=spec.repo_id, filename="config.json", **kwargs)),
    }
