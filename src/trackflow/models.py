"""Model registry and automatic Hugging Face ONNX weight downloading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

Precision = Literal["fp32", "fp16", "int8"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Descriptor for a built-in detector. Sizes are decimal MB at FP32."""

    name: str
    repo_id: str
    architecture: str
    license: str
    approx_size_mb: float
    default_input_size: tuple[int, int]
    description: str
    revision: str | None = None
    backend: Literal["onnx", "transformers"] = "onnx"
    onnx_file: str | None = None  # standalone FP32 graph with embedded post-processing
    normalize: bool = False  # force ImageNet normalization when sidecar omits or disables it
    # CoreML graph format for --device coreml. MLProgram covers more operators and gives the same
    # boxes as the CPU; NeuralNetwork is for graphs that MLProgram can't compile or crashes on.
    coreml_format: Literal["MLProgram", "NeuralNetwork"] = "MLProgram"
    letterbox: bool = False  # keep the aspect ratio and pad with gray, as YOLO is trained

    @property
    def precisions(self) -> tuple[Precision, ...]:
        if self.onnx_file:
            return ("fp32",)
        if self.backend == "transformers":
            return ("fp32", "fp16")
        return ("fp32", "fp16", "int8")


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
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class MaskModelSpec:
    """Descriptor for an optional Apache-2.0 video mask model used by ambiguity-triggered fallback."""

    name: str
    repo_id: str
    architecture: str
    license: str
    approx_size_mb: float
    description: str
    revision: str
    kind: Literal["sam2", "edgetam"] = "sam2"


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "rfdetr-nano": ModelSpec(
        name="rfdetr-nano",
        repo_id="onnx-community/rfdetr_nano-ONNX",
        architecture="RF-DETR Nano",
        license="Apache-2.0",
        approx_size_mb=108.1,
        default_input_size=(384, 384),
        description="384x384 DINOv2 detection transformer; fast on Apple Silicon and WebGPU.",
        revision="eae21cee0687a91bcf9fa071605c48d7705d2d91",
        normalize=True,  # onnx-community sidecar sets do_normalize=False, but RF-DETR expects ImageNet norm
    ),
    "dfine-n": ModelSpec(
        name="dfine-n",
        repo_id="onnx-community/dfine_n_coco-ONNX",
        architecture="D-FINE Nano",
        license="Apache-2.0",
        approx_size_mb=15.3,
        default_input_size=(640, 640),
        description="Compact 15 MB end-to-end transformer detector.",
        revision="380d2839c327efaf65dd0fe0c2c10ab7fadd5473",
    ),
    "dfine-s": ModelSpec(
        name="dfine-s",
        repo_id="onnx-community/dfine_s_coco-ONNX",
        architecture="D-FINE Small",
        license="Apache-2.0",
        approx_size_mb=41.5,
        default_input_size=(640, 640),
        description="42 MB COCO-only D-FINE transformer detector (Apache-2.0).",
        revision="a3cf03147a9b86c78475139115c8ac142577352d",
    ),
    "dfine-m": ModelSpec(
        name="dfine-m",
        repo_id="onnx-community/dfine_m_coco-ONNX",
        architecture="D-FINE Medium",
        license="Apache-2.0",
        approx_size_mb=78.6,
        default_input_size=(640, 640),
        description="79 MB COCO-only D-FINE detector; higher MOT recall than rtdetr-v2-r18.",
        revision="489756db2825cb068a588fe930af239a656d1fe1",
    ),
    "rtdetr-v2-r18": ModelSpec(
        name="rtdetr-v2-r18",
        repo_id="onnx-community/rtdetr_v2_r18vd-ONNX",
        architecture="RT-DETRv2 ResNet-18",
        license="Apache-2.0",
        approx_size_mb=81.1,
        default_input_size=(640, 640),
        description="Default detector; balanced accuracy and speed for crowded scenes.",
        revision="936f90b6a476c6da4dfe053fc521af55285976ba",
    ),
    "rfdetr-base": ModelSpec(
        name="rfdetr-base",
        repo_id="onnx-community/rfdetr_base-ONNX",
        architecture="RF-DETR Base",
        license="Apache-2.0",
        approx_size_mb=107.8,
        default_input_size=(560, 560),
        description="Higher-accuracy 560x560 detection transformer.",
        revision="7b18fa82ba7b1d6bc9446518cf2cc1d991097a3f",
        normalize=True,
        coreml_format="NeuralNetwork",  # MLProgram aborts the process in Metal's graph compiler
    ),
    "rfdetr-small": ModelSpec(
        "rfdetr-small",
        "onnx-community/rfdetr_small-ONNX",
        "RF-DETR Small",
        "Apache-2.0",
        114.7,
        (512, 512),
        "512px COCO detector.",
        "63463b68b200177d1fea7015f11f3cebb0ba4eeb",
        normalize=True,
    ),
    "rfdetr-medium": ModelSpec(
        "rfdetr-medium",
        "onnx-community/rfdetr_medium-ONNX",
        "RF-DETR Medium",
        "Apache-2.0",
        121.0,
        (576, 576),
        "576px COCO detector.",
        "8553cefdd984bbaeb9d9ca8743d6a6a1c3e29dce",
        normalize=True,
    ),
    "rfdetr-large": ModelSpec(
        "rfdetr-large",
        "Roboflow/rf-detr-large",
        "RF-DETR Large",
        "Apache-2.0",
        135.8,
        (704, 704),
        "Official 704px checkpoint. Install the transformers extra.",
        "f62f7dd5252b61097cbace33886045816dadbde9",
        backend="transformers",
    ),
    "deimv2-atto": ModelSpec(
        "deimv2-atto",
        "tatsuyaaaaaaa/deimv2-onnx",
        "DEIMv2 Atto",
        "DEIMv2 (non-commercial)",
        2.2,
        (320, 320),
        "HGNetv2 COCO edge detector.",
        "e1514db73b6a64566560cc54a2c28f3bf25a31da",
        onnx_file="deimv2_hgnetv2_atto_coco.onnx",
        coreml_format="NeuralNetwork",  # MLProgram rejects its GridSample coordinates
    ),
    "deimv2-s": ModelSpec(
        "deimv2-s",
        "tatsuyaaaaaaa/deimv2-onnx",
        "DEIMv2 Small",
        "DEIMv2 (non-commercial) + DINOv3",
        40.7,
        (640, 640),
        "DINOv3 COCO detector.",
        "e1514db73b6a64566560cc54a2c28f3bf25a31da",
        onnx_file="deimv2_dinov3_s_coco.onnx",
        normalize=True,
        coreml_format="NeuralNetwork",  # MLProgram rejects its GridSample coordinates
    ),
}

for size, mb, revision in (
    ("n", 9.9, "a8dc7e14743e1cea8ccd493bd99b4c2827de1acf"),
    ("s", 38.2, "37669b009f416cb1df28751257d6ec5f8e4b4e20"),
    ("m", 81.9, "a1db4877f0a3ed68554c231cdae958e2280087e3"),
    ("l", 99.6, "7a72d097ea1719e10eda2d89f75b6e9c009f7f6b"),
    ("x", 223.2, "032eec52a570b81e5f9d66485b9322bd9b7f4053"),
):
    name = f"yolo26{size}"
    MODEL_REGISTRY[name] = ModelSpec(
        name,
        f"onnx-community/{name}-ONNX",
        f"YOLO26{size}",
        "AGPL-3.0 / Ultralytics Enterprise",
        mb,
        (640, 640),
        "COCO detector converted to logits and normalized boxes.",
        revision,
        letterbox=True,
    )


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
        revision="6b7c523703ae26e39be4b03774966f8571c7f523",
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
        revision="4649052661e53c7000355844105f8a1792088239",
    ),
    "siglip2": VlmModelSpec(
        name="siglip2",
        repo_id="onnx-community/siglip2-base-patch16-224-ONNX",
        license="Apache-2.0",
        approx_size_mb=378.0,
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
        description="SigLIP 2 Base vision+text ONNX; strongest pure-ONNX preset (no PyTorch).",
        revision="ba1f3b0843f24bc5417d38e19c37b287d719b2f4",
    ),
    # PyTorch through sentence-transformers (the [transformers] extra), so no ONNX files.
    "embeddinggemma": VlmModelSpec(
        name="embeddinggemma",
        repo_id="google/embeddinggemma-2",
        license="Apache-2.0",
        approx_size_mb=1489.0,
        vision_files={},
        text_files={},
        description="EmbeddingGemma 2 text+image embeddings (PyTorch); best on attribute search.",
        revision="f9972ac22e696a105c482202646979e33e4b5a1e",
    ),
}

# Optional Apache-2.0 mask propagators for ambiguity-triggered fallback (install `[sam2]`).
# SAM 3 uses the non-OSI Meta SAM License and is intentionally excluded from this registry;
# use ``Sam3MaskPropagator`` externally if you accept Meta's terms.
MASK_REGISTRY: dict[str, MaskModelSpec] = {
    "sam2.1-tiny": MaskModelSpec(
        name="sam2.1-tiny",
        repo_id="facebook/sam2.1-hiera-tiny",
        architecture="SAM 2.1 Hiera-Tiny",
        license="Apache-2.0",
        approx_size_mb=155.9,
        description="Default mask fallback; strong occlusion resistance in ~1.9 GB peak VRAM.",
        revision="de431c4043854a71d8101e17995dfe596bf101a5",
        kind="sam2",
    ),
    "edgetam": MaskModelSpec(
        name="edgetam",
        repo_id="yonigozlan/EdgeTAM-hf",
        architecture="EdgeTAM (2D Spatial Perceiver)",
        license="Apache-2.0",
        approx_size_mb=55.9,
        description="Fastest mask fallback (~1.1 GB peak VRAM); lighter memory attention.",
        revision="c266ce53b3fc00f0f495b583f6a116c4e57f53bb",
        kind="edgetam",
    ),
    "sam2.1-small": MaskModelSpec(
        name="sam2.1-small",
        repo_id="facebook/sam2.1-hiera-small",
        architecture="SAM 2.1 Hiera-Small",
        license="Apache-2.0",
        approx_size_mb=184.3,
        description="SAM 2.1 Small checkpoint via Hugging Face Transformers.",
        revision="ee5bba1d82bb8749febdf90f45e84b687142ba03",
        kind="sam2",
    ),
    "sam2.1-base-plus": MaskModelSpec(
        name="sam2.1-base-plus",
        repo_id="facebook/sam2.1-hiera-base-plus",
        architecture="SAM 2.1 Hiera-Base+",
        license="Apache-2.0",
        approx_size_mb=323.5,
        description="Higher-capacity SAM 2.1 Base+ mask propagator.",
        revision="b7320756a13354e7530a63935656d35b2f91a290",
        kind="sam2",
    ),
    "sam2.1-large": MaskModelSpec(
        name="sam2.1-large",
        repo_id="facebook/sam2.1-hiera-large",
        architecture="SAM 2.1 Hiera-Large",
        license="Apache-2.0",
        approx_size_mb=897.9,
        description="Largest SAM 2.1 checkpoint for maximum mask quality on GPU.",
        revision="665f8e2ad61cf5f53d65644ff27c8ee525124610",
        kind="sam2",
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
    if precision not in PRECISION_TO_FILENAME:
        raise ValueError(f"Unknown precision {precision!r}. Choose fp32, fp16, or int8.")
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

    spec = MODEL_REGISTRY.get(model)
    if spec is None and (local_path.is_absolute() or model.endswith(".onnx")):
        raise FileNotFoundError(f"ONNX file not found: {model}")
    if spec is None and "/" not in model:
        raise ValueError(
            f"Unknown detector {model!r}. See 'trackflow models' for presets or use an owner/repo ID."
        )
    repo_id = spec.repo_id if spec else model
    kwargs = {"revision": spec.revision if spec else None, "cache_dir": cache_dir}
    if spec and spec.backend != "onnx":
        raise ValueError(
            f"{model} uses the {spec.backend} backend. Load it with create_detector()."
        )
    if spec and spec.onnx_file:
        if precision != "fp32":
            raise ValueError(f"{model} is available only in fp32.")
        return Path(hf_hub_download(repo_id, spec.onnx_file, **kwargs)), None, None
    onnx_filename = PRECISION_TO_FILENAME[precision]
    onnx_path = Path(hf_hub_download(repo_id=repo_id, filename=onnx_filename, **kwargs))

    config_path: Path | None = None
    prep_path: Path | None = None
    try:
        config_path = Path(hf_hub_download(repo_id=repo_id, filename="config.json", **kwargs))
    except EntryNotFoundError:
        if spec:
            raise
    try:
        prep_path = Path(
            hf_hub_download(repo_id=repo_id, filename="preprocessor_config.json", **kwargs)
        )
    except EntryNotFoundError:
        if spec:
            raise

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
    if precision not in spec.vision_files or precision not in spec.text_files:
        raise ValueError(f"Unsupported precision {precision!r} for {backend}.")
    vision_file = spec.vision_files[precision]
    text_file = spec.text_files[precision]
    kwargs = {"cache_dir": cache_dir, "revision": spec.revision}

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
