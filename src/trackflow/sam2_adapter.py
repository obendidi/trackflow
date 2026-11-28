"""Optional Apache-2.0 SAM 2.1 / EdgeTAM video mask propagator (requires ``trackflow-python[sam2]``)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from trackflow.mask_fallback import BBox
from trackflow.models import MASK_REGISTRY


def _import_sam2_deps() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError(
            "SAM 2.1 / EdgeTAM mask fallback requires the optional [sam2] extra: "
            "pip install 'trackflow-python[sam2]'"
        ) from exc
    return torch, transformers


def _resolve_torch_device(torch: Any, device: str) -> str:
    req = device.lower().strip()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req in ("cuda", "gpu", "tensorrt") or req.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA requested ({device!r}) for SAM 2.1 mask fallback, but no CUDA device is available."
            )
        return req if req.startswith("cuda:") else "cuda"
    if req in ("mps", "coreml"):
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError(
            f"MPS requested ({device!r}) for SAM 2.1 mask fallback, but MPS is unavailable on this host."
        )
    if req == "cpu":
        return "cpu"
    raise ValueError(
        f"Unsupported device {device!r} for SAM 2.1 mask fallback. "
        "Choose auto, cpu, cuda, cuda:<idx>, mps, coreml, or tensorrt."
    )


def _select_dtype(torch: Any, device: str) -> Any:
    if device.startswith("cuda"):
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class Sam2MaskPropagator:
    """Apache-2.0 SAM 2.1 / EdgeTAM propagator via Hugging Face Transformers.

    Keeps all frames strictly in RAM (``writes_frames_to_disk = False``), runs on a 1024x1024
    input grid, and returns native 256x256 boolean masks directly to avoid full-resolution
    upsampling overhead while keeping identical mask-to-box containment scores.
    """

    writes_frames_to_disk: bool = False

    def __init__(
        self,
        model: str = "sam2.1-tiny",
        *,
        device: str = "auto",
        cache_dir: str | Path | None = None,
    ) -> None:
        spec = MASK_REGISTRY.get(model)
        if spec is None and not model.startswith("hf:"):
            raise ValueError(
                f"Unknown mask fallback model {model!r}. "
                f"Available presets: {', '.join(MASK_REGISTRY)} (or 'hf:owner/repo')."
            )
        repo_id = spec.repo_id if spec is not None else model[3:].strip()
        if not repo_id:
            raise ValueError("Custom 'hf:' mask fallback model must include an owner/repo ID.")
        revision = spec.revision if spec is not None else None
        kind = (
            spec.kind
            if spec is not None
            else ("edgetam" if "edgetam" in repo_id.lower() else "sam2")
        )

        torch, transformers = _import_sam2_deps()
        self.model_name = model
        self.repo_id = repo_id
        self.revision = revision
        self.device = _resolve_torch_device(torch, device)
        self._torch = torch
        self._dtype = _select_dtype(torch, self.device)

        load_kwargs: dict[str, Any] = {"dtype": self._dtype, "trust_remote_code": False}
        if revision is not None:
            load_kwargs["revision"] = revision
        if cache_dir is not None:
            load_kwargs["cache_dir"] = str(cache_dir)

        proc_kwargs: dict[str, Any] = {"trust_remote_code": False}
        if revision is not None:
            proc_kwargs["revision"] = revision
        if cache_dir is not None:
            proc_kwargs["cache_dir"] = str(cache_dir)

        if kind == "edgetam":
            self._model = (
                transformers.EdgeTamVideoModel.from_pretrained(repo_id, **load_kwargs)
                .to(self.device)
                .eval()
            )
            self._processor = transformers.AutoProcessor.from_pretrained(repo_id, **proc_kwargs)
        else:
            self._model = (
                transformers.Sam2VideoModel.from_pretrained(repo_id, **load_kwargs)
                .to(self.device)
                .eval()
            )
            self._processor = transformers.Sam2VideoProcessor.from_pretrained(
                repo_id, **proc_kwargs
            )

    def propagate_window(
        self,
        frames_rgb: Sequence[np.ndarray],
        seeds: dict[int, tuple[int, BBox]],
        frame_size_wh: tuple[int, int],
    ) -> dict[int, dict[int, np.ndarray]]:
        if not frames_rgb or not seeds:
            return {}
        torch = self._torch
        orig_w = max(1.0, float(frame_size_wh[0]))
        orig_h = max(1.0, float(frame_size_wh[1]))
        sx = 1024.0 / orig_w
        sy = 1024.0 / orig_h

        frames_1024 = [
            cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            if frame.shape[:2] != (1024, 1024)
            else frame
            for frame in frames_rgb
        ]

        # Group prompts by local frame index so every conditioning frame runs once before propagation.
        by_lf: dict[int, tuple[list[int], list[list[float]]]] = {}
        for oid in sorted(seeds.keys()):
            lf, box = seeds[oid]
            lf = max(0, min(len(frames_1024) - 1, int(lf)))
            x1 = max(0.0, min(1023.0, float(box[0]) * sx))
            y1 = max(0.0, min(1023.0, float(box[1]) * sy))
            x2 = max(1.0, min(1024.0, float(box[2]) * sx))
            y2 = max(1.0, min(1024.0, float(box[3]) * sy))
            if x2 <= x1 + 1.0 or y2 <= y1 + 1.0:
                continue
            ids_l, boxes_l = by_lf.setdefault(lf, ([], []))
            ids_l.append(int(oid))
            boxes_l.append([x1, y1, x2, y2])

        if not by_lf:
            return {}

        device_type = "cuda" if self.device.startswith("cuda") else self.device
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device_type,
                dtype=self._dtype,
                enabled=(device_type == "cuda"),
            ),
        ):
            session = self._processor.init_video_session(
                video=frames_1024,
                inference_device=self.device,
                dtype=self._dtype,
            )
            for lf in sorted(by_lf.keys()):
                ids_l, boxes_l = by_lf[lf]
                self._processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=lf,
                    obj_ids=ids_l if len(ids_l) > 1 else ids_l[0],
                    input_boxes=[boxes_l],
                )
                _ = self._model(inference_session=session, frame_idx=lf)

            masks_by_lf: dict[int, dict[int, np.ndarray]] = {}
            start_lf = min(by_lf.keys())
            for out in self._model.propagate_in_video_iterator(session, start_frame_idx=start_lf):
                lf = int(out.frame_idx)
                pm = out.pred_masks > 0.0
                if pm.ndim == 4:
                    pm = pm[:, 0]
                pm_np = pm.detach().cpu().numpy()
                obj_ids = list(getattr(session, "obj_ids", ()))
                f_dict: dict[int, np.ndarray] = {}
                for oi, oid in enumerate(obj_ids):
                    if oi < pm_np.shape[0]:
                        f_dict[int(oid)] = pm_np[oi]
                masks_by_lf[lf] = f_dict
            del session
        return masks_by_lf


def create_mask_propagator(
    backend: str,
    *,
    device: str = "auto",
    cache_dir: str | Path | None = None,
) -> Sam2MaskPropagator:
    """Instantiate an optional Apache-2.0 SAM 2.1 / EdgeTAM propagator by preset or ``hf:`` id."""
    return Sam2MaskPropagator(model=backend, device=device, cache_dir=cache_dir)
