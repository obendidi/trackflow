# Models & Licenses

Trackflow's own source code is **Apache-2.0**. Model weights download separately from Hugging Face on first use, are pinned to exact commit revisions for reproducibility, and retain their upstream licenses. Check the license column before deploying a preset commercially.

Run `trackflow models` (or `trackflow models --json` for scripts) to inspect every built-in preset, Hugging Face repository ID, pinned commit hash, and supported precision.

## Built-in object detectors

All built-in detectors output the 80 COCO classes (`person`, `car`, `truck`, `bicycle`, `backpack`, etc.).

| Preset | Architecture & input size | Size (FP32) | Backend | License | Notes |
| :--- | :--- | ---: | :--- | :--- | :--- |
| `rtdetr-v2-r18` *(default)* | RT-DETRv2 ResNet-18 (640×640) | 81.1 MB | ONNX | Apache-2.0 | Balanced accuracy and speed; 111.7 FPS on T4 TensorRT FP16 |
| `dfine-n` | D-FINE Nano (640×640) | 15.3 MB | ONNX | Apache-2.0 | Smallest Apache-2.0 ONNX download |
| `dfine-s` | D-FINE Small (640×640) | 41.5 MB | ONNX | Apache-2.0 | Fast COCO transformer; 121.8 FPS on T4 TensorRT FP16 |
| `dfine-m` | D-FINE Medium (640×640) | 78.6 MB | ONNX | Apache-2.0 | Highest MOT17 HOTA (`46.5`) among Apache-2.0 ONNX presets |
| `rfdetr-nano` | RF-DETR Nano (384×384) | 108.1 MB | ONNX | Apache-2.0 | Fast on Apple Silicon and WebGPU; stays FP32 on TensorRT |
| `rfdetr-small` | RF-DETR Small (512×512) | 114.7 MB | ONNX | Apache-2.0 | Stays FP32 on TensorRT |
| `rfdetr-base` | RF-DETR Base (560×560) | 107.8 MB | ONNX | Apache-2.0 | Uses CoreML `NeuralNetwork` format on macOS; stays FP32 on TensorRT |
| `rfdetr-medium` | RF-DETR Medium (576×576) | 121.0 MB | ONNX | Apache-2.0 | Stays FP32 on TensorRT |
| `rfdetr-large` | RF-DETR Large (704×704) | 135.8 MB | Transformers | Apache-2.0 | Official Roboflow checkpoint; requires `trackflow-python[transformers]` |
| `yolo26n` | YOLO26 Nano (640×640) | 9.9 MB | ONNX | **AGPL-3.0 / Enterprise** | Commercial use requires an Ultralytics license |
| `yolo26s` | YOLO26 Small (640×640) | 38.2 MB | ONNX | **AGPL-3.0 / Enterprise** | Commercial use requires an Ultralytics license |
| `yolo26m` | YOLO26 Medium (640×640) | 81.9 MB | ONNX | **AGPL-3.0 / Enterprise** | Commercial use requires an Ultralytics license |
| `yolo26l` | YOLO26 Large (640×640) | 99.6 MB | ONNX | **AGPL-3.0 / Enterprise** | Commercial use requires an Ultralytics license |
| `yolo26x` | YOLO26 XLarge (640×640) | 223.2 MB | ONNX | **AGPL-3.0 / Enterprise** | Commercial use requires an Ultralytics license |
| `deimv2-atto` | DEIMv2 Atto (320×320) | 2.2 MB | ONNX | **Non-commercial** | Tiny FP32-only edge model |
| `deimv2-s` | DEIMv2 Small / DINOv3 (640×640) | 40.7 MB | ONNX | **Non-commercial + DINOv3** | FP32-only |

### Weight precision (`--precision`)

Most ONNX presets offer `fp32` (default), `fp16`, and `int8` weight files via `--precision`:

```bash
trackflow run street.mp4 --model dfine-n --precision int8
```

Two runtime rules apply automatically:

- **`deimv2-atto` and `deimv2-s`** ship standalone FP32 graphs only (`--precision fp32`).
- **TensorRT FP16 (`trackflow-python-tensorrt`)** compiles FP32 ONNX weights to FP16 Tensor Core engines automatically for `rtdetr-v2-r18`, `dfine-*`, `yolo26*`, and `deimv2-*`. Keep `--precision fp32` when running TensorRT so TensorRT quantizes cleanly from FP32 weights. For `rfdetr-*`, Trackflow disables TensorRT FP16 (`trt_fp16=False`) because DINOv2 activations overflow FP16 and drop detections.

## Person Re-ID and vision-language models

| Flag | Model | Size | License | Purpose |
| :--- | :--- | ---: | :--- | :--- |
| `--reid osnet` | OSNet-AIN x1.0 (`mprib/osnet-ain-x1-0-msmt17-onnx`) | 8.7 MB | MIT code; MSMT17 weights (check dataset terms for commercial use) | BoT-SORT / BoostTrack appearance matching and `--reid-merge` |
| `--reid color` | HSV color histogram | 0 MB | Apache-2.0 (built-in) | Offline heuristic fallback with no download (weaker than `osnet`) |
| `--vlm clip` | CLIP ViT-B/32 INT8 (`Xenova/clip-vit-base-patch32`) | 153 MB | MIT | Default for `--query` and `trackflow search` (fastest on CPU) |
| `--vlm siglip` | SigLIP Base INT8 (`Xenova/siglip-base-patch16-224`) | 205 MB | Apache-2.0 | Stronger clothing/attribute retrieval than CLIP |
| `--vlm siglip2` | SigLIP 2 Base INT8 (`onnx-community/siglip2-base-patch16-224-ONNX`) | 378 MB | Apache-2.0 | Strongest pure-ONNX search model (no PyTorch) |
| `--vlm embeddinggemma` | EmbeddingGemma 2 (`google/embeddinggemma-2`) | 1,489 MB | Apache-2.0 | Highest search mAP; PyTorch via `trackflow-python[transformers]` |

## Custom detectors

You are not restricted to the built-in presets or COCO's 80 classes.

### Local `.onnx` file or Hugging Face ONNX repo

Pass a local file path or any Hugging Face `owner/repo` that exports `onnx/model.onnx` (plus optional `config.json` and `preprocessor_config.json`):

```bash
trackflow run conveyor.mp4 --model ./weights/my_detector.onnx --classes all
trackflow run street.mp4 --model onnx-community/rtdetr_r50vd-ONNX
```

When given a local `.onnx` file, Trackflow looks for `config.json` and `preprocessor_config.json` in the same folder and its parent directory to pick up label names (`id2label`) and input preprocessing automatically.

### Hugging Face Transformers (`hf:owner/repo`)

With `pip install "trackflow-python[transformers]"`, prefix any Transformers object-detection checkpoint with `hf:`:

```bash
trackflow run crowd.mp4 --model hf:PekingU/rtdetr_v2_r50vd --device mps
```

### Custom Python `Detector`

In Python, pass any object implementing `detect(frame_bgr) -> list[Detection]` to `TrackflowPipeline(detector=...)`:

```python
import numpy as np
from trackflow import Detection, TrackflowPipeline

class MyDetector:
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        # Run your own model and return pixel xyxy boxes
        return [Detection(bbox=(100.0, 50.0, 220.0, 310.0), confidence=0.92, class_id=0, class_name="pallet")]

pipeline = TrackflowPipeline(detector=MyDetector(), tracker="botsort", classes=["pallet"])
summary = pipeline.run("warehouse.mp4")
```
