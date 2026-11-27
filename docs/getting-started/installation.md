# Installation & Hardware

Trackflow ships three separate PyPI packages built from the same source tree so each environment pulls the exact ONNX Runtime build it needs with no conflicting wheels or manual library paths.

## Choose your package

| Environment | Zero-install (`uvx`) | Persistent install (`pip`) | Bundled runtime |
| :--- | :--- | :--- | :--- |
| **CPU / macOS** *(default)* | `uvx trackflow-python run video.mp4` | `pip install trackflow-python` | `onnxruntime` (CPU + CoreML on macOS) |
| **macOS GPU (CoreML)** | `uvx trackflow-python run video.mp4 --device coreml` | `pip install trackflow-python` | `CoreMLExecutionProvider` |
| **NVIDIA CUDA** | `uvx trackflow-python-cuda run video.mp4` | `pip install trackflow-python-cuda` | `onnxruntime-gpu[cuda,cudnn]` (auto-selects CUDA) |
| **NVIDIA TensorRT** *(Python 3.11+)* | `uvx trackflow-python-tensorrt run video.mp4` | `pip install trackflow-python-tensorrt` | `onnxruntime-gpu[cuda,cudnn]>=1.28` + `tensorrt-cu13-libs>=10,<11` (defaults `--device` to `tensorrt`) |

!!! warning "Never mix `onnxruntime` and `onnxruntime-gpu` in one virtual environment"
    Both packages install into the same `onnxruntime` Python namespace. Installing `trackflow-python` alongside `trackflow-python-cuda` or `trackflow-python-tensorrt` overwrites shared files and silently disables GPU execution providers. If you switch variants in an existing environment, uninstall `onnxruntime`, `onnxruntime-gpu`, and the previous `trackflow-python*` package first, or create a clean venv.

All three packages provide the `trackflow` CLI command and the `import trackflow` Python module.

## Verify your environment

Run `trackflow info` after installing (or add `--json` for scripts):

```bash
trackflow info
```

`trackflow info` reports:

- Trackflow, Python, and ONNX Runtime versions
- Detected NVIDIA GPU (via `nvidia-smi`) or Apple Silicon performance core count
- Preloaded pip CUDA/cuDNN/TensorRT shared libraries
- Active ONNX Runtime execution providers (`CPUExecutionProvider`, `CoreMLExecutionProvider`, `CUDAExecutionProvider`, `TensorrtExecutionProvider`)
- A recommended GPU command if an NVIDIA GPU is present while only the CPU runtime is installed

## Hardware details

### Apple Silicon (macOS)

- **CPU (`--device cpu` or default `auto`)**: Trackflow reads `hw.perflevel0.physicalcpu` via `sysctlbyname` and pins ONNX Runtime intra-op threads to performance cores instead of spilling onto efficiency cores. On an Apple M4 Pro, `rtdetr-v2-r18` runs at ~12–14 FPS end-to-end.
- **CoreML (`--device coreml`)**: Offloads the detector and static-batch `osnet` Re-ID crops to the Apple GPU / Neural Engine. Each model specifies `coreml_format` (`MLProgram` by default; `NeuralNetwork` for `rfdetr-base` and `deimv2-*` where Apple's `MLProgram` compiler fails).

### NVIDIA CUDA (`trackflow-python-cuda`)

- Requires only an NVIDIA driver on the host — no system CUDA Toolkit or cuDNN installation.
- Before creating an ONNX Runtime session, Trackflow preloads the pip-installed CUDA and cuDNN shared libraries (`preload_gpu_libraries()`), so `CUDAExecutionProvider` initializes even when `/usr/local/cuda` is absent.
- Uses `cudnn_conv_algo_search="HEURISTIC"` for fast startup without exhaustive multi-second autotuning.
- Verified at **37.9 FPS** (`rtdetr-v2-r18`) to **86.8 FPS** (`yolo26n`) on an NVIDIA Tesla T4.

### NVIDIA TensorRT (`trackflow-python-tensorrt`)

- Requires **Python 3.11+** and an NVIDIA GPU driver.
- Bundles CUDA 13, cuDNN, and TensorRT 10 shared libraries (`tensorrt-cu13-libs>=10,<11`) and preloads `libnvinfer.so.10`, `libnvinfer_plugin.so.10`, and `libnvonnxparser.so.10` automatically before ORT initializes.
- The `trackflow-python-tensorrt` entrypoint sets `TRACKFLOW_DEFAULT_DEVICE=tensorrt` so `uvx trackflow-python-tensorrt run street.mp4` uses TensorRT without extra flags.
- **Engine cache**: The first run compiles an optimized FP16 TensorRT engine for each model and saves both the `.engine` and timing cache under `~/.cache/trackflow/trt_cache/`. Cold compilation takes ~3–4 minutes per detector on a T4; every subsequent run loads from cache in **<1 second**.
- **Automatic FP32 guard for `rfdetr-*`**: DINOv2 activations inside RF-DETR overflow FP16 in TensorRT and drop recall. Trackflow sets `trt_fp16=False` on `rfdetr-*` presets automatically so box accuracy stays identical to CPU (`1.000` mean IoU).
- **OSNet Re-ID**: Runs in fixed static batches of 8 crops on TensorRT (and CoreML) to avoid dynamic-shape recompilation.

## Optional extras

| Extra | Install command | Adds |
| :--- | :--- | :--- |
| `transformers` | `pip install "trackflow-python[transformers]"` | PyTorch + Transformers + Sentence-Transformers for `rfdetr-large`, `hf:owner/repo` detectors, and `--vlm embeddinggemma` |
| `display` | `pip install "trackflow-python[display]"` | OpenCV GUI (`opencv-python`) for the `--display` live preview window |
| `docs` | `pip install "trackflow-python[docs]"` | MkDocs Material + `mkdocstrings` for building this documentation locally |

To combine extras with a GPU variant, install the extra on that package name, for example `pip install "trackflow-python-cuda[transformers]"`.

## Docker

The repository includes a multi-target [`Dockerfile`](https://github.com/obendidi/trackflow/blob/main/Dockerfile) with `cpu`, `cuda`, and `tensorrt` targets. Mount both the Hugging Face cache and Trackflow's TensorRT cache so weights and compiled engines persist across container runs:

```bash
# Build target: cpu, cuda, or tensorrt
docker build --target tensorrt -t trackflow:tensorrt .

# Run on a local video with persistent weight and TensorRT caches
docker run --rm --gpus all \
  -v "$PWD:/work" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/trackflow:/root/.cache/trackflow" \
  trackflow:tensorrt run /work/street.mp4
```

For CPU-only Docker runs, build with `--target cpu` and drop `--gpus all`.

## Troubleshooting

- **`TensorrtExecutionProvider` not listed**: Confirm you installed `trackflow-python-tensorrt` (not `trackflow-python`), Python is `>= 3.11`, and `nvidia-smi` succeeds on the host. Check `trackflow info` to see which TensorRT shared libraries preloaded.
- **GPU ignored after installing another package**: Run `pip list | grep onnxruntime`. If both `onnxruntime` and `onnxruntime-gpu` appear, uninstall both and reinstall only your target `trackflow-python-cuda` or `trackflow-python-tensorrt` package.
- **Weight downloads behind a firewall or offline host**: Pre-populate `~/.cache/huggingface` on a connected machine (run once with `--max-frames 1`) and copy or mount the cache directory. All built-in presets are pinned to exact Hugging Face commit hashes.
