---
name: onnx-vision-deploy
description: Practical checklist for deploying and debugging ONNX computer-vision models (RT-DETR, RF-DETR, YOLO, Re-ID, CLIP/SigLIP) across CPU, CUDA, Apple Silicon CoreML, and browser WebGPU/WASM. Use when exporting vision models to ONNX, wiring ONNX Runtime execution providers, fixing box/accuracy mismatches across devices, or keeping Python and browser pipelines in parity.
---

# ONNX Vision Deployment

Empirical checklist for shipping ONNX vision models without silent accuracy regressions across backends.

## 1. Measure Numerical Parity Against CPU First

Never treat "session loaded and returned boxes" as proof that an execution provider works. Before shipping any non-CPU backend (CoreML, TensorRT, WebGPU, FP16/INT8):

- Run at least **5 real non-trivial frames** (different aspect ratios, crowded scenes) on both `CPUExecutionProvider` and the target EP.
- Compare **detection count**, **matched-box IoU** (target `>= 0.90` on matched pairs), and **max confidence delta** (or cosine similarity `>= 0.99` for embedding models like OSNet/CLIP).
- Fail loudly when a requested `--device` is unavailable, and verify operator placement / wall-clock latency (via profiling or verbose EP logs) so silent per-operator CPU fallback is never claimed as GPU acceleration.

## 2. Match Training Preprocessing — Do Not Blindly Trust Configs

Preprocessing errors move downstream metrics far more than model or tracker tweaks:

- **YOLO family**: trained on **letterboxed** frames (preserve aspect ratio, pad to square with gray `114`, then unletterbox predicted boxes back to original frame pixels). Community ONNX `preprocessor_config.json` files frequently specify direct resize/stretch, which compresses 16:9 people to half width and severely degrades recall.
- **DETR family (RT-DETRv2, RF-DETR, D-FINE)**: trained on **stretched** inputs — do not apply letterboxing globally without measuring on a labeled split (it can hurt MOTA on DETR models).
- **Color order & normalization**: verify BGR vs RGB (`cv2.imread` is BGR), `[0, 1]` scaling vs ImageNet mean/std, and whether the ONNX graph bakes preprocessing in or expects raw normalized NCHW tensors.

## 3. Apple Silicon CoreML (`CoreMLExecutionProvider`) Gotchas

ONNX Runtime's default CoreML settings (`NeuralNetwork` on `ALL` compute units) frequently fall back to CPU *and* can return wrong boxes silently:

- **Prefer `MLProgram` + `CPUAndGPU`** (`ModelFormat: MLProgram`, `MLComputeUnits: CPUAndGPU`, requires `onnxruntime>=1.20`) for modern detectors and Re-ID backbones — it covers far more ops and typically matches CPU boxes while cutting CPU core usage by ~2–3×.
- **Keep per-model format overrides**: some graphs abort in Metal's graph compiler or fail to compile `GridSample` under `MLProgram` and must stay on `NeuralNetwork`; INT8 CLIP/SigLIP graphs may fail to load under `MLProgram` or produce `NaN` on `CPUAndGPU`.
- **Benchmark before changing `auto`**: some architectures (e.g. RF-DETR) run 3–4× slower on CoreML GPU than on Apple Silicon CPU. Measure wall-clock ms *and* CPU core load per model family before making CoreML the default.

## 4. Keep Browser WebGPU / WASM in Automated Parity

When shipping both a Python pipeline and a browser WebGPU/WASM demo:

- Keep preprocessing (`letterbox`, `unletterbox`, coordinate scaling) and post-processing (thresholding, NMS, tracker association) as pure functions in JS.
- Add a **Node-backed parity test** in the Python test suite that feeds identical boxes/frame sizes through both Python and JS and asserts numerical equality. Do not rely on manual browser eyeballing to catch coordinate drift.
- Test the browser demo on WASM as well as WebGPU (headless browsers and Safari/Firefox often hit WASM fallback).

## 5. Pin Weights and Separate Code vs. Weight Licenses

- Pin every remote model (e.g. Hugging Face Hub) to an exact commit revision SHA so reruns produce identical weights.
- Keep model weight licenses explicit in code and docs: an Apache-2.0 wrapper does **not** re-license AGPL-3.0 (e.g. Ultralytics YOLO) or non-commercial weights.
- Cap CPU thread counts (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`) in benchmarks and batch jobs on laptops so runs remain reproducible and thermal throttling does not skew timings.
