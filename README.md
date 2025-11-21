# GEMM Micro-Bench (V1 • V2 • V3)

Lightweight, one-cell PyTorch micro-benchmarks for matrix multiply (**`A @ B`**) on NVIDIA GPUs (tested on **Tesla G4/T4**). Each version is self-contained and runnable in Colab or locally. The suite produces CSV logs and plots to help you compare **throughput (TFLOPS)**, **scaling vs. size**, and—starting in V3—**energy efficiency (TFLOPS/W)** across backends and dtypes.

> **Hardware used for the reference runs:** Tesla G4 (T4-class).
> **Notes:** V1/V2 compare FP32 vs FP16; **BF16 is introduced in V3** and auto-skips on GPUs that don’t support it.

---

## Quick Start

### Run in Colab

* Open the desired version file and run the single cell.
* Colab GPU: **Tesla G4** runtime (or any CUDA-capable GPU).

### Run locally

```bash
# (Optional) create venv/conda
torch pandas numpy matplotlib scikit-learn nvidia-ml-py3
# For V3, NVML must be available (NVIDIA driver on host)
```

---

## Version Overview

### V1 — Baseline

**File:** `gemm_microbench_v1.py`
**Focus:** Minimal matmul micro-bench to sanity-check TFLOPS.
**Key settings:**

* Op: `matmul`
* Dtypes: FP32 vs FP16 (CUDA)
* Sizes: M,N ∈ {128, 256, 512}, K ∈ {128, 256, 512}
* Iter/Warmup: 20 / 5
* TF32 allowed, `torch.compile=True`
* Artifacts: `results.csv`, `heatmap.png`

---

### V2 — Extended Size Sweep + Curves

**File:** `gemm_microbench_v2.py`
**Improvements:**

* Expanded grid: M,N,K ∈ {256, 512, 1024, 2048}
* Added **FP32 vs FP16 curve** (square GEMM)
* Simple **RandomForest TFLOPS predictor** (M,N,K,dtype → TFLOPS)

**Artifacts:**

* `results.csv`
* `heatmap.png`
* `curves_fp32_vs_fp16.png`

---

### V3 — Power + TFLOPS/W + Backends + BF16

**File:** `gemm_microbench_v3.py`
**Major upgrades:**

* **NVML power sampling** → adds **TFLOPS/W** metric
* **Backends:** `default`, `inductor`, `aot_eager`
* **BF16** support (skipped if unsupported on device)
* Efficiency bar chart (top-N TFLOPS/W)

**Artifacts:**

* `results_v3.csv`
* `heatmap_v3.png`
* `efficiency_v3.png`

---

## Outputs

| Version | CSV File         | Key Visuals                    | Notes                      |
| ------- | ---------------- | ------------------------------ | -------------------------- |
| **V1**  | `results.csv`    | Heatmap (TFLOPS)               | Baseline speed test        |
| **V2**  | `results.csv`    | Heatmap + FP32 vs FP16 curves  | Size scaling + predictor   |
| **V3**  | `results_v3.csv` | Heatmap + Efficiency bar chart | Adds power + backend sweep |

CSV columns include: `timestamp`, `device`, `dtype`, `M`, `N`, `K`, `iters`, `elapsed_s`, `tflops`, and in V3, also `avg_power_w`, `tflops_per_watt`, `backend`.

---

## Reproducing the Reference Runs

1. Launch Colab GPU runtime (Tesla G4 preferred).
2. Copy the code cell from the version file.
3. Run all cells sequentially.
4. Verify plots are saved in working directory.

Tips:

* For **V3**, NVML must be initialized — ensure drivers are visible to the runtime.
* **BF16** is only available on **Ampere+ GPUs (A10, A100, L4, etc.)**.

---

## FAQ

**Why mention CPU?**
Scripts can fall back to CPU for portability, but reference runs use **GPU only (Tesla G4/T4)**.

**Why do backends differ?**
Compilation overhead dominates at small sizes. For fair backend comparison, focus on 512–2048.

**What if NVML shows `n/a`?**
Means power metrics unavailable; TFLOPS/W will be skipped but TFLOPS will still log.

**What’s the RandomForest predictor?**
A toy model in V2 estimating TFLOPS from (M,N,K,dtype). Useful for direction, not absolute accuracy.

---

## License

MIT License — Free to use and modify.

---

### Citation

If you use this benchmark in reports or educational demos, cite as:

```
GEMM Micro-Bench v1–v3 (2025). Simple PyTorch GEMM micro-benchmark series for performance & efficiency profiling.
```
