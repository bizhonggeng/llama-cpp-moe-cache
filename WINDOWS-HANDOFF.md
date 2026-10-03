# WINDOWS-HANDOFF — build llama.cpp (MoE expert cache + budget) on Windows

Target machine handoff. All commands are PowerShell. **No GitHub CLI needed**.
The repo is public, so plain `git` (or a zip download) is enough.

What this produces: `llama-server`, `llama-cli`, `llama-bench`, `llama-quant`
for Windows x64 + CUDA 12, with the small-VRAM MoE expert-cache patch built in
(`--moe-expert-cache-budget <MiB>` flag).

---

## 0. Get the code (pick ONE, no gh)

Option A - git clone (needs git; gives you the full patched branch):
```powershell
git clone -b build-win-cuda12 https://github.com/bizhonggeng/llama-cpp-moe-cache.git
cd llama-cpp-moe-cache
```

Option B - zip download (needs NO git at all; same files):
```powershell
Invoke-WebRequest -Uri "https://github.com/bizhonggeng/llama-cpp-moe-cache/archive/refs/heads/build-win-cuda12.zip" -OutFile repo.zip
Expand-Archive repo.zip -DestinationPath .
cd; cd llama-cpp-moe-cache-build-win-cuda12
```

Either way you end up in a folder containing `src\llama-moecache.cpp`,
`build-win.ps1`, `moe-cache\`, `STATUS.md`. If you don't see
`src\llama-moecache.cpp`, you are on the wrong source.

---

## 1. Prerequisites (install once if missing)

- Visual Studio 2022 **Build Tools**, workload "Desktop development with C++"
  (or full VS 2022). Check with:  `where msbuild`
- NVIDIA **CUDA Toolkit 12.x**. Check nvcc:
  ```powershell
  nvcc --version
  ```
  If "not recognized", install from https://developer.nvidia.com/cuda-downloads.
- **CMake** >= 3.20. Check:  `cmake --version`
- **Git** (only for option A).
- **coreinfo.exe** (Sysinternals) for the AVX-512 check below:
  ```powershell
  Invoke-WebRequest -Uri "https://download.sysinternals.com/files/Coreinfo.zip" -OutFile coreinfo.zip
  Expand-Archive coreinfo.zip -DestinationPath .\coreinfo
  ```

---

## 2. Detect AVX-512 (IMPORTANT - decide before building)

```powershell
.\coreinfo\Coreinfo64.exe -f | findstr /i AVX
```
- If the row `AVX512` has an asterisk `*`, your CPU supports AVX-512 -> you may
  build an `avx512` build (faster on AVX-512 cores).
- If `AVX512` shows `-` or the row is missing, CPU has **no AVX-512** -> build AVX2 only.
- Can't run coreinfo? **Default to AVX2** (safe). You can later test avx512 by
  building both and benchmarking - a CPU without AVX-512 will crash or run
  slowly on the AVX-512 binary.

Also pick your GPU arch (`-CudaArch`):
- `86`  = RTX 3050 / 3050Ti / 3060 (and RTX 20-series remaps) - most common
- `89`  = RTX 40xx
- `75`  = RTX 20xx
- `61`  = GTX 10xx
Not sure? Build `86`; mismatch only affects codegen speed, not correctness.

---

## 3. Build

AVX2 baseline (always safe):
```powershell
powershell -ExecutionPolicy Bypass -File .\build-win.ps1 -CudaArch 86 -Avx512:$false
```

If the CPU supports AVX-512, also build the comparison version:
```powershell
powershell -ExecutionPolicy Bypass -File .\build-win.ps1 -CudaArch 86 -Avx512:$true
```

Notes:
- Each run configures into `build\`, then builds, then copies exes + CUDA DLLs
  into `dist\` with a printed summary. See `dist\` for the artifact set.
- First build can take 20-60 min (nvcc compile). Later ones are incremental.
- If the "Long Path" or signing warning appears, ignore it; build continues.

---

## 4. Smoke test

```powershell
.\dist\llama-cli.exe -m <model>.gguf -ngl 999 -ncmoe 999 --moe-expert-cache-budget 1500 -ctk q8_0 -ctv q8_0 --no-mmap -n 8 -p "Hello"
```
Good signs in the log: a "MoE cache" / budget line prints, and if tokens
were previously hit you see cache hits. Also confirm the flag exists:
```powershell
.\dist\llama-cli.exe --help | findstr /i "moe-expert-cache-budget"
```

---

## 5. Benchmark (A/B) and report back

Compare SAME model, SAME chunk+prompt, two runs -> one with
`--moe-expert-cache-budget 1500`, one without. Record for me:
1. `dist\` file list + sizes (MB) - prove the bundle is complete
2. `nvidia-smi` total VRAM vs the model you load (real VRAM footprint)
3. tokens/sec for both runs (A = with budget, B = without)
4. Any error / crash in the no-budget run (that is the point - budget should
   stop the OOM you hit)
5. The model file name + its GGUF dims (from the load log; key_dim, n_gqa,
   n_lora, n_expert_count, etc.)

If you can also run `llama-bench`:
```powershell
.\dist\llama-bench.exe -m <model>.gguf -n 64 -t 4
```

Send those back and route A/B/C follows per BUILD_HANDOFF.md section 10.

---
Assisted-by: Hermes Agent session (2026-10-04)