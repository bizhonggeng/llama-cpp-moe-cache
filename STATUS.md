# llama-cpp-moe-cache - Build Status & Takeover

## What the repo contains
Patched llama.cpp for the **MoE expert cache + small-VRAM budget** feature:
- Upstream PR #27861 (MoE expert cache) merged.
  Base `eec18f5d3` + merge commit `67f58f2ab` (src/CMakeLists.txt conflict
  resolved: main's `LLAMA_CORE_SOURCES` aggregate + `llama-moecache.cpp`).
- Budget patch `b4e04c3db` applied (9/9 hunks) exposing
  `--moe-expert-cache-budget <MiB>` bin/CI flag with a 2x top_k break-even
  floor. Applies to `include/llama.h`, `common/common.h`, `common/arg.cpp`,
  `src/llama-moecache.{h,cpp}`, `src/llama-context.cpp`.
- `moe-cache/` helper scripts + `BUILD_HANDOFF.md` (original task doc).
- `build-win.ps1` - build & bundle directly on a Windows machine.
- `.github/workflows/build-win-cuda12.yml` - CMake recipe reference.

## Branch layout
- `build-win-cuda12` - the build branch (HEAD carries everything above).
- `main` - mirrored (README + merge + patch + scripts + workflow).
Both are pushed to `github.com/bizhonggeng/llama-cpp-moe-cache` (public).

## Hard blocker found (why we pivoted)
The CUDA **silent installer does not run on GitHub-hosted windows-2022
runners**: it exits `0xE0E00019` (=-522190823) immediately, regardless of:
- Jimver/cuda-toolkit v0.2.30 vs direct network-installer (Start-Process -s)
- method local vs network, sub-packages (incl. dropping visual_studio_integration)
- cache on/off, TEMP redirected to D:, disks (C: had 82.7 GB free)

Dec 2025/2026: the hosted windows image is incompatible with NVIDIA's SFX
installer in silent mode. **Conclusion: do not fight this on hosted runners.**

## Pivot (agreed)
Build **on the target Windows machine** (or a self-hosted runner) where CUDA
is installed normally. `build-win.ps1` automates configure/build/bundle.

## Build on target machine
Prereqs: VS 2022 Build Tools (C++), CUDA Toolkit 12.x, CMake >= 3.20, git.
1. Detect AVX-512 first:
   `coreinfo.exe -f | findstr /i AVX`   (Sysinternals; blank AVX-512 row = AVX2 only)
2. Configure your CPU/card: `-CudaArch 86|89|75|61`
3. Build (AVX2 baseline first, then `-Avx512:$true` if the CPU supports it):
   ```
   git clone -b build-win-cuda12 https://github.com/bizhonggeng/llama-cpp-moe-cache
   cd llama-cpp-moe-cache
   powershell -ExecutionPolicy Bypass -File build-win.ps1 -CudaArch 86 -Avx512:$false
   ```
   Output lands in `dist\` (exes + cudart/cublas/cublasLt DLLs + LICENSE).

## SMOKE check on target machine
```
dist\llama-cli.exe -m <model>.gguf -ngl 999 -ncmoe 999 --moe-expert-cache-budget 1500 -ctk q8_0 -ctv q8_0 --no-mmap -n 8 -p "Hello"
```
Expected: decreases/budget printout in logs; Llama.cpp `--help` shows
`--moe-expert-cache-budget`.

## Benchmark / report back (from handoff §10.2)
Run A/B: budget vs no-budget on identical chunk+prompts. Record real VRAM &
the model dims I asked for in BUILD_HANDOFF. Route decision A/B/C per harvest.

Assisted-by: Hermes Agent session (pivot recorded 2026-10-04)