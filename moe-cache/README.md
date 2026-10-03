# MoE expert LRU cache — small-VRAM adaptation of llama.cpp PR #27861

## What this is

Upstream PR [ggml-org/llama.cpp#27861](https://github.com/ggml-org/llama.cpp/pull/27861)
adds a GPU-resident LRU cache for MoE expert weights that live in host memory.
It is the single largest decode win available for "experts on CPU, attention on GPU"
setups: **+84% at N=64, +52% at N=32** on Qwen3.8-Flash-Next (RFC #28248).

It has not been merged in ~2 months. This directory carries it plus one adaptation.

## Why the adaptation

PR #27861 exposes `--moe-expert-cache N` — you state a slot count yourself.
On a 4-8 GB card that is the wrong knob, for two measured reasons:

1. **Too small is worse than off.** RFC #28248: "N = top-k is a full-miss worst
   case and is measurably slower than master; gains start at ~2x top-k."
2. **Too large also inverts.** koren1712: 8 GiB slab 8.70 t/s vs 16 GiB slab
   7.16 t/s — *identical* hit counts and streamed bytes. Past working-set
   coverage, extra residency buys no fewer reads and only adds memory pressure.

So the user-facing knob should be a **device-memory budget**, and the allocator
should refuse to run below the break-even floor. That is what `apply_budget_patch.py` adds.

## Files

| file | what it is |
|---|---|
| `0001-moe-expert-cache-upstream-PR27861.patch` | PR #27861 verbatim (826 lines, 12 files) |
| `apply_budget_patch.py` | adds `--moe-expert-cache-budget MiB` + the 2x top_k floor |
| `budget.py` | tells you which number to pass for your card and model |

## Apply

```bash
cd /path/to/llama.cpp

# 1. the upstream PR
git fetch origin pull/27861/head:pr-27861
git merge pr-27861

# 2. the small-VRAM adaptation (idempotent)
python3 /path/to/apply_budget_patch.py

# 3. build
cmake -B build -DGGML_CUDA=ON && cmake --build build -j
```

`python3 apply_budget_patch.py --check` reports what would change without writing.

## Run

```bash
./build/bin/llama-server \
  -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 999 -ncmoe 999 \
  --moe-expert-cache-budget 2048 \
  -ctk q8_0 -ctv q8_0 \
  --no-mmap --mlock \
  -t 8
```

Expect a line at startup:

```
moecache: budget 2048 MiB -> 30 slots/layer across 40 layer(s), 1.69 MiB per expert (2025.6 MiB device, floor 16)
```

and every 512 steps:

```
moe-cache: steps=512 hits=... misses=... hit-rate=..%
```

If the budget is under the floor the cache **disables itself** and says so —
that is intentional, not a bug.

## Choosing the number

```bash
python3 budget.py --vram 4      # your 3050 Ti / 4 GB tier
python3 budget.py --vram 8
python3 budget.py --vram 8 --free 5.0    # override the free-VRAM estimate
```

Default assumes Qwen3.6-35B-A3B (40 MoE layers, 256 experts, top-8, 4.5 bpw).
Override with `--hidden --intermediate --layers --experts --topk --bpw` for another model.

Rule of thumb from the curve:

| budget | slots/layer on Qwen3.6-35B-A3B | expected gain |
|---|---|---|
| 1024 MiB | 15 | **below floor — disabled** |
| 1536 MiB | 22 | +33% |
| 2048 MiB | 30 | +49% |
| 4096 MiB | 60 | +79% |
| >6144 MiB | 91+ | +84% (saturated; stop here) |

## Caveats

- **Not compiled.** The patch is derived by textual rewriting against the PR #27861
  hunks. Build it and read the startup log before trusting the numbers.
- The gain curve in `budget.py` interpolates RFC #28248's four measured points
  (N=16/32/48/64 at top_k=8) onto a multiple-of-top_k axis. It is an estimate,
  not a measurement on your hardware.
- PR #27861 is decode-only (`n_tokens == 1`); MTP / speculative decoding currently
  bypasses the cache, so `--moe-expert-cache` and speculative gains do not stack.
- Long-session stability was verified by the RFC author (byte-identical anchor
  outputs over 4 runs, 2561 tokens); the CPU/device split is exact by construction.

## If you want to push further

The RFC author's own list of what the tier string cannot express yet:

1. an absolute VRAM rail (on 32 GB WDDM cards the paging line sits at ~0.8-0.9 GB
   free *at max context* — 83 MiB separates "no effect" from "TTFT x2.3")
2. cost-based rather than hit-rate-based admission: promote when
   `p_e * bytes_saved - upload_cost > 0`
3. per-layer, not per-model, sizing (routing entropy differs by layer:
   DeepSeek-V4-Flash measured 6.3-7.9 bits)
