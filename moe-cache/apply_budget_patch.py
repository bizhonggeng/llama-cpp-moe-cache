#!/usr/bin/env python3
"""
Small-VRAM adaptation for llama.cpp PR #27861 (MoE expert LRU cache).

PR #27861 exposes --moe-expert-cache N, i.e. the user must state a slot count.
On a 4-8 GB card that is unusable, for two measured reasons:

  (a) RFC ggml-org/llama.cpp#28248: N == top_k is a full-miss worst case and is
      measurably SLOWER than master; gains only start at roughly 2x top_k.
  (b) Over-residency inverts the result: koren1712 measured an 8 GiB slab at
      8.70 t/s vs a 16 GiB slab at 7.16 t/s with identical hit counts.

So the user-facing knob has to be a device-memory BUDGET, not a slot count,
and the allocator must refuse to run below the break-even floor.

This script rewrites the PR #27861 hunks in place. It is idempotent.

Usage:  cd /path/to/llama.cpp && python3 apply_budget_patch.py
        python3 apply_budget_patch.py --root /path/to/llama.cpp --check
"""

import argparse
import os
import sys

EDITS = []


def edit(relpath, old, new, desc):
    EDITS.append((relpath, old, new, desc))


# ---------------------------------------------------------------- 1. llama.h
edit(
    "include/llama.h",
    """        int32_t  n_moe_cache_slots;   // cache slots per host-resident expert layer (0 = disabled)
        int32_t  n_moe_cache_inserts; // max expert uploads per layer per decode step
""",
    """        int32_t  n_moe_cache_slots;   // cache slots per host-resident expert layer (0 = disabled)
        int32_t  n_moe_cache_inserts; // max expert uploads per layer per decode step
        float    n_moe_cache_budget_mib; // device-memory budget for the cache, MiB.
                                         // > 0: derive n_moe_cache_slots from it and refuse
                                         //      to run below the 2x top_k break-even floor.
                                         // <= 0: use n_moe_cache_slots verbatim (upstream behaviour).
""",
    "llama.h: add n_moe_cache_budget_mib to llama_context_params",
)

# ------------------------------------------------------------ 2. common.h
edit(
    "common/common.h",
    """    int32_t n_moe_cache_slots     = 0;    // GPU cache slots per host-resident MoE expert layer (0 = disabled)
    int32_t n_moe_cache_inserts   = 2;    // max expert uploads per layer per decode step
""",
    """    int32_t n_moe_cache_slots     = 0;    // GPU cache slots per host-resident MoE expert layer (0 = disabled)
    int32_t n_moe_cache_inserts   = 2;    // max expert uploads per layer per decode step
    float   n_moe_cache_budget_mib = 0.0f; // MiB budget for the cache; >0 derives slots and enforces the floor
""",
    "common.h: add n_moe_cache_budget_mib to common_params",
)

# ------------------------------------------------------------- 3. arg.cpp
edit(
    "common/arg.cpp",
    """    add_opt(common_arg(
        {"--moe-expert-cache-inserts"}, "N",
        string_format("max expert uploads per layer per decode step for the MoE expert cache (default: %d)", params.n_moe_cache_inserts),
        [](common_params & params, int value) {
            params.n_moe_cache_inserts = value;
        }
    ).set_env("LLAMA_ARG_MOE_EXPERT_CACHE_INSERTS"));
""",
    """    add_opt(common_arg(
        {"--moe-expert-cache-inserts"}, "N",
        string_format("max expert uploads per layer per decode step for the MoE expert cache (default: %d)", params.n_moe_cache_inserts),
        [](common_params & params, int value) {
            params.n_moe_cache_inserts = value;
        }
    ).set_env("LLAMA_ARG_MOE_EXPERT_CACHE_INSERTS"));
    add_opt(common_arg(
        {"--moe-expert-cache-budget"}, "MiB",
        string_format("device-memory budget for the MoE expert cache in MiB; derives the slot count "
                      "and disables the cache when the budget cannot reach the break-even floor "
                      "(2x n_expert_used) (default: %g)", params.n_moe_cache_budget_mib),
        [](common_params & params, float value) {
            params.n_moe_cache_budget_mib = value;
        }
    ).set_env("LLAMA_ARG_MOE_EXPERT_CACHE_BUDGET"));
""",
    "arg.cpp: add --moe-expert-cache-budget",
)

# ------------------------------------------------------- 4. moecache.h
edit(
    "src/llama-moecache.h",
    """void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts);
""",
    """void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts, float budget_mib);
""",
    "moecache.h: extend init signature",
)

# ----------------------------------------------------- 5. llama-context.cpp
edit(
    "src/llama-context.cpp",
    """    llama_moe_cache_init(model, params.n_moe_cache_slots, params.n_moe_cache_inserts);
""",
    """    llama_moe_cache_init(model, params.n_moe_cache_slots, params.n_moe_cache_inserts, params.n_moe_cache_budget_mib);
""",
    "llama-context.cpp: pass the budget through",
)

edit(
    "src/llama-context.cpp",
    """        /*.n_moe_cache_slots           =*/ 0,
        /*.n_moe_cache_inserts         =*/ 2,
""",
    """        /*.n_moe_cache_slots           =*/ 0,
        /*.n_moe_cache_inserts         =*/ 2,
        /*.n_moe_cache_budget_mib     =*/ 0.0f,
""",
    "llama-context.cpp: default the budget to 0 (upstream behaviour)",
)

# ------------------------------------------------------ 6. moecache.cpp
edit(
    "src/llama-moecache.cpp",
    """void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts) {
""",
    """void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts, float budget_mib) {
""",
    "moecache.cpp: extend init signature",
)

edit(
    "src/llama-moecache.cpp",
    """#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
""",
    """#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
""",
    "moecache.cpp: include <algorithm> for std::min",
)

# The real change: derive the slot count from a MiB budget, with a hard floor.
edit(
    "src/llama-moecache.cpp",
    """        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\\n", __func__, n_slots);
            delete mc;
            return;
        }
""",
    """        // ---- budget-driven slot sizing (small-VRAM adaptation) -------------
        //
        // Device cost of the pool is  N slots * L layers * bytes-per-expert,
        // where bytes-per-expert is the sum of the up/gate/down per-expert
        // strides. Two measured constraints decide N:
        //
        //   (a) floor: RFC ggml-org/llama.cpp#28248 reports N == top_k as a
        //       full-miss worst case that is measurably SLOWER than master,
        //       with gains appearing only from roughly 2x top_k upward.
        //       Below the floor we refuse to run rather than run a net loss.
        //   (b) ceiling: residency is never free. koren1712 measured 8 GiB
        //       slab 8.70 t/s vs 16 GiB slab 7.16 t/s at identical hit counts
        //       and identical streamed bytes - past working-set coverage,
        //       extra residency buys no fewer reads and only adds pressure.
        //       Hence a budget is the user-facing knob, not a slot count.
        if (budget_mib > 0.0f && !groups.empty()) {
            size_t n_cand = 0;
            double bpe_sum = 0.0; // bytes per expert (up + gate + down)

            for (auto & g : groups) {
                for (auto & c : g.second) {
                    bpe_sum += (double) c.l->ffn_up_exps->nb[2]
                             + (double) c.l->ffn_gate_exps->nb[2]
                             + (double) c.l->ffn_down_exps->nb[2];
                    n_cand++;
                }
            }

            if (n_cand == 0 || bpe_sum <= 0.0) {
                LLAMA_LOG_WARN("%s: cannot size the cache from the budget - disabled\\n", __func__);
                delete mc;
                g_init_done = true;
                return;
            }

            const double bpe    = bpe_sum / (double) n_cand;
            const double budget = (double) budget_mib * 1024.0 * 1024.0;

            // N slots on each of n_cand layers must fit in the budget
            const int32_t n_fit = (int32_t) (budget / (bpe * (double) n_cand));

            const int32_t top_k   = model.hparams.n_expert_used > 0 ? model.hparams.n_expert_used : 8;
            const int32_t n_floor = 2 * top_k;

            if (n_fit < n_floor) {
                LLAMA_LOG_WARN(
                    "%s: budget of %.0f MiB fits only %d slot(s)/layer across %zu layer(s) "
                    "(%.2f MiB per expert); %d are needed to beat master (2x top_k=%d) - cache disabled\\n",
                    __func__, (double) budget_mib, n_fit, n_cand,
                    bpe / 1024.0 / 1024.0, n_floor, top_k);
                delete mc;
                g_init_done = true;
                return;
            }

            n_slots = (n_slots > 0) ? std::min(n_slots, n_fit) : n_fit;
            mc->n_slots = n_slots;

            LLAMA_LOG_INFO("%s: budget %.0f MiB -> %d slots/layer across %zu layer(s), "
                           "%.2f MiB per expert (%.1f MiB device, floor %d)\\n",
                           __func__, (double) budget_mib, n_slots, n_cand,
                           bpe / 1024.0 / 1024.0,
                           bpe * (double) n_cand * (double) n_slots / 1024.0 / 1024.0,
                           n_floor);
        }

        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\\n", __func__, n_slots);
            delete mc;
            return;
        }
""",
    "moecache.cpp: budget-driven slot sizing with a break-even floor",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="path to a llama.cpp checkout with PR #27861 applied")
    ap.add_argument("--check", action="store_true", help="only report whether each hunk is present")
    args = ap.parse_args()

    failures = []
    for relpath, old, new, desc in EDITS:
        path = os.path.join(args.root, relpath)
        if not os.path.isfile(path):
            failures.append((relpath, "file not found"))
            print("MISSING  %-40s %s" % (relpath, desc))
            continue

        with open(path, "r", encoding="utf-8") as f:
            src = f.read()

        if new in src:
            print("ALREADY  %-40s %s" % (relpath, desc))
            continue

        if old not in src:
            failures.append((relpath, "anchor text not found -- is PR #27861 applied?"))
            print("FAILED   %-40s %s" % (relpath, desc))
            continue

        if args.check:
            print("OK       %-40s %s" % (relpath, desc))
            continue

        with open(path, "w", encoding="utf-8") as f:
            f.write(src.replace(old, new, 1))
        print("PATCHED  %-40s %s" % (relpath, desc))

    if failures:
        print("\n%d hunk(s) failed. Apply PR #27861 first:" % len(failures))
        print("  git fetch origin pull/27861/head:pr-27861 && git merge pr-27861")
        return 1

    print("\nAll hunks applied. Build and run:")
    print("  cmake -B build -DGGML_CUDA=ON && cmake --build build -j")
    print("  ./build/bin/llama-server -m <model>.gguf -ngl 999 -ncmoe 999 \\")
    print("      --moe-expert-cache-budget 1500 -ctk q8_0 -ctv q8_0 --no-mmap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
