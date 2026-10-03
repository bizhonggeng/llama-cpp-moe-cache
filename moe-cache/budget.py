#!/usr/bin/env python3
"""
Work out --moe-expert-cache-budget for your card and model.

Device cost of the pool = N slots * L MoE layers * bytes-per-expert,
bytes-per-expert = (hidden * intermediate * 3) * bpw / 8   (up + gate + down)

Usage:
    python3 budget.py                      # Qwen3.6-35B-A3B defaults
    python3 budget.py --vram 8 --free 4.0
    python3 budget.py --hidden 2048 --intermediate 512 --layers 40 --experts 256 --topk 8 --bpw 4.5
"""
import argparse

# measured decode gain vs slot count, from RFC ggml-org/llama.cpp#28248
# (RTX 4090 / Qwen3.8-Flash-Next / all experts on CPU); expressed as a
# fraction of top_k since that is what sets the miss rate.
GAIN_CURVE = [
    (0.0,  0.00),   # below the floor: disabled
    (2.0,  0.12),   # N = 2x top_k  (RFC: N=16, topk=8 -> +12%)
    (3.0,  0.40),   # interpolating N=16..32
    (4.0,  0.52),   # RFC: N=32 -> +52%
    (6.0,  0.65),   # RFC: N=48 -> +65%
    (8.0,  0.84),   # RFC: N=64 -> +84%; saturates here
]


def gain(mult):
    if mult <= GAIN_CURVE[1][0]:
        return GAIN_CURVE[1][1] * (mult / GAIN_CURVE[1][0])
    for i in range(1, len(GAIN_CURVE) - 1):
        x0, y0 = GAIN_CURVE[i]
        x1, y1 = GAIN_CURVE[i + 1]
        if x0 <= mult <= x1:
            return y0 + (y1 - y0) * (mult - x0) / (x1 - x0)
    return GAIN_CURVE[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden",       type=int,   default=2048)
    ap.add_argument("--intermediate", type=int,   default=512)
    ap.add_argument("--layers",       type=int,   default=40,  help="MoE (non-dense) layers")
    ap.add_argument("--experts",      type=int,   default=256)
    ap.add_argument("--topk",         type=int,   default=8)
    ap.add_argument("--bpw",          type=float, default=4.5, help="quant bits per weight")
    ap.add_argument("--vram",         type=float, default=8.0, help="card VRAM, GiB")
    ap.add_argument("--free",         type=float, default=None,
                    help="GiB actually free for the cache (default: 45%% of VRAM)")
    args = ap.parse_args()

    free = args.free if args.free is not None else args.vram * 0.45

    bpe = args.hidden * args.intermediate * 3 * args.bpw / 8.0        # bytes
    per_slot = bpe * args.layers                                       # bytes per slot across all layers
    floor_n = 2 * args.topk
    floor_mib = floor_n * per_slot / 1024.0 / 1024.0

    print()
    print("  model: %d MoE layers, %d experts, top-%d, %.2f bpw" %
          (args.layers, args.experts, args.topk, args.bpw))
    print("  %.2f MiB per expert (up+gate+down), %.1f MiB per slot across all layers"
          % (bpe / 1024 / 1024, per_slot / 1024 / 1024))
    print()
    print("  card: %.1f GiB VRAM, %.1f GiB budget for the cache" % (args.vram, free))
    print()

    if free * 1024 < floor_mib:
        print("  !! BUDGET TOO SMALL")
        print("     the break-even floor is %d slots (2x top_k) = %.0f MiB,"
              % (floor_n, floor_mib))
        print("     you have %.0f MiB. The cache would be a net LOSS (RFC #28248:"
              % (free * 1024))
        print("     N == top_k is measurably slower than master).")
        print("     -> run without --moe-expert-cache-budget, or free more VRAM.")
        print()
        return 1

    best_n = int(free * 1024 * 1024 * 1024 / per_slot)
    mult = best_n / args.topk
    g = gain(mult)

    print("  -> --moe-expert-cache-budget %.0f" % (free * 1024))
    print("     = %d slots/layer  (%.1fx top_k)" % (best_n, mult))
    print()

    print("  %-10s %-14s %-16s %s" % ("budget MiB", "slots/layer", "x top_k", "expected decode gain"))
    print("  " + "-" * 62)
    for mb in [512, 1024, 1536, 2048, 3072, 4096, 6144]:
        n = int(mb * 1024.0 * 1024.0 / per_slot)
        if n < floor_n:
            row = "disabled (below floor %d)" % floor_n
            gv = ""
        else:
            row = "+%.0f%%" % (100 * gain(n / args.topk))
            gv = row
        mark = "   <= yours" if abs(mb - free * 1024) < 300 else ""
        print("  %-10d %-14d %-16.1f %s%s" % (mb, n, n / args.topk, row, mark))
    print()
    print("  Note: gain saturates near 8x top_k. Past that, extra residency buys")
    print("  no extra hits and starts costing you (koren1712: 8 GiB slab 8.70 t/s")
    print("  vs 16 GiB slab 7.16 t/s at identical hit counts). Do not over-allocate.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
