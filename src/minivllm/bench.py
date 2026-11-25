"""Serving throughput on a GPU: static vs continuous batching, and prefix caching.

    python -m minivllm.bench --out results

The model is GPT-2 (124M) in fp16 with greedy decoding. Each workload has 256 requests
arriving at once. Prompt lengths are uniform in [16, 512] tokens and output lengths
uniform in [16, 256]; outputs are a fixed number of tokens (no EOS), so every
configuration does identical work. Token ids are random: content does not affect the
cost of a forward pass. Reported per configuration: wall time, generated tokens/s, mean
and p90 request latency, mean time to first token, scheduler steps, mean running batch,
preemptions and tokens served from the prefix cache.

The ``shared-prefix`` workload prepends the same 256-token system prompt to every
request (with prompts of 16-384 tokens after it, to stay inside GPT-2's 1,024 positions),
which is the case prefix caching is for.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from .engine import Engine
from .model import PagedGPT2


def workload(n=256, shared=0, seed=0, vocab=50257, max_prompt=512):
    rng = random.Random(seed)
    sys = [rng.randrange(vocab) for _ in range(shared)]
    return [
        (
            sys + [rng.randrange(vocab) for _ in range(rng.randint(16, max_prompt))],
            rng.randint(16, 256),
        )
        for _ in range(n)
    ]


def run(model, reqs, **kw) -> dict:
    e = Engine(model, **kw)
    e.generate(reqs[:4])  # warm-up (allocator and kernels), then a fresh engine
    e = Engine(model, **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = e.generate(reqs)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    lat = np.array([s.finish_time - t0 for s in e.finished])
    ttft = np.array([s.first_token_time - t0 for s in e.finished])
    gen = sum(len(o) for o in outs)
    return {
        **kw,
        "seconds": wall,
        "gen_tok_per_s": gen / wall,
        "latency_mean": float(lat.mean()),
        "latency_p90": float(np.percentile(lat, 90)),
        "ttft_mean": float(ttft.mean()),
        "steps": e.stats.steps,
        "mean_batch": float(np.mean(e.stats.batch_sizes)),
        "preemptions": e.stats.preemptions,
        "computed_tokens": e.stats.computed_tokens,
        "prefix_hit_tokens": e.stats.prefix_hit_tokens,
    }


def report(rows) -> str:
    out = [
        "| workload | policy | max seqs | KV blocks | prefix cache | tok/s "
        "| latency mean / p90 (s) | TTFT (s) | steps | mean batch | preempt | prefix-hit tokens |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['workload']} | {r['policy']} | {r['max_seqs']} | {r['num_blocks']} | "
            f"{'on' if r['prefix_caching'] else 'off'} | {r['gen_tok_per_s']:.0f} | "
            f"{r['latency_mean']:.1f} / {r['latency_p90']:.1f} | {r['ttft_mean']:.1f} | "
            f"{r['steps']} | {r['mean_batch']:.1f} | {r['preemptions']} | "
            f"{r['prefix_hit_tokens']:,} |"
        )
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="gpt2")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    model = PagedGPT2.from_pretrained(a.model)
    base = dict(block_size=16, max_tokens=2048, prefix_caching=True)
    configs = [
        ("random", dict(policy="static", max_seqs=32, num_blocks=4096)),
        ("random", dict(policy="continuous", max_seqs=32, num_blocks=4096)),
        ("random", dict(policy="continuous", max_seqs=128, num_blocks=4096)),
        ("random", dict(policy="continuous", max_seqs=128, num_blocks=1024)),  # memory-bound
        (
            "shared-prefix",
            dict(policy="continuous", max_seqs=128, num_blocks=4096, prefix_caching=False),
        ),
        ("shared-prefix", dict(policy="continuous", max_seqs=128, num_blocks=4096)),
    ]
    path = os.path.join(a.out, "runs.jsonl")
    rows = [json.loads(s) for s in open(path)] if os.path.exists(path) else []
    done = {
        (r["workload"], r["policy"], r["max_seqs"], r["num_blocks"], r["prefix_caching"])
        for r in rows
    }
    loads = {"random": workload(), "shared-prefix": workload(shared=256, seed=1, max_prompt=384)}
    for name, kw in configs:
        kw = {**base, **kw}
        key = (name, kw["policy"], kw["max_seqs"], kw["num_blocks"], kw["prefix_caching"])
        if key in done:
            continue
        r = {
            "workload": name,
            "device": torch.cuda.get_device_name(),
            "model": a.model,
            **run(model, loads[name], **kw),
        }
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        rows.append(r)
        print(json.dumps(r), flush=True)
    text = report(rows)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
