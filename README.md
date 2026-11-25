# mini-vllm

[![CI](https://github.com/kishorhange111/mini-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/mini-vllm/actions/workflows/ci.yml)

A small LLM serving engine written from scratch, built around the ideas that make vLLM
fast: **paged KV memory**, **continuous batching** with chunked prefill and preemption,
and **automatic prefix caching**. It serves GPT-2 from a hand-written forward pass. Its
output is checked to be **token-for-token identical to plain sequential decoding**,
including under memory pressure that forces preemption.

## Pieces

| file | what it does |
|---|---|
| [`blocks.py`](src/minivllm/blocks.py) | KV memory as a pool of fixed-size blocks with reference counts. Each sequence has a block table, so memory grows one block at a time instead of reserving `max_len`. Full blocks are keyed by a hash of their tokens chained with the parent block's hash; a later prompt with the same prefix **reuses** them. Freed cached blocks are evicted LRU-first, and only after every uncached free block is gone |
| [`model.py`](src/minivllm/model.py) | GPT-2 on a **ragged batch**. All new tokens of all sequences (prefill chunks and single decode tokens) are flattened into one matrix, so the projections and MLP are one large matmul per layer. K/V are written straight into pool slots. Attention gathers each sequence's context through its block table, with decode and prefill sequences attended as separate groups so short queries aren't padded to long ones |
| [`engine.py`](src/minivllm/engine.py) | The scheduler. Each step gives every sequence `len(tokens) − n_cached` tokens, capped by a token budget, which treats prefill and decode alike as in vLLM V1 / Sarathi. Running sequences go first. If one needs a block and none is free, the *newest* running sequence is **preempted** (blocks freed, recomputed later, often from the prefix cache). Then waiting sequences are admitted FCFS. `policy="static"` is the baseline: a batch is admitted only when the previous one has finished entirely |

## Tests

There are 9, on CPU, with a random GPT-2 in float64. They check:
- paged, chunked prefill plus decode equals a plain causal forward;
- **engine output equals sequential greedy decoding for 20 mixed requests** under four
  configurations: ample memory, a pool small enough to force preemption and chunking (the
  test asserts preemption happened), static batching, and prefix caching off;
- a shared 32-token prompt is computed once: four later requests take 128 tokens from the
  prefix cache, and the saving in computed tokens is exactly that;
- with one long request among short ones, continuous batching needs fewer steps than
  static batching;
- allocator reference counts, LRU eviction order and the free/used invariant;
- a request larger than the whole pool raises instead of hanging.

## Benchmark ([`bench.py`](src/minivllm/bench.py))

GPT-2 (124M), fp16, on a T4. There are 256 requests arriving at once, with prompts of 16–512
tokens and outputs of 16–256 tokens. The configurations are:
- static batching (32 sequences);
- continuous batching (32 and 128 sequences);
- continuous batching with a 4× smaller KV pool (memory-bound, so it preempts);
- a shared 256-token system prompt with prefix caching off and on.

Reported for each: generated tokens/s, mean and p90 latency, time to first token, mean
running batch, preemptions and prefix-cache hits.

```
pytest -q                               # 9 tests, CPU
python -m minivllm.bench --out results  # GPU, resumable (runs.jsonl)
```

**Results are pending** from the Colab run.
