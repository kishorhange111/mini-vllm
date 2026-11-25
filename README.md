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

## Results (Tesla T4, Colab)

The table and raw rows are in [`results/`](results/). All configurations do identical work:
256 requests and the same generated tokens.

| workload | policy | max seqs | KV blocks | prefix cache | tok/s | latency mean / p90 (s) | TTFT (s) | mean batch | preempt | tokens through the model |
|---|---|---|---|---|---|---|---|---|---|---|
| random | static | 32 | 4096 | – | 356 | 49.3 / 88.5 | 47.7 | 4.1 | 0 | 105,058 |
| random | continuous | 32 | 4096 | – | 1468 | 12.7 / 21.9 | 9.9 | 28.9 | 0 | 105,058 |
| random | continuous | 128 | 4096 | – | **1838** | 12.1 / 18.0 | 4.4 | 80.7 | 0 | 105,058 |
| random | continuous | 128 | 1024 | – | 1568 | 12.2 / 20.8 | 8.5 | 39.7 | 84 | 126,764 |
| shared prefix | continuous | 128 | 4096 | off | 1488 | 15.3 / 23.2 | 6.1 | 79.8 | 21 | 161,924 |
| shared prefix | continuous | 128 | 4096 | on | 1677 | 13.4 / 20.4 | 4.6 | 84.5 | 0 | 88,935 |

What the numbers show:
- **Continuous batching is 4.1× faster than static batching at the same 32-sequence limit**
  (1468 vs 356 tok/s). A static batch runs until its longest output finishes, so its slots
  sit idle: its mean running batch is 4.1 of 32. Continuous batching refills a slot the
  step it frees, which keeps the mean at 28.9. Mean latency falls from 49 s to 13 s.
- Raising the limit to 128 sequences adds another 25% (1838 tok/s) and cuts time to first
  token from 9.9 s to 4.4 s.
- **A 4× smaller KV pool causes 84 preemptions.** Preempted sequences free their blocks and
  recompute their tokens later, which puts 21% more tokens through the model (126,764 vs
  105,058). Throughput drops 15%, to 1568 tok/s, and time to first token nearly doubles.
- **Prefix caching on a shared 256-token system prompt puts 45% fewer tokens through the
  model** (88,935 vs 161,924), and 64,000 tokens are served from the cache. Throughput rises 13% and
  time to first token falls from 6.1 s to 4.6 s. Without caching, each request holds its
  own copy of the prefix's KV blocks, and the pool runs short (21 preemptions). With
  sharing it doesn't (0).
