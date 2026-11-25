"""Continuous batching with chunked prefill, prefix caching and preemption.

Every step the scheduler gives each scheduled sequence a number of tokens to compute.
Prompt chunks and single decode tokens are treated alike: for each sequence it is
``len(tokens) − n_cached``, capped by a per-step token budget. This is the unified scheduling
of vLLM V1 / Sarathi.

1. **Running sequences first**, oldest first. If a sequence needs a new KV block and none
   is free, the *newest* running sequence is **preempted**: its blocks are freed and it
   goes back to the front of the waiting queue, to be recomputed later. Its blocks often
   survive in the prefix cache, so recomputation is cheap.
2. **Then admit waiting sequences** (FCFS) while the token budget, the sequence limit and
   free blocks allow. Full prompt blocks already in the prefix cache are reused, not
   recomputed.
3. One ragged forward pass, a greedy next token for every sequence whose context is now
   fully computed, then newly full blocks are registered in the prefix cache.

``policy="static"`` is the baseline. It admits a batch only when nothing is running, and
the whole batch then runs until its *longest* request finishes, so short requests leave
their slots idle. That is how plain ``generate(batch)`` serving behaves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .blocks import BlockAllocator, block_hash
from .model import Batch, PagedGPT2


@dataclass
class Sequence:
    id: int
    prompt: list[int]
    max_new: int
    out: list[int] = field(default_factory=list)
    table: list[int] = field(default_factory=list)
    hashes: list[int] = field(default_factory=list)
    n_cached: int = 0
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.out

    @property
    def done(self) -> bool:
        return len(self.out) >= self.max_new


@dataclass
class Stats:
    steps: int = 0
    preemptions: int = 0
    computed_tokens: int = 0
    prefix_hit_tokens: int = 0
    batch_sizes: list[int] = field(default_factory=list)


class Engine:
    def __init__(
        self,
        model: PagedGPT2,
        num_blocks: int,
        block_size: int = 16,
        max_seqs: int = 64,
        max_tokens: int = 2048,
        policy: str = "continuous",
        prefix_caching: bool = True,
    ):
        self.m, self.bs = model, block_size
        self.alloc = BlockAllocator(num_blocks, block_size, prefix_caching)
        model.allocate_pool(num_blocks, block_size)
        self.max_seqs, self.max_tokens, self.policy = max_seqs, max_tokens, policy
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        self.stats = Stats()

    # ---------------------------------------------------------------------------- memory
    def _blocks_needed(self, s: Sequence, upto: int) -> int:
        return max(0, -(-upto // self.bs) - len(s.table))

    def _free(self, s: Sequence):
        for b in s.table:
            self.alloc.release(b)
        s.table, s.hashes, s.n_cached = [], [], 0

    def _preempt(self, s: Sequence):
        self._free(s)
        self.running.remove(s)
        self.waiting.insert(0, s)
        self.stats.preemptions += 1

    def _reuse_prefix(self, s: Sequence):
        """Attach cached full blocks covering the start of ``s.tokens``; at least one token
        is always left to compute so the step produces logits."""
        toks, parent = s.tokens, None
        for k in range((len(toks) - 1) // self.bs):
            h = block_hash(parent, tuple(toks[k * self.bs : (k + 1) * self.bs]))
            b = self.alloc.lookup(h)
            if b is None:
                break
            s.table.append(b)
            s.hashes.append(h)
            parent = h
        s.n_cached = len(s.table) * self.bs

    # ------------------------------------------------------------------------- schedule
    def _schedule(self) -> list[tuple[Sequence, int]]:
        budget, plan = self.max_tokens, []
        for s in list(self.running):
            if s not in self.running:  # preempted earlier in this loop
                continue
            n = min(len(s.tokens) - s.n_cached, budget)
            if n == 0:
                continue
            while self._blocks_needed(s, s.n_cached + n) > self.alloc.num_free:
                victim = self.running[-1]
                self._preempt(victim)
                if victim is s:
                    break
            if s not in self.running:
                continue
            for _ in range(self._blocks_needed(s, s.n_cached + n)):
                s.table.append(self.alloc.allocate())
            plan.append((s, n))
            budget -= n
        admit = self.policy == "continuous" or not self.running
        while admit and self.waiting and budget > 0 and len(self.running) < self.max_seqs:
            s = self.waiting[0]
            self._reuse_prefix(s)
            n = min(len(s.tokens) - s.n_cached, budget)
            if self.policy == "static" and n < len(s.tokens) - s.n_cached and plan:
                self._free(s)  # static batching prefills a request in one go
                break
            if self._blocks_needed(s, s.n_cached + n) > self.alloc.num_free:
                self._free(s)
                break
            for _ in range(self._blocks_needed(s, s.n_cached + n)):
                s.table.append(self.alloc.allocate())
            self.waiting.pop(0)
            self.running.append(s)
            plan.append((s, n))
            self.stats.prefix_hit_tokens += s.n_cached
            budget -= n
        return plan

    # ----------------------------------------------------------------------------- step
    def _batch(self, plan) -> Batch:
        dev, bs = self.m.device, self.bs
        toks, pos, starts, ends = [], [], [0], []
        for s, n in plan:
            a, b = s.n_cached, s.n_cached + n
            toks += s.tokens[a:b]
            pos += range(a, b)
            starts.append(starts[-1] + n)
            ends.append(b)
        width = max(len(s.table) for s, _ in plan)
        tables = torch.tensor([s.table + [0] * (width - len(s.table)) for s, _ in plan])
        ctx_lens = torch.tensor(ends)
        ar = torch.arange(int(ctx_lens.max()))
        ctx_slots = tables[:, ar // bs] * bs + ar % bs  # (S, c_max); padding is masked
        positions = torch.tensor(pos)
        seq_of = torch.repeat_interleave(torch.arange(len(plan)), torch.diff(torch.tensor(starts)))
        slots = tables[seq_of, positions // bs] * bs + positions % bs
        return Batch(
            tokens=torch.tensor(toks, device=dev),
            positions=positions.to(dev),
            slots=slots.to(dev),
            seq_starts=starts,
            ctx_slots=ctx_slots.to(dev),
            ctx_lens=ctx_lens.to(dev),
        )

    def step(self):
        plan = self._schedule()
        if not plan:
            if self.waiting:
                raise MemoryError("a single request does not fit in the KV pool")
            return
        logits = self.m.forward(self._batch(plan))
        now = time.perf_counter()
        self.stats.steps += 1
        self.stats.batch_sizes.append(len(plan))
        for (s, n), lg in zip(plan, logits, strict=True):
            s.n_cached += n
            self.stats.computed_tokens += n
            while len(s.hashes) < s.n_cached // self.bs:  # register newly full blocks
                k = len(s.hashes)
                h = block_hash(
                    s.hashes[-1] if s.hashes else None,
                    tuple(s.tokens[k * self.bs : (k + 1) * self.bs]),
                )
                self.alloc.register(s.table[k], h)
                s.hashes.append(h)
            if s.n_cached == len(s.tokens):  # context complete: emit a token
                s.out.append(int(lg.argmax()))
                s.first_token_time = s.first_token_time or now
        for s, _ in plan:
            if s.done and s in self.running:
                s.finish_time = now
                self._free(s)
                self.running.remove(s)
                self.finished.append(s)

    def generate(self, requests: list[tuple[list[int], int]]) -> list[list[int]]:
        seqs = [Sequence(i, list(p), n) for i, (p, n) in enumerate(requests)]
        self.waiting += seqs
        while self.waiting or self.running:
            self.step()
        return [s.out for s in seqs]


def reference_generate(model: PagedGPT2, prompt: list[int], n: int) -> list[int]:
    toks = list(prompt)
    for _ in range(n):
        toks.append(int(model.reference_logits(toks)[-1].argmax()))
    return toks[len(prompt) :]
