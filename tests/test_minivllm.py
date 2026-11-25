import random

import pytest
import torch

from minivllm.blocks import BlockAllocator, block_hash
from minivllm.engine import Engine, reference_generate
from minivllm.model import random_gpt2


def _requests(n, seed=0, shared=0, vocab=64):
    rng = random.Random(seed)
    sys = [rng.randrange(vocab) for _ in range(shared)]
    return [
        (sys + [rng.randrange(vocab) for _ in range(rng.randint(1, 30))], rng.randint(1, 25))
        for _ in range(n)
    ]


@pytest.fixture(scope="module")
def model():
    return random_gpt2()


@pytest.fixture(scope="module")
def reference(model):
    cache = {}

    def ref(prompt, n):
        key = (tuple(prompt), n)
        if key not in cache:
            cache[key] = reference_generate(model, prompt, n)
        return cache[key]

    return ref


def test_paged_forward_matches_plain_forward_for_prefill_and_decode(model):
    torch.manual_seed(0)
    toks = torch.randint(0, 64, (21,)).tolist()
    e = Engine(model, num_blocks=8, block_size=4, max_tokens=7)  # prefill in 7-token chunks
    out = e.generate([(toks, 5)])[0]
    ref = model.reference_logits(toks + out)
    assert out == ref[len(toks) - 1 : -1].argmax(-1).tolist()


@pytest.mark.parametrize(
    "kw",
    [
        dict(num_blocks=200, block_size=4),  # plenty of memory
        dict(num_blocks=24, block_size=4, max_tokens=16),  # tight: preemption + chunking
        dict(num_blocks=200, block_size=8, policy="static", max_seqs=6),
        dict(num_blocks=60, block_size=4, prefix_caching=False),
    ],
)
def test_engine_output_equals_sequential_greedy(model, reference, kw):
    reqs = _requests(20, seed=1, shared=9)
    e = Engine(model, **kw)
    outs = e.generate(reqs)
    for (p, n), o in zip(reqs, outs, strict=True):
        assert o == reference(p, n)
    e.alloc.check()
    assert sum(r > 0 for r in e.alloc.ref) == 0  # everything released
    if kw.get("max_tokens") == 16:
        assert e.stats.preemptions > 0  # the tight config really exercised preemption


def test_prefix_cache_skips_recomputing_a_shared_prompt(model, reference):
    shared = list(range(32))  # 8 full blocks of 4
    reqs = [(shared + [40 + i], 3) for i in range(5)]
    on = Engine(model, num_blocks=100, block_size=4, max_seqs=1)  # one at a time
    off = Engine(model, num_blocks=100, block_size=4, max_seqs=1, prefix_caching=False)
    assert on.generate(reqs) == off.generate(reqs) == [reference(p, n) for p, n in reqs]
    assert on.stats.prefix_hit_tokens == 4 * 32  # every request after the first
    assert off.stats.computed_tokens - on.stats.computed_tokens == 4 * 32


def test_continuous_batching_takes_fewer_steps_than_static(model):
    reqs = [([1, 2, 3], 40)] + [([4, 5], 2)] * 15  # one long request, many short ones
    cont = Engine(model, num_blocks=200, block_size=4, max_seqs=4)
    stat = Engine(model, num_blocks=200, block_size=4, max_seqs=4, policy="static")
    assert cont.generate(reqs) == stat.generate(reqs)
    assert cont.stats.steps < stat.stats.steps


def test_allocator_refcounts_lru_eviction_and_invariants():
    a = BlockAllocator(3, 4)
    b0, b1 = a.allocate(), a.allocate()
    h = block_hash(None, (1, 2, 3, 4))
    a.register(b0, h)
    a.release(b0)  # cached and evictable, not free
    assert a.num_free == 2 and a.lookup(h) == b0 and a.ref[b0] == 1
    a.release(b0)
    a.release(b1)
    # uncached free blocks are handed out before any cached block is evicted
    assert {a.allocate(), a.allocate()} == {0, 1, 2} - {b0}
    assert h in a.cached
    assert a.allocate() == b0  # now the cached block is evicted and reused
    assert a.lookup(h) is None
    with pytest.raises(MemoryError):
        a.allocate()
    a.check()


def test_request_too_large_for_the_pool(model):
    e = Engine(model, num_blocks=2, block_size=4)
    with pytest.raises(MemoryError):
        e.generate([(list(range(20)), 2)])
