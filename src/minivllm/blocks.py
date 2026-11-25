"""Paged KV-cache memory (Kwon et al. 2023, "PagedAttention"), with automatic prefix caching.

KV memory is a pool of fixed-size *blocks* of ``block_size`` token slots. A sequence owns
a *block table*, a list of block ids, so its cache needn't be contiguous and memory is
allocated one block at a time as it grows. At most ``block_size − 1`` slots per sequence
are wasted, against reserving ``max_len`` slots up front.

**Prefix caching.** A block that is *full* is identified by a hash of its tokens and of
everything before it (the parent block's hash). A new sequence whose prompt starts with
the same full blocks reuses them, incrementing their reference count instead of
recomputing them. Freed blocks with a hash stay cached until memory is needed, and are
then evicted least recently used first.
"""

from __future__ import annotations

from collections import OrderedDict


def block_hash(parent: int | None, tokens: tuple[int, ...]) -> int:
    return hash((parent, tokens))


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int, prefix_caching: bool = True):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.prefix_caching = prefix_caching
        self.ref = [0] * num_blocks
        self.free: list[int] = list(range(num_blocks - 1, -1, -1))  # never-used blocks
        self.cached: dict[int, int] = {}  # hash -> block id
        self.hash_of: dict[int, int] = {}  # block id -> hash
        self.evictable: OrderedDict[int, None] = OrderedDict()  # ref 0 and cached, LRU first
        self.hits = self.lookups = 0

    # ------------------------------------------------------------------------- counting
    @property
    def num_free(self) -> int:
        """Blocks that can be handed out now: never used plus evictable cached ones."""
        return len(self.free) + len(self.evictable)

    def check(self):
        used = sum(r > 0 for r in self.ref)
        assert used + self.num_free == self.num_blocks, (used, self.num_free)
        assert all(self.ref[b] == 0 for b in self.free + list(self.evictable))

    # ----------------------------------------------------------------------- allocation
    def allocate(self) -> int:
        if self.free:
            b = self.free.pop()
        elif self.evictable:
            b, _ = self.evictable.popitem(last=False)  # evict the LRU cached block
            del self.cached[self.hash_of.pop(b)]
        else:
            raise MemoryError("out of KV blocks")
        self.ref[b] = 1
        return b

    def release(self, b: int):
        self.ref[b] -= 1
        if self.ref[b] == 0:
            if b in self.hash_of:
                self.evictable[b] = None
            else:
                self.free.append(b)

    def lookup(self, h: int) -> int | None:
        """Reuse the cached block with hash ``h`` if there is one (ref count +1)."""
        if not self.prefix_caching:
            return None
        self.lookups += 1
        b = self.cached.get(h)
        if b is None:
            return None
        self.hits += 1
        if self.ref[b] == 0:
            del self.evictable[b]
        self.ref[b] += 1
        return b

    def register(self, b: int, h: int):
        """Mark block ``b`` as full with content hash ``h`` so later prompts can reuse it."""
        if self.prefix_caching and h not in self.cached:
            self.cached[h], self.hash_of[b] = b, h
