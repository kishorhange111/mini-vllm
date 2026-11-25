"""GPT-2 whose KV cache lives in a paged pool, run on a *ragged* batch.

One forward call takes any mix of sequences, with any number of new tokens each: prompt
chunks for sequences being prefilled, one token for sequences being decoded. All new
tokens are flattened into one ``(N, d)`` matrix, so every projection and MLP runs as a
single large matmul over the whole batch. That is where continuous batching gets its
throughput. Only attention is per-sequence. Each sequence's context is gathered from its
block table into a padded ``(S, H, C_max, d_h)`` view, and one masked SDPA call covers
the batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

NAMES = (
    "ln_1.weight",
    "ln_1.bias",
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "attn.c_proj.bias",
    "ln_2.weight",
    "ln_2.bias",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "mlp.c_proj.bias",
)


@dataclass
class Batch:
    """What to compute this step, for S sequences.

    tokens, positions, slots: (N,) the new tokens, their positions, and the pool slot each
        one's K/V is written to.
    seq_starts: (S + 1,) offsets of each sequence's new tokens within the N.
    ctx_slots: (S, C_max) pool slots of each sequence's whole context (cached + new),
        padded; ctx_lens (S,) the true lengths.
    """

    tokens: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    seq_starts: list[int]
    ctx_slots: torch.Tensor
    ctx_lens: torch.Tensor


class PagedGPT2:
    def __init__(self, sd: dict, n_layer: int, n_head: int, eps: float = 1e-5):
        self.n_layer, self.n_head, self.eps = n_layer, n_head, eps
        self.wte, self.wpe = sd["transformer.wte.weight"], sd["transformer.wpe.weight"]
        self.d = self.wte.shape[1]
        self.hd = self.d // n_head
        self.lnf = (sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"])
        self.layers = [{n: sd[f"transformer.h.{i}.{n}"] for n in NAMES} for i in range(n_layer)]
        self.device, self.dtype = self.wte.device, self.wte.dtype
        self.k_pool = self.v_pool = None

    @classmethod
    def from_pretrained(cls, name: str, device="cuda", dtype=torch.float16):
        from transformers import GPT2LMHeadModel

        hf = GPT2LMHeadModel.from_pretrained(name, torch_dtype=dtype).to(device)
        sd = {k: v.detach() for k, v in hf.state_dict().items()}
        return cls(sd, hf.config.n_layer, hf.config.n_head, hf.config.layer_norm_epsilon)

    def allocate_pool(self, num_blocks: int, block_size: int):
        shape = (self.n_layer, num_blocks * block_size, self.n_head, self.hd)
        self.k_pool = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.v_pool = torch.zeros(shape, device=self.device, dtype=self.dtype)

    def kv_bytes_per_token(self) -> int:
        return 2 * self.n_layer * self.d * self.wte.element_size()

    def _ln(self, x, w, b):
        return F.layer_norm(x, (self.d,), w, b, self.eps)

    @torch.no_grad()
    def forward(self, b: Batch) -> torch.Tensor:
        """Returns logits (S, vocab) at each sequence's *last* new token."""
        st = torch.tensor(b.seq_starts, device=self.device)
        n_new = torch.diff(st)
        # Decode sequences (1 new token) and prefill chunks are attended separately, so a
        # long prompt chunk doesn't pad every decode query to its length.
        groups = []
        for sel in (n_new == 1, n_new > 1):
            seqs = sel.nonzero()[:, 0]
            if len(seqs) == 0:
                continue
            nn_ = n_new[seqs]
            rows = torch.repeat_interleave(torch.arange(len(seqs), device=self.device), nn_)
            first = st[seqs][rows]
            tok = (
                first
                + torch.arange(len(rows), device=self.device)
                - torch.repeat_interleave(torch.cumsum(nn_, 0) - nn_, nn_)
            )
            cols = tok - first
            q_max = int(nn_.max())
            ctx = b.ctx_slots[seqs][:, : int(b.ctx_lens[seqs].max())]
            q_pos = torch.zeros(len(seqs), q_max, dtype=torch.long, device=self.device)
            q_pos[rows, cols] = b.positions[tok]
            kpos = torch.arange(ctx.shape[1], device=self.device)
            mask = (kpos[None, None] <= q_pos[:, :, None]) & (
                kpos[None, None] < b.ctx_lens[seqs][:, None, None]
            )
            groups.append((tok, rows, cols, q_max, ctx, mask[:, None]))
        x = self.wte[b.tokens] + self.wpe[b.positions]  # (N, d)
        H, hd = self.n_head, self.hd
        for li, L in enumerate(self.layers):
            h = self._ln(x, L["ln_1.weight"], L["ln_1.bias"])
            q, k, v = (h @ L["attn.c_attn.weight"] + L["attn.c_attn.bias"]).split(self.d, -1)
            self.k_pool[li, b.slots] = k.view(-1, H, hd)
            self.v_pool[li, b.slots] = v.view(-1, H, hd)
            q = q.view(-1, H, hd)
            y = torch.empty_like(q)
            for tok, rows, cols, q_max, ctx, mask in groups:
                qp = q.new_zeros(int(rows.max()) + 1, q_max, H, hd)
                qp[rows, cols] = q[tok]
                kc = self.k_pool[li][ctx].transpose(1, 2)  # (S_g, H, c_max, hd)
                vc = self.v_pool[li][ctx].transpose(1, 2)
                yg = F.scaled_dot_product_attention(qp.transpose(1, 2), kc, vc, attn_mask=mask)
                y[tok] = yg.transpose(1, 2)[rows, cols]
            x = x + y.reshape(-1, self.d) @ L["attn.c_proj.weight"] + L["attn.c_proj.bias"]
            h = self._ln(x, L["ln_2.weight"], L["ln_2.bias"])
            h = F.gelu(h @ L["mlp.c_fc.weight"] + L["mlp.c_fc.bias"], approximate="tanh")
            x = x + h @ L["mlp.c_proj.weight"] + L["mlp.c_proj.bias"]
        last = st[1:] - 1
        return (self._ln(x[last], *self.lnf) @ self.wte.t()).float()

    @torch.no_grad()
    def reference_logits(self, tokens: list[int]) -> torch.Tensor:
        """Plain causal forward over one sequence, no cache: logits at every position."""
        idx = torch.tensor(tokens, device=self.device)
        x = self.wte[idx] + self.wpe[: len(tokens)]
        for L in self.layers:
            h = self._ln(x, L["ln_1.weight"], L["ln_1.bias"])
            q, k, v = (h @ L["attn.c_attn.weight"] + L["attn.c_attn.bias"]).split(self.d, -1)
            q, k, v = (t.view(-1, self.n_head, self.hd).transpose(0, 1) for t in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(0, 1)
            x = x + y.reshape(-1, self.d) @ L["attn.c_proj.weight"] + L["attn.c_proj.bias"]
            h = self._ln(x, L["ln_2.weight"], L["ln_2.bias"])
            h = F.gelu(h @ L["mlp.c_fc.weight"] + L["mlp.c_fc.bias"], approximate="tanh")
            x = x + h @ L["mlp.c_proj.weight"] + L["mlp.c_proj.bias"]
        return (self._ln(x, *self.lnf) @ self.wte.t()).float()


def random_gpt2(vocab=64, d=32, n_layer=2, n_head=4, ctx=256, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)

    def r(*s, std=0.2):
        return torch.randn(*s, generator=g) * std

    sd = {
        "transformer.wte.weight": r(vocab, d, std=1.0),
        "transformer.wpe.weight": r(ctx, d),
        "transformer.ln_f.weight": 1 + r(d),
        "transformer.ln_f.bias": r(d),
    }
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        sd |= {
            p + "ln_1.weight": 1 + r(d),
            p + "ln_1.bias": r(d),
            p + "attn.c_attn.weight": r(d, 3 * d),
            p + "attn.c_attn.bias": r(3 * d),
            p + "attn.c_proj.weight": r(d, d),
            p + "attn.c_proj.bias": r(d),
            p + "ln_2.weight": 1 + r(d),
            p + "ln_2.bias": r(d),
            p + "mlp.c_fc.weight": r(d, 4 * d),
            p + "mlp.c_fc.bias": r(4 * d),
            p + "mlp.c_proj.weight": r(4 * d, d),
            p + "mlp.c_proj.bias": r(d),
        }
    return PagedGPT2({k: v.to(dtype) for k, v in sd.items()}, n_layer, n_head)
