"""The architecture: per-frame slot encoder -> causal sequence model over
the episode -> action head. See `docs/design.md` §4 for why it takes this
shape; this file is the implementation of that decision.

Four pieces, top to bottom:

  1. `SlotAttention` / `FrameEncoder` — decomposes one settled frame into
     `config.num_slots` learned object slots (Slot Attention, Locatello et
     al. 2020) and pools them into a single per-frame token. This is what
     satisfies C10: the model learns its own object decomposition instead
     of receiving one from hand-coded segmentation.
  2. `RotaryEmbedding` / `CausalSelfAttention` / `TransformerBlock` — a
     standard pre-LN causal transformer block with RoPE positional
     encoding. RoPE is used (rather than learned absolute positions)
     because it doesn't fix a maximum sequence length in advance, which
     matters since episode lengths vary a lot and aren't known ahead of
     time (§4.4).
  3. The action head — legal-action logits plus (x, y) logits for ACTION6,
     as two independent categoricals over `config.coord_range` rather than
     a single joint 4096-way softmax or a continuous regression (see the
     architecture discussion in the project log: a discrete-cell click
     doesn't benefit from joint (x, y) correlation modeling, and regression
     would let the model land on the average of two valid but distant
     cells).
  4. `ARCPolicyNetwork` — composes the above and applies the one hard game
     rule that belongs in code, not learned: illegal actions get -inf
     logits (`obs.action_mask`).

Every position in the sequence gets a prediction (teacher forcing): the
network always returns `PolicyOutput` shaped [B, T, ...], not just a
decision for the last step. Online play only reads the last position (see
`model.adapters.decode_action`); training reads all of them at once.

No KV-caching: each call to `ARCPolicyNetwork.forward` recomputes the whole
sequence from scratch. This is the standard, simplest-correct way to train
a causal transformer (one parallel forward pass per trajectory) and it's
also what online play does today, recomputing the whole episode-so-far on
every step. That's O(T) redundant work per step at inference — acceptable
for now (compute isn't the constraint, correctness is), and isolated behind
this module if it's ever worth adding a cache.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig, Observation, PolicyOutput

# ── 1. Slot encoder ──────────────────────────────────────────────────────


class SlotAttention(nn.Module):
    """Iterative attention that routes a set of input features to
    `config.num_slots` slots, softly competing for explanation of the
    input — the learned stand-in for hand-coded object segmentation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_slots = config.num_slots
        self.slot_dim = config.slot_dim
        self.iters = config.slot_iters
        self.scale = config.slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, config.slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, config.slot_dim))

        self.norm_inputs = nn.LayerNorm(config.slot_dim)
        self.norm_slots = nn.LayerNorm(config.slot_dim)
        self.norm_mlp = nn.LayerNorm(config.slot_dim)

        self.to_q = nn.Linear(config.slot_dim, config.slot_dim, bias=False)
        self.to_k = nn.Linear(config.slot_dim, config.slot_dim, bias=False)
        self.to_v = nn.Linear(config.slot_dim, config.slot_dim, bias=False)

        self.gru = nn.GRUCell(config.slot_dim, config.slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.slot_dim, config.slot_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.slot_dim * 2, config.slot_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: [N, L, slot_dim] (N = batch*time, L = spatial positions)
        -> slots [N, num_slots, slot_dim].
        """
        n, _, _ = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        # Slots are re-initialized from noise on every call (as in the
        # original Slot Attention), including at inference: it's the
        # symmetry-breaking that lets otherwise-identical slots specialize.
        mu = self.slots_mu.expand(n, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(n, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots)) * self.scale
            attn_logits = torch.einsum("nkd,nld->nkl", q, k)
            attn = attn_logits.softmax(dim=1)  # competition across slots, per input position
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)  # weights sum to 1 over inputs, per slot
            updates = torch.einsum("nkl,nld->nkd", attn, v)  # attention-weighted mean of inputs
            slots = self.gru(
                updates.reshape(-1, self.slot_dim), slots_prev.reshape(-1, self.slot_dim)
            ).reshape(n, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class FrameEncoder(nn.Module):
    """One settled 64x64 color-index frame -> one token for the sequence
    model. A small conv stack turns the frame into a spatial feature map;
    Slot Attention decomposes that into objects; mean pooling over slots
    collapses them into a single per-frame summary.

    Mean pooling (rather than concatenating the K slots in order) is
    deliberate: Slot Attention doesn't guarantee a consistent slot
    ordering/identity between separate forward calls (this frame's "slot 3"
    isn't necessarily the same object as the previous frame's "slot 3"), so
    concatenation would bind the sequence model to an ordering that isn't
    stable across time. Mean pooling is permutation-invariant, at the cost
    of not giving the transformer object-indexed structure to reason about
    relations between specific objects across steps — revisit if that turns
    out to matter.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.color_embed = nn.Embedding(config.num_colors, config.color_embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(config.color_embed_dim, config.cnn_channels, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.cnn_channels, config.cnn_channels, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.cnn_channels, config.cnn_channels, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.cnn_channels, config.slot_dim, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # Three stride-2 convs: grid_size -> grid_size/8 spatially (64 -> 8).
        feat_size = config.grid_size // 8
        self.pos_embed = nn.Parameter(torch.randn(1, config.slot_dim, feat_size, feat_size) * 0.02)
        self.slot_attention = SlotAttention(config)
        self.slot_to_token = nn.Linear(config.slot_dim, config.d_model)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """grid: [N, H, W] long -> [N, d_model] float (N = batch*time)."""
        x = self.color_embed(grid).permute(0, 3, 1, 2)  # [N, color_embed_dim, H, W]
        x = self.conv(x)  # [N, slot_dim, H/8, W/8]
        x = x + self.pos_embed
        n, c, hf, wf = x.shape
        x = x.flatten(2).transpose(1, 2)  # [N, Hf*Wf, slot_dim]
        slots = self.slot_attention(x)  # [N, num_slots, slot_dim]
        pooled = slots.mean(dim=1)  # [N, slot_dim]
        return self.slot_to_token(pooled)


# ── 2. Causal transformer over the episode ──────────────────────────────


class RotaryEmbedding(nn.Module):
    """Precomputes RoPE's cos/sin tables for a given sequence length,
    per-call rather than capped at a fixed maximum — see the module
    docstring for why RoPE was chosen over learned absolute positions.
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, num_heads, T, head_dim]; cos/sin: [T, head_dim]."""
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("config.d_model must be divisible by config.num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]; attn_mask: bool [B, 1, T, T], True = allowed to attend."""
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, T, Hd]

        cos, sin = self.rope(t, x.device, x.dtype)
        cos, sin = cos[None, None], sin[None, None]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        hidden = config.d_model * config.ffn_mult
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), attn_mask))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


# ── 3 & 4. Action head + composed network ───────────────────────────────


class ARCPolicyNetwork(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.frame_encoder = FrameEncoder(config)
        # +1 slot in the embedding table for the "start of episode" token
        # (the step whose producing action was RESET, not a policy action).
        self.action_embed = nn.Embedding(config.num_simple_actions + 1, config.d_model)
        self.level_up_embed = nn.Embedding(2, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.action_head = nn.Linear(config.d_model, config.num_simple_actions)
        self.x_head = nn.Linear(config.d_model, config.coord_range)
        self.y_head = nn.Linear(config.d_model, config.coord_range)

    def forward(self, obs: Observation) -> PolicyOutput:
        b, t, h, w = obs.grid.shape
        frame_tokens = self.frame_encoder(obs.grid.reshape(b * t, h, w)).reshape(b, t, -1)
        tokens = (
            frame_tokens + self.action_embed(obs.prev_actions) + self.level_up_embed(obs.level_ups)
        )
        tokens = self.input_norm(tokens)

        attn_mask = self._build_attn_mask(obs.padding_mask)
        x = tokens
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_norm(x)

        action_logits = self.action_head(x)
        action_logits = action_logits.masked_fill(~obs.action_mask, float("-inf"))
        x_logits = self.x_head(x)
        y_logits = self.y_head(x)
        return PolicyOutput(action_logits=action_logits, x_logits=x_logits, y_logits=y_logits)

    @staticmethod
    def _build_attn_mask(padding_mask: torch.Tensor) -> torch.Tensor:
        """padding_mask: bool [B, T], True = real step -> bool [B, 1, T, T],
        True = query i is allowed to attend to key j (causal: j <= i, and j
        must be a real, non-padding step).

        Padding query rows are forced to allow at least the diagonal so
        `scaled_dot_product_attention` never sees an all-False row (which
        would soft-max into NaN); their output is unused downstream since
        training excludes padding positions from the loss.
        """
        t = padding_mask.shape[1]
        causal = ~torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=padding_mask.device), diagonal=1
        )
        key_ok = padding_mask[:, None, None, :]  # [B, 1, 1, T]
        allowed = causal[None, None] & key_ok
        diag = torch.eye(t, dtype=torch.bool, device=padding_mask.device)[None, None]
        return allowed | diag
