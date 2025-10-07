import math

import torch
import torch.nn as nn


# Lean 4 pseudo-typing guide for this file
# variables (B S E H d_h V : Nat)
# -- E = emb_dim, H = num_heads, d_h = E / H, V = vocab_size
# -- Tensor B S E denotes torch.Tensor with shape [B, S, E]

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    A simplified alternative to Layer Normalization that only uses RMS statistics
    """

    def __init__(self, emb_dim: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))  # Learnable scale parameter
        self.epsilon = epsilon  # Small constant for numerical stability

    # Lean4 (shape):
    # def forward (x : Tensor B S E) : Tensor B S E
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute root mean square normalization
        squared_x = x ** 2
        mean_squared = torch.mean(squared_x, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_squared + self.epsilon)

        # Normalize and scale
        x_normalized = x / rms
        output = x_normalized * self.scale
        return output


class AttentionHead(nn.Module):
    def __init__(self, emb_dim: int, d_h: int) -> None:
        super().__init__()
        self.W_Q = nn.Parameter(torch.eye(emb_dim, d_h))
        self.W_K = nn.Parameter(torch.eye(emb_dim, d_h))
        self.W_V = nn.Parameter(torch.eye(emb_dim, d_h))
        self.d_h: int = d_h

    # Lean4 (shape):
    # def forward (x : Tensor B S E) (mask : Tensor S S) (kv_cache? : Option {k,v}) : Tensor B S d_h
    # Supports optional KV cache for incremental decoding. When provided, appends
    # new K/V to the cache and uses all cached keys/values for attention.
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        kv_cache: dict | None = None,
    ) -> torch.Tensor:
        # x: [B, S, E]
        # project queries/keys/values for the new tokens only
        Q_new = x @ self.W_Q  # [B, S, d_h]
        K_new = x @ self.W_K  # [B, S, d_h]
        V_new = x @ self.W_V  # [B, S, d_h]

        # Determine how many tokens are already cached (per head)
        past_len = 0
        if kv_cache is not None and kv_cache.get("k") is not None:
            past_len = kv_cache["k"].size(1)

        # Apply RoPE with a position offset to Q and V to match sequence positions
        # (K remains unrotated here to stay consistent with the existing implementation.)
        Q_new = rope(Q_new, pos_offset=past_len)
        V_new = rope(V_new, pos_offset=past_len)

        # Concatenate with cached K/V if provided
        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                K = torch.cat([kv_cache["k"], K_new], dim=1)
                V = torch.cat([kv_cache["v"], V_new], dim=1)
            else:
                K, V = K_new, V_new
            # Update cache in-place
            kv_cache["k"], kv_cache["v"] = K, V
        else:
            K, V = K_new, V_new

        # Attention scores: Q_new attends over all keys (cached + new)
        scores = Q_new @ K.transpose(-2, -1) / math.sqrt(self.d_h)  # [B, S, S_total]

        # If a mask is provided and shapes differ (e.g., during cached decoding),
        # build a correct causal mask for [S_query x S_total].
        if mask is None or mask.shape[-2] != scores.shape[-2] or mask.shape[-1] != scores.shape[-1]:
            # Build causal mask that permits attending to all past tokens plus
            # current position for each query row.
            B, S_q, S_total = scores.shape
            device = scores.device
            # Each row t (0..S_q-1) can attend up to index past_len + t
            i = (torch.arange(S_q, device=device).unsqueeze(1) + past_len)  # [S_q,1]
            j = torch.arange(S_total, device=device).unsqueeze(0)  # [1,S_total]
            causal = (j <= i).to(scores.dtype)  # [S_q,S_total]
        else:
            causal = mask.to(scores.dtype)

        masked_scores = scores.masked_fill(causal == 0, float("-inf"))
        attention_weights = torch.softmax(masked_scores, dim=-1)

        # output for the new tokens only, using all values
        return attention_weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int) -> None:
        super().__init__()
        d_h = emb_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(emb_dim, d_h) for _ in range(num_heads)])
        # learnable projection matrix
        self.W_O = nn.Parameter(torch.empty(emb_dim, emb_dim))

    # Lean4 (shape):
    # def forward (x : Tensor B S E) (mask : Tensor S S) (kv_cache? : Option (Array H {k,v})) : Tensor B S E
    # Each head receives its own optional cache dict {k,v}.
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, kv_cache: list | None = None) -> torch.Tensor:
        # x: [B, S, E]
        head_outputs = []
        for i, head in enumerate(self.heads):
            cache_i = None
            if kv_cache is not None:
                # Expect a list of length num_heads containing dicts for 'k'/'v'
                cache_i = kv_cache[i]
            head_outputs.append(head(x, mask, kv_cache=cache_i))
        x = torch.cat(head_outputs, dim=-1)
        return x @ self.W_O


class MLP(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(emb_dim, emb_dim * 4))
        self.B_1 = nn.Parameter(torch.empty(emb_dim * 4))
        self.W_2 = nn.Parameter(torch.empty(emb_dim * 4, emb_dim))
        self.B_2 = nn.Parameter(torch.empty(emb_dim))

    # Lean4 (shape):
    # def forward (x : Tensor B S E) : Tensor B S E
    # let h : Tensor B S (4*E) := relu (x ⬝ W_1 + B_1)
    # in h ⬝ W_2 + B_2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.W_1 + self.B_1
        x = torch.relu(x)
        x = x @ self.W_2 + self.B_2
        return x


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.norm2 = RMSNorm(emb_dim)
        self.mlp = MLP(emb_dim)

    # Lean4 (shape):
    # def forward (x : Tensor B S E) (mask : Tensor S S) (kv_cache? : Option (Array H {k,v})) : Tensor B S E
    # let a := attn (norm1 x) mask kv_cache
    # let x₁ := x + a; in x₁ + mlp (norm2 x₁)
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, kv_cache: list | None = None) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), mask, kv_cache=kv_cache)
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x


class DecoderLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_heads: int, num_blocks: int, pad_idx: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.layers = nn.ModuleList([DecoderBlock(emb_dim, num_heads) for _ in range(num_blocks)])
        self.output = nn.Parameter(torch.rand(emb_dim, vocab_size))

    # Lean4 (shape):
    # def forward (tok : Tensor B S) : Tensor B S V
    # Training path (no cache): regular causal mask.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        _, seq_len, _ = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        for layer in self.layers:
            x = layer(x, mask, kv_cache=None)
        return x @ self.output

    # Incremental path with KV cache. Expects `kv_caches` to be a list with one
    # element per decoder block; each element is a list (len=num_heads) of dicts
    # with optional 'k'/'v' tensors. Pass only the new tokens in `x`.
    @torch.no_grad()
    def forward_with_cache(self, x: torch.Tensor, kv_caches: list[list[dict]]) -> torch.Tensor:
        # x: [B, S_new] token ids
        x = self.embedding(x)  # [B,S_new,E]
        # Build a causal mask for [S_new x (past+S_new)] on the fly in heads
        mask = None  # constructed per-head based on cache lengths
        for layer, cache_for_layer in zip(self.layers, kv_caches):
            x = layer(x, mask, kv_cache=cache_for_layer)
        return x @ self.output


def rope(x: torch.Tensor, theta_base: float = 10000.0, pos_offset: int = 0) -> torch.Tensor:
    # Lean4 (shape):
    # def rope (x : Tensor B S E_even) (theta_base : Float := 10000.0) : Tensor B S E_even
    # split x into x1,x2 : Tensor B S (E/2); rotate pairs with sin/cos; recombine
    """
    Implements Rotary Position Embedding (RoPE) for transformer attention.
    RoPE encodes position information through rotation matrices applied to pairs of dimensions.

    Args:
        x: Input tensor of shape (batch_size, seq_len, emb_dim)
        theta_base: Base for computing rotation frequencies (default: 10000.0)

    Returns:
        Tensor with position information encoded through rotations
    """
    batch_size, seq_len, emb_dim = x.size()
    assert emb_dim % 2 == 0, "Embedding dimensionality must be even for RoPE"

    # Generate sequence position indices with optional offset
    pos = torch.arange(pos_offset, pos_offset + seq_len, dtype=torch.float32, device=x.device)
    pos = pos.unsqueeze(0).expand(batch_size, seq_len)

    # Compute frequency bands for each dimension pair
    # Modified: frequencies start from p=1 and use (p-1) in exponent
    p = torch.arange(1, emb_dim // 2 + 1, dtype=torch.float32, device=x.device)
    theta_p = 1.0 / (theta_base ** (2 * (p - 1) / emb_dim))

    # Compute rotation angles for each position and frequency
    pos = pos.unsqueeze(-1)
    theta = pos * theta_p

    # Compute rotation components
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    # Split input into alternating dimensions
    x1 = x[..., 0::2]  # Dimensions at indices 0,2,4,...
    x2 = x[..., 1::2]  # Dimensions at indices 1,3,5,...

    # Apply 2D rotations to each pair
    x_rotated_1 = x1 * cos_theta - x2 * sin_theta
    x_rotated_2 = x1 * sin_theta + x2 * cos_theta

    # Recombine rotated pairs into final output
    x_rotated = torch.stack((x_rotated_1, x_rotated_2), dim=-1).reshape(batch_size, seq_len, emb_dim)

    return x_rotated
