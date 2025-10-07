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
    # def forward (x : Tensor B S E) (mask : Tensor S S) : Tensor B S d_h
    # let Q : Tensor B S d_h := x ⬝ W_Q
    # let K : Tensor B S d_h := x ⬝ W_K
    # let V : Tensor B S d_h := x ⬝ W_V
    # let scores : Tensor B S S := Q ⬝ Kᵀ / √(d_h)
    # let masked : Tensor B S S := scores ⊙ mask  -- mask broadcast over B
    # let attn : Tensor B S S := softmax masked
    # in attn ⬝ V
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: batch_size x seq_len x emb_dim
        # Q, K, V: batch_size x seq_len x d_h
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        Q, V = rope(Q), rope(V)
        # attention scores
        # swaps the last two dims of K: batch_size x d_h x seq_len
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_h)
        # apply the causal mask
        masked_scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = torch.softmax(masked_scores, dim=-1)
        # ^ each row is a probability distribution
        # output: batch_size x seq_len x d_h
        return attention_weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int) -> None:
        super().__init__()
        d_h = emb_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(emb_dim, d_h) for _ in range(num_heads)])
        # learnable projection matrix
        self.W_O = nn.Parameter(torch.empty(emb_dim, emb_dim))

    # Lean4 (shape):
    # def forward (x : Tensor B S E) (mask : Tensor S S) : Tensor B S E
    # let heads : List (Tensor B S d_h) := map (·.forward x mask) self.heads
    # let concat : Tensor B S (H * d_h) := cat heads  -- H = |self.heads|
    # in concat ⬝ W_O  -- (H * d_h = E)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: batch_size x seq_len x emb_dim
        head_outputs = [head(x, mask) for head in self.heads]
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
    # def forward (x : Tensor B S E) (mask : Tensor S S) : Tensor B S E
    # let a : Tensor B S E := self.attn (self.norm1 x) mask
    # let x₁ : Tensor B S E := x + a
    # let m : Tensor B S E := self.mlp (self.norm2 x₁)
    # in x₁ + m
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), mask)
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
    # let x : Tensor B S E := embedding tok
    # let mask : Tensor S S := tril(1)  -- broadcast over B
    # let y : Tensor B S E := foldl (λ acc blk => blk.forward acc mask) x self.layers
    # in y ⬝ output  -- projects to V
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        _, seq_len, _ = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        for layer in self.layers:
            x = layer(x, mask)
        return x @ self.output


def rope(x: torch.Tensor, theta_base: float = 10000.0) -> torch.Tensor:
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

    # Generate sequence position indices
    pos = torch.arange(0, seq_len, dtype=torch.float32, device=x.device)
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
