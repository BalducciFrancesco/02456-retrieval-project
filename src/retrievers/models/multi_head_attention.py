import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d = emb_dim // n_heads  # numbers per token, per head
        self.W_q = nn.Linear(emb_dim, emb_dim)
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_o = nn.Linear(emb_dim, emb_dim)  # mixes the heads at the end

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, T, _ = x.shape
        return x.view(batch, T, self.n_heads, self.d).transpose(1, 2)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        batch, T, _ = x.shape
        q = self.split_heads(self.W_q(x))
        k = self.split_heads(self.W_k(x))
        v = self.split_heads(self.W_v(x))

        scores = q @ k.transpose(-2, -1) / self.d ** 0.5
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        z = (weights @ v).transpose(1, 2).reshape(batch, T, -1)  # put the heads side by side again
        return self.W_o(z), weights
