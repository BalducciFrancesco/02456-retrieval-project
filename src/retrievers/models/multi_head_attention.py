import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int):
        super().__init__()
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.d = emb_dim // n_heads  # numbers per token, per head
        self.W_q = nn.Linear(emb_dim, emb_dim)
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_o = nn.Linear(emb_dim, emb_dim)  # mixes the heads at the end

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, T, _ = x.shape                       # [batch, T, emb_dim]
        x = x.view(batch, T, self.n_heads, self.d)  # [batch, T, heads, d]
        #  transpose done because before the matrix was grouped by token
        #  instead of by head  but q,k,v should be grouped by head
        return x.transpose(1, 2)                    # [batch, heads, T, d]

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        batch, T, _ = x.shape                 # [batch, T, emb_dim]
        q = self.separate_heads(self.W_q(x))  # [batch, heads, T, d]
        k = self.separate_heads(self.W_k(x))  # [batch, heads, T, d]
        v = self.separate_heads(self.W_v(x))  # [batch, heads, T, d]

        # [batch, heads, T, d] @ [batch, heads, d, T] -> [batch, heads, T, T]
        # we transpose because we want the final two dimensions to be tokenxtoken
        scores = q @ k.transpose(-2, -1) / self.d ** 0.5
        if attention_mask is not None:
            # mask [batch, T] -> [batch, 1, 1, T], so it covers every head and row
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # [batch, heads, T, T], rows sum to 1
        projected_v = weights @ v                # [batch, heads, T, d]
        #group again by token and we flatten the head outputs into one vector per token
        z = projected_v.transpose(1, 2).reshape(batch, T, -1)  # [batch, T, heads, d] -> [batch, T, emb_dim]
        return self.W_o(z), weights              # [batch, T, emb_dim], [batch, heads, T, T]
